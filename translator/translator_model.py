import math
import torch
from torch import nn

from config import ModelConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class FeedForward(nn.Module):
    # MLP
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.dim, config.hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config.hidden_dim, config.dim, bias=True)
        self.dropout = nn.Dropout(config.dropout_rate)
 
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(self.dropout(x))
        return x

class EncoderBlock(nn.Module):
    # or we can call it as Transformer Block
    # according to the paper
    # Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm
    # Position Embedding is outside the Transformer Block
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_dropout = nn.Dropout(config.dropout_rate)
        
        self.feed_forward = FeedForward(config)
        self.feed_forward_norm = RMSNorm(config.dim, config.norm_eps)
        self.feed_forward_dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, x, e_mask):
        # we can choose whether to apply normalization before or after attention
        # while llama3 apply normalization before attention
        x_norm = self.attention_norm(x)
        x = x + self.attention_dropout(self.attention(x_norm, x_norm, x_norm, e_mask))

        x_norm = self.feed_forward_norm(x)
        x = x + self.feed_forward_dropout(self.feed_forward(x_norm))
        return x

class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = config.n_encoder_layers
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(self.layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
    
    def forward(self, x, e_mask):
        for i in range(self.layers):
            x = self.blocks[i](x, e_mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Masked self-attention
        self.self_att = Attention(config)
        self.self_att_norm = RMSNorm(config.dim, config.norm_eps)
        self.self_att_dropout = nn.Dropout(config.dropout_rate)

        # Cross-attention
        self.cross_att = Attention(config)
        self.cross_att_norm = RMSNorm(config.dim, config.norm_eps)
        self.cross_att_dropout = nn.Dropout(config.dropout_rate)

        # Feed Forward
        self.feed_forward = FeedForward(config)
        self.feed_forward_norm = RMSNorm(config.dim, config.norm_eps)
        self.feed_forward_dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, e_output, d_mask, e_mask):
        # Masked Self-Attention
        x_norm = self.self_att_norm(x)
        x = x + self.self_att_dropout(self.self_att(x_norm, x_norm, x_norm, d_mask))

        # Cross-Attention
        x_norm = self.cross_att_norm(x)
        x = x + self.cross_att_dropout(self.cross_att(x_norm, e_output, e_output, e_mask))

        # Feed Forward
        x_norm = self.feed_forward_norm(x)
        x = x + self.feed_forward_dropout(self.feed_forward(x_norm))

        return x

class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = config.n_decoder_layers
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(self.layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x, e_output, d_mask, e_mask):
        for i in range(self.layers):
            x = self.blocks[i](x, e_output, d_mask, e_mask)
        return self.norm(x)




class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        # W_q, W_k, W_v
        self.wq = nn.Linear(config.dim, config.dim)
        self.wk = nn.Linear(config.dim, config.dim)
        self.wv = nn.Linear(config.dim, config.dim)

        # TODO 做什么用的
        self.wo = nn.Linear(config.dim, config.dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, q, k, v, mask=None):
        '''
        q: [batch_size, seq_len, dim]
        k: [batch_size, seq_len, dim]
        v: [batch_size, seq_len, dim]
        '''
        batch_size = q.shape[0]
        q, k, v = self.wq(q), self.wk(k), self.wv(v)
        
        # [batch_size, seq_len, dim] -> [batch_size, seq_len, n_heads, head_dim]
        # according to below comment, we want [batch_size, n_heads, seq_len, head_dim], so why can't we just view as this shape?
        # TODO
        q = q.view(batch_size, -1, self.n_heads, self.head_dim)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim)

        # For llama3, RoPE here, and without Tensor.transpose()

        # [batch_size, seq_len, n_heads, head_dim] -> [batch_size, n_heads, seq_len, head_dim]
        # transpose is hard to understand, we can think in this way:
        # the dimension of tensor has its own meaning
        # like q[a][b][c] means a-th sequence, b-th query vector in this sequence, in c-th head
        # now we transpose 1-th dimension and 2-th dimension, q[a][b][c] -> q[a][c][b]
        # so it is easy to understand, new tensor will be [batch_size, n_heads, seq_len, head_dim]
        # because we care the seq_len * head_dim as a whole
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_values = self.attention(q, k, v, mask) # [batch_size, n_heads, seq_len, head_dim]
        # concat all the heads
        # since we want to concat, it is a good idea that arrange the sub-tensor one by one in memory
        # that is why we use contiguous()
        # assuming i-th token, we want its sub-tensor in all heads arranged one by one in memory
        # so sub-tensor of i-th token from different heads are contiguous in memory
        concat_output = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.dim) # [batch_size, seq_len, dim]

        return self.wo(concat_output)
        
    def attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim) # Q * K^T / sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1e9)
        
        attn_dist = self.softmax(attn_scores) # softmax(Q * K^T / sqrt(d_k))
        attn_dist = self.dropout(attn_dist)
        attn_values = torch.matmul(attn_dist, v) # softmax(Q * K^T / sqrt(d_k)) * V
        return attn_values

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig, src_pad_id, tgt_pad_id, device):
        super().__init__()
        self.src_vocab_size = config.src_vocab_size
        self.tgt_vocab_size = config.tgt_vocab_size
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.device = device
        self.max_seq_len = config.max_seq_len
        
        self.src_embedding = nn.Embedding(self.src_vocab_size, config.dim)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, config.dim)
        
        self.encoder = Encoder(config)
        
        self.decoder = Decoder(config)

        self.softmax = nn.Softmax(dim=-1)


    
    def generate_subsequent_mask(self, size):
        "Generates an upper-triangular matrix of -inf, with zeros on diag."
        return torch.tril(torch.ones((1, size, size), device=self.device, dtype=torch.bool)) ##################################
    
    def forward(self, src, tgt):
        '''
        src: [batch_size, seq_len], e.g. [[1, 2, 3, 4], [5, 6, 7, 8]] `1` can refer to token `I`
        tgt: [batch_size, seq_len], e.g. [[1, 2, 3, 4], [5, 6, 7, 8]] `1` can refer to token `我`
        '''
        src = self.src_embedding(src) # [batch_size, seq_len, dim]
        # positional embadding
        tgt = self.tgt_embedding(tgt) # [batch_size, seq_len, dim]

        src_pad_mask = (src != self.src_pad_id).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_seq_len]
        tgt_pad_mask = (tgt != self.tgt_pad_id).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_seq_len]

        tgt_seq_len = self.max_seq_len
        subsequent_mask = self.generate_subsequent_mask(tgt_seq_len)  # [1, tgt_seq_len, tgt_seq_len]
        subsequent_mask = subsequent_mask.unsqueeze(1)  # [1, 1, tgt_seq_len, tgt_seq_len]

        d_mask = tgt_pad_mask & (subsequent_mask)

        # Encoder output
        e_output = self.encoder(src, src_pad_mask)  # [batch_size, src_seq_len, dim]

        # Decoder output
        d_output = self.decoder(tgt, e_output, d_mask, src_pad_mask)

        # Output layer
        output = self.softmax(d_output)  # [batch_size, tgt_seq_len, tgt_vocab_size]

        return output
