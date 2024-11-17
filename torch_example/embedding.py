import torch
from torch import nn

if __name__ == '__main__':
    embedding = nn.Embedding(10, 2) # 一共有10个词，每个词用2维词向量表示
    input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]]) # 两个句子，每个句子有4个词
    output = embedding(input)
    print(output.size()) # sentence_num * seq_len * embedding_dim
    # 实际使用中，每次输入的sentence_num是预设的batch_size