import torch
from tokenizer import Tokenizer
from translator_model import Transformer
from config import ModelConfig
from torch import nn

class Translator:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = Tokenizer('./data/sp/src_sp.model', './data/sp/tgt_sp.model')
        self.model_config = ModelConfig(
            src_vocab_size=self.tokenizer.src_vocab_size,
            tgt_vocab_size=self.tokenizer.tgt_vocab_size,
        )
        self.transformer = Transformer(self.model_config, self.tokenizer.src_pad_id, self.tokenizer.tgt_pad_id, self.device)

    def train(self, train_dataloader, num_epochs=10):
        self.transformer.train()
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.tgt_pad_token_id)

        for epoch in range(num_epochs):
            for batch in train_dataloader:
                src_batch = batch['src'].to(self.device)
                tgt_batch = batch['tgt'].to(self.device)

                # Prepare input and target sequences
                tgt_input = tgt_batch[:, :-1]  # Exclude the last token
                tgt_target = tgt_batch[:, 1:]  # Exclude the first token

                # # Create masks
                # src_pad_mask = (src_batch != self.tokenizer.src_pad_token_id).unsqueeze(1).unsqueeze(2)
                # tgt_pad_mask = (tgt_input != self.tokenizer.tgt_pad_token_id).unsqueeze(1).unsqueeze(2)
                # tgt_seq_len = tgt_input.size(1)
                # subsequent_mask = torch.tril(torch.ones((1, 1, tgt_seq_len, tgt_seq_len), device=self.device)).bool()
                # tgt_mask = tgt_pad_mask & subsequent_mask

                # # Forward pass
                # optimizer.zero_grad()
                # encoder_output = self.transformer.encoder(src_batch, src_pad_mask)
                # decoder_output = self.transformer.decoder(tgt_input, encoder_output, tgt_mask, src_pad_mask)
                # output_logits = self.transformer.output_layer(decoder_output)

                optimizer.zero_grad()
                output_logits = self.transformer(src_batch, tgt_input)

                # Compute loss
                # Notice that here we are using crossentropy loss. In this case, we cannot add softmax inside the transformer,
                # since crossentropy requires raw logits
                
                loss = criterion(output_logits.view(-1, self.tokenizer.tgt_vocab_size), tgt_target.contiguous().view(-1))

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

        # Save the trained model
        torch.save(self.transformer.state_dict(), 'trained_transformer.pth')

    def _preprocess_sequence(self, sequence):
        if len(sequence) < self.model_config.max_seq_len:
            sequence += [self.tokenizer.src_pad_id] * (self.model_config.max_seq_len - len(sequence))
        else:
            sequence = sequence[:self.model_config.max_seq_len]
        return sequence
        
    def translate(self, src_sentence: str) -> str:
        src_tokens = self.tokenizer.encode_src(src_sentence, False, False)
        src_tokens = torch.LongTensor(self._preprocess_sequence(src_tokens)).unsqueeze(0).to(self.device)
        src_pad_mask = (src_tokens != self.tokenizer.src_pad_id).unsqueeze(1).to(self.device)
        src_tokens = self.transformer.src_embedding(src_tokens)
        # src_tokens = self.transformer.positional_encoding(src_tokens)
        e_output = self.transformer.encoder(src_tokens, src_pad_mask)
        
        result_sequence = torch.LongTensor([self.tokenizer.tgt_pad_id]).to(self.device)
        result_sequence[0] = self.tokenizer.tgt_bos_id
        cur_len = 1
        
        for i in range(self.model_config.max_seq_len):
            d_mask = (result_sequence.unsqueeze(0) != self.tokenizer.tgt_pad_id).unsqueeze(1).to(self.device)
            nopeak_mask = torch.ones([1, self.model_config.max_seq_len, self.model_config.max_seq_len], dtype=torch.bool).to(self.device)
            nopeak_mask = torch.tril(nopeak_mask)
            d_mask = d_mask & nopeak_mask
            
            tgt_tokens = self.transformer.tgt_embedding(result_sequence.unsqueeze(0))
            # tgt_tokens = self.transformer.positional_encoding(tgt_tokens)
            decoder_output = self.transformer.decoder(tgt_tokens, e_output, d_mask, src_pad_mask)
            
            output = self.transformer.softmax()
            output = torch.argmax(output, dim=-1)
            latest_word_id = output[0][i].item()
            
            if i < self.model_config.max_seq_len - 1:
                result_sequence[i + 1] = latest_word_id
                cur_len += 1
            if latest_word_id == self.tokenizer.tgt_eos_id:
                break

        return self.tokenizer.decode_tgt(result_sequence)
    
if __name__ == '__main__':
    
    
    translator = Translator()
    
    input_sentence = "I am a student."
    translated_sentence = translator.translate(input_sentence)
    print(translated_sentence)