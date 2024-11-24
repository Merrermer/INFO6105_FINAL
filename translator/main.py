from tokenizer import Tokenizer
from translator_model import Transformer
from config import ModelConfig
import torch

class Translator:
    def __init__(self):
        self.tokenizer = Tokenizer('./data/sp/src_sp.model', './data/sp/tgt_sp.model')
        self.model_config = ModelConfig(
            src_vocab_size=self.tokenizer.src_vocab_size,
            tgt_vocab_size=self.tokenizer.tgt_vocab_size,
        )
        self.transformer = Transformer(self.model_config)
    
    def train(self):
        pass
        
    def translate(self, src_sentence: str) -> str:
        tokens = self.tokenizer.encode_src(src_sentence, True, True)
        print(tokens)
        res = self.transformer.forward(tokens)

        res = self.tokenizer.decode_tgt(tokens)
        print(res)
        return "Default translated sentence"
    
if __name__ == '__main__':
    
    
    translator = Translator()
    
    input_sentence = "I am a student."
    translated_sentence = translator.translate(input_sentence)
    print(translated_sentence)