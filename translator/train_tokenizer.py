import os
from data import ds
from sentencepiece import SentencePieceTrainer
from tqdm import tqdm

def extract_text(dataset, src_lang='en', tgt_lang='zh', split='train'):
    src_texts = []
    tgt_texts = []
    for item in dataset[split]:
        src_texts.append(item['translation'][src_lang])
        tgt_texts.append(item['translation'][tgt_lang])
    return src_texts, tgt_texts

def train_tokenizer(input_file, model_prefix, vocab_size=16000, character_coverage=1.0):
    SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type='unigram',
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
        train_extremely_large_corpus=True,
        shuffle_input_sentence=True,
        max_sentence_length=10000,
        # input_sentence_size=25984574,
        input_sentence_size=10000,
    )
def main():
    if not os.path.exists('./data/sp'):
        os.makedirs('./data/sp')
    src_text_file = './data/src.txt'
    tgt_text_file = './data/tgt.txt'
    if not os.path.exists(src_text_file) or not os.path.exists(tgt_text_file):
        src_texts, tgt_texts = extract_text(ds)
        
        print("Writing source texts...")
        with open('./data/src.txt', 'w', encoding='utf-8') as f:
            for text in tqdm(src_texts, desc="Writing source texts"):
                f.write(text + '\n')

        print("Writing target texts...")
        with open('./data/tgt.txt', 'w', encoding='utf-8') as f:
            for text in tqdm(tgt_texts, desc="Writing target texts"):
                f.write(text + '\n')
    train_tokenizer(
        input_file=src_text_file,
        model_prefix='data/sp/src_sp',
    )
    train_tokenizer(
        input_file=tgt_text_file,
        model_prefix='data/sp/tgt_sp',
    )
if __name__ == '__main__':
    main()