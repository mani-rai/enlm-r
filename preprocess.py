import os.path

from transformers import XLMRobertaTokenizer
from datasets import load_dataset

from data import EnlmrLanguageSpecificDataset

def preprocess(dataset, output_file_name):
    if not os.path.exists("data/prod/processed"):
        os.makedirs("data/prod/processed")
    with open("data/processed/" + output_file_name, "w", encoding="utf-8") as f:
        for sample in dataset:
            line = ""
            for code in sample:
                line += str(code) + " "
            f.write(line)
            f.write("\n")

def main():
    tokenizer = XLMRobertaTokenizer('sentencepiece/enlm-r.spm.model',
                                    sp_model_kwargs={'enable_sampling': True, 'nbest_size': 64, 'alpha': 0.1},
                                    model_max_length=512, name_or_path='enlm-r-base')
    en_train_raw_ds = load_dataset('text', data_files="train.en", split='train')
    en_train_ds = EnlmrLanguageSpecificDataset('English Train Dataset', en_train_raw_ds, tokenizer, buffer_size=1,
                                               is_valid=True)
    preprocess(en_train_ds, "data/test/processed/train.en")

    # ne_train_raw_ds = load_dataset('text', data_files="train.ne", split='train')
    # ne_train_ds = EnlmrLanguageSpecificDataset('Nepali Train Dataset', ne_train_raw_ds, tokenizer, buffer_size=1,
    #                                            is_valid=True)
    # preprocess(ne_train_ds, "train.ne")
    #
    # en_valid_raw_ds = load_dataset('text', data_files="valid.en", split='train')
    # en_valid_ds = EnlmrLanguageSpecificDataset('English Valid Dataset', en_valid_raw_ds, tokenizer, is_valid=True,
    #                                            buffer_size=1)
    # preprocess(en_valid_ds, "valid.en")
    #
    # ne_valid_raw_ds = load_dataset('text', data_files="valid.ne", split='train')
    # ne_valid_ds = EnlmrLanguageSpecificDataset('Nepali Valid Dataset', ne_valid_raw_ds, tokenizer, is_valid=True,
    #                                            buffer_size=1)
    # preprocess(ne_valid_ds, "valid.ne")

if __name__ == '__main__':
    main()