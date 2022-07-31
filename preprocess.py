import os.path
import sys

from datasets import load_dataset
from transformers import XLMRobertaTokenizer

from data import EnlmrLanguageSpecificDataset


def preprocess(dataset, output_file_name):
    if not os.path.exists(os.path.dirname(output_file_name)):
        os.makedirs(os.path.dirname(output_file_name))
    with open(output_file_name, "w", encoding="utf-8") as f:
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
    raw_ds = load_dataset('text', data_files=sys.argv[1], split='train')
    ds = EnlmrLanguageSpecificDataset('Dataset', raw_ds, tokenizer, buffer_size=1, is_valid=True)
    preprocess(ds, sys.argv[2])


if __name__ == '__main__':
    main()
