import logging
from random import choice

import torch
from datasets import load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.utils.data import IterableDataset
from datetime import datetime


class EnlmrLanguageSpecificDataset(IterableDataset):

    def __init__(self, name, dataset, tokenizer, buffer_size=10000, max_token=512, is_valid=False):
        self.name = name
        self.setup_logging()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.max_token = max_token
        self.dataset_iterator = iter(self.dataset)
        self.buffer = []
        self.rem = None
        self.is_valid = is_valid
        self.initialize_buffer()

    def setup_logging(self):
        self.logger = logging.getLogger(self.name)
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                            level=logging.INFO)

    def initialize_buffer(self):
        self.logger.info("Starting buffer initialization.")
        while len(self.buffer) < self.buffer_size:
            sample = self.get_sample()
            if sample is None:
                break
            if len(sample) > self.max_token:
                continue
            self.buffer.append(sample)
        self.logger.info("Buffer initialized.")

    def get_sample(self):
        seq = [self.tokenizer.convert_tokens_to_ids('<s>')]
        bos = True
        eod = False
        while True:
            if self.rem is not None:
                seq = seq + self.rem
                self.rem = None
                bos = False
            try:
                line_sample = next(self.dataset_iterator)
            except StopIteration:
                seq.append(self.tokenizer.convert_tokens_to_ids('</s>'))
                if len(seq) < 3:
                    if self.is_valid:
                        return
                    else:
                        seq = [self.tokenizer.convert_tokens_to_ids('<s>')]
                        self.dataset_iterator = iter(self.dataset)
                        self.logger.info("Dataset ended.")
                        continue
                else:
                    if self.is_valid:
                        return seq
                    else:
                        self.dataset_iterator = iter(self.dataset)
                        continue
            sen = line_sample['text']
            sen = sen.strip()
            if len(sen) == 0:
                if bos:
                    continue
                elif not eod:
                    if len(seq) + 1 <= self.max_token:
                        seq.append(self.tokenizer.convert_tokens_to_ids('</s>'))
                    eod = True
                continue
            encodes = self.tokenizer.encode(sen, add_special_tokens=False)
            encodes.append(self.tokenizer.convert_tokens_to_ids('</s>'))
            if len(encodes) + 1 > self.max_token:
                continue
            eod = False
            bos = False
            if len(seq) + len(encodes) <= self.max_token:
                seq = seq + encodes
            else:
                self.rem = encodes
                break
        return seq

    def __iter__(self):
        while True:
            if len(self.buffer) == 0:
                return
            index = choice(list(range(len(self.buffer))))
            sample = self.buffer[index]
            new_sample = self.get_sample()
            if new_sample is not None:
                self.buffer[index] = new_sample
            else:
                self.buffer = self.buffer[:index] + self.buffer[index + 1:]
            yield sample

    def __next__(self):
        if len(self.buffer) == 0:
            raise StopIteration
        index = choice(list(range(len(self.buffer))))
        sample = self.buffer[index]
        new_sample = self.get_sample()
        if new_sample is not None:
            self.buffer[index] = new_sample
        else:
            self.buffer = self.buffer[:index] + self.buffer[index + 1:]
        return sample


class EnlmrCombinedDataset(IterableDataset):

    def __init__(self, datasets, batch_size=32, max_tokens=512):
        self.datasets = datasets
        self.batch_size = batch_size
        self.max_tokens = max_tokens

    def __iter__(self):
        indexes = list(range(len(self.datasets)))
        index = None
        count = 1
        while True:
            if len(indexes) == 0:
                return
            if count == 1 or index not in indexes or index is None:
                index = choice(indexes)
            try:
                sample = next(self.datasets[index])
            except StopIteration:
                indexes.remove(index)
                continue
            if count >= self.batch_size:
                count = 1
            else:
                count += 1
            sample_tensor = torch.tensor(sample)
            yield {'input_ids': sample_tensor, 'attention_mask': torch.ones_like(sample_tensor)}


class ConvertedDataset:

    def __init__(self, dataset):
        print("Converting dataset.", datetime.now())
        self.dataset = list(dataset)
        print("Dataset converted.", datetime.now())

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    en_raw_ds = load_dataset('text', data_files="data/cc100-10k.en", split='train')
    ne_raw_ds = load_dataset('text', data_files="data/cc100-10k.ne", split='train')

    tokenizer = ByteLevelBPETokenizer('configs/enlm-r-vocab.json', 'configs/enlm-r-merges.txt')

    en_ds = EnlmrLanguageSpecificDataset('en', en_raw_ds, tokenizer)
    ne_ds = EnlmrLanguageSpecificDataset('ne', ne_raw_ds, tokenizer)

    data = iter(EnlmrCombinedDataset([en_ds, ne_ds], tokenizer, 3))

    for sample in data:
        print(tokenizer.decode(sample.tolist()))
