from random import choice
from typing import List, Union, Iterable

from datasets import load_dataset
from math import ceil
from torch.utils.data import Sampler, RandomSampler


class BatchSampler(Sampler[List[int]]):

    def __init__(self, samplers: List[Union[Sampler[int], Iterable[int]]], batch_size: int) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.samplers = samplers
        self.batch_size = batch_size

    def __iter__(self):
        sampler_indexes = list(range(len(self.samplers)))
        sampler_iterators = [iter(sampler) for sampler in self.samplers]
        while True:
            if len(sampler_indexes) == 0:
                return
            batch = []
            sampler_index = choice(sampler_indexes)
            sampler_iter = sampler_iterators[sampler_index]
            offset = 0
            for index in range(sampler_index):
                offset += len(self.samplers[index])
            while len(batch) < self.batch_size:
                try:
                    sample_index = next(sampler_iter)
                    sample_index += offset
                    batch.append(sample_index)
                except StopIteration:
                    sampler_indexes.remove(sampler_index)
                    break
            if len(batch) == 0:
                continue
            yield batch

    def __len__(self):
        length = 0
        for sampler in self.samplers:
            length += ceil(len(sampler) / self.batch_size)
        return length


def main():
    en_train_raw_ds = load_dataset('text', data_files="data/train.en", split='train')
    ne_train_raw_ds = load_dataset('text', data_files="data/train.ne", split='train')

    en_rand_sampler = RandomSampler(en_train_raw_ds)
    ne_rand_sampler = RandomSampler(ne_train_raw_ds)

    batch_sampler = iter(BatchSampler([en_rand_sampler, ne_rand_sampler], 5))

    overall = []
    count = 0
    for batch in batch_sampler:
        print("Batch max min:", min(batch), max(batch), len(batch))
        count += 1
        overall += batch

    print(count, max(overall), min(overall), len(set(overall)))


if __name__ == '__main__':
    main()
