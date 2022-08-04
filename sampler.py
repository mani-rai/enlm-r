from typing import List, Union, Iterable

from math import ceil
from torch.utils.data import Sampler


class BatchSampler(Sampler[List[int]]):

    def __init__(self, samplers: List[Union[Sampler[int], Iterable[int]]], cumulative_sizes, batch_size: int,
                 gradient_accumulation_steps: int) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.samplers = samplers
        self.batch_size = batch_size
        self.cumulative_sizes = cumulative_sizes
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def __iter__(self):
        sampler_indexes = list(range(len(self.samplers)))
        sampler_iterators = [iter(sampler) for sampler in self.samplers]
        sampler_indexes_idx = 0
        gradient_accumulation_steps_count = 0
        while True:
            if len(sampler_indexes) == 0:
                return
            batch = []
            if gradient_accumulation_steps_count >= self.gradient_accumulation_steps:
                if (sampler_indexes_idx + 1) < len(sampler_indexes):
                    sampler_indexes_idx += 1
                    gradient_accumulation_steps_count = 0
                else:
                    sampler_indexes_idx = 0
                    gradient_accumulation_steps_count = 0
            sampler_index = sampler_indexes[sampler_indexes_idx]
            sampler_iter = sampler_iterators[sampler_index]
            offset = 0 if sampler_index == 0 else self.cumulative_sizes[sampler_index - 1]
            while len(batch) < self.batch_size:
                try:
                    sample_index = next(sampler_iter)
                    sample_index += offset
                    batch.append(sample_index)
                except StopIteration:
                    sampler_indexes.remove(sampler_index)
                    gradient_accumulation_steps_count = self.gradient_accumulation_steps
                    break
            if len(batch) == 0:
                continue
            gradient_accumulation_steps_count += 1
            yield batch

    def __len__(self):
        length = 0
        for sampler in self.samplers:
            length += ceil(len(sampler) / self.batch_size)
        return length
