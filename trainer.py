from typing import Optional

import torch
import transformers
from packaging import version
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.trainer_pt_utils import DistributedSamplerWithLoop
from transformers.trainer_utils import seed_worker, has_length
from transformers.training_args import ParallelMode

from sampler import BatchSampler

_is_torch_generator_available = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True


class Trainer(transformers.Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_samplers = self._get_train_samplers()
        batch_sampler = BatchSampler(train_samplers, self.train_dataset.cumulative_sizes,
                                     self.args.per_device_train_batch_size, self.args.gradient_accumulation_steps)

        return DataLoader(
            self.train_dataset,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
            batch_sampler=batch_sampler
        )

    def _get_train_samplers(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        samplers = []
        if self.args.world_size <= 1:
            if _is_torch_generator_available:
                for train_dataset in self.train_dataset.datasets:
                    samplers.append(RandomSampler(train_dataset, generator=generator))
                return samplers
            else:
                for train_dataset in self.train_dataset.datasets:
                    samplers.append(RandomSampler(train_dataset))
                return samplers
        elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
        ):
            # Use a loop for TPUs when drop_last is False to have all batches have the same size.
            for train_dataset in self.train_dataset.datasets:
                samplers.append(
                    DistributedSamplerWithLoop(
                        train_dataset,
                        batch_size=self.args.per_device_train_batch_size,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                    )
                )
            return samplers
        else:
            for train_dataset in self.train_dataset.datasets:
                samplers.append(
                    DistributedSampler(
                        train_dataset,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                    )
                )
            return samplers

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        eval_samplers = self._get_eval_samplers(eval_dataset)
        batch_sampler = BatchSampler(eval_samplers, self.eval_dataset.cumulative_sizes,
                                     self.args.per_device_eval_batch_size, self.args.gradient_accumulation_steps)

        return DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            batch_sampler=batch_sampler
        )

    def _get_eval_samplers(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        samplers = []
        if self.args.world_size <= 1:
            if _is_torch_generator_available:
                for dataset in eval_dataset.datasets:
                    samplers.append(RandomSampler(dataset, generator=generator))
                return samplers
            else:
                for dataset in eval_dataset.datasets:
                    samplers.append(RandomSampler(dataset))
                return samplers
        elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
        ):
            # Use a loop for TPUs when drop_last is False to have all batches have the same size.
            for dataset in eval_dataset.datasets:
                samplers.append(
                    DistributedSamplerWithLoop(
                        dataset,
                        batch_size=self.args.per_device_train_batch_size,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                    )
                )
            return samplers
        else:
            for dataset in eval_dataset.datasets:
                samplers.append(
                    DistributedSampler(
                        dataset,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                    )
                )
            return samplers
