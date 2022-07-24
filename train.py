from datetime import datetime

import transformers
from datasets import load_dataset
from transformers import XLMRobertaTokenizer, DataCollatorForLanguageModeling, XLMRobertaConfig, XLMRobertaForMaskedLM, \
    TrainingArguments, SchedulerType, IntervalStrategy

import wandb
from data import EnlmrCombinedDataset, EnlmrLanguageSpecificDataset, ConvertedDataset


class Trainer:

    def __init__(self):
        self.batch_size = 2048
        self.max_token = 512

    def load_datasets(self, tokenizer):
        en_train_raw_ds = load_dataset('text', data_files="train.en", split='train')
        ne_train_raw_ds = load_dataset('text', data_files="train.ne", split='train')
        en_valid_raw_ds = load_dataset('text', data_files="valid.en", split='train')
        ne_valid_raw_ds = load_dataset('text', data_files="valid.ne", split='train')

        en_train_ds = EnlmrLanguageSpecificDataset('English Train Dataset', en_train_raw_ds, tokenizer)
        en_valid_ds = EnlmrLanguageSpecificDataset('English Valid Dataset', en_valid_raw_ds, tokenizer, is_valid=True)
        ne_train_ds = EnlmrLanguageSpecificDataset('Nepali Train Dataset', ne_train_raw_ds, tokenizer)
        ne_valid_ds = EnlmrLanguageSpecificDataset('Nepali Valid Dataset', ne_valid_raw_ds, tokenizer, is_valid=True)

        train_ds = EnlmrCombinedDataset([en_train_ds, ne_train_ds], batch_size=self.batch_size)
        valid_ds = ConvertedDataset(EnlmrCombinedDataset([en_valid_ds, ne_valid_ds], batch_size=self.batch_size))

        return train_ds, valid_ds

    def train(self):
        tokenizer = XLMRobertaTokenizer('enlm-r.spm.model',
                                        sp_model_kwargs={'enable_sampling': True, 'nbest_size': 64, 'alpha': 0.1},
                                        model_max_length=self.max_token, name_or_path='enlm-r-base')
        train_ds, valid_ds = self.load_datasets(tokenizer)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
        config = XLMRobertaConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=514, type_vocab_size=1,
                                  layer_norm_eps=1e-05, output_past=True)
        model = XLMRobertaForMaskedLM(config=config)

        run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        training_args = TrainingArguments(
            max_steps=50000,
            # fp16=True,

            output_dir="checkpoints",
            overwrite_output_dir=True,
            warmup_steps=10000,
            optim='adamw_torch',
            logging_steps=10,
            eval_steps=50,
            save_steps=50,
            tpu_num_cores=8,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-06,
            evaluation_strategy='steps',
            weight_decay=0.01,
            learning_rate=0.0006,
            gradient_accumulation_steps=16,
            eval_accumulation_steps=16,
            push_to_hub=True,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            lr_scheduler_type=SchedulerType.POLYNOMIAL,
            logging_dir='logs',
            logging_strategy=IntervalStrategy.STEPS,
            save_strategy='steps',
            save_total_limit=4,
            seed=1,
            run_name=run_name,
            report_to=['wandb'],
            hub_model_id="enlm-r",
            hub_strategy='checkpoint',
            hub_token='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',
            hub_private_repo=True,
        )

        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            tokenizer=tokenizer,
        )

        trainer.train(resume_from_checkpoint="checkpoints/last-checkpoint")


def main():
    wandb.init(project="enlm-r", id="3i7jxf5t", resume="must")
    trainer = Trainer()
    trainer.train()


import os

os.environ['XLA_USE_BF16'] = "1"
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'
os.environ["WANDB_DISABLED"] = "false"

if __name__ == '__main__':
    main()
