import os
from datetime import datetime

import transformers
from datasets import load_dataset
from transformers import XLMRobertaTokenizer, DataCollatorForLanguageModeling, XLMRobertaConfig, XLMRobertaForMaskedLM, \
    TrainingArguments, SchedulerType, IntervalStrategy

import wandb
from data import EnlmrCombinedDataset, EnlmrLanguageSpecificDataset, ConvertedDataset


class Trainer:

    def __init__(self):
        self.batch_size = 256
        self.max_token = 512

    def load_datasets(self, tokenizer):
        en_ds = load_dataset('text', data_files="data/cc100-en-4gb.txt", split='train').train_test_split(test_size=0.1,
                                                                                                         shuffle=False)
        ne_ds = load_dataset('text', data_files="data/cc100-ne-4gb.txt", split='train').train_test_split(test_size=0.1,
                                                                                                         shuffle=False)

        en_train_ds = EnlmrLanguageSpecificDataset('English Train Dataset', en_ds['train'], tokenizer)
        en_valid_ds = EnlmrLanguageSpecificDataset('English Valid Dataset', en_ds['test'], tokenizer, is_valid=True)
        ne_train_ds = EnlmrLanguageSpecificDataset('Nepali Train Dataset', ne_ds['train'], tokenizer)
        ne_valid_ds = EnlmrLanguageSpecificDataset('Nepali Valid Dataset', ne_ds['test'], tokenizer, is_valid=True)

        train_ds = EnlmrCombinedDataset([en_train_ds, ne_train_ds], batch_size=self.batch_size)
        valid_ds = ConvertedDataset(EnlmrCombinedDataset([en_valid_ds, ne_valid_ds], batch_size=self.batch_size))

        return train_ds, valid_ds

    def train(self):
        tokenizer = XLMRobertaTokenizer('sentencepiece/enlm-r.spm.model',
                                        sp_model_kwargs={'enable_sampling': True, 'nbest_size': 64, 'alpha': 0.1},
                                        model_max_length=self.max_token, name_or_path='enlm-r-base')
        train_ds, valid_ds = self.load_datasets(tokenizer)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
        config = XLMRobertaConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=514, type_vocab_size=1,
                                  layer_norm_eps=1e-05, output_past=True)
        model = XLMRobertaForMaskedLM(config=config)

        run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        training_args = TrainingArguments(
            output_dir="checkpoints",
            overwrite_output_dir=True,
            warmup_steps=1000,
            max_steps=10000,
            logging_steps=1000,
            save_steps=200,
            bf16=True,
            # fp16=True,
            tpu_num_cores=8,
            push_to_hub=True,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,

            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-06,
            evaluation_strategy='steps',
            weight_decay=0.01,
            learning_rate=0.0006,
            lr_scheduler_type=SchedulerType.POLYNOMIAL,
            log_level='info',
            logging_dir='logs',
            logging_strategy=IntervalStrategy.STEPS,
            save_strategy='steps',
            save_total_limit=20,
            seed=1,
            run_name=run_name,
            load_best_model_at_end=True,
            greater_is_better=False,
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

        trainer.train()


def main():
    wandb.init(project="enlm-r")
    trainer = Trainer()
    trainer.train()


os.environ["WANDB_DISABLED"] = "false"

if __name__ == '__main__':
    main()
