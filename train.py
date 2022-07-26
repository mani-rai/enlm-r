from datasets import load_dataset
from torch.utils.data import ConcatDataset
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    XLMRobertaConfig, XLMRobertaTokenizer, SchedulerType, AutoModelForMaskedLM, )

from data import EnlmrDataset
from trainer import Trainer
import os
# import torch_xla.distributed.xla_multiprocessing as xmp

max_token = 512
batch_size = 8192
num_of_devices = 4
per_device_batch_size = 16
data_dir = "data"
gradient_accumulation_steps = batch_size // (per_device_batch_size * num_of_devices)
num_train_epochs = 20
logging_steps = 10
save_steps = 97
eval_steps = 160
report_to = ["tensorboard"]
push_to_hub = True
resume_from_checkpoint = False


def main():
    # Loading datasets
    en_train_raw_ds = load_dataset('text', data_files=data_dir + "/train.en", split='train')
    en_preprocessed_train_ds = EnlmrDataset("English Train Dataset", en_train_raw_ds, max_token)
    ne_train_raw_ds = load_dataset('text', data_files=data_dir + "/train.ne", split='train')
    ne_preprocessed_train_ds = EnlmrDataset("Nepali Train Dataset", ne_train_raw_ds, max_token)
    en_valid_raw_ds = load_dataset('text', data_files=data_dir + "/valid.en", split='train')
    en_preprocessed_valid_ds = EnlmrDataset("English Valid Dataset", en_valid_raw_ds, max_token)
    ne_valid_raw_ds = load_dataset('text', data_files=data_dir + "/valid.ne", split='train')
    ne_preprocessed_valid_ds = EnlmrDataset("Nepali Valid Dataset", ne_valid_raw_ds, max_token)
    train_ds = ConcatDataset([en_preprocessed_train_ds, ne_preprocessed_train_ds])
    valid_ds = ConcatDataset([en_preprocessed_valid_ds, ne_preprocessed_valid_ds])

    # Setting up the model
    tokenizer = XLMRobertaTokenizer('sentencepiece/enlm-r.spm.model',
                                    sp_model_kwargs={'enable_sampling': False, 'nbest_size': 0, 'alpha': 0.1},
                                    model_max_length=max_token, name_or_path='enlm-roberta')
    training_args = TrainingArguments(
        output_dir="outputs",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        prediction_loss_only=True,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=0.00006,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-06,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type=SchedulerType.POLYNOMIAL,
        warmup_ratio=0.0,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        tpu_num_cores=8,
        eval_steps=eval_steps,
        ignore_data_skip=True,
        optim="adamw_torch",
        report_to=report_to,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
        push_to_hub=push_to_hub,
        hub_model_id="enlm-roberta-130",
        hub_strategy="end",
        hub_token='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',
        skip_memory_metrics=False,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )
    config = XLMRobertaConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=514, type_vocab_size=1,
                              layer_norm_eps=1e-05, output_past=True)
    # model = XLMRobertaForMaskedLM(config=config)
    model = AutoModelForMaskedLM.from_pretrained('manirai91/enlm-roberta-final');

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.push_to_hub:
        commit_message = f"Training in progress, step {trainer.state.global_step}"
        trainer.push_to_hub(commit_message=commit_message)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    os.environ['XLA_USE_BF16'] = '1'
    os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '1000000000'
    main()


if __name__ == "__main__":
    # xmp.spawn(_mp_fn, args=(), nprocs=8, start_method='fork')
    main()
