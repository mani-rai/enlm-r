import datasets
import numpy as np
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

max_tokens = 512

imdb_ds = datasets.load_dataset("imdb", split="train+test")
imdb_ds = imdb_ds.train_test_split(test_size=0.1, shuffle=False)

raw_train_ds = imdb_ds['train']
raw_valid_ds = imdb_ds['test']
raw_test_ds = datasets.load_dataset("manirai91/yt-nepali-movie-reviews", split="train")

# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
# tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
tokenizer = AutoTokenizer.from_pretrained("manirai91/enlm-roberta-final")


def tokenize(samples):
    return tokenizer(samples["text"], truncation=True, max_length=max_tokens)


tokenized_train_ds = raw_train_ds.map(tokenize, batched=True)
tokenized_valid_ds = raw_valid_ds.map(tokenize, batched=True)
tokenized_test_ds = raw_test_ds.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=max_tokens)
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
# model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained("manirai91/enlm-roberta-final", num_labels=2)

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


epochs = 10
learning_rate = 1e-5
batch_size = 32
num_of_devices = 2
per_device_batch_size = 16
gradient_accumulation_steps = batch_size // (per_device_batch_size * num_of_devices)
warmup_ratio = 0.06
save_strategy = 'epoch'
save_total_limit = 3
logging_steps = 20
report_to = ["tensorboard"]
push_to_hub = True
resume_from_checkpoint = False

training_args = TrainingArguments(
    output_dir="outputs",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=per_device_batch_size,
    per_device_eval_batch_size=per_device_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    weight_decay=0.1,
    num_train_epochs=epochs,
    warmup_ratio=warmup_ratio,
    logging_steps=logging_steps,
    save_strategy=save_strategy,
    save_total_limit=save_total_limit,
    tpu_num_cores=num_of_devices,
    optim="adamw_torch",
    report_to=report_to,
    push_to_hub=push_to_hub,
    resume_from_checkpoint=resume_from_checkpoint,
    hub_model_id="enlm-roberta-imdb-final",
    hub_strategy="end",
    hub_token="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",
    do_train=True,
    do_eval=True,
    load_best_model_at_end=True,
    metric_for_best_model="ne_accuracy",
    greater_is_better=True,
    lr_scheduler_type="linear"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset={"en": tokenized_valid_ds, "ne": tokenized_test_ds},
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

train_result = trainer.train(training_args.resume_from_checkpoint)
metrics = train_result.metrics
metrics["train_samples"] = len(tokenized_train_ds)

trainer.save_model()  # Saves the tokenizer too for easy upload

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
