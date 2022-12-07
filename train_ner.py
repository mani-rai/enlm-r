import datasets
import numpy as np
from datasets import load_metric
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification

# Load datasets
conll2003_ds = datasets.load_dataset("conll2003", split="train+validation+test")
conll2003_ds = conll2003_ds.train_test_split(test_size=0.1, shuffle=False)

raw_train_ds = conll2003_ds['train']
raw_eval_ds = conll2003_ds['test']
raw_test_ds = datasets.load_dataset("manirai91/ebiquity-v2", split="train")

max_tokens = 512
# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
# tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
tokenizer = AutoTokenizer.from_pretrained("manirai91/enlm-roberta-130")

# Preprocessing
def preprocess(samples):
    tokenized_inputs = tokenizer(samples['tokens'], is_split_into_words=True, max_length=max_tokens, truncation=True)
    labels = []
    for i, label in enumerate(samples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

train_ds = raw_train_ds.map(preprocess, batched=True)
eval_ds = raw_eval_ds.map(preprocess, batched=True)
test_ds = raw_test_ds.map(preprocess, batched=True)

# Loading metric
label_list = train_ds.features[f'ner_tags'].feature.names
seqeval_metic = load_metric("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metic.compute(predictions=true_predictions, references=true_labels)
    return {
        "accuracy": results["overall_accuracy"],
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"]
    }

# Load model and tokenizer
# model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=9)
# model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=9)
model = AutoModelForTokenClassification.from_pretrained("manirai91/enlm-roberta-130", num_labels=9)

# Defining data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding='max_length', max_length=max_tokens)

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
    hub_model_id="enlm-roberta-130-conll2003",
    hub_strategy="end",
    hub_token="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",
    do_train=True,
    do_eval=True,
    load_best_model_at_end=True,
    metric_for_best_model="ne_f1",
    greater_is_better=True,
    lr_scheduler_type="linear"
)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset={"en": eval_ds, "ne": test_ds},
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

train_result = trainer.train(training_args.resume_from_checkpoint)
metrics = train_result.metrics
metrics["train_samples"] = len(train_ds)

trainer.save_model()  # Saves the tokenizer too for easy upload

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
