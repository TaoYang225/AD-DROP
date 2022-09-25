import numpy as np
from datasets import load_dataset, load_metric
from self_transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer
from self_transformers import DataCollatorForTokenClassification
import torch

task = 'ner'
model_checkpoint = 'google/electra-base-discriminator'
# model_checkpoint = "../../../aModel/bert-base-uncased"
batch_size = 32

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
datasets = load_dataset("conll2003")
label_list = datasets["train"].features[f"{task}_tags"].feature.names
metric = load_metric("seqeval")
label_all_tokens = True

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list), output_attentions=True)

args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy = "epoch",
    save_strategy = 'no',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=9,
    g_dropout=0.8,
    keep_rate=0.9,
    seed = 526,
    # predict_with_generate=True,
    do_mask=True,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train(ignore_keys_for_eval=['attentions'])

# trainer.evaluate(ignore_keys=['attentions'])
#
# predictions, labels, _ = trainer.predict(tokenized_datasets["test"], ignore_keys=['attentions'])
# predictions = np.argmax(predictions, axis=2)
#
# # Remove ignored index (special tokens)
# true_predictions = [
#     [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
#     for prediction, label in zip(predictions, labels)
# ]
# true_labels = [
#     [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
#     for prediction, label in zip(predictions, labels)
# ]
#
# results = metric.compute(predictions=true_predictions, references=true_labels)
# print('test result')
# print(results)

