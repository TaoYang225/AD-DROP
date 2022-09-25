from datasets import load_dataset, load_metric
from self_transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
import torch
import numpy as np

raw_data = load_dataset("wmt16", "tr-en")
metric = load_metric("sacrebleu")
source_language = "tr"
target_language = "en"
model_name = 'Helsinki-NLP/opus-mt-tr-en'

if model_name in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "translate English to Romanian: "
else:
    prefix = ""
seed = 42
# meteor = load_metric('meteor')

# customizing compute_metrics function to display bleu score, mean prediction length and meteor score
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result




tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False)


max_input_length = 128
max_target_length = 128

def preprocess(instances):
   input = [prefix + i[source_language] for i in instances["translation"]]
   target = [i[target_language] for i in instances["translation"]]
   tokenized_inputs = tokenizer(input, max_length=max_input_length, truncation=True)
   # Setup the tokenizer for target
   with tokenizer.as_target_tokenizer():
       label = tokenizer(target, max_length=max_target_length, truncation=True)
   tokenized_inputs["labels"] = label["input_ids"]
   return tokenized_inputs

#Applying the pre processing on the entire dataset
tokenized_datasets = raw_data.map(preprocess, batched=True)
# train_dataset = tokenized_datasets["train"].shuffle(seed=seed).select(range(100))
# # eval_dataset = tokenized_datasets["validation"].shuffle(seed=seed).select(range(100))
# # test_dataset = tokenized_datasets["test"].shuffle(seed=seed).select(range(100))
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"]

print(len(train_dataset))
print(len(eval_dataset))
print(len(test_dataset))

# device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, output_attentions=True)
# model.to(device)
batch_size = 16

#defining training attributes
args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy = "epoch",
    save_strategy = 'no',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=5,
    g_dropout=0.6,
    keep_rate=0.9,
    predict_with_generate=True,
    do_mask=True,
)

#pad inputs and label them
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

#training object with customized parameters
trainer = Seq2SeqTrainer(
   model,
   args,
   train_dataset=train_dataset,
   eval_dataset=test_dataset,
   data_collator=data_collator,
   tokenizer=tokenizer,
   compute_metrics=compute_metrics
)

trainer.train()

