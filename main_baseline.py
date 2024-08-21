
# %%
from transformers import DataCollatorForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModel, AutoConfig, AutoTokenizer, TrainingArguments, Trainer
from transformers.modeling_utils import PreTrainedModel

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import cuda

import wandb
import random
import argparse
import numpy as np
import os 
import shutil 

from datasets import load_from_disk, Dataset, DatasetDict

from utils.models import *
from utils.eval_metrics import *
from utils.processing import *
from utils.early_stopping import *


import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--weight_decay', type=float, required=True)
parser.add_argument('--num_epochs', type=int, required=True)
parser.add_argument('--es_threshold', type=float, required=True)
parser.add_argument('--es_tolerance', type=int, required=True)
parser.add_argument('--dropout', type=float, required=True)
parser.add_argument('--jobname', type=str,  required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--tag', type=str, required=True)
parser.add_argument('--project', type=str, required=True)
parser.add_argument('--sequence_length', type=int, required=True)
parser.add_argument('--model_type_train', type=str, required=True)
parser.add_argument('--model_type_test', type=str, required=True)


args = parser.parse_args()


# %%

def load_model_class_train(model_type):
    if model_type == "model_baseline":
        return model_baseline
    else:
        raise ValueError("Invalid model type. Please specify a valid model type.")


def load_model_class_test(model_type):
    if model_type == "model_baseline":
        return model_baseline        
    else:
        raise ValueError("Invalid model type. Please specify a valid model type.")

# %%

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

checkpoint = args.model
save_path = 'path to save models'
path = args.path

jobname = f'{args.jobname}_{args.seed}'

wandb.init(
  project=f"{args.project}",
  tags=[f"{args.tag}"],
  name=f"{jobname}"
)

def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

set_seed(args.seed)

# %%

dataset = load_dataset(args.dataset)
label_list = dataset['train'].features['ner_tags'].feature.names
label2id = {tag: id for id, tag in enumerate(label_list)}     
id2label =  {id: tag for tag, id in label2id.items()}
label_names = label_list.copy()
# dataset = dataset.shuffle(seed=42)
print(dataset)

# %%
config = AutoConfig.from_pretrained(checkpoint)
model_class_train = load_model_class_train(args.model_type_train)
model = model_class_train(checkpoint, config,                      ##### Check model
                              num_labels =len(label_list),
                              id2label = id2label,
                              label2id = label2id,
                              dropout = args.dropout)

model.to(device)

# %%
if args.model[:7] == 'roberta':
      tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
else:
      tokenizer = AutoTokenizer.from_pretrained(args.model)

# %%
def tokenize_and_align_labels(examples, sequence_length):
    tokenized_inputs = tokenizer(
        examples["tokens"], max_length=sequence_length, truncation=True, is_split_into_words=True, padding='max_length')
    
    labels = []
    
    for i, (label) in enumerate(examples[f"ner_tags"]):
        # Map tokens to their respective word.
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            # Only label the first token of a given word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(0)     ### check to calculate on subwords
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels

    return tokenized_inputs

# %%
tokenized = dataset.map(tokenize_and_align_labels, fn_kwargs={'sequence_length': args.sequence_length},  batched=True)
tokenized.set_format(
    'torch', columns=["input_ids", "attention_mask", "labels"])
# %%
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
# %%

metric = evaluate.load("seqeval")

train_dataloader = DataLoader(
    tokenized['train'],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=args.batch_size,
)

eval_dataloader = DataLoader(
   tokenized['validation'] , collate_fn=data_collator, batch_size=args.batch_size
)

# %%

from torch.optim import AdamW
optimizer = AdamW(model.parameters(),betas=(0.9, 0.999), lr=args.lr, weight_decay=args.weight_decay)

# %%

from transformers import get_scheduler

num_train_epochs = args.num_epochs
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps,
)

# %%

early_stopping = EarlyStopping(tolerance=args.es_tolerance, min_delta=args.es_threshold)

# %%
from tqdm.auto import tqdm
eval_f1_score = []
for epoch in tqdm(range(num_train_epochs)):
    # Training
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        batch.to(device)
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        train_loss += loss
        optimizer.step()
        lr_scheduler.step()

    # Evaluation
    model.eval()
    val_loss = 0
    for batch in eval_dataloader:
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)

        eval_loss = outputs.loss
        val_loss += eval_loss

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        true_predictions, true_labels = postprocess(predictions, labels, label_list)

        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    
    wandb.log({"eval_loss": val_loss,
                "train_loss": train_loss,
                "eval_f1": results['overall_f1'],
                "eval_precision": results['overall_precision'],
                "eval_recall": results['overall_recall'],
                "epoch": epoch})
    
    model.save_pretrained(save_path + f'/{args.project}' + f'/{jobname}' + f'/model_{epoch}')
    eval_f1_score.append(results['overall_f1'])

    early_stopping(train_loss, val_loss)
    if early_stopping.early_stop:
      print(f"Final model at  Epoch : {epoch}")
      break

best_f1_score_epoch = np.array(eval_f1_score).argmax()
print(f"Best epoch {best_f1_score_epoch}")
checkpoint = save_path + f'/{args.project}' + f'/{jobname}' +  f'/model_{best_f1_score_epoch}'

# %% TEST MODE

metric = evaluate.load("seqeval")
model_class_test = load_model_class_test(args.model_type_test)
config = AutoConfig.from_pretrained(checkpoint)
model = model_class_test(checkpoint, config,
                              num_labels=len(label_list),
                              id2label = id2label,
                              label2id = label2id,
                              dropout = args.dropout,
                              )

model.load_state_dict(torch.load(checkpoint + '/pytorch_model.bin'))
model.to(device)
model.eval()

# %%

test_dataloader5e-5 = DataLoader(
   tokenized['test'] , collate_fn=data_collator, batch_size=32)

for batch in test_dataloader:
    batch = { k: v.to(device) for k, v in batch.items() }
    with torch.no_grad():
        outputs = model(**batch)

    predictions = outputs.logits.argmax(dim=-1)
    labels = batch["labels"]

    true_predictions, true_labels = postprocess(predictions, labels, label_list)
    metric.add_batch(predictions=true_predictions, references=true_labels)
    
results_test = metric.compute()

wandb.log({
        "test_f1": results_test['overall_f1']*100,
        "test_precision": results_test['overall_precision']*100,
        "test_recall": results_test['overall_recall']*100,
        "results_test":results_test,
        "test_epoch": epoch-3})

metric = evaluate.load("seqeval")
batch = []

for batch in eval_dataloader:
    batch = { k: v.to(device) for k, v in batch.items() }
    with torch.no_grad():
        outputs = model(**batch)

    predictions = outputs.logits.argmax(dim=-1)
    labels = batch["labels"]

    true_predictions, true_labels = postprocess(predictions, labels, label_list)
    metric.add_batch(predictions=true_predictions, references=true_labels)

results_eval = metric.compute()

wandb.log({
        "best_eval_f1": results_eval['overall_f1']*100,
        "best_eval_epoch": epoch-3})

# %%

os.listdir(save_path + f'/{args.project}' + f'/{jobname}' )
shutil.rmtree(save_path + f'/{args.project}' + f'/{jobname}')
