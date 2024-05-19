
import random
import torch
from torch import cuda
import numpy as np
import os
import shutil 
os.chdir('/home/ulhaq/project/cpda')

from transformers import AutoTokenizer, DataCollatorForTokenClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk, load_dataset, DatasetDict, Dataset, concatenate_datasets
import wandb
import argparse
from PIL import Image

from utils.models import *
from utils.eval_metrics import *
from utils.processing import *
from utils.early_stopping import *

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# %%

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--shuffle_seed', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--dataset_size', type=int, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--weight_decay', type=float, required=True)
parser.add_argument('--num_epochs', type=int, required=True)
parser.add_argument('--early_stopping_tolerance', type=int, required=True)
parser.add_argument('--early_stopping_threshold', type=float, required=True)
parser.add_argument('--jobname', type=str,  required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--project', type=str, required=True)
parser.add_argument('--lamda', type=float, required=True)
parser.add_argument('--logger', type=str, required=True)

args = parser.parse_args()

# %%

model_save_path = '/home/ulhaq/project/cpda/models/'

jobname = f'{args.jobname}'

if args.logger=='True':
    wandb.init(
    project=f"{args.project}",
    name=f"{jobname}"
    )

device = 'cuda' if cuda.is_available() else "cpu"
print('Running on ...', device)
print('Lamda .... ', args.lamda)
print('Logging....', args.logger)
##### Set seed for reproducibility

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(args.seed)

###### Load Dataset ############

dataset = load_dataset(args.dataset)

if args.shuffle_seed is not 0:
    dataset = dataset.shuffle(seed=args.shuffle_seed)
dataset = dataset.filter(lambda e, i: i<args.dataset_size, with_indices=True)

label_list = dataset['train'].features['ner_tags'].feature.names
label2id = {tag: id for id, tag in enumerate(label_list)}     
id2label =  {id: tag for tag, id in label2id.items()}
entities = [x[2:] for x in label_list[1:]]
entities = sorted(set(entities), key=entities.index)
prompt = [f'<{x}>' for x in entities]
print('Prompt Tokens ....', prompt)


# Define TOKENIZER

checkpoint = args.model
tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)
tokenizer.add_tokens(prompt)       

# Load DATASET
dataset_processor = DatasetProcessor(dataset)
dataset = dataset.map(dataset_processor.add_prompt, fn_kwargs={'id2label': id2label, 'prompt': prompt, 'tokenizer': tokenizer})
dataset = DatasetProcessor.dataset_extend(dataset)

print('Dataset', dataset)
# %%

# Define MODEL
prompt_mapping = {}
encodings = []
for entity in prompt:
    entity_encoding = tokenizer.encode(entity, add_special_tokens=False)
    encodings.append(entity_encoding[0])

model = roberta_mlm.from_pretrained(checkpoint, encodings)
model.resize_token_embeddings(len(tokenizer))

for idx, entity in enumerate(prompt,-len(prompt)):
    entity_encoding = tokenizer.encode(entity, add_special_tokens=False)
    print(idx, entity, entity_encoding)
    prompt_mapping.update({f"{entity}": entity_encoding[0]})

    with torch.no_grad():
        model.roberta.embeddings.word_embeddings.weight[idx, :] += model.roberta.embeddings.word_embeddings.weight[entity_encoding[0], :].clone()    
model.to(device)


# TOKENIZE Data
tokenized = dataset.map(tokenize_and_align_labels, fn_kwargs={'id2label': id2label, 'prompt_mapping': prompt_mapping, 'tokenizer': tokenizer},  batched=True,)
tokenized.set_format('torch', columns=[
                     "input_ids", "attention_mask", "masked_input_ids", "tagged_sent"])
# %%
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


train_dataloader = DataLoader(
    tokenized['train'],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=args.batch_size,
)

eval_dataloader = DataLoader(
    tokenized['validation'], collate_fn=data_collator, batch_size=args.batch_size
)

optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

num_train_epochs = args.num_epochs
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps,
)

metric = evaluate.load("seqeval")
early_stopping = EarlyStopping(tolerance=args.early_stopping_tolerance, min_delta=args.early_stopping_threshold)

# %%

val_losses = []
previous_loss = 0

for epoch in range(num_train_epochs):
    # Training
    model.train()
    train_loss=0
    positive_loss_train=0
    negative_loss_train=0
    contrastive_loss_train=0
    mlm_loss_train=0
    embeddings_raw = []
    anchors = []
    print(f"Training Epochs {epoch}....")
    for batch in tqdm(train_dataloader):
        batch.to(device)
        input_ids = batch['masked_input_ids']
        labels = batch['input_ids']
        attention_mask = batch['attention_mask']
        tagged = batch['tagged_sent']
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, tagged=tagged, lamda=args.lamda)
        loss = outputs['loss_total']
        loss.backward()
        train_loss += loss
        contrastive_loss_train += outputs['loss_contrastive']
        positive_loss_train += outputs['positive_similarity']
        negative_loss_train += outputs['negative_similarity']
        mlm_loss_train += outputs['mlm_loss']
        optimizer.step()
        lr_scheduler.step()

        if args.logger=='True':
            wandb.log({
                    "train_loss": outputs['loss_total'],
                    "loss_contrastive_train": outputs['loss_contrastive'],
                    "similarity_positive_train": outputs['positive_similarity'],
                    "similarity_negative_train": outputs['negative_similarity'],
                    "mlm_loss_train": outputs['mlm_loss'],
                    })

    model.eval()
    val_loss = 0
    positive_loss_eval=0
    negative_loss_eval=0
    contrastive_loss_eval=0
    mlm_loss_eval=0

    print("Evaluating....")
    for batch in tqdm(train_dataloader):
        batch.to(device)
        input_ids = batch['masked_input_ids']
        labels = batch['input_ids']
        attention_mask = batch['attention_mask']
        tagged = batch['tagged_sent']
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, tagged=tagged, lamda=args.lamda)
            logits = outputs['logits']
        eval_loss = outputs['loss_total']
        contrastive_loss_eval += outputs['loss_contrastive']
        positive_loss_eval += outputs['positive_similarity']
        negative_loss_eval += outputs['negative_similarity']
        mlm_loss_eval += outputs['mlm_loss']
        val_loss += eval_loss

        if args.logger=='True':
            wandb.log({"eval_loss": outputs['loss_total'],
                    "loss_contrastive_eval": outputs['loss_contrastive'],
                    "similarity_positive_eval": outputs['positive_similarity'],
                    "similarity_negative_eval": outputs['negative_similarity'],
                    "mlm_loss_eval": outputs['mlm_loss'],
                    })

    if args.logger=='True':
         wandb.log({
                    "epoch": epoch,
                    })
                
    early_stopping(train_loss, val_loss)
    if val_loss.item() < previous_loss:
       print(f'saving best model at epoch {epoch}')
       model.save_pretrained(model_save_path + f'/{args.project}' + f'/{jobname}' + f'/model_best')
    previous_loss = val_loss.item()

    if early_stopping.early_stop:
      print(f"Final model at  Epoch : {epoch}")
      break
