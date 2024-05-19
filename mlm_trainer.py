
import random
import torch
from torch import cuda
import numpy as np
import os
import shutil 
os.chdir('/home/ulhaq/project/contrastive_loss')

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
parser.add_argument('--jobname', type=str,  required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--project', type=str, required=True)
parser.add_argument('--lamda', type=float, required=True)
parser.add_argument('--logger', type=str, required=True)

args = parser.parse_args()

# %%

save_path = '/home/ulhaq/project/contrastive_loss/models/'

jobname = f'{args.jobname}'
print(args.logger)

if args.logger=='True':
    wandb.init(
    project=f"{args.project}",
    name=f"{jobname}"
    )

device = 'cuda' if cuda.is_available() else "cpu"
print(device)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(args.seed)

# %% WNUT 17 dataset]
dataset = load_dataset(args.dataset)
if args.shuffle_seed is not 0:
    dataset = dataset.shuffle(seed=args.shuffle_seed)
dataset_org = dataset.filter(lambda e, i: i<args.dataset_size, with_indices=True)
label_list = dataset['train'].features['ner_tags'].feature.names

label2id = {tag: id for id, tag in enumerate(label_list)}     
id2label =  {id: tag for tag, id in label2id.items()}
label_names = label_list.copy()
entities = [x[2:] for x in label_names[1:]]
entities = sorted(set(entities), key=entities.index)
prompt = [f'<{x}>' for x in entities]
print(prompt)

def process(examples):
    tokens, ner_tags = examples['tokens'], examples['ner_tags']
    labels = [id2label[x] for x in ner_tags]
    tagged_sent = tokens.copy()
    tokens_org = []
    ner_tags_org = []
    masked_ner_tags_org = []
    labels_org = []
    tagged_sent_org = []
    masked_tokens_org = []

    for idx, (s, l) in enumerate(zip(tokens, labels)):
        if l != 'O':
            tagged_sent[idx] = f'<{l[2:]}>'
            
    special_tokens = [x for x in tagged_sent if x in prompt]
    special_tokens = sorted(set(special_tokens), key=special_tokens.index)
    tokens_org_ = []
    if special_tokens == []:
        tokens_org.append(tokens)
        ner_tags_org.append(ner_tags)
        masked_ner_tags_org.append(ner_tags) ### If there are no tags maskedsent==tagged_sent
        
    for special_token in special_tokens:
        
        ##### Masked Sentences #####
        masked_tokens_temp = [tokenizer.mask_token if token in special_token else token for token in tagged_sent]
        special_token_indexes = [i for i, token in enumerate(masked_tokens_temp) if token == tokenizer.mask_token]
        ########################
        
        _masked_ner_tags = [tag if idx in special_token_indexes else 0 for idx, tag in enumerate(ner_tags)]
        _masked_ner_tags = [-100] + _masked_ner_tags
        masked_ner_tags_org.append(_masked_ner_tags)

        _ner_tags_org = [-100] + ner_tags
        ner_tags_org.append(_ner_tags_org)
        
        tokens_temp = [special_token] + tokens
        tokens_org.append(tokens_temp)
        
    examples['tokens']  = tokens_org
    examples['ner_tags']  = ner_tags_org
    examples['masked_ner_tags']  = masked_ner_tags_org

    return examples

dataset = dataset_org.map(process)

# %%

def dataset_process(dataset):
   
    dataset_entity = {}

    for item in dataset:
        tokens = []
        masked_ner_tags = []
        ner_tags = []
        masked_sentences = []
        tagged_sentences = []
        for token, tag, masked_ner_tag in zip(dataset[f'{item}']['tokens'], dataset[f'{item}']['ner_tags'], dataset[f'{item}']['masked_ner_tags']):
            tokens.extend(token)
            ner_tags.extend(tag)
            masked_ner_tags.extend(masked_ner_tag)

        dataset_new_tokens = pd.DataFrame({"tokens": tokens, "ner_tags": ner_tags, "masked_ner_tags":masked_ner_tags})
        dataset_new_tokens = Dataset.from_pandas(dataset_new_tokens)
        dataset_entity.update({f'{item}': dataset_new_tokens
                })
        
    dataset_dict = DatasetDict({
        'train': dataset_entity['train'],
        'validation':  dataset_entity['validation'],
        'test': dataset_entity['test'],
            })
        
    return dataset_dict

dataset = dataset_process(dataset)
print(dataset)
# %%

prompt_mapping = {}
checkpoint = args.model
tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)
tokenizer.add_tokens(prompt)       
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

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], max_length=128, truncation=True, is_split_into_words=True, padding='max_length')
    
    labels = []
    masked_input_ids = []
    masked_labels = []
    tagged_sent = []
    attn_mask = []
    label_input_ids= []
    for i, (label, tokens, masked_ner_tag) in enumerate(zip(examples[f"ner_tags"], examples['tokens'], examples['masked_ner_tags'])):
        # Map tokens to their respective word.
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        input_ids = torch.tensor(tokenized_inputs.input_ids[i])
        previous_word_idx = None
        label_ids = []
        masked_label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
                masked_label_ids.append(-100)
            # Only label the first token of a given word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
                masked_label_ids.append(masked_ner_tag[word_idx])
            else:
                label_ids.append(0)
                masked_label_ids.append(0)
            previous_word_idx = word_idx
                   
        label_ids = torch.tensor(label_ids)
        labels.append(label_ids)
        
        masked_label_ids = torch.tensor(masked_label_ids)
        
        mask = (masked_label_ids != -100) & (masked_label_ids != 0)
        masked_id = [tokenizer.mask_token_id if m else i.item() for i, m in zip(input_ids, mask)]
        masked_input_ids.append(masked_id) 
        masked_labels.append(masked_label_ids)

        tags = [f'<{id2label[x.item()][2:]}>' if x != -100 and x != 0 else "O" for x in label_ids]
        tagged_sent_temp = [prompt_mapping[word] if word in prompt_mapping else 0 for word in tags]
        tagged_sent.append(tagged_sent_temp)
                
    tokenized_inputs["masked_input_ids"] = torch.tensor(masked_input_ids)
    tokenized_inputs["tagged_sent"] = torch.tensor(tagged_sent)
    
    return tokenized_inputs

# %%

tokenized = dataset.map(tokenize_and_align_labels, batched=True)
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
early_stopping = EarlyStopping(tolerance=10, min_delta=0.03)

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
        embeddings_raw.append(outputs['positive_tokens_dict'])
        anchors.append(outputs['anchors_dict'])
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
       model.save_pretrained(save_path + f'/{args.project}' + f'/{jobname}' + f'/model_best')
    previous_loss = val_loss.item()

    if early_stopping.early_stop:
      print(f"Final model at  Epoch : {epoch}")
      break
