import random
import torch
from torch import cuda
import numpy as np
import os
os.chdir('project directory')

from transformers import AutoTokenizer, DataCollatorForTokenClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import wandb
from datasets import load_from_disk, load_dataset, DatasetDict, Dataset, concatenate_datasets
from random import randrange

from utils.models import *
from utils.eval_metrics import *
from utils.processing import *

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# %%

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--shuffle_seed', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--path_to_save', type=str, required=True)
parser.add_argument('--out_file', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=False)
parser.add_argument('--dataset_size', type=int, required=True)
parser.add_argument('--k', type=int, required=True)
parser.add_argument('--sampling_size', type=int, required=True)
parser.add_argument('--sequence_length', type=int, required=True)
parser.add_argument('--lamda', type=float, required=False)
parser.add_argument('--masking_rate', type=float, required=True)
parser.add_argument('--project', type=str, required=True)
parser.add_argument('--jobname', type=str, required=True)
parser.add_argument('--context_augmentation', type=str, required=True)

args = parser.parse_args()

# %%
device = 'cuda' if cuda.is_available() else "cpu"
print('Running on ...', device)
print('Data Generation .... ')
print('Masking rate....', args.masking_rate)
print('Dataset....', args.dataset)

##### Set seed for Reproducibility

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(args.seed)

###### Load Dataset ############

dataset_base = load_dataset(args.dataset)

if args.shuffle_seed is not 0:
    dataset_small = dataset_base.shuffle(seed=args.shuffle_seed)
dataset_small = dataset_small.filter(lambda e, i: i<args.dataset_size, with_indices=True)

label_list = dataset_base['train'].features['ner_tags'].feature.names
label2id = {tag: id for id, tag in enumerate(label_list)}     
id2label =  {id: tag for tag, id in label2id.items()}
entities = [x[2:] for x in label_list[1:]]
entities = sorted(set(entities), key=entities.index)
prompt = [f'<{x}>' for x in entities]
print('Prompt Tokens ....', prompt)

######################################################################################

checkpoint = args.checkpoint
tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
tokenizer.add_tokens(prompt)

# %%%

def process(examples, masking_rate):
    
    sent, tags = examples['tokens'], examples['ner_tags']
    label = [id2label[x] for x in tags]
    masked_sent = sent.copy()
    
    # Calculate the number of tokens to mask (30% of the length of the sentence)
    length_sentence = len(masked_sent)
    num_tokens_to_mask = int(masking_rate * length_sentence)
    
    # Randomly select indices to mask
    indices_to_mask = random.sample(range(length_sentence), num_tokens_to_mask)
    
    # Mask selected indices
    
    for i in indices_to_mask:
        
        if tags[i] == 0: 
            masked_sent[i] = tokenizer.mask_token
            
    examples['tokens'] = masked_sent

    return examples

dataset = dataset_small.map(process, fn_kwargs={'masking_rate': args.masking_rate})

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

tokenized = dataset.map(tokenize_and_align_labels, fn_kwargs={'sequence_length': args.sequence_length}, batched=True)
tokenized.set_format(
    'torch', columns=["input_ids", "attention_mask", "labels"])

##### Model for Context Augmentation

model_context = AutoModelForMaskedLM.from_pretrained(args.model)
model_context.to(device)
model_context.eval()

# %%

def generate_topk(model, tokenized, sampling_size):

    model.eval()
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,  output_attentions=True)
        logits = outputs['logits']
        top_logits, topk = torch.topk(logits, k=sampling_size, dim=-1)  # get top 5 tokens
        mask = (input_ids==50264)
        topks = [t[m] for t, m in zip(topk, mask)]
        topk_ids = topk[mask]
        tokens_to_replace  = [[tokenizer.convert_ids_to_tokens(x) for x in top_k] for top_k in topks]
        tokens_to_replace = [[[token[1:] if token.startswith('Ġ') else token for token in sentence] for sentence in sentences] for sentences in tokens_to_replace]

    return tokens_to_replace


def augmentation(sentence, ner_tags, tokens_to_replace, ks=args.k):
    selected_tokens = [[sublist[i] for sublist in tokens_to_replace] for i in range(ks)] 

    augmented_sentences = []
    augmented_ner_tags  = []
    for k in range(ks):
        _sentence = sentence.copy()
        _ner_tags = ner_tags.copy()
      
        special_token_indexes = [i for i, token in enumerate(_sentence) if token == tokenizer.mask_token ]
        _selected_tokens = selected_tokens[k]

        for index, replacement in zip(special_token_indexes, _selected_tokens):
            _sentence[index] = replacement

        augmented_sentences.append(_sentence)
        augmented_ner_tags.append(_ner_tags)
        
        assert len(_sentence) == len(_ner_tags)

    return augmented_sentences, augmented_ner_tags

augmentated_sentences = []
augmented_ner_tags = []

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# %%

train_dataloader = DataLoader(
    tokenized['train'],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=args.batch_size)

tokens_to_replace = []
for batch in train_dataloader:
    batch.to(device)
    _tokens_to_replace = generate_topk(model=model_context, tokenized=batch, sampling_size=args.sampling_size)
    tokens_to_replace.extend(_tokens_to_replace)

for i in tqdm(range(len(dataset['train']))):
    tokens, ner_tags = dataset['train']['tokens'][i], dataset['train']['ner_tags'][i]
    _tokens_to_replace = tokens_to_replace[i]
    sentence_context_augmented, ner_tags_context_augmented = augmentation(sentence=tokens, ner_tags=ner_tags, tokens_to_replace=_tokens_to_replace)    
    augmentated_sentences.extend(sentence_context_augmented)  
    augmented_ner_tags.extend(ner_tags_context_augmented)

# %%

augmented_dataset = pd.DataFrame({"tokens":augmentated_sentences+dataset_small['train']['tokens'], "ner_tags":augmented_ner_tags+dataset_small['train']['ner_tags']})
augmented_dataset = Dataset.from_pandas(augmented_dataset)

# %%

dataset = DatasetDict({"train": augmented_dataset,
                       "validation": dataset_small['validation'],
                       "test":dataset_base['test']})

# %%
dataset.save_to_disk(args.path_to_save + '/' + args.out_file)
print("Prompt Augmented Dataset..............", dataset)
