import random
import torch
from torch import cuda
import numpy as np
import os
os.chdir('/home/ulhaq/project/cpda/')

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
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--dataset_size', type=int, required=True)
parser.add_argument('--k', type=int, required=True)
parser.add_argument('--sampling_size', type=int, required=True)
parser.add_argument('--sequence_length', type=int, required=True)
parser.add_argument('--lamda', type=float, required=True)
parser.add_argument('--project', type=str, required=True)
parser.add_argument('--jobname', type=str, required=True)
parser.add_argument('--context_augmentation', type=str, required=True)



args = parser.parse_args()

# %%
device = 'cuda' if cuda.is_available() else "cpu"
print('Running on ...', device)
print('Data Generation .... ')
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

dataset_processor = DatasetProcessor(dataset_small)
dataset = dataset_small.map(dataset_processor.add_prompt_generate, fn_kwargs={'id2label': id2label, 'prompt': prompt, 'tokenizer': tokenizer}, load_from_cache_file=False)

# %%%

prompt_mapping = {}
encodings = []
for entity in prompt:
    entity_encoding = tokenizer.encode(entity, add_special_tokens=False)
    encodings.append(entity_encoding[0])
    prompt_mapping.update({f"{entity}": entity_encoding[0]})

##### Model for Prompt Augmentation

model = roberta_mlm.from_pretrained(checkpoint, encodings)
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load(checkpoint + '/pytorch_model.bin'))
model.to(device)

##### Model for Context Augmentation
model_context = AutoModelForMaskedLM.from_pretrained(args.model)
model_context.to(device)
model_context.eval()

# %%

def generate_topk(model, tokenized, sampling_size):

    model.eval()
    input_ids = torch.tensor(tokenized['input_ids']).to(device)
    attention_mask = torch.tensor(tokenized['attention_mask']).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,  output_attentions=True)
        logits = outputs['logits']
        top_logits, topk = torch.topk(logits, k=sampling_size, dim=-1)  # get top 5 tokens
        mask = (input_ids==50264)
        topk_ids = topk[mask]
        tokens_to_replace  = [tokenizer.convert_ids_to_tokens(x) for x in topk_ids]
        tokens_to_replace = [[token[1:] if token.startswith('Ġ') else token for token in sentence] for sentence in tokens_to_replace]
    return tokens_to_replace

# %%

def augmentation(model, tokenized, sampling_size, sentence, ner_tags):

    tokens_to_replace = generate_topk(model=model, tokenized=tokenized_data, sampling_size=args.sampling_size)
    tokens_to_replace = [[token for token in sentence if token not in prompt] for sentence in tokens_to_replace]
    # selected_tokens = [[sublist[i] for sublist in tokens_to_replace] for i in range(args.k)]  # model probability    
    augmented_sentences = []
    augmented_ner_tags  = []
    for k in range(args.k):
        _sentence  = sentence.copy()
        _ner_tags  = ner_tags.copy()

        special_token_indexes = [i for i, token in enumerate(_ner_tags) if token !=0 ]
        selected_tokens = [random.choice(lst) for lst in tokens_to_replace]  #### Uniform Probability

        for index, replacement in zip(special_token_indexes, selected_tokens):
            _sentence[index] = replacement

        augmented_sentences.append(_sentence)
        augmented_ner_tags.append(_ner_tags)
        assert len(_sentence) == len(_ner_tags)

    return augmented_sentences, augmented_ner_tags

augmentated_sentences = []
augmented_ner_tags = []

for i in range(len(dataset['train'])):
    
    _tokens, _ner_tags = dataset_small['train']['tokens'], dataset_small['train']['ner_tags']
    
    tokens = _tokens[i]
    ner_tags = _ner_tags[i]

    if dataset['train']['tagged_tokens_org'][i] == []:
        if  args.context_augmentation == 'True':
            masked_context = dataset_processor.mask_context_tokens(tokens=tokens, ner_tags=ner_tags, masking_rate=0.3, tokenizer=tokenizer)
        
            tokenized_data = tokenizer(
                [masked_context], max_length=args.sequence_length, truncation=True, is_split_into_words=True, padding='max_length')

            ner_tags_context = [30 if token == '<mask>' else 0 for token in masked_context]

            sentence_context, ner_tags = augmentation(model=model_context, tokenized=tokenized_data, sampling_size=args.sampling_size, sentence=tokens, ner_tags=ner_tags_context)    
            augmentated_sentences.extend(sentence_context)  
            augmented_ner_tags.extend(ner_tags)

            continue

        else:
            continue

    tokenized_data = tokenizer(
         dataset['train']['masked_tokens_org'][i], max_length=args.sequence_length, truncation=True, is_split_into_words=True, padding='max_length')

    sentence, ner_tags = augmentation(model=model, tokenized=tokenized_data, sampling_size=args.sampling_size, sentence=tokens, ner_tags=ner_tags)    

    augmentated_sentences.extend(sentence)  
    augmented_ner_tags.extend(ner_tags)

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
