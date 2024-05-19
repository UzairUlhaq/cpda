
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets
import os
from transformers import AutoModel, AutoTokenizer
import torch
from torch import cuda
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
from PIL import Image

device = 'cuda' if cuda.is_available() else 'cpu'




def get_class_tokens(dataset, a, b):
    tokens = []
    for token,label in zip(dataset['train']['tokens'],dataset['train']['ner_tags']):
        for t, l in zip(token, label):
            if l == a or l==b:
                tokens.append(t)
    training_tokens = list(set(tokens))
    
    return training_tokens

def get_embeddings(dataset, model):

    person = get_class_tokens(dataset, 1, 2)
    org = get_class_tokens(dataset, 3, 4)
    loc = get_class_tokens(dataset, 5, 6)
    misc = get_class_tokens(dataset, 7, 8)

    tokens_dict = {'person': person,
                   'org': org,
                   'loc': loc,
                   'misc': misc}

    df = pd.DataFrame(columns=["Class", "Value"])
    embedding_dict = []
    for item in tokens_dict:
        embeddings_all = []
        for text in tokens_dict[item]:
            df = df.append({'Class':item, 'Value': text}, ignore_index=True )

    tokens = [x for x in df['Value']]
    tokenized_inputs = tokenizer(tokens, return_tensors='pt', padding=True, truncation=True)
    tokenized_inputs.to('cuda')
    with torch.no_grad():
        outputs = model(**tokenized_inputs, is_eval=True, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]
        embeddings = embeddings.mean(dim=1)

    embeddings_np = embeddings.cpu().numpy()
    return embeddings_np, df

def tsne_plot(embeddings, df, perplexity, lr, n_iter):
    
    tsne = TSNE(perplexity=perplexity, n_components=2,
            random_state=42, learning_rate=lr, n_iter=n_iter)
    embeddings_tsne = tsne.fit_transform(embeddings)
    df['x'] = embeddings_tsne[:,0]
    df['y'] = embeddings_tsne[:,1]
    return df

def save_plots(df, epoch):
    fig = plt.figure(figsize=(8, 8))
    # Map each class to a color
    color_mapping = {'Person': 'green', 'Organization': 'red', 'Location': 'blue', 'Miscellaneous': 'orange', 'Anchor_Person': 'green', 'Anchor_Organization': 'red', 'Anchor_Location': 'blue', 'Anchor_Miscellaneous': 'orange'}
    df['color'] = df['Class'].map(color_mapping)

    color_mapping = {'Person': 'green', 'Organization': 'red', 'Location': 'blue', 'Miscellaneous': 'orange'}
    # Create scatter plot with different colors
    for class_label, color in color_mapping.items():
        class_data = df[df['Class'] == class_label]
        plt.scatter(class_data['x'], class_data['y'], label=class_label, color=color)

    color_mapping = {'Anchor_Person': 'green', 'Anchor_Organization': 'red', 'Anchor_Location': 'blue', 'Anchor_Miscellaneous': 'orange'}
    # # Create scatter plot with different colors
    for class_label, color in color_mapping.items():
        class_data = df[df['Class'] == class_label]
        plt.scatter(class_data['x'], class_data['y'], label=class_label, color=color, marker='D')
        
    # Add labels and legend
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Embeddings")
    plt.legend()
    
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    pil_image = Image.frombytes("RGB", (width, height), buf)
    pil_image.save(f'/home/ulhaq/project/contrastive_loss/figures/{epoch}.png')
    plt.close()

def get_embeddings_test(embeddings_raw, anchors, prompts):
    data_dict = {prompt_id: [] for prompt_id in prompts}
    ids_tokens = []
    embeddings_tokens = []
    ids_anchors = []
    embeddings_anchors = []

    for item in embeddings_raw:
        for key, value in item.items():  # Iterate directly over key-value pairs
            if value:  # Check if the value list is not empty
                x = []
                [x.extend(item) for item in value]
                array = [f'{key}']*len(x)
                ids_tokens.extend(array)
                embeddings_tokens.extend(x)

    for item in anchors:
        for key, value in item.items():  # Iterate directly over key-value pairs
            if value:  # Check if the value list is not empty
                x = []
                [x.extend(item) for item in value]
                array = [f'{key}']*len(x)
                ids_anchors.extend(array)
                embeddings_anchors.extend(x)            

    mapping_tokens = {'50265': 'Person','50266': 'Organization', '50267': 'Location', '50268': 'Miscellaneous'}
    mapping_anchors = {'50265': 'Anchor_Person','50266': 'Anchor_Organization', '50267': 'Anchor_Location', '50268': 'Anchor_Miscellaneous'}

    ids_tokens_mapping = [mapping_tokens[value] for value in ids_tokens]
    ids_anchors_mapping = [mapping_anchors[value] for value in ids_anchors]
    
    df_tokens = pd.DataFrame(columns=["Class", "Value"])
    for key, value in zip(ids_tokens_mapping, embeddings_tokens):
        df_tokens = pd.concat([df_tokens, pd.DataFrame({'Class': [key], 'Value': [value]})], ignore_index=True)
    
    df_anchors = pd.DataFrame(columns=["Class", "Value"])
    for key, value in zip(ids_anchors_mapping, embeddings_anchors):
        df_anchors = pd.concat([df_anchors, pd.DataFrame({'Class': [key], 'Value': [value]})], ignore_index=True)
    df = pd.concat([df_tokens, df_anchors])
    
    return df



