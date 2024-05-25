
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets
import torch

class DatasetProcessor:
    def __init__(self, dataset):
        self.dataset = dataset

    # @staticmethod
    def add_prompt(self, examples, id2label, prompt, tokenizer):
        tokens, ner_tags = examples['tokens'], examples['ner_tags']
        labels = [id2label[x] for x in ner_tags]
        tagged_sent = tokens.copy()
        tokens_org = []
        ner_tags_org = []
        masked_tokens_org = []
        masked_ner_tags_org = []
        tagged_sent_org = []

        for idx, (s, l) in enumerate(zip(tokens, labels)):
            if l != 'O':
                tagged_sent[idx] = f'<{l[2:]}>'

        special_tokens = [x for x in tagged_sent if x in prompt]
        special_tokens = sorted(set(special_tokens), key=special_tokens.index)

        if special_tokens == []:
            tokens_org.append(tokens)
            ner_tags_org.append(ner_tags)
            masked_ner_tags_org.append(ner_tags)  # If there are no tags, masked_sent == tagged_sent
            masked_tokens_org.append(tokens)

        for special_token in special_tokens:
            masked_tokens_temp = [tokenizer.mask_token if token in special_token else token for token in tagged_sent]
            special_token_indexes = [i for i, token in enumerate(masked_tokens_temp) if token == tokenizer.mask_token]
            masked_tokens = [tokenizer.mask_token if idx in special_token_indexes else word for idx, word in enumerate(tokens)]
            masked_tokens_temp = [special_token] + masked_tokens
            masked_tokens_org.append(masked_tokens_temp)
            
            _masked_ner_tags = [tag if idx in special_token_indexes else 0 for idx, tag in enumerate(ner_tags)]
            _masked_ner_tags = [-100] + _masked_ner_tags
            masked_ner_tags_org.append(_masked_ner_tags)

            _ner_tags_org = [-100] + ner_tags
            ner_tags_org.append(_ner_tags_org)

            tokens_temp = [special_token] + tokens
            tokens_org.append(tokens_temp)

            tagged_sent_org.append(tagged_sent)

        examples['tokens'] = tokens_org
        examples['ner_tags'] = ner_tags_org
        examples['masked_ner_tags'] = masked_ner_tags_org
        examples['masked_tokens_org'] = masked_tokens_org
        examples['tagged_tokens_org'] = tagged_sent_org

        return examples

    def add_prompt_generate(self, examples, id2label, prompt, tokenizer):
        tokens, ner_tags = examples['tokens'], examples['ner_tags']
        labels = [id2label[x] for x in ner_tags]
        tagged_sent_org = []
        tokens_prompt = []
        masked_tokens_org = []
        ner_tags_prompt = []
        masked_ner_tags = []
    
        for idx, (s, l, n) in enumerate(zip(tokens, labels, ner_tags)):
            tagged_sent = tokens.copy()
            if l != 'O':
                tagged_sent[idx] = f'<{l[2:]}>'
                tagged_sent_org.append(tagged_sent)
                _tokens = [f'<{l[2:]}>'] + tokens
                tokens_prompt.append(tokens)
                ner_tags_prompt = [-100] + ner_tags
                _masked_tokens = [tokenizer.mask_token if token in prompt else token for idx, token in enumerate(tagged_sent)]
                _masked_tokens = [f'<{l[2:]}>'] + _masked_tokens
                masked_tokens_org.append(_masked_tokens)
                masked_ner_tags.append(ner_tags_prompt)    
    
        examples.update({
            'tokens': tokens_prompt,
            'ner_tags': ner_tags_prompt,
            'masked_tokens_org': masked_tokens_org,
            'tagged_tokens_org': tagged_sent_org,
            'masked_ner_tags': masked_ner_tags
        })
    
        return examples
       

    @staticmethod
    def dataset_extend(dataset):
   
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

def tokenize_and_align_labels(examples, tokenizer, id2label, prompt_mapping, masking_type, sequence_length):
    
    tokenized_inputs = tokenizer(
        examples[f"{masking_type}"], max_length=sequence_length, truncation=True, is_split_into_words=True, padding='max_length')

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