

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput, MaskedLMOutput
from typing import List, Optional, Tuple, Union
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForMaskedLM
from torch.nn import functional as F

# %%

class model_baseline(PreTrainedModel):

    def __init__(self, checkpoint, config, num_labels, id2label, label2id, dropout):
        super(model_baseline, self).__init__(config)
        self.num_labels = num_labels
        self.label2id = label2id
        self.id2label = id2label
        self.dropout = dropout
        self.model = AutoModel.from_pretrained(checkpoint,
                                               config=AutoConfig.from_pretrained(checkpoint, output_attention=True,
                                                                                 output_hidden_state=True,
                                                                                 label2id=label2id,
                                                                                 id2label=id2label))
                
        # New Layer
        self.drop_out = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
       
        sequence_output = outputs[0]
        sequence_output = self.drop_out(sequence_output)
                
        logits = self.classifier(sequence_output)

        loss = None
        loss_pooler = None
        if labels is not None:
            loss_func = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))

            return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

# %%

class roberta_mlm(RobertaPreTrainedModel):

    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config, encodings):
        super().__init__(config)
        self.prompt = encodings   ## prompt encodings
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.loss =  nn.CrossEntropyLoss(ignore_index=1, reduction='mean')

        # Initialize weights and apply final processing
        self.post_init()
        
    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,      
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        tagged: Optional[bool] = None,
        label_inputs_ids:Optional[bool]=None,
        is_eval:Optional[bool]=None,
        lamda:Optional[torch.LongTensor]=None,

    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]                                                    ### [bz, N, 768]
        prediction_scores = self.lm_head(sequence_output)

        if labels is None:                                                                     ### For Generation Only
            return {
            "logits": prediction_scores,
            }

        # Initialize losses
        contrastive_loss = ContrastiveLoss()        
        loss_cumulative = 0
        loss_contrastive = 0
        mlm_loss = 0
        d_pos = 0
        d_neg = 0

        for i in range(sequence_output.shape[0]):                                         ### iterate over each sentence to get anchor, positve tokens and negative tokens to calculate loss
            sequence_normalize_parameter = sum(attention_mask[i])                         ### +ve number           
            prompt_id = input_ids[i][1].item()                                            ### extract prompt token which is '[CLS] [Prompt_ID] [Sentence] [SEP]'
            if prompt_id not in self.prompt:                                              ### If sequence doesnot have any class then calculate mlm loss only
                mlm_loss += self.loss(prediction_scores[i].view(-1, self.config.vocab_size), labels[i].view(-1))
                continue  

            anchor = sequence_output[i][input_ids[i].eq(prompt_id)]                       ### [1, E]
            positive_tokens_mask = (tagged[i] == prompt_id) 
            positive_tokens_mask_expanded = positive_tokens_mask.unsqueeze(-1).expand_as(sequence_output[i])         ### [1, N, E]         
            positive_tokens = sequence_output[i][positive_tokens_mask_expanded].view(-1, 768)                        ### [N, E] #Get tokens belonging to positive class
            negative_anchors = list(set(self.prompt)-set([prompt_id]))                                               ### select one placeholder at time to calculate loss
            ####  If dataset has only one CLASS ### 
          
            if len(self.prompt)==1: 
                negative_tokens_mask = (tagged[i] != prompt_id)                            ### Generate mask for negative placeholder 
                negative_tokens_mask_expanded = negative_tokens_mask.unsqueeze(-1).expand_as(sequence_output[i])      
                negative_tokens = sequence_output[i][negative_tokens_mask_expanded].view(-1, 768)    ### [Ne, E] #Get tokens belonging to negtive class
                entity_length = positive_tokens.shape[0]  ### high == max entity length
               
                if negative_tokens.shape[0] < entity_length:
                    entity_length = entity_length-negative_tokens.shape[0]
                
                start_index = torch.randint(low=0, high=negative_tokens.shape[0]-entity_length, size=(1,1)).item()
                end_index = start_index + entity_length 
                negative_tokens = negative_tokens[start_index:end_index]
                if negative_tokens.shape[0] != 0 and positive_tokens.shape[0] != 0:
                    loss, distance_positive, distance_negative = contrastive_loss(anchor, positive_tokens, negative_tokens)
                    # distance_normalize_parameter = len(distance_positive) + len(distance_negative)
                    loss_contrastive += loss#/distance_normalize_parameter
                    d_pos += (distance_positive.sum())#/distance_normalize_parameter)
                    d_neg += (distance_negative.sum())#/distance_normalize_parameter)

            #############################################################################################        
           
            ####  If dataset have Multiple CLASSES  ###
            else:     
                for p in negative_anchors:                                                ### iterate over placeholders belonging to negative classes
                    negative_tokens_mask = (tagged[i] == p)                               ### Generate mask for negative placeholder 
                    negative_tokens_mask_expanded = negative_tokens_mask.unsqueeze(-1).expand_as(sequence_output[i])      
                    negative_tokens = sequence_output[i][negative_tokens_mask_expanded].view(-1, 768)    ### [Ne, E] #Get tokens belonging to negtive class
                    if negative_tokens.shape[0] != 0 and positive_tokens.shape[0] != 0:
                        loss, distance_positive, distance_negative = contrastive_loss(anchor, positive_tokens, negative_tokens)
                        # distance_normalize_parameter = len(distance_positive) + len(distance_negative)
                        loss_contrastive += loss #/distance_normalize_parameter
                        d_pos += (distance_positive.sum())#/distance_normalize_parameter)
                        d_neg += (distance_negative.sum())#/distance_normalize_parameter)

            mlm_loss += self.loss(prediction_scores[i].view(-1, self.config.vocab_size), labels[i].view(-1))
        
        positive_distance = d_pos/sequence_output.shape[0]                                 ### Normalize with batch size 
        negative_distance = d_neg/sequence_output.shape[0]
        mlm_loss_normalized = mlm_loss/sequence_output.shape[0]
        loss_contrastive_normalized = loss_contrastive/sequence_output.shape[0]

        mlm_loss_normalized = (1-lamda)*mlm_loss_normalized   
        loss_contrastive_normalized = lamda*loss_contrastive_normalized

        loss_total = None

        if labels is not None:
            loss_total = loss_contrastive_normalized + mlm_loss_normalized                  ### Lamda is balancing factor

        return {
            "loss_total": loss_total,
            "loss_contrastive": loss_contrastive_normalized,
            "positive_similarity": positive_distance,
            "negative_similarity": negative_distance,
            "mlm_loss": mlm_loss_normalized,
            "logits": prediction_scores,
            }

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
        
    def forward(self, features, **kwargs):

        x = self.dense(features)
        x = self.layer_norm(x)
        x = self.decoder(x)

        return x

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, anchor, positive, negative):
        # Compute cosine similarities
        similarity_positive = F.cosine_similarity(anchor.unsqueeze(1), positive.unsqueeze(0), dim=-1)
        similarity_negative = F.cosine_similarity(anchor.unsqueeze(1), negative.unsqueeze(0), dim=-1)

        # Compute exponentials of the similarities
        exp_sim_pos = torch.exp(similarity_positive)
        exp_sim_neg = torch.exp(torch.pow(similarity_negative,2))

        # Compute loss
        loss = -torch.log(exp_sim_pos.mean() / (exp_sim_pos.mean() + exp_sim_neg.mean()))

        return loss, similarity_positive[0], similarity_negative[0]

