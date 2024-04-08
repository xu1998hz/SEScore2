import torch
from train.feedforward import FeedForward
from transformers import (
    AutoModel,
    DataCollatorWithPadding
)
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

def preprocess_data(triplets_score_dict, tokenizer, max_length, batch_size, shuffle=True, sampler=True, mode='train'):
    if mode == 'train':
        ds = Dataset.from_dict({"pivot": triplets_score_dict['pivot'], 'mt': triplets_score_dict['mt'], 'score': triplets_score_dict['score']})
    else:
        ds = Dataset.from_dict({"pivot": triplets_score_dict['pivot'], 'mt': triplets_score_dict['mt']})

    def preprocess_function(examples):
        model_inputs = {}
        # pivot examples added into dataloader, one pivot per instance
        pivot = tokenizer(examples['pivot'], max_length=max_length, padding='max_length', truncation=True)
        model_inputs['input_ids'], model_inputs['pivot_attn_masks'] = pivot['input_ids'], pivot['attention_mask']
        # mt examples added into dataloader, one mt per instance
        mt = tokenizer(examples['mt'], max_length=max_length, padding='max_length', truncation=True)
        model_inputs['mt_input_ids'], model_inputs['mt_attn_masks'] = mt["input_ids"], mt['attention_mask']
        # store the labels in model inputs
        if mode == 'train':
            model_inputs['score'] = examples['score']
        return model_inputs

    processed_datasets = ds.map(
        preprocess_function,
        batched=True,
        num_proc=None,
        remove_columns=ds.column_names,
        desc="Running tokenizer on dataset",
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='max_length',
        max_length=max_length,
        pad_to_multiple_of=None,
        return_tensors = 'pt'
    )

    if sampler:
        data_sampler = torch.utils.data.distributed.DistributedSampler(processed_datasets, shuffle=shuffle)
        dataloader = DataLoader(processed_datasets, batch_size=batch_size, collate_fn=data_collator, sampler=data_sampler)
    else:
        dataloader = DataLoader(processed_datasets, batch_size=batch_size, collate_fn=data_collator, shuffle=shuffle)
    return dataloader

class exp_config():
    max_length = 256
    temp = 0.1
    drop_out=0.1
    activation="Tanh"
    final_activation=None

def sent_emb(hidden_states, emb_type, attention_mask):
    if emb_type == 'last_layer':
        sen_embed = (hidden_states[-1]*attention_mask.unsqueeze(-1)).sum(1)/attention_mask.sum(-1).unsqueeze(-1)
    elif emb_type == 'avg_first_last':
        sen_embed = ((hidden_states[-1]+hidden_states[0])/2.0*attention_mask.unsqueeze(-1)).sum(1)/attention_mask.sum(-1).unsqueeze(-1)
    else:
        print(f"{emb_type} sentence emb type is not supported!")
        exit(1)
    return sen_embed

def pool(model, encoded_input, attention_mask, emb_type):
    encoded_input, attention_mask = encoded_input.to(exp_config.device_id), attention_mask.to(exp_config.device_id)
    outputs = model(input_ids=encoded_input, attention_mask=attention_mask, output_hidden_states=True)
    pool_embed = sent_emb(outputs.hidden_states, emb_type, attention_mask)
    return pool_embed

class Regression_XLM_Roberta(nn.Module):
    def __init__(self, model_addr): 
        super().__init__()
        self.xlm = AutoModel.from_pretrained(model_addr)
        # initialize the feedforward to process the festures
        self.estimator = FeedForward(
            in_dim=exp_config.hidden_size * 4,
            hidden_sizes=exp_config.hidden_size_FNN,
            activations=exp_config.activation,
            dropout=exp_config.drop_out,
            final_activation=exp_config.final_activation,
        )

    def freeze_xlm(self) -> None:
        """Frezees the all layers in XLM weights."""
        for param in self.xlm.parameters():
            param.requires_grad = False

    def unfreeze_xlm(self) -> None:
        """Unfrezees the entire encoder."""
        for param in self.xlm.parameters():
            param.requires_grad = True

    def forward(self, batch, emb_type):
        pivot_pool_embed = pool(self.xlm, batch['input_ids'], batch['pivot_attn_masks'], emb_type)
        mt_pool_embed = pool(self.xlm, batch['mt_input_ids'], batch['mt_attn_masks'], emb_type)
        # compute diff between two embeds
        diff_ref = torch.abs(mt_pool_embed - pivot_pool_embed)
        prod_ref = mt_pool_embed * pivot_pool_embed
        # concatenate emebddings of mt and ref and derived features from them
        embedded_sequences = torch.cat(
            (mt_pool_embed, pivot_pool_embed, prod_ref, diff_ref), dim=1
        )
        return self.estimator(embedded_sequences)