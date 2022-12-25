import torch
import click
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
import math
from datasets import Dataset
import os

"""python3 syn_data/biimt_gen_data.py -prefix mt_infilling_data -batch_size 64 -model_checkpoint mt_infilling_weights_zh-en_2M_1660497817/epoch4.ckpt
-tok_checkpoint mbart-large-50-many-to-one-mmt-tok -src_file zh-en-mask.txt"""

"""Yield batch sized list of indices."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

def preprocess_data(inp_data, tokenizer, max_length, batch_size, shuffle=True, sampler=True):
    ds = Dataset.from_dict({"src": inp_data})
    # generate input data
    def preprocess_function(examples):
        model_inputs = {}
        srcs = tokenizer(examples['src'], max_length=max_length, padding='max_length', truncation=True)
        model_inputs['input_ids'], model_inputs['src_attn_masks'] = srcs['input_ids'], srcs['attention_mask']
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
        return_tensors='pt'
    )

    if sampler:
        data_sampler = torch.utils.data.distributed.DistributedSampler(processed_datasets, shuffle=shuffle)
        dataloader = DataLoader(processed_datasets, batch_size=batch_size, collate_fn=data_collator, sampler=data_sampler)
    else:
        dataloader = DataLoader(processed_datasets, batch_size=batch_size, collate_fn=data_collator, shuffle=shuffle)
    return dataloader

@click.command()
@click.option('-batch_size', type=int, default=80)
@click.option('-model_checkpoint', type=str)
@click.option('-tok_checkpoint', type=str, default="mbart-large-50-many-to-one-mmt-tok")
@click.option('-src_file', type=str, help="addr to the src file", default='')
@click.option('-save_dir', type=str)
@click.option('-prefix', type=str)
@click.option('-part_index', type=int, help="from 0 to 7")
def main(batch_size, model_checkpoint, tok_checkpoint, src_file, save_dir, prefix, part_index):
    torch.cuda.empty_cache()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(tok_checkpoint)
    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': ['<sep>']})
    model = torch.load(model_checkpoint).to(device)
    model.eval()
    # load in dev data
    mask_inp = open(src_file, 'r').readlines()
    mask_inp = [sen[:-1] for sen in mask_inp]
    file_loc_index = list(range(0, len(mask_inp), math.ceil(len(mask_inp)/8)))+[len(mask_inp)]
    start_index, end_index = file_loc_index[part_index], file_loc_index[part_index+1]
    dataloader = preprocess_data(mask_inp[start_index:end_index], tokenizer, 256, batch_size, shuffle=False, sampler=False)

    gen_file = open(f'{save_dir}/{prefix}_{start_index}_{end_index}.outputs', 'w')
    with torch.no_grad():
        for data in dataloader:
            translated_tokens = model.generate(data['input_ids'].to(device))
            batch_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            for text in batch_texts:
                gen_file.write(text+'\n')
    print(f"All outputs are saved at {prefix}.outputs")

if __name__ == "__main__":
    main()
