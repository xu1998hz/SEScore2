import click
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    AdamW
)
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from scipy.stats import poisson
import nltk
from nltk.tokenize import word_tokenize
from random import randrange
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm.auto import tqdm
import random

nltk.download('punkt')

class exp_config():
    max_length = 256
    hidden_size = 1024
    temp = 0.05
    lr = 3e-05

def construct_uniform_mask_data(src_data, ref_data, mask_token, sep_token, mask_ratio, aug_factor):
    src_data, ref_data = [data[:-1] for data in src_data], [data[:-1] for data in ref_data]
    mask_inp_data_ls, mask_ref_data_ls = [], []
    with tqdm(total=len(src_data)*aug_factor) as pbar:
        for src_sen, ref_sen in zip(src_data, ref_data):
            for i in range(aug_factor):
                mask_inp_data, mask_tar_data = [], []
                ref_words_ls = word_tokenize(ref_sen)
                mask_len = int(len(ref_words_ls) * mask_ratio)
                start_index = randrange(0, len(ref_words_ls)-mask_len+1)
                end_index = start_index + mask_len
                mask_inp_data+=ref_words_ls[:start_index] + [mask_token] + ref_words_ls[end_index:]
                mask_tar_data+=ref_words_ls[start_index:end_index]
                mask_inp_data_ls.append(src_sen +f' {sep_token} '+ TreebankWordDetokenizer().detokenize(mask_inp_data))
                mask_ref_data_ls.append(TreebankWordDetokenizer().detokenize(mask_tar_data))
                pbar.update(1)
    return mask_inp_data_ls, mask_ref_data_ls

# in data preprocess, we need to randomnized the mask function in every epoch
def preprocess_data(src_data, ref_data, tokenizer, max_length, batch_size, shuffle=True, sampler=True):
    ds = Dataset.from_dict({"src": src_data, 'tar': ref_data})
    # generate input data
    def preprocess_function(examples):
        model_inputs = {}
        srcs = tokenizer(examples['src'], max_length=max_length, padding='max_length', truncation=True)
        model_inputs['input_ids'], model_inputs['src_attn_masks'] = srcs['input_ids'], srcs['attention_mask']
        targets = tokenizer(examples['tar'], max_length=max_length, padding='max_length', truncation=True)
        model_inputs['labels'] = targets["input_ids"]
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

def store_nli_loss(model, dev_dataloader, train_batch):
    model.eval()
    with torch.no_grad():
        train_loss = model(train_batch['input_ids'], attention_mask=train_batch['src_attn_masks'], labels=train_batch['labels']).loss
        dev_loss = 0
        # store all dev loss
        for dev_batch in dev_dataloader:
            dev_loss += model(dev_batch['input_ids'], attention_mask=dev_batch['src_attn_masks'], labels=dev_batch['labels']).loss
        dev_loss = dev_loss/len(dev_dataloader)
        # log both training and dev losses in wandb
        wandb.log({
                   "training loss": train_loss.item(),
                   "dev loss": dev_loss.item()
                  })
    model.train()
    return dev_loss

@click.command()
@click.option('-aug_factor', type=int)
@click.option('-tar_lang')
@click.option('-mask_ratio', type=float, default=0.7)
@click.option('-gradient_accumulation_steps', default=8, type=int)
@click.option('-src_train_file', type=str, help="addr to the src train file", default='mt_infilling_data/zh-en/2M/zh_train_2M.txt')
@click.option('-ref_train_file', type=str, help="addr to the ref train file", default='mt_infilling_data/zh-en/2M/en_train_2M.txt')
@click.option('-src_dev_file', type=str, help="addr to the src dev file", default='mt_infilling_data/zh-en/2M/zh_dev_2M.txt')
@click.option('-ref_dev_file', type=str, help="addr to the ref dev file", default='mt_infilling_data/zh-en/2M/en_dev_2M.txt')
@click.option('-batch_size', type=int, help="batch size for contrastive learning", default=32)
@click.option('-num_epoch', type=int, help="Number of epoches to train", default=5)
@click.option('-eval_step', type=int, help="Number of steps to evaluate", default=500)
@click.option('-num_warmup_steps', type=int, help="Number of steps to warm up", default=0)
@click.option('-save_dir_name', type=str, help="the dir name of weights being saved", default=None)
def main(aug_factor, tar_lang, mask_ratio, src_train_file, ref_train_file, src_dev_file, ref_dev_file, batch_size, \
num_epoch, eval_step, num_warmup_steps, gradient_accumulation_steps, save_dir_name):
    random.seed(10)
    # initalize the process
    dist.init_process_group(backend="nccl")
    rank = int(os.environ['LOCAL_RANK'])
    # only main process initalize wandb
    if rank == 0:
        # initalize the project parameters into Wandb, store experiment specific parameters
        wandb.init(project="ContraScore", config=
        {
            "epoch": num_epoch,
            "eval_step": eval_step,
            "batch size": batch_size * gradient_accumulation_steps,
            "num_warmup_steps": num_warmup_steps,
            "mask ratio": mask_ratio,
            "aug factor": aug_factor,
            "seed": 10
        })
    exp_config.device_id = rank % torch.cuda.device_count()

    if tar_lang == 'en_XX':
        model_checkpoint = "mbart-large-50-many-to-one-mmt-model"
        tok_checkpoint = "mbart-large-50-many-to-one-mmt-tok"
    elif tar_lang == 'de_DE':
        model_checkpoint = "mbart-large-50-one-to-many-mmt-model"
        tok_checkpoint = "mbart-large-50-one-to-many-mmt-tok"
    else:
        print("We currently only support en_XX and de_DE for target language")
        exit(1)

    # set cuda device with rank and clear ram cache to ensure balanced ram allocations
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(tok_checkpoint)
    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': ['<sep>']})
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(exp_config.device_id)

    print("load in models")

    # parallelize the pipeline into multiple gpus
    optimizer = AdamW(model.parameters(), lr=exp_config.lr)
    # load in src,ref and mt data if available
    src_train_data, src_dev_data = open(src_train_file, 'r').readlines(), open(src_dev_file, 'r').readlines()
    ref_train_data, ref_dev_data = open(ref_train_file, 'r').readlines(), open(ref_dev_file, 'r').readlines()
    mask_inp_train, mask_ref_train = construct_uniform_mask_data(src_train_data, ref_train_data, tokenizer.mask_token, '<sep>', mask_ratio, aug_factor)
    mask_inp_dev, mask_ref_dev = construct_uniform_mask_data(src_dev_data, ref_dev_data, tokenizer.mask_token, '<sep>', mask_ratio, aug_factor)

    print('construct masked data!')

    train_dataloader = preprocess_data(mask_inp_train, mask_ref_train, tokenizer, exp_config.max_length, batch_size)
    dev_dataloader = preprocess_data(mask_inp_dev, mask_ref_dev, tokenizer, exp_config.max_length, batch_size)

    model = DDP(model, device_ids=[exp_config.device_id])
    model.train()

    max_train_steps = math.ceil(len(train_dataloader) / gradient_accumulation_steps) * num_epoch
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    if not os.path.isdir(f'{save_dir_name}') and rank == 0:
        os.makedirs(f'{save_dir_name}')

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epoch):
            # store best checkpoint and end-of-epoch checkpoint at each epoch folder
            if not os.path.isdir(f'{save_dir_name}/epoch{epoch}') and rank == 0:
                os.makedirs(f'{save_dir_name}/epoch{epoch}')
                dev_loss_file = open(f"{save_dir_name}/epoch{epoch}/epoch{epoch}_dev.loss", 'w')
            step_best_dev_loss = float("inf")
            torch.cuda.empty_cache() # empty cache in gpus
            train_dataloader.sampler.set_epoch(epoch) # set the sampler at each epoch
            for step, train_batch in enumerate(train_dataloader):
                # evaluate at every eval_step (includes the beginning loss)
                if (step % (eval_step * gradient_accumulation_steps) == 0) and rank == 0:
                    print(f"In the eval step: {step}")
                    # store all the losses in wandb
                    cur_dev_loss = store_nli_loss(model, dev_dataloader, train_batch)
                    if cur_dev_loss < step_best_dev_loss:
                        step_best_dev_loss = cur_dev_loss
                        torch.save(model.module, f'{save_dir_name}/epoch{epoch}/epoch{epoch}_best.ckpt')
                        dev_loss_file.write(f'Step {step}, {cur_dev_loss}\n')

                train_loss = model(train_batch['input_ids'], attention_mask=train_batch['src_attn_masks'], labels=train_batch['labels']).loss
                # accumulate losses at weights, each is normalized by accumulation steps
                train_loss = train_loss / gradient_accumulation_steps
                train_loss.backward()

                if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad() # clear the grads

                # save at end of epoch and at main processls
                if step == len(train_dataloader) - 1 and rank == 0:
                    cur_dev_loss = store_nli_loss(model, dev_dataloader, train_batch)
                    torch.save(model.module, f'{save_dir_name}/epoch{epoch}/epoch{epoch}_end.ckpt')
                    dev_loss_file.write(f'End of Epoch: {cur_dev_loss}')
                    dev_loss_file.close()
                    print(f"Saved entire model at current epoch {epoch}!")

if __name__ == "__main__":
    main()
