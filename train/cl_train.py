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
    AutoModelForMaskedLM,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    AdamW
)
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

class exp_config():
    max_length = 256
    hidden_size = 1024
    temp = 0.05
    lr = 3e-05

class XLM_Roberta_MLP():
    """
    Load the XLM-Roberta large pretrained weights and at the same time perform concatenations
    We store additional projection matrix: projecting max_length * hidden state to hidden state, then pass to ReLU
    This model returns the outputs in (batch size * hidden states), no pooling is required!
    """
    def __init__(self, model_addr):
        super().__init__()
        self.xlm = AutoModelForMaskedLM.from_pretrained(model_addr)
        self.dense = nn.Linear(exp_config.max_length * exp_config.hidden_size, exp_config.hidden_size)
        self.activation = nn.ReLU() # Use RELU to extract useful features

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.xlm(input_ids=encoded_input, attention_mask=attention_mask, output_hidden_states=True)
        features = torch.flatten(outputs.hidden_states[-1], start_dim=1)
        x = self.dense(features)
        x = self.activation(x)

        return x

def sent_emb(hidden_states, emb_type, attention_mask):
    if emb_type == 'last_layer':
        sen_embed = (hidden_states[-1]*attention_mask.unsqueeze(-1)).sum(1)/attention_mask.sum(-1).unsqueeze(-1)
    elif emb_type == 'avg_first_last':
        sen_embed = ((hidden_states[-1]+hidden_states[0])/2.0*attention_mask.unsqueeze(-1)).sum(1)/attention_mask.sum(-1).unsqueeze(-1)
    elif emb_type == 'states_concat':
        sen_embed = outputs
    else:
        print(f"{emb_type} sentence emb type is not supported!")
    return sen_embed

def pool(model, encoded_input, attention_mask, emb_type):
    encoded_input, attention_mask = encoded_input.to(exp_config.device_id), attention_mask.to(exp_config.device_id)
    outputs = model(input_ids=encoded_input, attention_mask=attention_mask, output_hidden_states=True)
    pool_embed = sent_emb(outputs.hidden_states, emb_type, attention_mask)
    return pool_embed

class CL_XLM_Roberta(nn.Module):
    def __init__(self, model_addr):
        super().__init__()
        self.xlm = AutoModelForMaskedLM.from_pretrained(model_addr)

    def forward(self, batch, strategy, emb_type):
        src_pool_embed = pool(self.xlm, batch['input_ids'], batch['src_attn_masks'], emb_type)
        pos_pool_embed = pool(self.xlm, batch['tar'], batch['tar_attn_masks'], emb_type)

        if strategy == 'margin_src_ref_batch':
            neg_batch_embed = torch.cat((pos_pool_embed[1:, :], pos_pool_embed[:1, :]), dim=0)
            return src_pool_embed, pos_pool_embed, neg_batch_embed
        elif strategy == 'margin_src_ref_mt1':
            mt1_pool_embed = pool(self.xlm, batch['mt1'], batch['mt1_attn_masks'], emb_type)
            return src_pool_embed, pos_pool_embed, mt1_pool_embed
        elif strategy == 'margin_src_ref_mt1_batch':
            mt1_pool_embed = pool(self.xlm, batch['mt1'], batch['mt1_attn_masks'], emb_type)
            neg_batch_embed = torch.cat((pos_pool_embed[1:, :], pos_pool_embed[:1, :]), dim=0)
            return src_pool_embed, pos_pool_embed, mt1_pool_embed, neg_batch_embed
        else:
            print("Your training strategy is not supported!")
            exit(1)

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return -self.cos(x, y) / self.temp

def preprocess_data(src_data, ref_data, tokenizer, max_length, batch_size, mt1_data=None, shuffle=True, sampler=True):
    if mt1_data:
        ds = Dataset.from_dict({"src": src_data, 'tar': ref_data, 'mt1': mt1_data})
    else:
        ds = Dataset.from_dict({"src": src_data, 'tar': ref_data})

    def preprocess_function(examples):
        model_inputs = {}
        srcs = tokenizer(examples['src'], max_length=max_length, padding='max_length', truncation=True)
        model_inputs['input_ids'], model_inputs['src_attn_masks'] = srcs['input_ids'], srcs['attention_mask']
        targets = tokenizer(examples['tar'], max_length=max_length, padding='max_length', truncation=True)
        model_inputs['tar'], model_inputs['tar_attn_masks'] = targets["input_ids"], targets['attention_mask']
        if mt1_data:
            mt1_texts = tokenizer(examples['mt1'], max_length=max_length, padding='max_length', truncation=True)
            model_inputs['mt1'], model_inputs['mt1_attn_masks'] = mt1_texts["input_ids"], mt1_texts['attention_mask']
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

class loss_funct():
    def __init__(self, strategy, score_type):
        self.strategy = strategy
        if score_type == "cosine":
            self.score = Similarity(temp=exp_config.temp)
            margin = 0.01
        elif score_type == "L2":
            self.score = nn.PairwiseDistance(p=2)
            margin = 1
        else:
            print("We only support two distance metrics: cosine and L2")
            exit(1)

        # parameter order: anchor, positive, negative
        self.loss_fct_1 = nn.TripletMarginWithDistanceLoss(distance_function=self.score, margin=margin)
        if strategy == 'margin_src_ref_mt1_batch':
            self.loss_fct_2 = nn.TripletMarginWithDistanceLoss(distance_function=self.score, margin=margin*2)

    def loss_compute(self, train_batch, model, emb_type):
        if self.strategy == 'margin_src_ref_batch':
            src_pool_embed, pos_pool_embed, neg_batch_embed = model(train_batch, self.strategy, emb_type)
            return [self.loss_fct_1(src_pool_embed, pos_pool_embed, neg_batch_embed)]
        elif self.strategy == 'margin_src_ref_mt1':
            src_pool_embed, pos_pool_embed, mt1_pool_embed = model(train_batch, self.strategy, emb_type)
            return [self.loss_fct_1(src_pool_embed, pos_pool_embed, mt1_pool_embed)]
        elif self.strategy == 'margin_src_ref_mt1_batch':
            src_pool_embed, pos_pool_embed, mt1_pool_embed, neg_batch_embed = model(train_batch, self.strategy, emb_type)
            loss_1 = self.loss_fct_1(src_pool_embed, pos_pool_embed, mt1_pool_embed)
            loss_2 = self.loss_fct_2(src_pool_embed, pos_pool_embed, neg_batch_embed)
            loss_3 = self.loss_fct_1(pos_pool_embed, mt1_pool_embed, neg_batch_embed)
            return [loss_1, loss_2, loss_3]
        else:
            print("Your training strategy is not supported!")
            exit(1)

def store_cl_loss(model, dev_dataloader, loss_manager, train_batch, emb_type):
    model.eval()
    with torch.no_grad():
        train_loss_ls = loss_manager.loss_compute(train_batch, model, emb_type)
        train_loss = torch.stack(train_loss_ls, dim=0).sum(dim=0)
        loss_size = len(train_loss_ls)
        # store all training loss
        if loss_size == 3:
            dev_loss1, dev_loss2, dev_loss3 = 0, 0, 0
        dev_loss = 0
        # store all dev loss
        for dev_batch in dev_dataloader:
            dev_loss_ls = loss_manager.loss_compute(dev_batch, model, emb_type)
            if loss_size == 3:
                dev_loss1=dev_loss1+dev_loss_ls[0]
                dev_loss2=dev_loss2+dev_loss_ls[1]
                dev_loss3=dev_loss3+dev_loss_ls[2]
            dev_loss = dev_loss+torch.stack(dev_loss_ls, dim=0).sum(dim=0)

        dev_loss = dev_loss/len(dev_dataloader)

        if loss_size == 3:
            dev_loss1, dev_loss2, dev_loss3 = dev_loss1/len(dev_dataloader), dev_loss2/len(dev_dataloader), dev_loss3/len(dev_dataloader)
            wandb.log({"training loss (src, pos, mt1)": train_loss_ls[0].item(),
                       "training loss (src, pos, batchNeg)": train_loss_ls[1].item(),
                       "training loss (pos, mt, batchNeg)": train_loss_ls[2].item(),
                       "training loss": torch.stack(train_loss_ls, dim=0).sum(dim=0).item(),
                       "dev loss (src, pos, mt1)": dev_loss1.item(),
                       "dev loss (src, pos, batchNeg)": dev_loss2.item(),
                       "dev loss (pos, mt, batchNeg)": dev_loss3.item(),
                       "dev loss": dev_loss.item()
                      })
        else:
            wandb.log({
                       "training loss": torch.stack(train_loss_ls, dim=0).sum(dim=0).item(),
                       "dev loss": dev_loss.item()
                      })
    model.train()

@click.command()
@click.option('-gradient_accumulation_steps', default=1, type=int)
@click.option('-src_train_file', type=str, help="addr to the src train file")
@click.option('-ref_train_file', type=str, help="addr to the ref train file")
@click.option('-src_dev_file', type=str, help="addr to the src dev file")
@click.option('-ref_dev_file', type=str, help="addr to the ref dev file")
@click.option('-mt_file_train_1', type=str, help="addr to the first MT train output file", default=None)
@click.option('-mt_file_dev_1', type=str, help="addr to the first MT dev output file", default=None)
@click.option('-batch_size', type=int, help="batch size for contrastive learning")
@click.option('-emb_type', type=str, help="choose from last_layer, avg_first_last and states_concat")
@click.option('-strategy', type=str, help="choose from margin_src_ref_batch, margin_src_ref_mt1 and margin_src_ref_mt1_batch")
@click.option('-num_epoch', type=int, help="Number of epoches to train", default=5)
@click.option('-eval_step', type=int, help="Number of steps to evaluate", default=4)
@click.option('-num_warmup_steps', type=int, help="Number of steps to warm up", default=0)
@click.option('-save_dir_name', type=str, help="the dir name of weights being saved", default=None)
@click.option('-score_type', type=str, help="choose between L2 and cosine")
def main(src_train_file, ref_train_file, src_dev_file, ref_dev_file, batch_size, emb_type, strategy, mt_file_train_1, mt_file_dev_1,
num_epoch, eval_step, num_warmup_steps, gradient_accumulation_steps, save_dir_name, score_type):
    # initalize the process
    dist.init_process_group(backend="nccl")
    rank = int(os.environ['LOCAL_RANK'])
    # only main process initalize wandb
    if rank == 0:
        # initalize the project parameters into Wandb, store experiment specific parameters
        wandb.init(project="ContraScore", config=
        {
            "strategy": strategy,
            "epoch": num_epoch,
            "eval_step": eval_step,
            "emb_type": emb_type,
            "batch size": batch_size * gradient_accumulation_steps,
            "margin": 1 if score_type == 'L2' else 0.01,
            "score_type": score_type
        })
    exp_config.device_id = rank % torch.cuda.device_count()
    # set cuda device with rank and clear ram cache to ensure balanced ram allocations
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    # we will build bilingual sentence embedding models. XLM for both en-de and zh-en
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-tok")
    if emb_type == 'last_layer':
        model = CL_XLM_Roberta("xlm-roberta-large-model").to(exp_config.device_id)
    elif emb_type == 'states_concat':
        model = XLM_Roberta_MLP("xlm-roberta-large-model").to(exp_config.device_id)
    else:
        print("Incorrect model type!")

    # parallelize the pipeline into multiple gpus
    optimizer = AdamW(model.parameters(), lr=exp_config.lr)
    # load in src,ref and mt data if available
    src_train_data, src_dev_data = open(src_train_file, 'r').readlines(), open(src_dev_file, 'r').readlines()
    ref_train_data, ref_dev_data = open(ref_train_file, 'r').readlines(), open(ref_dev_file, 'r').readlines()
    src_train_data, ref_train_data = [data[:-1] for data in src_train_data], [data[:-1] for data in ref_train_data]
    src_dev_data, ref_dev_data = [data[:-1] for data in src_dev_data], [data[:-1] for data in ref_dev_data]
    # if mt_file_1 is none means there is no hard negative sample
    mt1_train_data, mt1_dev_data = None, None
    if mt_file_train_1:
        mt1_train_data, mt1_dev_data = open(mt_file_train_1, 'r').readlines(), open(mt_file_dev_1, 'r').readlines()
        mt1_train_data, mt1_dev_data = [data[:-1] for data in mt1_train_data], [data[:-1] for data in mt1_dev_data]
    train_dataloader = preprocess_data(src_train_data, ref_train_data, tokenizer, exp_config.max_length, batch_size, mt1_data=mt1_train_data)
    dev_dataloader = preprocess_data(src_dev_data, ref_dev_data, tokenizer, exp_config.max_length, batch_size, mt1_data=mt1_dev_data)

    model = DDP(model, device_ids=[exp_config.device_id], find_unused_parameters=True)
    model.train()

    max_train_steps = math.ceil(len(train_dataloader) / gradient_accumulation_steps) * num_epoch
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    loss_manager = loss_funct(strategy, score_type)

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epoch):
            torch.cuda.empty_cache() # empty cache in gpus
            train_dataloader.sampler.set_epoch(epoch) # set the sampler at each epoch
            for step, train_batch in enumerate(train_dataloader):
                # evaluate at every eval_step and also at the end of epoch (includes the beginning loss)
                if (step % (eval_step * gradient_accumulation_steps) == 0) and rank == 0:
                    # store all the losses in wandb
                    store_cl_loss(model, dev_dataloader, loss_manager, train_batch, emb_type)

                train_loss_ls = loss_manager.loss_compute(train_batch, model, emb_type)
                train_loss = torch.stack(train_loss_ls, dim=0).sum(dim=0)

                # accumulate losses at weights, each is normalized by accumulation steps
                train_loss = train_loss / gradient_accumulation_steps
                train_loss.backward()

                if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad() # clear the grads

                # save at end of epoch and at main processls
                if not os.path.isdir(f'{save_dir_name}'):
                    os.makedirs(f'{save_dir_name}')

                if step == len(train_dataloader) - 1 and rank == 0:
                    store_cl_loss(model, dev_dataloader, loss_manager, train_batch, emb_type)
                    torch.save(model.module, f'{save_dir_name}/epoch{epoch}.ckpt')
                    print(f"Saved entire model at current epoch {epoch}!")

if __name__ == "__main__":
    main()
