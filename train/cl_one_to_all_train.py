import click
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    XLMRobertaTokenizerFast,
    AutoModelForMaskedLM,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    AdamW
)
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import random
from mt_metrics_eval import data
import time

# 1) obtain the triplet pairs (ranking difference and use it to determine margins)
# 2) use margin ranking loss for all triplets, based on rankings

human_mapping_dict = {
    "wmt21.news": {
        'en-de': ['refA', 'refD'],
        'en-ru': ['refB'],
        'zh-en': ['refA']
    },
    "wmt20": {
        'en-de': 'refb',
        'zh-en': 'refb'
    }
}

class exp_config():
    max_length = 256
    hidden_size = 1024
    temp = 0.1
    lr = 3e-05

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

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

class Stratified_CL_XLM_Roberta(nn.Module):
    def __init__(self, model_addr):
        super().__init__()
        self.xlm = AutoModelForMaskedLM.from_pretrained(model_addr)

    def forward(self, batch, emb_type, src_based, eval_mode):
        pivot_pool_embed = pool(self.xlm, batch['input_ids'], batch['pivot_attn_masks'], emb_type)
        if eval_mode:
            pos_pool_embed = pool(self.xlm, batch['pos'], batch['pos_attn_masks'], emb_type)
            neg_pool_embed = pool(self.xlm, batch['neg'], batch['neg_attn_masks'], emb_type)
            return pivot_pool_embed, pos_pool_embed, neg_pool_embed
        else:
            if src_based:
                pos_pool_embed = pool(self.xlm, batch['pos'], batch['pos_attn_masks'], emb_type)
            else:
                # positive sample embedding is generated through dropout rate 0.1
                pos_pool_embed = pool(self.xlm, batch['input_ids'], batch['pivot_attn_masks'], emb_type)
            num_bacth_neg, num_attn_neg_masks = torch.transpose(batch['neg'], 0, 1), torch.transpose(batch['neg_attn_masks'], 0, 1)
            batch_pos_neg_ls = [pos_pool_embed]
            # compute one neg embedding for each batch per time
            for batch_neg, batch_attn_neg_masks in zip(num_bacth_neg, num_attn_neg_masks):
                ele_neg_pool_embed = pool(self.xlm, batch_neg, batch_attn_neg_masks, emb_type)
                batch_pos_neg_ls.append(ele_neg_pool_embed)
            pos_neg_pool_embed = torch.stack(batch_pos_neg_ls, 0)
            pos_neg_pool_embed = torch.transpose(pos_neg_pool_embed, 0, 1)
            return pivot_pool_embed, pos_neg_pool_embed

def preprocess_data(triplets_score_dict, tokenizer, max_length, batch_size, src_based, shuffle=True, sampler=True, eval_mode=False):
    if src_based:
        ds = Dataset.from_dict({"pivot": triplets_score_dict['pivot'], 'pos': triplets_score_dict['pos'], \
        'neg': triplets_score_dict['neg']})
    else:
        ds = Dataset.from_dict({"pivot": triplets_score_dict['pivot'], 'neg': triplets_score_dict['neg']})

    def preprocess_function(examples):
        model_inputs = {}
        # pivot examples added into dataloader, one pivot per instance
        pivot = tokenizer(examples['pivot'], max_length=max_length, padding='max_length', truncation=True)
        model_inputs['input_ids'], model_inputs['pivot_attn_masks'] = pivot['input_ids'], pivot['attention_mask']
        # only src based examples need postive instance
        if src_based:
            # pos examples added into dataloader, one pos per instance
            pos = tokenizer(examples['pos'], max_length=max_length, padding='max_length', truncation=True)
            model_inputs['pos'], model_inputs['pos_attn_masks'] = pos["input_ids"], pos['attention_mask']
        if eval_mode:
            # in eval mode, one neg per instance
            neg = tokenizer(examples['neg'], max_length=max_length, padding='max_length', truncation=True)
            model_inputs['neg'], model_inputs['neg_attn_masks'] = neg["input_ids"], neg['attention_mask']
        else:
            # neg examples added into dataloader, k negs per instance
            neg_ls, attns_ls = [], []
            for result in examples['neg']:
                neg = tokenizer(result, max_length=max_length, padding='max_length', truncation=True)
                neg_ls.append(neg["input_ids"])
                attns_ls.append(neg['attention_mask'])
            model_inputs['neg'], model_inputs['neg_attn_masks'] = neg_ls, attns_ls
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
    def __init__(self, score_type):
        # be cautiously when applying cosine because order of pos neg will be opposite (targets=1)
        if score_type == "cosine":
            self.score = Similarity(temp=exp_config.temp)
            self.margin = 0.01
            self.loss_fct = torch.nn.CrossEntropyLoss()
        # targets = -1 when applying L2 distance
        elif score_type == "L2":
            self.score = nn.PairwiseDistance(p=2)
            self.margin = 1
            self.loss_fct = torch.nn.MarginRankingLoss(margin=self.margin)
        else:
            print("We only support two distance metrics: cosine and L2")
            exit(1)

    # If src_based, ref will be used as pos samples (pos:0, batchNeg: 6)->max_diff=6. If ref_based, max_diff=5
    def loss_compute(self, batch, model, emb_type, src_based):
        pivot_pool_embed, pos_neg_pool_embed = model(batch, emb_type, src_based=src_based, eval_mode=False)
        batch_scores = self.score(pivot_pool_embed.unsqueeze(1), pivot_pool_embed.unsqueeze(0))
        n = batch_scores.size(0)
        # remove the diagonal of scores
        batch_scores = batch_scores.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
        hard_scores = self.score(pivot_pool_embed.unsqueeze(1), pos_neg_pool_embed)
        scores = torch.cat((hard_scores, batch_scores), dim=1)
        labels = torch.zeros(scores.size(0)).long().to(exp_config.device_id)
        return self.loss_fct(scores, labels)

def baselines_cl_eval(srcs, mt_outs_dict, refs, emb_type, model, batch_size, score, tokenizer):
    with torch.no_grad():
        # load tokenizer and models, already specified addr for tokenizer
        mt_scores_dict = {'src': {}, 'ref': {}, 'src_ref': {}}
        score_funct = Similarity(temp=1) if score=='cosine' else nn.PairwiseDistance(p=2)
        # generate src embeddings
        for mt_name, mt_outs in mt_outs_dict.items():
            for key in mt_scores_dict:
                mt_scores_dict[key][mt_name] = []

            cur_data_dict = {'pivot': srcs, 'pos': refs, 'neg': mt_outs}
            cur_data_loader = preprocess_data(cur_data_dict, tokenizer, exp_config.max_length, 200, shuffle=False, sampler=False, src_based=True, eval_mode=True)
            for batch in cur_data_loader:
                # generate a batch of src, ref, mt embeddings
                src_pool_embeds, ref_pool_embeds, mt_pool_embeds = model(batch, emb_type, src_based=True, eval_mode=True)
                # the score is computed between (src and mt) and (ref and mt)
                src_ls, ref_ls = score_funct(src_pool_embeds, mt_pool_embeds).tolist(), score_funct(ref_pool_embeds, mt_pool_embeds).tolist()
                h_mean_ls = [2*src_score*ref_score/(src_score+ref_score) for src_score, ref_score in zip(src_ls, ref_ls)]
                if score == 'L2':
                    src_ls = [1/(1+score) for score in src_ls]
                    ref_ls = [1/(1+score) for score in ref_ls]
                    h_mean_ls = [1/(1+score) for score in h_mean_ls]
                mt_scores_dict['src'][mt_name].extend(src_ls)
                mt_scores_dict['ref'][mt_name].extend(ref_ls)
                mt_scores_dict['src_ref'][mt_name].extend(h_mean_ls)

        return mt_scores_dict

def store_cl_loss(model, loss_manager, train_batch, emb_type, src_based):
    print("start to eval loss")
    with torch.no_grad():
        save_dict = {}
        train_loss = loss_manager.loss_compute(train_batch, model, emb_type, src_based)
        save_dict['total_train_loss'] = train_loss.item()
    return save_dict

def store_corr_eval(evs, mt_scores_dict, mode, wmt, lang):
    # process the ground truth human ratings
    mqm_scores = evs.Scores(mode, 'mqm')
    qm_no_human = set(mqm_scores) - set(evs.all_refs)
    qm_human = qm_no_human.copy()
    qm_human.update(human_mapping_dict[wmt][lang])
    save_dict, temp_ls = {}, []

    for eval_type, scores_dict in mt_scores_dict.items():
        if mode == 'sys':
            # compute system-level scores (overwrite) otherwise seg scores are available already
            scores_dict = {mt_name: [sum(scores)/len(scores)] for mt_name, scores in scores_dict.items()}
        mqm_bp = evs.Correlation(mqm_scores, scores_dict, qm_human)
        mqm_bp_no = evs.Correlation(mqm_scores, scores_dict, qm_no_human)

        if mode == 'seg':
            save_dict[f'{eval_type}_seg_system_human']=mqm_bp.Kendall()[0]
            save_dict[f'{eval_type}_seg_system']=mqm_bp_no.Kendall()[0]
            temp_ls.append(mqm_bp.Kendall()[0])
        elif mode == 'sys':
            save_dict[f'{eval_type}_sys_system_human']=mqm_bp.Pearson()[0]
            save_dict[f'{eval_type}_sys_system']=mqm_bp_no.Pearson()[0]
            temp_ls.append(mqm_bp.Pearson()[0])
        else:
            print('Please choose between seg and sys!')
            exit(1)
    return max(temp_ls), save_dict


@click.command()
@click.option('-src_based', type=bool, help="if not specify or false means ref-based", default=False)
@click.option('-lang_dir', type=str, help="en_de or zh_en", default="zh_en")
@click.option('-gradient_accumulation_steps', default=1, type=int)
@click.option('-batch_size', type=int, help="train and eval batch size for contrastive learning", default=12)
@click.option('-emb_type', type=str, help="choose from last_layer, avg_first_last and states_concat", default="last_layer")
@click.option('-num_epoch', type=int, help="Number of epoches to train", default=5)
@click.option('-eval_step', type=int, help="Number of steps to evaluate", default=400)
@click.option('-num_warmup_steps', type=int, help="Number of steps to warm up", default=0)
@click.option('-save_dir_name', type=str, help="the dir name of weights being saved", default=None)
@click.option('-score_type', type=str, help="choose between L2 and cosine", default="cosine")
@click.option('-enable_loss_eval', type=bool, help="If given true, we will evaluate loss per eval step", default=True)
@click.option('-pivot_file', type=str, help="wiki_raw_2M.en", default="zh_en_2000000_train.txt")
@click.option('-pos_file', type=str, help="Optioanl: src will be pivot and target will be pos", default=None)
@click.option('-neg_file', type=str, help="wiki_cl_2M_5_zh_en.txt", default="mt_cl_5_zh_en.txt")
def main(src_based, lang_dir, gradient_accumulation_steps, batch_size, emb_type, num_epoch, eval_step, \
            num_warmup_steps, save_dir_name, score_type, enable_loss_eval, pivot_file, pos_file, neg_file):
    # load in eval data
    wmt, lang = 'wmt21.news', lang_dir.replace('_', '-')
    evs = data.EvalSet(wmt, lang)
    srcs, mt_outs_dict, refs = evs.src, evs.sys_outputs, evs.all_refs[evs.std_ref]
    # initalize the process
    dist.init_process_group(backend="nccl")
    rank = int(os.environ['LOCAL_RANK'])
    # only main process initalize wandb
    if rank == 0:
        # initalize the project parameters into Wandb, store experiment specific parameters
        wandb.init(project="ContraScore", config=
        {
            "src_based": src_based,
            "strategy": "Stratified CL",
            "epoch": num_epoch,
            "eval_step": eval_step,
            "emb_type": emb_type,
            "train batch size": batch_size * gradient_accumulation_steps * 8,
            "margin": 1 if score_type == 'L2' else 0.01,
            "score_type": score_type
        })

    exp_config.device_id = rank % torch.cuda.device_count()
    # set cuda device with rank and clear ram cache to ensure balanced ram allocations
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    # we will build bilingual sentence embedding models. XLM for both en-de and zh-en
    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large-tok")
    if emb_type == 'last_layer':
        model = Stratified_CL_XLM_Roberta("xlm-roberta-large-model").to(exp_config.device_id)
    else:
        print("Incorrect model type!")
        exit(1)

    # parallelize the pipeline into multiple gpus
    optimizer = AdamW(model.parameters(), lr=exp_config.lr)
    raw_neg_lines = open(neg_file, 'r').readlines()
    neg_samples = [ele[:-1].split('\t') for ele in raw_neg_lines]

    def file_to_dict(file_type):
        raw_pivot_lines = open(pivot_file, 'r').readlines()
        pivot_samples = [ele[:-1] for ele in raw_pivot_lines]
        if src_based:
            raw_pos_lines = open(pos_file, 'r').readlines()
            pos_samples = [ele[:-1] for ele in raw_pos_lines]
            triplets_score_dict = {'pivot': pivot_samples, 'pos': pos_samples, 'neg': neg_samples}
        else:
            triplets_score_dict = {'pivot': pivot_samples, 'neg': neg_samples}
        return triplets_score_dict

    triplets_score_train_dict = file_to_dict('train')
    train_dataloader = preprocess_data(triplets_score_train_dict, tokenizer, exp_config.max_length, batch_size, shuffle=True, sampler=True, \
        src_based=False, eval_mode=False)

    model = DDP(model, device_ids=[exp_config.device_id], find_unused_parameters=True)
    model.train()

    max_train_steps = math.ceil(len(train_dataloader) / gradient_accumulation_steps) * num_epoch
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    loss_manager = loss_funct(score_type)

    # save at end of epoch and at main processls
    if not os.path.isdir(f'{save_dir_name}') and rank == 0:
        os.makedirs(f'{save_dir_name}')

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epoch):
            # reset the best correlation each time at start of epoch
            cur_best_cor = float('-inf')
            torch.cuda.empty_cache() # empty cache in gpus
            train_dataloader.sampler.set_epoch(epoch) # set the sampler at each epoch
            for step, train_batch in enumerate(train_dataloader):
                # evaluate at every eval_step and also at the end of epoch (includes the beginning loss)
                if ((step % (eval_step * gradient_accumulation_steps) == 0) or (step == len(train_dataloader) - 1)) and rank == 0:
                    # store all the losses in wandb
                    print("start to evaluate!")
                    model.eval()
                    wandb_temp_dict = {}
                    # evaluate on the seg and sys correlations
                    start_eval_time = time.time()
                    mt_scores_dict = baselines_cl_eval(srcs, mt_outs_dict, refs, emb_type, model, batch_size, score_type, tokenizer)
                    step_seg_cor, save_seg_dict = store_corr_eval(evs, mt_scores_dict, 'seg', wmt, lang)
                    _, save_sys_dict = store_corr_eval(evs, mt_scores_dict, 'sys', wmt, lang)
                    wandb_temp_dict.update(save_seg_dict)
                    wandb_temp_dict.update(save_sys_dict)
                    # monitor the training loss
                    if enable_loss_eval:
                        save_loss_dict = store_cl_loss(model, loss_manager, train_batch, emb_type, src_based=src_based)
                        wandb_temp_dict.update(save_loss_dict)
                    wandb.log(wandb_temp_dict)
                    print("Testing Duration: ", time.time()-start_eval_time)
                    # save at the best epoch step
                    if step_seg_cor > cur_best_cor:
                        cur_best_cor=step_seg_cor
                        torch.save(model.module, f'{save_dir_name}/epoch{epoch}_best.ckpt')
                        print(f"Saved best model at current epoch {epoch}!")
                    model.train()
                    torch.cuda.empty_cache()

                train_loss = loss_manager.loss_compute(train_batch, model, emb_type, src_based)

                # accumulate losses at weights, each is normalized by accumulation steps
                train_loss = train_loss / gradient_accumulation_steps
                train_loss.backward()

                if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad() # clear the grads

if __name__ == "__main__":
    random.seed(10)
    main()
