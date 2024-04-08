import click
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    AdamW,
)
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import random
# from mt_metrics_eval import data
import time
from train.feedforward import FeedForward

# 1) obtain the triplet pairs (ranking difference and use it to determine margins)
# 2) use margin ranking loss for all triplets, based on rankings

human_mapping_dict = {
    "wmt21.news": {
        'en-de': ['refA', 'refD'],
        'en-ru': ['refB'],
        'zh-en': ['refA']
    },
    "wmt20": {
        'en-de': ['refb'],
        'zh-en': ['refb']
    },
}

class exp_config():
    max_length = 256
    temp = 0.1
    drop_out=0.1
    activation="Tanh"
    final_activation=None

def read_ja_eval_data():
    # load in all the evaluation data
    ref_lines = open('mqm_en_ja/mqm_en_ja_ref.txt', 'r').readlines()
    ref_lines = [line[:-1] for line in ref_lines]
    out_lines = open('mqm_en_ja/mqm_en_ja_sys.txt', 'r').readlines()
    out_lines = [line[:-1] for line in out_lines]
    gt_lines = open('mqm_en_ja/mqm_en_ja_scores.txt', 'r').readlines()
    gt_lines = [float(line[:-1]) for line in gt_lines]

    gt_score_dict = {}
    gt_score_dict['sys1'] = gt_lines[:118]
    gt_score_dict['sys2'] = gt_lines[118:118*2]
    gt_score_dict['sys3'] = gt_lines[118*2:118*3]
    gt_score_dict['sys4'] = gt_lines[118*3:]
    return ref_lines, out_lines, gt_score_dict

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

class loss_funct():
    def __init__(self):
        # be cautiously when applying cosine because order of pos neg will be opposite (targets=1)
        self.loss_fct = nn.MSELoss()

    # If src_based, ref will be used as pos samples (pos:0, batchNeg: 6)->max_diff=6. If ref_based, max_diff=5
    def loss_compute(self, batch, model, emb_type):
        score = model(batch, emb_type)
        return self.loss_fct(score, batch['score'].unsqueeze(1).to(exp_config.device_id))

def baselines_cl_eval(mt_outs_dict, refs, emb_type, model, batch_size, tokenizer):
    with torch.no_grad():
        # load tokenizer and models, already specified addr for tokenizer
        mt_scores_dict = {}
        # generate src embeddings
        for mt_name, mt_outs in mt_outs_dict.items():
            mt_scores_dict[mt_name] = []
            cur_data_dict = {'pivot': refs, 'mt': mt_outs}
            cur_data_loader = preprocess_data(cur_data_dict, tokenizer, exp_config.max_length, batch_size, shuffle=False, sampler=False, mode='test')
            for batch in cur_data_loader:
                # generate a batch of ref, mt embeddings
                score = model(batch, emb_type).squeeze(1).tolist()
                mt_scores_dict[mt_name].extend(score)
        return mt_scores_dict

def baselines_cl_eval_en_ja(mt_outs, refs, emb_type, model, batch_size, tokenizer):
    with torch.no_grad():
        mt_score_dict = {}
        seg_scores = []
        cur_data_dict = {'pivot': refs, 'mt': mt_outs}
        cur_data_loader = preprocess_data(cur_data_dict, tokenizer, exp_config.max_length, batch_size, shuffle=False, sampler=False, mode='test')
        for batch in cur_data_loader:
            # generate a batch of ref, mt embeddings
            score = model(batch, emb_type).squeeze(1).tolist()
            seg_scores.extend(score)

        mt_score_dict['sys1'] = seg_scores[:118]
        mt_score_dict['sys2'] = seg_scores[118:118*2]
        mt_score_dict['sys3'] = seg_scores[118*2:118*3]
        mt_score_dict['sys4'] = seg_scores[118*3:]
        return mt_score_dict

def store_regression_loss(model, loss_manager, train_batch, emb_type):
    with torch.no_grad():
        save_dict = {}
        train_loss = loss_manager.loss_compute(train_batch, model, emb_type)
        save_dict['total_train_loss'] = train_loss.item()
    return save_dict

def store_corr_eval_en_ja(evs, mt_scores_dict, gt_scores_dict):
    save_dict = {}
    mqm_bp = evs.Correlation(gt_scores_dict, mt_scores_dict, gt_scores_dict.keys())
    save_dict['seg']=mqm_bp.Kendall()[0]
    return mqm_bp.Kendall()[0], save_dict

def store_corr_eval_ted(evs, mt_scores_dict, mode, lang):
    mqm_scores = evs.Scores(mode, 'mqm')
    qm_no_human = set(mqm_scores) - set(evs.all_refs)

    save_dict = {}
    mqm_bp_no = evs.Correlation(mqm_scores, mt_scores_dict, qm_no_human)
    save_dict['seg_ted']=mqm_bp_no.Kendall()[0]
    return mqm_bp_no.Kendall()[0], save_dict

def store_corr_eval(evs, mt_scores_dict, mode, wmt, lang):
    # process the ground truth human ratings
    mqm_scores = evs.Scores(mode, 'mqm')
    qm_no_human = set(mqm_scores) - set(evs.all_refs)
    qm_human = qm_no_human.copy()
    qm_human.update(human_mapping_dict[wmt][lang])
    save_dict, temp_ls = {}, []

    if mode == 'sys':
        # compute system-level scores (overwrite) otherwise seg scores are available already
        mt_scores_dict = {mt_name: [sum(scores)/len(scores)] for mt_name, scores in mt_scores_dict.items()}
    mqm_bp = evs.Correlation(mqm_scores, mt_scores_dict, qm_human)
    mqm_bp_no = evs.Correlation(mqm_scores, mt_scores_dict, qm_no_human)

    if mode == 'seg':
        save_dict['seg_system_human']=mqm_bp.Kendall()[0]
        save_dict['seg_system']=mqm_bp_no.Kendall()[0]
        temp_ls.append(mqm_bp.Kendall()[0])
    elif mode == 'sys':
        save_dict['sys_system_human']=mqm_bp.Pearson()[0]
        save_dict['sys_system']=mqm_bp_no.Pearson()[0]
        temp_ls.append(mqm_bp.Pearson()[0])
    else:
        print('Please choose between seg and sys!')
        exit(1)
    return max(temp_ls), save_dict


@click.command()
@click.option('-lr', type=float, help="learning rate", default=1e-5)
@click.option('-lang_dir', type=str, help="en_de or zh_en", default="zh_en")
@click.option('-gradient_accumulation_steps', default=1, type=int)
@click.option('-batch_size', type=int, help="train and eval batch size for contrastive learning", default=40)
@click.option('-emb_type', type=str, help="choose from last_layer, avg_first_last and states_concat", default="last_layer")
@click.option('-num_epoch', type=int, help="Number of epoches to train", default=1)
@click.option('-eval_step', type=int, help="Number of steps to evaluate", default=400)
@click.option('-num_warmup_steps', type=int, help="Number of steps to warm up", default=0)
@click.option('-save_dir_name', type=str, help="the dir name of weights being saved", default=None)
@click.option('-enable_loss_eval', type=bool, help="If given true, we will evaluate loss per eval step", default=False)
@click.option('-data_file', type=str, help="fixed_wmt_news_cl_10_en_zh_rand_True_5.txt", default="fixed_wmt_news_cl_10_en_zh_rand_True_5.txt")
@click.option('-gradual_unfrozen', type=bool, help="Specify whether xlm needs to be frozen at beginning")
@click.option('-model_base', type=str, help="Specify which model base we should use: rembert or xlm-roberta-large")
@click.option('-hidden_size', type=int, help="model base's hidden size dim: 1024 or 1152")
@click.option('-model_addr', type=str, help="The addr of the model weight")
@click.option('-load_file_enable', type=bool)
@click.option('-test_ted_enable', type=bool)
@click.option('-alpha', type=float)
def main(lang_dir, gradient_accumulation_steps, batch_size, emb_type, num_epoch, eval_step, \
            num_warmup_steps, save_dir_name, enable_loss_eval, data_file, gradual_unfrozen, lr, model_base, hidden_size, \
            model_addr, load_file_enable, test_ted_enable, alpha):
    # load in eval data
    exp_config.lr = lr
    exp_config.hidden_size = hidden_size
    exp_config.hidden_size_FNN=[hidden_size*2, hidden_size]
    wmt, lang = 'wmt21.news', lang_dir.replace('_', '-')
    evs = data.EvalSet(wmt, lang)
    if lang == 'en-ja':
        ref_lines, out_lines, gt_score_dict = read_ja_eval_data()
    else:
        mt_outs_dict, refs = evs.sys_outputs, evs.all_refs[evs.std_ref]
        evs_ted = data.EvalSet('wmt21.tedtalks', lang)
        mt_outs_dict_ted, refs_ted = evs_ted.sys_outputs, evs_ted.all_refs[evs_ted.std_ref]
    # initalize the process
    dist.init_process_group(backend="nccl")
    rank = int(os.environ['LOCAL_RANK'])
    # only main process initalize wandb
    if rank == 0:
        # initalize the project parameters into Wandb, store experiment specific parameters
        wandb.init(project="ContraScore", config=
        {
            "strategy": "Regression",
            "epoch": num_epoch,
            "eval_step": eval_step,
            "emb_type": emb_type,
            "train batch size": batch_size * gradient_accumulation_steps * 8,
            "lr": exp_config.lr,
        })

    exp_config.device_id = rank % torch.cuda.device_count()
    # set cuda device with rank and clear ram cache to ensure balanced ram allocations
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    # we will build bilingual sentence embedding models. XLM for both en-de and zh-en
    tokenizer = AutoTokenizer.from_pretrained(f"{model_base}-tok")

    if load_file_enable:
        model = torch.load(model_addr, map_location="cpu").to(exp_config.device_id)
    else:
        if emb_type == 'last_layer':
            model = Regression_XLM_Roberta(f"{model_base}-model").to(exp_config.device_id)
        else:
            print("Incorrect model type!")
            exit(1)

    # parallelize the pipeline into multiple gpus
    optimizer = AdamW(model.parameters(), lr=exp_config.lr)

    data_lines = open(data_file, 'r').readlines()
    ref_ls, mt_ls, score_ls = [], [], []
    data_lines = [ele[:-1] for ele in data_lines]
    for line in data_lines:
        line_ls = line.split('\t')
        ref, mt, score = line_ls[0], line_ls[1], line_ls[2]
        ref_ls += [ref]
        mt_ls += [mt]
        score_ls += [float(score)]

    triplets_score_train_dict = {'pivot': ref_ls, 'mt': mt_ls, 'score': score_ls}
    train_dataloader = preprocess_data(triplets_score_train_dict, tokenizer, exp_config.max_length, batch_size, shuffle=True, sampler=True, mode='train')

    model = DDP(model, device_ids=[exp_config.device_id], find_unused_parameters=True)
    model.train()

    max_train_steps = math.ceil(len(train_dataloader) / gradient_accumulation_steps) * num_epoch
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    loss_manager = loss_funct()

    # save at end of epoch and at main processls
    if not os.path.isdir(f'{save_dir_name}') and rank == 0:
        os.makedirs(f'{save_dir_name}')

    global_step = 0 # monitor the overall steps to decide to unfreeze xlm layers
    if gradual_unfrozen:
        model.module.freeze_xlm() # freeze the xlm parameters

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epoch):
            # reset the best correlation each time at start of epoch
            cur_best_cor = float('-inf')
            torch.cuda.empty_cache() # empty cache in gpus
            train_dataloader.sampler.set_epoch(epoch) # set the sampler at each epoch
            for step, train_batch in enumerate(train_dataloader):
                # reset the best correlation at 25 * eval steps
                if (step % (eval_step * gradient_accumulation_steps * 5) == 0):
                    cur_best_cor = float('-inf')
                # evaluate at every eval_step and also at the end of epoch (includes the beginning loss)
                if ((step % (eval_step * gradient_accumulation_steps) == 0) or (step == len(train_dataloader) - 1)) and rank == 0:
                    # store all the losses in wandb
                    print("start to evaluate!")
                    model.eval()
                    wandb_temp_dict = {}
                    # evaluate on the seg and sys correlations
                    start_eval_time = time.time()
                    if lang == 'en-ja': # evs, mt_scores_dict, gt_scores_dict
                        mt_scores_dict = baselines_cl_eval_en_ja(out_lines, ref_lines, emb_type, model, 200, tokenizer)
                        step_seg_cor, save_seg_dict = store_corr_eval_en_ja(evs, mt_scores_dict, gt_score_dict)
                        wandb_temp_dict.update(save_seg_dict)
                    else:
                        mt_scores_dict = baselines_cl_eval(mt_outs_dict, refs, emb_type, model, 200, tokenizer)
                        step_seg_cor, save_seg_dict = store_corr_eval(evs, mt_scores_dict, 'seg', wmt, lang)
                        _, save_sys_dict = store_corr_eval(evs, mt_scores_dict, 'sys', wmt, lang)
                        wandb_temp_dict.update(save_seg_dict)
                        wandb_temp_dict.update(save_sys_dict)
                        if test_ted_enable:
                            mt_scores_dict_ted = baselines_cl_eval(mt_outs_dict_ted, refs_ted, emb_type, model, 200, tokenizer)
                            step_seg_cor_ted, save_seg_dict_ted = store_corr_eval_ted(evs_ted, mt_scores_dict_ted, 'seg', lang)
                            wandb_temp_dict.update(save_seg_dict_ted)
                    # monitor the training loss
                    if enable_loss_eval:
                        save_loss_dict = store_regression_loss(model, loss_manager, train_batch, emb_type)
                        wandb_temp_dict.update(save_loss_dict)
                    wandb.log(wandb_temp_dict)
                    print("Testing Duration: ", time.time()-start_eval_time)
                    # save at the best epoch step
                    if step_seg_cor > cur_best_cor:
                        cur_best_cor=step_seg_cor
                        torch.save(model.module, f'{save_dir_name}/epoch{epoch}_best_{step}.ckpt')
                        print(f"Saved best model at current epoch {epoch}!")
                    model.train()
                    torch.cuda.empty_cache()

                if gradual_unfrozen and (global_step == int(len(train_dataloader)*alpha)):
                    # unfreeze the model paramters after the first evaluation step
                    model.module.unfreeze_xlm()

                train_loss = loss_manager.loss_compute(train_batch, model, emb_type)

                # accumulate losses at weights, each is normalized by accumulation steps
                train_loss = train_loss / gradient_accumulation_steps
                train_loss.backward()

                if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad() # clear the grads

                global_step += 1

if __name__ == "__main__":
    random.seed(10)
    main()
