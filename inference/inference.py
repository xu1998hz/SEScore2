"""This file contains all codes for inference for baseline models and ContraScore"""
"""CUDA_VISIBLE_DEVICES=0 python3 inference/inference.py -lang zh-en -wmt wmt21.news -model_addr new_margin_loss_weights/margin_src_ref_batch_last_layer_cosine_weights_1659947050 -emb_type last_layer -batch_size 250 -score L2"""

import click
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
)
import torch
# from train.cl_train import preprocess_data,exp_config
from train.cl_stratified_train import preprocess_data, exp_config
from mt_metrics_eval import data
import torch.nn as nn
import json
import wandb
import torch.nn as nn
import os
import glob

human_mapping_dict = {
    "wmt21.news": {
        'en-de': ['refA', 'refD'],
        'en-ru': ['refB'],
        'zh-en': ['refA']
    },
    "wmt20": {
        'en-de': 'refb',
        'zh-en': 'refb'
    },
    "wmt21.tedtalks": {
        
    }
}

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

class config():
    max_length = 256

def baselines_cl_eval(srcs, mt_outs_dict, refs, emb_type, model, batch_size, score, tokenizer):
    with torch.no_grad():
        # load tokenizer and models, already specified addr for tokenizer
        model.eval()
        mt_scores_dict = {'src': {}, 'ref': {}, 'src_ref': {}}
        score_funct = Similarity(temp=0.05) if score=='cosine' else nn.PairwiseDistance(p=2)
        # generate src embeddings
        for mt_name, mt_outs in mt_outs_dict.items():
            for key in mt_scores_dict:
                mt_scores_dict[key][mt_name] = []

            cur_data_dict = {'pivot': srcs, 'pos': refs, 'neg': mt_outs, 'margin': [0]*len(mt_outs)}
            cur_data_loader = preprocess_data(cur_data_dict, tokenizer, exp_config.max_length, batch_size, shuffle=False, sampler=False)
            for batch in cur_data_loader:
                # generate a batch of src, ref, mt embeddings
                pivot_pool_embeds, pos_pool_embeds, neg_pool_embeds = model(batch, emb_type)

                src_ls, ref_ls = score_funct(pivot_pool_embeds, pos_pool_embeds).tolist(), score_funct(pivot_pool_embeds, neg_pool_embeds).tolist()
                h_mean_ls = [2*src_score*ref_score/(src_score+ref_score) for src_score, ref_score in zip(src_ls, ref_ls)]
                if score == 'L2':
                    src_ls = [1/(1+score) for score in src_ls]
                    ref_ls = [1/(1+score) for score in ref_ls]
                    h_mean_ls = [1/(1+score) for score in h_mean_ls]
                mt_scores_dict['src'][mt_name].extend(src_ls)
                mt_scores_dict['ref'][mt_name].extend(ref_ls)
                mt_scores_dict['src_ref'][mt_name].extend(h_mean_ls)

        return mt_scores_dict

def corr_eval(evs, mt_scores_dict, mode, wmt, lang, outFile):
    # process the ground truth human ratings
    mqm_scores = evs.Scores(mode, 'mqm')
    qm_no_human = set(mqm_scores) - set(evs.all_refs)
    qm_human = qm_no_human.copy()
    qm_human.update(human_mapping_dict[wmt][lang])

    for eval_type, scores_dict in mt_scores_dict.items():
        print(f"Use {eval_type} for metric evaluation:", file=outFile)
        if mode == 'sys':
            # compute system-level scores (overwrite) otherwise seg scores are available already
            scores_dict = {mt_name: [sum(scores)/len(scores)] for mt_name, scores in scores_dict.items()}
        mqm_bp = evs.Correlation(mqm_scores, scores_dict, qm_human)
        mqm_bp_no = evs.Correlation(mqm_scores, scores_dict, qm_no_human)

        if mode == 'seg':
            print(f"MQM Segment-level Kendall for system + human: {mqm_bp.Kendall()[0]}", file=outFile)
            print(f"MQM Segment-level Kendall for system: {mqm_bp_no.Kendall()[0]}", file=outFile)
        elif mode == 'sys':
            print(f"MQM System-level Pearson for system + human: {mqm_bp.Pearson()[0]}", file=outFile)
            print(f"MQM System-level Pearson for system: {mqm_bp_no.Pearson()[0]}", file=outFile)
        else:
            print('Please choose between seg and sys!')
            exit(1)

@click.command()
@click.option('-lang', type=str, help="choose from zh-en, en-de and en-ru", default='zh-en')
@click.option('-wmt', type=str, help="choose from wmt21.news, wmt21.tedtalks. and wmt20", default='wmt21.news')
@click.option('-batch_size', type=int, help="batch size for each gpu")
@click.option('-score_file', type=str, help="load scores from a json file, addr of the file", default=None)
# belows are optional parameters for baseline cls
@click.option('-model_addr', default=None, help='locations of model weights')
@click.option('-emb_type', default=None, help="last_layer or states_concat")
@click.option('-score', default='L2')
def main(score_file, batch_size, lang, wmt, model_addr, emb_type, score):
    # load in data from WMT testing set
    evs = data.EvalSet(wmt, lang)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-tok")
    exp_config.device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if score_file:
        mt_scores_dict = json.load(open(score_file))
    else:
        srcs, mt_outs_dict, refs = evs.src, evs.sys_outputs, evs.all_refs[evs.std_ref]
        if os.path.isdir(f'{model_addr}'):
            for file_addr in glob.glob(f'{model_addr}/*'):
                saveFile = open(f'{file_addr}.json', 'w')
                outFile = open(f'{file_addr}.out', 'w')
                print(f"Current evaluation is at {wmt} for {lang}: ", file=outFile)
                model = torch.load(file_addr).to(exp_config.device_id)
                mt_scores_dict = baselines_cl_eval(srcs, mt_outs_dict, refs, emb_type, model, batch_size, score)
                saveFile.write(json.dumps(mt_scores_dict))
                print(f'{file_addr}.json is saved!', file=outFile)

                # compute the segment-level correlations
                corr_eval(evs, mt_scores_dict, 'seg', wmt, lang, outFile)
                # compute the system-level correlations
                corr_eval(evs, mt_scores_dict, 'sys', wmt, lang, outFile)

if __name__ == "__main__":
    main()
