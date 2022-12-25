import click
from mt_metrics_eval import data
import json
import wandb
from inference import *

# WANDB_NAME=margin_src_ref_mt1_batch_last_layer_L2_zh_en_sys python3 inference/log_scores_wandb.py -model_addr new_margin_loss_weights/margin_src_ref_mt1_batch_last_layer_L2_weights_1659948126 -mode sys -wmt wmt21.news -lang zh-en
# WMT21 en-de src, ref, src_ref based System-level Correlation With Human

def corr_eval_wandb(evs, mt_scores_dict, mode, wmt, lang, model_addr):
    # process the ground truth human ratings
    mqm_scores = evs.Scores(mode, 'mqm')
    qm_no_human = set(mqm_scores) - set(evs.all_refs)
    qm_human = qm_no_human.copy()
    qm_human.update(human_mapping_dict[wmt][lang])

    final_dict = {}
    for eval_type, scores_dict in mt_scores_dict.items():
        print(f"Use {eval_type} at {mode} for metric evaluation:")
        if mode == 'sys':
            # compute system-level scores (overwrite) otherwise seg scores are available already
            scores_dict = {mt_name: [sum(scores)/len(scores)] for mt_name, scores in scores_dict.items()}
        mqm_bp = evs.Correlation(mqm_scores, scores_dict, qm_human)
        mqm_bp_no = evs.Correlation(mqm_scores, scores_dict, qm_no_human)

        if mode == 'seg':
            final_dict[eval_type]=[mqm_bp.Kendall()[0], mqm_bp_no.Kendall()[0]]
        elif mode == 'sys':
            final_dict[eval_type]=[mqm_bp.Pearson()[0], mqm_bp_no.Pearson()[0]]
        else:
            print("your mode is wrong!")
            exit(1)

    wandb.log({
        f"src_{wmt}_{lang}_{mode}_{model_addr}_with_human": final_dict['src'][0],
        f"src_{wmt}_{lang}_{mode}_{model_addr}_without_human": final_dict['src'][1],
        f"ref_{wmt}_{lang}_{mode}_{model_addr}_with_human": final_dict['ref'][0],
        f"ref_{wmt}_{lang}_{mode}_{model_addr}_without_human": final_dict['ref'][1],
        f"src_ref_{wmt}_{lang}_{mode}_{model_addr}_with_human": final_dict['src_ref'][0],
        f"src_ref_{wmt}_{lang}_{mode}_{model_addr}_without_human": final_dict['src_ref'][1],
    })

@click.command()
@click.option('-model_addr')
@click.option('-mode')
@click.option('-wmt', default="wmt21.news")
@click.option('-lang', default="en-de")
def main(model_addr, mode, wmt, lang):
    wandb.init(project="ContraScore", config=
    {
        'mode': mode,
        'model_type': model_addr,
        'type': "eval_score_mode"
    })

    evs = data.EvalSet('wmt21.news', lang)
    for i in range(5):
        mt_scores_dict = json.load(open(f'{model_addr}/epoch{i}.ckpt.json'))
        print(f"load in {model_addr}/epoch{i}.ckpt.json")

        corr_eval_wandb(evs, mt_scores_dict, mode, wmt, lang, model_addr.split('/')[1])

if __name__ == "__main__":
    main()
