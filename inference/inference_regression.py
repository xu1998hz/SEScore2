import click
import torch
import json
from train.regression import *

# python3 inference/inference_regression.py -wmt wmt21.news -lang zh-en -model_addr epoch0_best_3616_zhen_pos_neg_sep20.ckpt
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

def store_corr_eval(evs, mt_scores_dict, mode, wmt, lang):
    # process the ground truth human ratings
    mqm_scores = evs.Scores(mode, 'mqm')
    qm_no_human = set(mqm_scores) - set(evs.all_refs)
    qm_human = qm_no_human.copy()
    if wmt != 'wmt21.tedtalks':
        qm_human.update(human_mapping_dict[wmt][lang])

    if mode == 'sys':
        # compute system-level scores (overwrite) otherwise seg scores are available already
        mt_scores_dict = {mt_name: [sum(scores)/len(scores)] for mt_name, scores in mt_scores_dict.items()}
    mqm_bp = evs.Correlation(mqm_scores, mt_scores_dict, qm_human)
    mqm_bp_no = evs.Correlation(mqm_scores, mt_scores_dict, qm_no_human)

    if mode == 'seg':
        print("seg_system_human: ", mqm_bp.Kendall()[0])
        print("seg_system: ", mqm_bp_no.Kendall()[0])
    elif mode == 'sys':
        print("sys_system_human: ", mqm_bp.Pearson()[0])
        print("sys_system: ", mqm_bp_no.Pearson()[0])
    else:
        print('Please choose between seg and sys!')
        exit(1)

@click.command()
@click.option('-wmt')
@click.option('-lang')
@click.option('-load_file', default=None)
@click.option('-model_base')
@click.option('-model_addr', default=None, help='locations of model weights')
def main(lang, wmt, model_addr, load_file, model_base):
    evs = data.EvalSet(wmt, lang)
    mt_outs_dict, refs = evs.sys_outputs, evs.all_refs[evs.std_ref]
    exp_config.device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not load_file:
        model = torch.load(model_addr).to(exp_config.device_id)
        model.eval()
        print("model is loaded!")
        tokenizer = AutoTokenizer.from_pretrained(f"{model_base}-tok")
        mt_scores_dict = baselines_cl_eval(mt_outs_dict, refs, 'last_layer', model, 200, tokenizer)
        with open(f'{lang}_{wmt}.json', 'w') as f:
            json.dump(mt_scores_dict, f)
        print("result is saved!")
    else:
        mt_scores_dict = json.load(open(load_file))
    store_corr_eval(evs, mt_scores_dict, 'seg', wmt, lang)
    store_corr_eval(evs, mt_scores_dict, 'sys', wmt, lang)

if __name__ == "__main__":
    main()
