import click
import sys
sys.path.insert(0, '/home/wendaxu/SEScore2')
# from nltk.tokenize import word_tokenize
import nltk
# from train.regression import *
import json
import pandas as pd
# import evaluate
# nltk.download('punkt')

# speech translation:
# 1) baseline: CUDA_VISIBLE_DEVICES=0 python3 inference/test_all_metrics.py -metric_name comet-da-20 -benchmark_name st-en-ja
# 2) sescore2: CUDA_VISIBLE_DEVICES=0 python3 inference/test_all_metrics.py -metric_name sescore2 -benchmark_name st-en-ja -ckpt ckpt -model_base google/rembert

# CUDA_VISIBLE_DEVICES=0 python3 inference/test_all_metrics.py -metric_name sescore2 -benchmark_name x-en-wmt20-da -ckpt epoch0_best_en_de_rembert.ckpt -model_base rembert -lp
def test_save_metric_benchmark(metric_name, benchmark_name, ref_lines, out_lines, lang, ckpt, model_base=None, tot_src_ls=None):
    if metric_name == 'comet-da-20':
        from comet import download_model, load_from_checkpoint
        model_path = download_model("wmt20-comet-da")
        model = load_from_checkpoint(model_path)
        if benchmark_name == 'st-en-ja':
            src_lines = open('mqm_en_ja/mqm_en_ja_src.txt', 'r').readlines()
            src_lines = [line[:-1] for line in src_lines]
            dataset = [{"src": src, "ref": ref, "mt": out} for src, ref, out in zip(src_lines, ref_lines, out_lines)]
            seg_scores, _ = model.predict(dataset, batch_size=16, gpus=1)
        elif benchmark_name == 'mt-de-en':
            src_lines = open('de_en_benchmark/en_de.src', 'r').readlines()
            src_lines = [line[:-1] for line in src_lines]
            dataset = [{"src": src, "ref": ref, "mt": out} for src, ref, out in zip(src_lines, ref_lines, out_lines)]
            seg_scores, _ = model.predict(dataset, batch_size=16, gpus=1)
        elif benchmark_name == 'x-en-wmt20-da':
            src_lines = tot_src_ls
            dataset = [{"src": src, "ref": ref, "mt": out} for src, ref, out in zip(src_lines, ref_lines, out_lines)]
            seg_scores, _ = model.predict(dataset, batch_size=16, gpus=1)
        else:
            print("Your current benchmark doesn't support comet!")
            exit(1)
    elif metric_name == 'bleurt-20':
        from bleurt import score
        checkpoint = "bleurt/BLEURT-20"
        scorer = score.BleurtScorer(checkpoint)
        seg_scores = scorer.score(references=ref_lines, candidates=out_lines)
    elif metric_name == 'sent-bleu':
        seg_scores=[]
        sacrebleu = evaluate.load("sacrebleu")
        for out_line, ref_line in zip(out_lines, ref_lines):
            if lang == 'ja':
                results = sacrebleu.compute(predictions=[out_line], references=[[ref_line]], tokenize='ja-mecab')
            else:
                results = sacrebleu.compute(predictions=[out_line], references=[[ref_line]])
            seg_scores += [results["score"]]
    elif metric_name == 'chrf':
        seg_scores=[]
        chrf = evaluate.load("chrf")
        for ref, hyp in zip(ref_lines, out_lines):
            results = chrf.compute(predictions=[hyp], references=[[ref]])
            seg_scores+=[results['score']]
    elif metric_name == 'ter':
        seg_scores=[]
        ter = evaluate.load("ter")
        for ref, hyp in zip(ref_lines, out_lines):
            if lang == 'ja':
                results = ter.compute(predictions=[hyp], references=[[ref]], normalized=True, support_zh_ja_chars=True)
            else:
                results = ter.compute(predictions=[hyp], references=[[ref]], case_sensitive=True)
            seg_scores+=[results['score']]
    elif metric_name == 'prism':
        import os
        from prism import Prism
        prism = Prism(model_dir=os.environ['MODEL_DIR'], lang=lang)
        seg_scores=prism.score(cand=out_lines, ref=ref_lines, segment_scores=True)
    elif metric_name == 'bartscore':
        from bart_score import BARTScorer
        bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large')
        seg_scores=bart_scorer.score(out_lines, ref_lines, batch_size=16)
    elif metric_name == 'bertscore':
        from bert_score import score
        seg_scores=[]
        if lang != 'en':
            bertscore_lang = 'others'
        else:
            bertscore_lang = 'en'
        _, _, F = score(out_lines, ref_lines, lang=bertscore_lang)
        seg_scores=F.tolist()
    elif metric_name == 'sescore':
        from comet import load_from_checkpoint
        if lang == 'en':
            model = load_from_checkpoint('sescore_ckpt/zh_en/checkpoint/english.ckpt')
        elif lang == 'de':
            model = load_from_checkpoint('sescore_ckpt/en_de/checkpoint/sescore_german.ckpt')
        else:
            print("SEScore only supports en and de!")
        dataset = [{"src": ref, "mt": out} for ref, out in zip(ref_lines, out_lines)]
        seg_scores, _ = model.predict(dataset, batch_size=16, gpus=1)
    elif metric_name == 'sescore2':
        exp_config.device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = torch.load(ckpt).to(exp_config.device_id)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(f"{model_base}-tok")
        seg_scores=[]
        with torch.no_grad():
            cur_data_dict = {'pivot': ref_lines, 'mt': out_lines}
            cur_data_loader = preprocess_data(cur_data_dict, tokenizer, exp_config.max_length, 200, shuffle=False, sampler=False, mode='test')
            for batch in cur_data_loader:
                # generate a batch of ref, mt embeddings
                score = model(batch, 'last_layer').squeeze(1).tolist()
                seg_scores.extend(score)
    else:
        print("Your current metric name is not supported!")
        exit(1)

    return seg_scores

def s2t_en_ja_benchmark(seg_scores, gt_lines):
    from mt_metrics_eval import data
    evs = data.EvalSet('wmt21.news', 'zh-en')
    mt_score_dict, gt_score_dict = {}, {}

    mt_score_dict['sys1'] = seg_scores[:78]
    mt_score_dict['sys2'] = seg_scores[78:78*2]
    mt_score_dict['sys3'] = seg_scores[78*2:78*3]
    mt_score_dict['sys4'] = seg_scores[78*3:]

    gt_score_dict['sys1'] = gt_lines[:78]
    gt_score_dict['sys2'] = gt_lines[78:78*2]
    gt_score_dict['sys3'] = gt_lines[78*2:78*3]
    gt_score_dict['sys4'] = gt_lines[78*3:]

    mqm_bp = evs.Correlation(gt_score_dict, mt_score_dict, mt_score_dict.keys())
    print(mqm_bp.Kendall())

def mt_en_de_benchmark(seg_scores, gt_lines):
    from mt_metrics_eval import data
    evs = data.EvalSet('wmt21.news', 'zh-en')
    mt_score_dict, gt_score_dict = {}, {}

    for index, i in enumerate(range(0, 700, 100)):
        mt_score_dict[f'sys{index}'] = seg_scores[i:i+100]
        gt_score_dict[f'sys{index}'] = gt_lines[i:i+100]

    mqm_bp = evs.Correlation(gt_score_dict, mt_score_dict, mt_score_dict.keys())
    print(mqm_bp.Kendall())

@click.command()
@click.option('-metric_name')
@click.option('-benchmark_name')
@click.option('-ckpt', help="only used to sescore2")
@click.option('-load_file', type=bool, default=False)
@click.option('-file_first')
@click.option('-file_sec')
@click.option('-caption_file')
@click.option('-model_base')
@click.option('-lp')
@click.option('-lang', help="use en, ja or de")
def main(metric_name, benchmark_name, load_file, ckpt, file_first, file_sec, caption_file, model_base, lp, lang):
    if load_file:
        if benchmark_name == 'st-en-ja':
            gt_lines = open('mqm_en_ja/mqm_en_ja_scores.txt', 'r').readlines()
            gt_lines = [float(line[:-1]) for line in gt_lines]
            seg_scores = open(file_first, 'r').readlines()
            seg_scores = [float(score[:-1]) for score in seg_scores]
            s2t_en_ja_benchmark(seg_scores, gt_lines)
        elif benchmark_name == 'mt-de-en':
            gt_lines = open('de_en_benchmark/en_de_tot.scores', 'r').readlines()
            gt_lines = [float(line[:-1]) for line in gt_lines]
            seg_scores = open(file_first, 'r').readlines()
            seg_scores = [float(score[:-1]) for score in seg_scores]
            mt_en_de_benchmark(seg_scores, gt_lines)
        elif benchmark_name == 'paws':
            better = open(file_first, 'r').readlines()
            better = [float(ele[:-1]) for ele in better]
            worse = open(file_sec, 'r').readlines()
            worse = [float(ele[:-1]) for ele in worse]
            import numpy as np
            diff = np.array(better)-np.array(worse)
            concordant, total = np.sum(diff>0), len(better)
            discordant = total-concordant
            print(concordant)
            print((concordant-discordant)/total)
        elif benchmark_name == 'webnlg2020':
            import numpy as np
            model_dicts = json.load(open(file_first))
            gt_dicts = json.load(open(file_sec))
            gt_dicts_cor = {sys: list(gt_dicts[sys]['Correctness'].values()) for sys in gt_dicts}
            gt_dicts_cov = {sys: list(gt_dicts[sys]['DataCoverage'].values()) for sys in gt_dicts}
            gt_dicts_flu = {sys: list(gt_dicts[sys]['Fluency'].values()) for sys in gt_dicts}
            gt_dicts_rel = {sys: list(gt_dicts[sys]['Relevance'].values()) for sys in gt_dicts}
            gt_dicts_str = {sys: list(gt_dicts[sys]['TextStructure'].values()) for sys in gt_dicts}

            gt_dicts_tot = {sys: (np.array(list(gt_dicts[sys]['Correctness'].values())) + \
            np.array(list(gt_dicts[sys]['DataCoverage'].values())) + \
            np.array(list(gt_dicts[sys]['Fluency'].values())) + np.array(list(gt_dicts[sys]['Relevance'].values())) + \
            np.array(list(gt_dicts[sys]['TextStructure'].values())))/5 for sys in gt_dicts}

            gt_dicts_tot = {sys: gt_dicts_tot[sys].tolist() for sys in gt_dicts_tot}

            from mt_metrics_eval import data
            evs = data.EvalSet('wmt21.news', 'zh-en')
            mqm_bp = evs.Correlation(gt_dicts_cor, model_dicts, model_dicts.keys())
            print("Correctness: ", mqm_bp.Kendall())
            mqm_bp = evs.Correlation(gt_dicts_cov, model_dicts, model_dicts.keys())
            print("Coverage: ", mqm_bp.Kendall())
            mqm_bp = evs.Correlation(gt_dicts_flu, model_dicts, model_dicts.keys())
            print("Fluency: ", mqm_bp.Kendall())
            mqm_bp = evs.Correlation(gt_dicts_rel, model_dicts, model_dicts.keys())
            print("Relevance: ", mqm_bp.Kendall())
            mqm_bp = evs.Correlation(gt_dicts_str, model_dicts, model_dicts.keys())
            print("Text Stucture: ", mqm_bp.Kendall())
            mqm_bp = evs.Correlation(gt_dicts_tot, model_dicts, model_dicts.keys())
            print("Total: ", mqm_bp.Kendall())
        elif benchmark_name == 'webnlg':
            model_dicts = json.load(open(file_first))
            new_model_dicts, flu_gt_dicts, gra_gt_dicts, sem_gt_dicts = {}, {}, {}, {}
            df = pd.read_csv('./webnlg-human-evaluation/all_data_final_averaged.csv')
            for team, seg_score_dict in model_dicts.items():
                new_model_dicts[team] = [seg_score_dict[str(i)] for i in range(223)]
                flu_gt_dicts[team] = list(df.loc[df['team'] == team]['fluency'])
                gra_gt_dicts[team] = list(df.loc[df['team'] == team]['grammar'])
                sem_gt_dicts[team] = list(df.loc[df['team'] == team]['semantics'])
            from mt_metrics_eval import data
            evs = data.EvalSet('wmt21.news', 'zh-en')
            mqm_bp = evs.Correlation(flu_gt_dicts, new_model_dicts, new_model_dicts.keys())
            print("fluency: ", mqm_bp.Kendall())
            mqm_bp = evs.Correlation(gra_gt_dicts, new_model_dicts, new_model_dicts.keys())
            print("gra: ", mqm_bp.Kendall())
            mqm_bp = evs.Correlation(sem_gt_dicts, new_model_dicts, new_model_dicts.keys())
            print("sem: ", mqm_bp.Kendall())
    else:
        if benchmark_name == 'st-en-ja':
            ref_lines = open('mqm_en_ja/mqm_en_ja_ref.txt', 'r').readlines()
            ref_lines = [line[:-1] for line in ref_lines]
            out_lines = open('mqm_en_ja/mqm_en_ja_sys.txt', 'r').readlines()
            out_lines = [line[:-1] for line in out_lines]
            save_name = f'{benchmark_name}_{metric_name}_results.txt'
            save_file = open(save_name, 'w')
            seg_scores = test_save_metric_benchmark(metric_name, benchmark_name, ref_lines, out_lines, lang, ckpt, model_base)
            seg_scores = [str(score)+'\n' for score in seg_scores]
            save_file.writelines(seg_scores)
            print("file is saved!")
        elif benchmark_name == 'mt-de-en':
            ref_lines = open('de_en_benchmark/en_de.ref', 'r').readlines()
            ref_lines = [line[:-1] for line in ref_lines]
            out_lines = open('de_en_benchmark/en_de.out', 'r').readlines()
            out_lines = [line[:-1] for line in out_lines]
            save_name = f'{benchmark_name}_{metric_name}_results.txt'
            save_file = open(save_name, 'w')
            seg_scores = test_save_metric_benchmark(metric_name, benchmark_name, ref_lines, out_lines, lang, ckpt, model_base)
            seg_scores = [str(score)+'\n' for score in seg_scores]
            save_file.writelines(seg_scores)
            print("file is saved!")
        elif benchmark_name == 'paws':
            ref_lines = open('paws_ref.txt', 'r').readlines()
            ref_lines = [line[:-1] for line in ref_lines]
            out_lines = open('paws_better.txt', 'r').readlines()
            out_lines = [line[:-1] for line in out_lines]
            # save the better scores
            save_name = f'{benchmark_name}_{metric_name}_better_results.txt'
            seg_scores = test_save_metric_benchmark(metric_name, benchmark_name, ref_lines, out_lines, lang, ckpt, model_base)
            out_lines = open('paws_worse.txt', 'r').readlines()
            out_lines = [line[:-1] for line in out_lines]
            # save the worse scores
            save_name = f'{benchmark_name}_{metric_name}_worse_results.txt'
            save_file = open(save_name, 'w')
            seg_scores = test_save_metric_benchmark(metric_name, benchmark_name, ref_lines, out_lines, lang, ckpt, model_base)
            seg_scores = [str(score)+'\n' for score in seg_scores]
            save_file.writelines(seg_scores)
            print("files are saved!")
        elif benchmark_name == 'caption':
            ref_out_lines = open(caption_file, 'r').readlines()
            file_index = caption_file.split('.')[0][-1]
            cap_prefix = caption_file.split('.')[0].split('_')[0]
            index_ls = [ele.split('\t')[0] for ele in ref_out_lines]
            ref_ls = [ele.split('\t')[1] for ele in ref_out_lines]
            out_ls = [ele.split('\t')[2][:-1] for ele in ref_out_lines]
            save_name = f'{benchmark_name}_{cap_prefix}_{metric_name}_results_{model_base}_{file_index}.txt'
            save_file = open(save_name, 'w')
            seg_scores = test_save_metric_benchmark(metric_name, benchmark_name, ref_ls, out_ls, lang, ckpt, model_base)
            seg_scores = [str(score)+'\n' for score in seg_scores]
            save_file.writelines(seg_scores)
        elif benchmark_name == 'webnlg2020':
            sys_score_output_dict = json.load(open('webnlg2020.json'))
            sys_seg_scores_dict = {}
            for sys in sys_score_output_dict:
                out_ls = list(sys_score_output_dict[sys]['Output'].values())
                ref_ls = list(sys_score_output_dict[sys]['Ref'].values())
                seg_scores = test_save_metric_benchmark(metric_name, benchmark_name, ref_ls, out_ls, lang, ckpt, model_base)
                sys_seg_scores_dict[sys] = seg_scores
            with open(f'{benchmark_name}_{metric_name}_results.json', 'w') as f:
                json.dump(sys_seg_scores_dict, f)
            print("Metric outputs are saved!")
        elif benchmark_name == 'webnlg':
            df = pd.read_csv('./webnlg-human-evaluation/all_data_final_averaged.csv')
            team_segid_scores = {}
            for i in range(3):
                ref1_ls = open(f'./webnlg-human-evaluation/gold-sample-reference{i}.lex', 'r').readlines()
                for team in set(df['team']):
                    cur_data = df.loc[df['team'] == team]
                    outs = []
                    refs = []
                    srcs = open('./webnlg-human-evaluation/MRs.txt', 'r').readlines()
                    assert(len(srcs)==223)

                    assert(len(ref1_ls)==223)
                    index_ls = []
                    for index, sen in enumerate(ref1_ls):
                        if sen != '\n':
                            index_ls += [index]
                    srcs = [src[:-1] for src in srcs[:-1]]+[srcs[-1]]
                    src_ref_mapping = {src.strip(): ref for src, ref in zip(srcs, ref1_ls)}

                    print(len(cur_data['mr']))
                    for mr_src, out_ele in zip(cur_data['mr'], cur_data['text']):
                        if src_ref_mapping[mr_src.strip()] != '\n':
                            refs.append(src_ref_mapping[mr_src])
                            if pd.isna(out_ele):
                                outs.append('')
                            else:
                                outs.append(out_ele)

                    seg_scores = test_save_metric_benchmark(metric_name, benchmark_name, refs, outs, lang, ckpt, model_base)

                    print("after seg gen")

                    if team not in team_segid_scores:
                        team_segid_scores[team] = {}
                    for index, ele_score in zip(index_ls, seg_scores):
                        if index not in team_segid_scores[team]:
                            team_segid_scores[team][index] = []
                        team_segid_scores[team][index]+=[ele_score]

            final_dict = {}
            # print(team_segid_scores)
            for team, id_score_dict in team_segid_scores.items():
                if team not in final_dict:
                    final_dict[team] = {}
                for id, score_ls in id_score_dict.items():
                    final_dict[team][id] = max(score_ls)

            save_name = f'{benchmark_name}_{metric_name}_results_{model_base}.json'
            with open(save_name, 'w') as f:
                json.dump(final_dict, f)

            print(f"{save_name} is saved!")
        elif benchmark_name == 'x-en-wmt20-da':
            import numpy as np
            data = pd.read_csv('2020-daRR.csv').dropna()
            dict_data = pd.read_csv('2020-da.csv')
            print(set(data['lp']))

            cur_lang_data = data.loc[data['lp'] == lp]
            cur_dict_data = dict_data.loc[dict_data['lp'] == lp]

            new_dict_data={}
            for mt, mt_score in zip(cur_dict_data['mt'], cur_dict_data['score']):
                if mt not in new_dict_data:
                    new_dict_data[mt]=[]
                new_dict_data[mt]+=[float(mt_score)]

            cur_lang_data_1 = cur_lang_data[cur_lang_data['better.model']=='OPPO.1360']
            cur_lang_data_1 = cur_lang_data_1[cur_lang_data_1['worse.model']=='Tohoku-AIP-NTT.1442']

            cur_lang_data_2 = cur_lang_data[cur_lang_data['better.model']=='Tohoku-AIP-NTT.1442']
            cur_lang_data_2 = cur_lang_data_2[cur_lang_data_2['worse.model']=='OPPO.1360']

            cur_lang_data = pd.concat([cur_lang_data_1, cur_lang_data_2], ignore_index=True, axis=0)

            print(cur_lang_data)

            new_dict_data = {key: sum(val_ls)/len(val_ls) for key, val_ls in new_dict_data.items()}
            tot_better_ls, tot_worse_ls, tot_ref_ls, tot_src_ls = [], [], [], []
            print(len(cur_lang_data['lp']))
            for better_ele, worse_ele, ref_ele, src_ele in zip(cur_lang_data['better'], cur_lang_data['worse'], \
                cur_lang_data['ref'], cur_lang_data['src']):
                if new_dict_data[better_ele] > new_dict_data[worse_ele]+25.0:
                    tot_better_ls+=[better_ele]
                    tot_worse_ls+=[worse_ele]
                    tot_ref_ls+=[ref_ele]
                    tot_src_ls+=[src_ele]

            print("length of evaluation: ", len(tot_better_ls))
            refs, better_outs, worse_outs = tot_ref_ls, tot_better_ls, tot_worse_ls  # cur_lang_data['ref'], cur_lang_data['better'], cur_lang_data['worse']

            better_segs = test_save_metric_benchmark(metric_name, benchmark_name, refs, better_outs, ckpt, tot_src_ls, model_base)
            save_better_segs = [str(score)+'\n' for score in better_segs]
            save_better = open(f'{benchmark_name}_{metric_name}_{model_base}_better.txt', 'w')
            save_better.writelines(save_better_segs)

            worse_segs = test_save_metric_benchmark(metric_name, benchmark_name, refs, worse_outs, ckpt, tot_src_ls, model_base)
            save_worse_segs = [str(score)+'\n' for score in worse_segs]
            save_worse = open(f'{benchmark_name}_{metric_name}_{model_base}_worse.txt', 'w')
            save_worse.writelines(save_worse_segs)

            # calculate using Kendall Tau-like correlations
            diff = np.array(better_segs)-np.array(worse_segs)
            concordant, total = np.sum(diff>0), len(better_segs)
            discordant = total-concordant
            print(concordant)
            print(total)
            print("Kendall Tau-like: ", (concordant-discordant)/total)
        else:
            print("benchmark is not supported!")

if __name__ == "__main__":
    main()
