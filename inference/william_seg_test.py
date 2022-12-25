import click
import numpy as np
import scipy

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
    return mqm_bp.Kendall()[0]

def WilliamsTest(r12, r13, r23, n, one_sided=True):
    """Return Williams test p-value for given Pearson correlations."""
    k = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r23 * r13
    rbar = ((r12 + r13) / 2)
    tnum = (r12 - r13) * np.sqrt((n - 1) * (1 + r23))
    tden = np.sqrt(2 * (n - 1) / (n - 3) * k + rbar**2 * (1 - r23)**3)
    p = scipy.stats.t.sf(np.abs(tnum / tden), n - 3)
    return p if one_sided else 2 * p

@click.command()
@click.option('-file_first')
@click.option('-file_sec')
def main(file_first, file_sec):
    gt_lines = open('mqm_en_ja/mqm_en_ja_scores.txt', 'r').readlines()
    gt_lines = [-float(line[:-1]) for line in gt_lines]
    seg_scores_1 = open(file_first, 'r').readlines()
    seg_scores_1 = [float(score[:-1]) for score in seg_scores_1]
    r1 = s2t_en_ja_benchmark(seg_scores_1, gt_lines)

    seg_scores_2 = open(file_sec, 'r').readlines()
    seg_scores_2 = [float(score[:-1]) for score in seg_scores_2]
    r2 = s2t_en_ja_benchmark(seg_scores_2, gt_lines)

    r12 = s2t_en_ja_benchmark(seg_scores_2, seg_scores_1)
    n = len(gt_lines)

    print(WilliamsTest(r1, r2, r12, n, one_sided=True))

if __name__ == "__main__":
    main()
