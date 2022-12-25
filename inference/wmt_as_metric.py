from mt_metrics_eval import data

human_mapping_dict = {
    "wmt21.news": {
        'en-de': ['refA'],
        'en-ru': ['refB'],
        'zh-en': ['refA']
    },
    "wmt20": {
        'en-de': ['refb'],
        'zh-en': ['refb']
    },
}

wmt, lang = 'wmt21.news', 'en-de'
evs = data.EvalSet(wmt, lang)

mqm_scores = evs.Scores('seg', 'mqm')
wmt_scores = evs.Scores('seg', 'wmt-z')

new_index_set = set()
for sys in wmt_scores:
    if sys in mqm_scores:
        index_set = set()
        for temp_index, (ele_mqm, ele_wmt) in enumerate(zip(mqm_scores[sys], wmt_scores[sys])):
            if ele_mqm != None and ele_wmt != None:
                index_set.add(temp_index)

        if len(new_index_set) == 0:
            new_index_set.update(index_set)
        else:
            new_index_set = new_index_set & index_set

print(new_index_set)
new_wmt_scores, new_mqm_scores = {}, {}
for sys in wmt_scores:
    if sys in mqm_scores:
        temp_wmt_ls, temp_mqm_ls = [], []
        for index in sorted(new_index_set):
            temp_wmt_ls.append(wmt_scores[sys][index])
            temp_mqm_ls.append(mqm_scores[sys][index])
        new_wmt_scores[sys]=[sum(temp_wmt_ls)/len(temp_wmt_ls)]
        new_mqm_scores[sys]=[sum(temp_mqm_ls)/len(temp_mqm_ls)]

qm_no_human = set(new_mqm_scores) - set(evs.all_refs)
qm_human = qm_no_human.copy()
qm_human.update(human_mapping_dict[wmt][lang])

print(new_mqm_scores)
print(new_wmt_scores)
mqm_bp = evs.Correlation(new_mqm_scores, new_wmt_scores, qm_human & set(new_wmt_scores))
mqm_bp_no = evs.Correlation(new_mqm_scores, new_wmt_scores, qm_no_human & set(new_wmt_scores))

print(mqm_bp.Kendall()[0])
print(mqm_bp_no.Kendall()[0])
print()
print(mqm_bp.Pearson()[0])
print(mqm_bp_no.Pearson()[0])
