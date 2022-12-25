import pandas as pd
import itertools

src_ref_lines = open('ted_42604.en-ja.txt', 'r').readlines()
src_lines = [line[:-1].split('\t')[0]+'\n' for line in src_ref_lines]
ref_lines = [line[:-1].split('\t')[1]+'\n' for line in src_ref_lines]
data = pd.read_excel('和訳評価_TED_42604_eval.xlsx', index_col=0)
print(data.keys())
# remove addtional nan rows
data = data[data['原文'].notna()]
data = data[data['エラー点数'].notna()]
data = data[data['エラー点数2'].notna()]
data = data[data['エラー点数3'].notna()]
data = data[data['エラー点数4'].notna()]
fixed_src_lines, fixed_ref_lines = [], []
for index in data['原文\n番号']:
    # print(src_lines[index])
    # print(ref_lines[index])
    fixed_src_lines+=[src_lines[index]]
    fixed_ref_lines+=[ref_lines[index]]

sys_names_ls, scores_ls, final_src_ls, final_ref_ls = [], [], [], []

assert(len(list(data['System1']))==78)
sys_names_ls.extend(list(data['System1']))
assert(len(list(data['エラー点数']))==78)
scores_ls.extend(list(data['エラー点数']))
final_src_ls.extend(fixed_src_lines)
final_ref_ls.extend(fixed_ref_lines)

assert(len(list(data['System2']))==78)
sys_names_ls.extend(list(data['System2']))
assert(len(list(data['エラー点数2']))==78)
scores_ls.extend(list(data['エラー点数2']))
final_src_ls.extend(fixed_src_lines)
final_ref_ls.extend(fixed_ref_lines)

assert(len(list(data['System3']))==78)
sys_names_ls.extend(list(data['System3']))
assert(len(list(data['エラー点数3']))==78)
scores_ls.extend(list(data['エラー点数3']))
final_src_ls.extend(fixed_src_lines)
final_ref_ls.extend(fixed_ref_lines)

assert(len(list(data['System4']))==78)
sys_names_ls.extend(list(data['System4']))
assert(len(list(data['エラー点数4']))==78)
scores_ls.extend(list(data['エラー点数4']))
final_src_ls.extend(fixed_src_lines)
final_ref_ls.extend(fixed_ref_lines)

save_score = open('mqm_en_ja_scores.txt', 'w')
save_sys = open('mqm_en_ja_sys.txt', 'w')
save_src = open('mqm_en_ja_src.txt', 'w')
save_ref = open('mqm_en_ja_ref.txt', 'w')

sys_names_ls = [sys+'\n' for sys in sys_names_ls]
scores_ls = [str(score)+'\n' for score in scores_ls]

save_sys.writelines(sys_names_ls)
save_score.writelines(scores_ls)
save_src.writelines(final_src_ls)
save_ref.writelines(final_ref_ls)

print("All files are saved!")
