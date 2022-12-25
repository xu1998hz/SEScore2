import pandas as pd
import math

df = pd.read_excel("和訳評価_TED_42604_eval.xlsx")

system_outputs = []
human_scores = []

# for system 1
for i in range(1, 5):
    for id in set(df['原文\n番号']):
        cur_id_data = df.loc[df['原文\n番号'] == id]
        system_outputs.append(list(cur_id_data[f'System{i}'])[0]+'\n')
        if i == 1:
            score = -sum(list(cur_id_data['エラー点数'].dropna()))
        else:
            score = -sum(list(cur_id_data[f'エラー点数{i}'].dropna()))

        human_scores.append(str(score)+'\n')

src_ref_lines = open('ted_42604.en-ja-fixed.txt', 'r').readlines()
src_lines = [line.split('\t')[0]+'\n' for line in src_ref_lines]*4
ref_lines = [line.split('\t')[1] for line in src_ref_lines]*4

en_ja_src_file = open('mqm_en_ja_src.txt', 'w')
en_ja_ref_file = open('mqm_en_ja_ref.txt', 'w')
en_ja_sys_file = open('mqm_en_ja_sys.txt', 'w')
en_ja_scores_file = open('mqm_en_ja_scores.txt', 'w')

en_ja_src_file.writelines(src_lines)
en_ja_ref_file.writelines(ref_lines)
en_ja_sys_file.writelines(system_outputs)
en_ja_scores_file.writelines(human_scores)

print("files are saved!")
