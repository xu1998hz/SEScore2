import glob
import pandas as pd

files_ls = sorted(glob.glob('annotated_outputs/*'))
src_lines = open('src.de', 'r').readlines()
src_lines = [line[:-1] for line in src_lines]*10
tot_lines = []

for file_name in files_ls:
    cur_lines = open(file_name, 'r').readlines()
    cur_lines = [line[:-1] for line in cur_lines]
    tot_lines += cur_lines

df_dict = {'source': src_lines, 'translation': tot_lines, 'score': ['']*1000, 'error_type': ['']*1000}
df = pd.DataFrame.from_dict(df_dict)
df.to_excel("annotated_en_de_1000.xlsx")

print("file is generated!") 
