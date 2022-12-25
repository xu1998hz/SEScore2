import pandas as pd
from nltk.tokenize import word_tokenize

language, severity = 'german', 'Major'
print(language)
print(severity)
if severity == 'Minor':
    thres = -1
elif severity == 'Major':
    thres = -5
else:
    print("Wrong severity measures!")

if language == 'english':
    df_span = pd.read_csv('wmt-mqm-human-evaluation/newstest2020/zhen/mqm_newstest2020_zhen.tsv', delimiter='\t')
    df_score = pd.read_csv('wmt-mqm-human-evaluation/newstest2020/zhen/mqm_newstest2020_zhen.avg_seg_scores.tsv', delimiter='\t')
    save_addr="severity_analysis/zh_en"
elif language == 'german':
    df_span = pd.read_csv('wmt-mqm-human-evaluation/newstest2020/ende/mqm_newstest2020_ende.tsv', delimiter='\t')
    df_score = pd.read_csv('wmt-mqm-human-evaluation/newstest2020/ende/mqm_newstest2020_ende.avg_seg_scores.tsv', delimiter='\t')
    save_addr="severity_analysis/en_de"

# save into three files
src_file = open(f'{save_addr}/{severity}/{language}_{severity}_2020_src_text.txt', 'w')
tar_file = open(f'{save_addr}/{severity}/{language}_{severity}_2020_tar_text.txt', 'w')
loc_file = open(f'{save_addr}/{severity}/{language}_{severity}_mined_sen_op_loc.txt', 'w')
cont_file = open(f'{save_addr}/{severity}/{language}_{severity}_mined_cont.txt', 'w')

examples = []

for line in df_score['system mqm_avg_score seg_id']:
    line_ls = line.split()
    if float(line.split()[1]) == thres:
        examples.append([line_ls[0], line_ls[2]])

sen_index,count=0,0
# extract all examples
for example in examples:
    # match with id, system and severity
    span_df = df_span.loc[(df_span['system'] == example[0]) & (df_span['seg_id'] == int(example[1])) \
    & (df_span['severity'] == severity)]
    if len(list(span_df['target']))>0:
        # all operations will be annotated as 2 (replace)
        for src_ele, tar_ele in zip(span_df['source'], span_df['target']):
            if '<v>' in tar_ele:
                original_sen = tar_ele.replace('<v>', ' ').replace('</v>', ' ')
                original_sen = original_sen.strip()
                src_sen = src_ele.replace('<v>', '' ).replace('</v>', ' ')
                src_sen = src_sen.strip()
                start_index, span_len = len(word_tokenize(tar_ele.split('<v>')[0])), len(word_tokenize(tar_ele.split('<v>')[1].split('</v>')[0]))
                cont = tar_ele.split('<v>')[1].split('</v>')[0]
                if word_tokenize(cont) == word_tokenize(original_sen)[start_index:start_index+span_len]:
                    src_file.write(src_sen+'\n')
                    tar_file.write(original_sen+'\n')
                    loc_file.write(str(sen_index)+'\t'+str(2)+'\t'+str(start_index)+"_"+str(start_index+span_len)+'\n')
                    cont_file.write(cont+'\n')
                    sen_index+=1
                else:
                    print(word_tokenize(cont))
                    print(word_tokenize(original_sen)[start_index:start_index+span_len])
                    print()
                    count+=1
print(count)
print("All files are generated!")
