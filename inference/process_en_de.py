import pandas as pd

# tar -cvf - sescore_ckpt | pigz -p 8 > sescore_ckpt.gz 
# tar -xvf sescore_ckpt

labels = list(pd.read_excel('annotated_en_de_1000_second.xlsx')['score'])
labels = [str(label)+'\n' for label in labels]
# outputs = list(pd.read_excel('annotated_en_de_1000.xlsx')['translation'])
# outputs = [output+'\n' for output in outputs]
# src_lines = open('src.de', 'r').readlines()
# src_lines = [line for line in src_lines]*10
# ref_lines = open('ref.en', 'r').readlines()
# ref_lines = [line for line in ref_lines]*10

save_score = open('en_de_sec.scores', 'w')
save_score.writelines(labels)
# save_out = open('en_de.out', 'w')
# save_out.writelines(outputs)
# save_src = open('en_de.src', 'w')
# save_src.writelines(src_lines)
# save_ref = open('en_de.ref', 'w')
# save_ref.writelines(ref_lines)
# print("all the files are saved!")
