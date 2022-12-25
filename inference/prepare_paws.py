from datasets import load_dataset

sen_pair_dict = {}

test_dataset = load_dataset('paws', 'labeled_final')

for sen_1, sen_2, label in zip(test_dataset['test']['sentence1'],  \
    test_dataset['test']['sentence2'], test_dataset['test']['label']):
    if sen_1 not in sen_pair_dict:
        sen_pair_dict[sen_1] = {}
        sen_pair_dict[sen_1][0] = []
        sen_pair_dict[sen_1][1] = []
    sen_pair_dict[sen_1][label].append(sen_2)

better_ls, worse_ls, ref_ls = [], [], []
for sen1, cur_dict in sen_pair_dict.items():
    if len(cur_dict[0])>0 and len(cur_dict[1])>0:
        better_ls.append(cur_dict[1][0]+'\n')
        worse_ls.append(cur_dict[0][0]+'\n')
        ref_ls.append(sen1+'\n')

save_better = open('paws_better.txt', 'w')
save_worse = open('paws_worse.txt', 'w')
save_ref = open('paws_ref.txt', 'w')

save_better.writelines(better_ls)
save_worse.writelines(worse_ls)
save_ref.writelines(ref_ls)
print("All files are saved!")
