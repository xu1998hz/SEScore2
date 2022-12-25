prefix = 'crowdflower'

lines0 = open(f'caption_{prefix}_contrascore_results_rembert_0.txt', 'r').readlines()
lines1 = open(f'caption_{prefix}_contrascore_results_rembert_1.txt', 'r').readlines()
lines2 = open(f'caption_{prefix}_contrascore_results_rembert_2.txt', 'r').readlines()
lines3 = open(f'caption_{prefix}_contrascore_results_rembert_3.txt', 'r').readlines()
lines4 = open(f'caption_{prefix}_contrascore_results_rembert_4.txt', 'r').readlines()

score_dict = {}
lines_ls = [lines0, lines1, lines2, lines3, lines4]
for cur_lines in lines_ls:
    for ele in cur_lines:
        if int(ele.split('\t')[0]) not in score_dict:
            score_dict[int(ele.split('\t')[0])] = []
        score_dict[int(ele.split('\t')[0])].append(float((ele.split('\t')[1])))

scores_ls = []
for i in range(len(score_dict)):
    scores_ls.append(str(max(score_dict[i]))+'\n')

savefile = open(f'caption_{prefix}_contrascore_rembert_results_max.txt', 'w')
savefile.writelines(scores_ls)
