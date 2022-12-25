from itertools import groupby
import click
from nltk.tokenize import word_tokenize
import spacy
from tqdm.auto import tqdm
import nltk
import math
nltk.download('punkt')

def editDistDP(str1, str2, m, n):
    # Create a table to store results of subproblems
    dp = [[{'dist': 0, 'path': [], 'mask': []} for x in range(n + 1)] for x in range(m + 1)]
    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):
            # If first string is empty, only option is to insert all characters of second string
            if i == 0:
                dp[i][j]['dist'] = j    # Min. operations = j
                # dp[i][j]['path'] += ['insert_all']

            # If second string is empty, only option is to remove all characters of second string
            elif j == 0:
                dp[i][j]['dist'] = i    # Min. operations = i
                # dp[i][j]['path'] += ['delete_all']

            # If characters are same, ignore last char and recur for remaining string
            elif str1[i-1] == str2[j-1]:
                dp[i][j]['dist'] = dp[i-1][j-1]['dist']
                dp[i][j]['path'] = dp[i-1][j-1]['path']+['no_ops']
                dp[i][j]['mask'] += dp[i-1][j-1]['mask']+[0]

            # If last character are different, consider all possibilities and find minimum
            else:
                # insert remove replace
                if dp[i][j-1]['dist'] == min(dp[i][j-1]['dist'], dp[i-1][j]['dist'], dp[i-1][j-1]['dist']):
                    dp[i][j]['dist'] = 1 + dp[i][j-1]['dist']
                    dp[i][j]['path'] = dp[i][j-1]['path'] + [{'insert': str2[j-1]}]
                    dp[i][j]['mask'] = dp[i][j-1]['mask'] + [1]
                elif dp[i-1][j]['dist'] == min(dp[i][j-1]['dist'], dp[i-1][j]['dist'], dp[i-1][j-1]['dist']):
                    dp[i][j]['dist'] = 1 + dp[i-1][j]['dist']
                    dp[i][j]['path'] = dp[i-1][j]['path'] + [{'delete': ''}]
                    dp[i][j]['mask'] = dp[i-1][j]['mask'] + [1]
                else:
                    dp[i][j]['dist'] = 1 + dp[i-1][j-1]['dist']
                    dp[i][j]['path'] = dp[i-1][j-1]['path'] + [{'replace': str2[j-1]}]
                    dp[i][j]['mask'] = dp[i-1][j-1]['mask'] + [1]
    return dp[m][n]

@click.command()
@click.option("-pair_addr")
@click.option("-process_index", type=int)
@click.option("-tot_p", type=int)
@click.option("-language", type=str)
def main(pair_addr, process_index, tot_p, language):
    pair_lines = open(pair_addr, 'r').readlines()
    batch_size = math.ceil(len(pair_lines)/tot_p)
    start_end_pair = [[i, min(i+batch_size, len(pair_lines))] for i in range(0, len(pair_lines), batch_size)]
    start_index, end_index = start_end_pair[process_index][0], start_end_pair[process_index][1]
    pair_lines = pair_lines[start_index:end_index]

    save_file = open(f'raw_ssl_data_sep16/cont_{start_index}_{end_index}_cand_phrase.txt', 'w')

    if language == 'japanese':
        nlp = spacy.blank('ja')

    # with tqdm(total=100000) as pbar:
    for pair_line in pair_lines:
        pair_ls = pair_line[:-1].split('\t')
        sen_index, str1, str2 = pair_ls[0], pair_ls[1], pair_ls[2]
        if language == 'japanese':
            words_ls_1 = [word.text for word in nlp(str1)]
            words_ls_2 = [word.text for word in nlp(str2)]
        else:
            words_ls_1 = word_tokenize(str1)
            words_ls_2 = word_tokenize(str2)

        best_ops_path = editDistDP(words_ls_1, words_ls_2, len(words_ls_1), len(words_ls_2))
        path, mask = best_ops_path['path'], best_ops_path['mask']

        cur_start_index, real_cur_start_index, err_locs, real_err_locs = 0, 0, [], []
        for k, g in groupby(mask, lambda x:x):
            span_len = len(list(g))
            # span has to be within 6-gram
            if k == 1:
                real_span_len = 0
                check_insert = True
                for ele in path[cur_start_index:cur_start_index+span_len]:
                    if list(ele.keys())[0] != 'insert':
                        check_insert = False
                        real_span_len+=1
                if span_len <= 6 and real_cur_start_index < len(words_ls_1):
                    # when operation is insert, the whole span has to be insert
                    if check_insert:
                        err_locs.append([cur_start_index, cur_start_index+span_len, 'insert'])
                        real_err_locs.append([real_cur_start_index, real_cur_start_index+real_span_len, 'insert'])
                    else:
                        err_locs.append([cur_start_index, cur_start_index+span_len, 'others'])
                        real_err_locs.append([real_cur_start_index, real_cur_start_index+real_span_len, 'others'])
                    if real_cur_start_index >= len(words_ls_1):
                        print("Index bugs!")
                        exit(1)
                real_cur_start_index+=real_span_len
            else:
                real_cur_start_index+=span_len
            cur_start_index+=span_len


        repl_contents = []
        for err_span in err_locs:
            temp_ls = []
            for loc_index in range(err_span[0], err_span[1]):
                if list(path[loc_index].keys())[0] == 'replace':
                    temp_ls.append(path[loc_index]['replace'])
                elif list(path[loc_index].keys())[0] == 'insert':
                    temp_ls.append(path[loc_index]['insert'])
                else:
                    temp_ls.append(path[loc_index]['delete'])

            repl_contents.append(' '.join(temp_ls).strip())

        save_conts_ls=[]
        for err_loc, repl_cont in zip(real_err_locs, repl_contents):
            if repl_cont != '':
                if err_loc[2] == 'insert':
                    save_conts_ls.append('-'.join([str(err_loc[0]), str(err_loc[0])]+[repl_cont]))
                else:
                    save_conts_ls.append('-'.join([str(err_loc[0]), str(err_loc[1])]+[repl_cont]))

        if len(save_conts_ls) > 0:
            save_file.write('\t'.join([sen_index] + save_conts_ls)+'\n')
        # pbar.update(1)
    print("File is saved!")

if __name__ == "__main__":
    main()
