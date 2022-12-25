from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random
import click
import nltk
import click
import spacy
nltk.download('punkt')
# 1) Sample five non-overlapping locations in a sentence to construct synthesized dataset
# 2) For each location, choose operations from Repeat, Delete, Swap, Insert and Replace
#    (For Replace operaiton, we use both in batch and also dictionary replacement)
# 3) Use two losses: multi-level margin losses and compactness losses

"""code to run: python3 syn_data/ssl_data_rep_insert_del_construct.py -inp_addr sep17_zh_en_ssl_data/train.en
 -rep_addr cont_total.txt -num_turns 5 -language english"""

# construct synthesized data from delete, swap, repeat, insert and replace operations
def ssl_stratified_data_schedule(words_ls, num_turns, max_span, schedule_ls, random_cont_enable):
    indices_ls = list(range(len(words_ls)))
    # dynamically store the indices that can be modified
    temp_ls = indices_ls.copy()
    # dynamically store the amount of operations that can be added on each position
    edit_queue = [len(words_ls)-ele for ele in indices_ls]

    # 0: delete, 1: insert_mt, 2: replace_mt
    operation_type = [0, 1]
    weights=[1,2]

    # randomly select operation at each turn
    ops_ls = random.choices(operation_type, weights=weights, k=num_turns)
    pre_ops_ls = []
    if sum(ops_ls) >= len(schedule_ls):
        pre_ops_ls = schedule_ls
    else:
        pre_ops_ls = random.choices(schedule_ls, k=sum(ops_ls))

    # randomly select edit position but non-overlapping to the last operation
    final_ops_ls, edit_locs, edit_conts = [], [], []
    all_cont_ls = [pre_op.split('-')[2] for pre_op in pre_ops_ls]
    if random_cont_enable:
        # randomly insert/replace content in each location
        random.shuffle(all_cont_ls)

    # format: 13-15-facilitating
    for pre_op, cur_cont in zip(pre_ops_ls, all_cont_ls):
        pre_op_cur_ls = pre_op.split('-')

        start_index, end_index = int(pre_op_cur_ls[0]), int(pre_op_cur_ls[1])
        edit_conts.append(cur_cont)
        # updates the tracking table
        for i in range(start_index-1, -1, -1):
            if edit_queue[i] == 0:
                break
            edit_queue[i] -= edit_queue[start_index]
        # for insert operation
        if start_index == end_index:
            temp_ls = list(set(temp_ls) - set([indices_ls[start_index]]))
            edit_queue[start_index] = 0
            edit_locs.append([start_index])
            final_ops_ls.append(1) # insert
        # for replace operation
        else:
            span_len = end_index-start_index
            temp_ls = list(set(temp_ls) - set(indices_ls[start_index:end_index]))
            edit_queue[start_index:end_index] = [0]*span_len
            edit_locs.append([start_index, end_index])
            final_ops_ls.append(2) # replace

    num_dels = len(ops_ls) - sum(ops_ls)
    # rest operations are for delete
    for op_index in range(num_dels):
        if len(temp_ls) > 0:
            rand_index = random.choice(temp_ls)
            for i in range(rand_index-1, -1, -1):
                if edit_queue[i] == 0:
                    break
                edit_queue[i] -= edit_queue[rand_index]

            if min(max_span, edit_queue[rand_index]) > 1:
                span_len = random.randrange(1, min(max_span, edit_queue[rand_index]))
            else:
                span_len = 1
            temp_ls = list(set(temp_ls) - set(indices_ls[rand_index:rand_index+span_len]))
            edit_queue[rand_index:rand_index+span_len] = [0]*span_len
            edit_locs.append([rand_index, rand_index+span_len])
            final_ops_ls.append(0)
        else:
            # number of turns is not supported by current sentence
            ret_ops = final_ops_ls[:sum(ops_ls)+op_index]
            return ret_ops, edit_locs, edit_conts

    return final_ops_ls, edit_locs, edit_conts

@click.command()
@click.option('-inp_addr', type=str)
@click.option('-rep_addr', type=str)
@click.option('-num_turns', type=int)
@click.option('-language', type=str, help="english or german")
@click.option('-rand_insert_replace', type=bool, help="If enabled, we randomly insert/replace span of tokens")
def main(inp_addr, rep_addr, num_turns, language, rand_insert_replace):
    max_span = 7
    lines = open(inp_addr, 'r').readlines()
    cont_lines = open(rep_addr, 'r')

    save_files_mined_ls, save_files_del_ls, save_files_mined_cont_ls = [], [], []
    num_turn_choices = list(range(1, num_turns+1))

    if language == 'japanese':
        nlp = spacy.blank('ja')
    # generate save file
    for i in range(1, num_turns+1, 1):
        save_files_mined_ls.append(open(f'mined_sen_op_loc_{i}.txt', 'w'))
    for i in range(1, num_turns+1, 1):
        save_files_del_ls.append(open(f'del_sen_op_loc_{i}.txt', 'w'))

    for i in range(1, num_turns+1, 1):
        save_files_mined_cont_ls.append(open(f'{language}_{i}_mined_cont_rand_{rand_insert_replace}.txt', 'w'))

    for cont_line_ls in cont_lines:
        temp_line_ls = cont_line_ls[:-1].split('\t')
        if len(temp_line_ls) > 1:
            cont_line_ls = temp_line_ls[1:]
            sen_index = int(temp_line_ls[0])
            line = lines[sen_index][:-1]
            cur_num_turns = random.choice(num_turn_choices)
            if language == 'japanese':
                words_ls = [word.text for word in nlp(line)]
            else:
                words_ls = word_tokenize(line)

            ops_ls, edit_locs, edit_conts = ssl_stratified_data_schedule(words_ls, cur_num_turns, max_span, \
                cont_line_ls, rand_insert_replace)
            # save as (sen_index, op, edit_locs)
            for file_index, (op, loc) in enumerate(zip(ops_ls, edit_locs)):
                if len(loc) == 1:
                    cur_loc = f'{loc[0]}'
                elif len(loc) == 2:
                    cur_loc = f'{loc[0]}_{loc[1]}'
                else:
                    print("Errors in sentence locations")
                    exit(1)

                if op == 0:
                    save_files_del_ls[file_index].write(str(sen_index)+'\t'+str(op)+'\t'+cur_loc+'\n')
                else:
                    save_files_mined_ls[file_index].write(str(sen_index)+'\t'+str(op)+'\t'+cur_loc+'\n')

            for file_index, cont in enumerate(edit_conts):
                save_files_mined_cont_ls[file_index].write(cont+'\n')

if __name__ == "__main__":
    main()
