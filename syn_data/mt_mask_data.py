from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random
import click
from dbdict import DbDict
import nltk
import click
nltk.download('punkt')
# 1) Sample five non-overlapping locations in a sentence to construct synthesized dataset
# 2) For each location, choose operations from Repeat, Delete, Swap, Insert and Replace
#    (For Replace operaiton, we use both in batch and also dictionary replacement)
# 3) Use two losses: multi-level margin losses and compactness losses

"""code to run: python3 syn_data/mt_mask_data.py -inp_addr train_400k.en -num_turns 5"""

# construct synthesized data from delete, swap, repeat, insert and replace operations
def ssl_stratified_data_schedule(words_ls, num_turns, max_span):
    indices_ls = list(range(len(words_ls)))
    # dynamically store the indices that can be modified
    temp_ls = indices_ls.copy()
    # dynamically store the amount of operations that can be added on each position
    edit_queue = [len(words_ls)-ele for ele in indices_ls]
    # 0: delete, 1: insert_mt, 3: replace_mt
    operation_type = [0, 1, 2]
    weights=[1,1,1]

    # randomly select operation at each turn
    ops_ls = []
    for i in range(num_turns):
        cur_op = random.choices(operation_type, weights=weights, k=1)[0]
        # the last operation is swap
        ops_ls.append(cur_op)
    # randomly select edit position but non-overlapping to the last operation
    edit_locs = []
    for op_index, op in enumerate(ops_ls):
        # this is for operation that needs two spans (like swap) in one sentence
        if len(temp_ls) > 0:
            rand_index = random.choice(temp_ls)
            for i in range(rand_index-1, -1, -1):
                if edit_queue[i] == 0:
                    break
                edit_queue[i] -= edit_queue[rand_index]
            # for insert operation only because insert has no error span from original sentence
            if op == 1:
                # insert position should be fixed
                temp_ls = list(set(temp_ls) - set([indices_ls[rand_index]]))
                edit_queue[rand_index] = 0
                edit_locs.append([rand_index])
            else:
                if min(max_span, edit_queue[rand_index]) > 1:
                    span_len = random.randrange(1, min(max_span, edit_queue[rand_index]))
                else:
                    span_len = 1
                temp_ls = list(set(temp_ls) - set(indices_ls[rand_index:rand_index+span_len]))
                edit_queue[rand_index:rand_index+span_len] = [0]*span_len
                edit_locs.append([rand_index, rand_index+span_len])
        else:
            # number of turns is not supported by current sentence
            ret_ops = ops_ls[:op_index]
            return ret_ops, edit_locs
    return ops_ls, edit_locs

@click.command()
@click.option('-inp_addr', type=str)
@click.option('-num_turns', type=int)
@click.option('-language', type=str, help="english or german")
def main(inp_addr, num_turns, language):
    max_span = 7
    lines = open(inp_addr, 'r').readlines()
    lines = [line[:-1] for line in lines]
    save_files_mt_ls, save_files_del_ls = [], []
    num_turn_choices = list(range(1, num_turns+1))
    # generate save file
    for i in range(1, num_turns+1, 1):
        save_files_mt_ls.append(open(f'mt_sen_op_loc_{i}.txt', 'w'))
    for i in range(1, num_turns+1, 1):
        save_files_del_ls.append(open(f'del_sen_op_loc_{i}.txt', 'w'))

    for sen_index, line in enumerate(lines):
        cur_num_turns = random.choice(num_turn_choices)
        words_ls = word_tokenize(line, language)
        ops_ls, edit_locs = ssl_stratified_data_schedule(words_ls, cur_num_turns, max_span)
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
                save_files_mt_ls[file_index].write(str(sen_index)+'\t'+str(op)+'\t'+cur_loc+'\n')

if __name__ == "__main__":
    main()
