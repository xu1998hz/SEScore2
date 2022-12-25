from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random
import click
from dbdict import DbDict
import nltk
nltk.download('punkt')
# 1) Sample five non-overlapping locations in a sentence to construct synthesized dataset
# 2) For each location, choose operations from Repeat, Delete, Swap, Insert and Replace
#    (For Replace operaiton, we use both in batch and also dictionary replacement)
# 3) Use two losses: multi-level margin losses and compactness losses

"""python3 syn_data/data.py -num_gens 3 -max_span 7 -num_turns 5 -corpus_addr wiki_test_1000.en -src_lang en -trg_lang zh -enable_nsp_err True -save_name test.txt -mode single -random_num_turns False"""

# construct synthesized data from delete, swap, repeat, insert and replace operations
def ssl_stratified_data_schedule(words_ls, num_turns, max_span, enable_nsp_err=False):
    indices_ls = list(range(len(words_ls)))
    # dynamically store the indices that can be modified
    temp_ls = indices_ls.copy()
    # dynamically store the amount of operations that can be added on each position
    edit_queue = [len(words_ls)-ele for ele in indices_ls]
    # 0: delete, 1: repeat, 2: insert_nsp, 3: replace_nsp, 4: insert_dict, 5: replace_dict, 6: swap
    if enable_nsp_err:
        operation_type = [0, 1, 2, 3, 4, 5, 6]
        weights=[1,1,1,1,2,2,1]
    else:
        # delete, replace
        operation_type = [0, 5]
        weights=[1, 2]
    # randomly select operation at each turn
    ops_ls = []
    for i in range(num_turns):
        cur_op = cur_op = random.choices(operation_type, weights=weights, k=1)[0]
        # the last operation is swap
        if cur_op == 6:
            ops_ls.append([operation_type[-1], operation_type[-1]])
        else:
            ops_ls.append([cur_op])
    # randomly select edit position but non-overlapping to the last operation
    edit_locs = []
    for op_index, ops in enumerate(ops_ls):
        temp_edit_locs = []
        # this is for operation that needs two spans (like swap) in one sentence
        for op in ops:
            if len(temp_ls) > 0:
                rand_index = random.choice(temp_ls)
                for i in range(rand_index-1, -1, -1):
                    if edit_queue[i] == 0:
                        break
                    edit_queue[i] -= edit_queue[rand_index]
                if op == 2:
                    temp_edit_locs.append([rand_index])
                else:
                    if min(max_span, edit_queue[rand_index]) > 1:
                        span_len = random.randrange(1, min(max_span, edit_queue[rand_index]))
                    else:
                        span_len = 1
                    temp_ls = list(set(temp_ls) - set(indices_ls[rand_index:rand_index+span_len]))
                    edit_queue[rand_index:rand_index+span_len] = [0]*span_len
                    temp_edit_locs.append([rand_index, rand_index+span_len])
            else:
                # number of turns is not supported by current sentence
                ret_ops = ops_ls[:op_index]
                return ret_ops, edit_locs
        edit_locs.append(sorted(temp_edit_locs))
    return ops_ls, edit_locs

@click.command()
@click.option('-mode', type=str, help='single or double')
@click.option('-num_gens', default=1, type=int, help="Number of hard negative samples per instance")
@click.option('-max_span', default=7, type=int, help="Max span for each operation, less than max_span")
@click.option('-random_num_turns', type=bool, help="Whether to randomnize the number of turns")
@click.option('-num_turns', type=int, help="Number of max turns per sample")
@click.option('-corpus_addr', type=str, help="Address of pretraining corpus")
@click.option('-src_lang', type=str, default='en', help='src lang for bilingual dictionary, en')
@click.option('-trg_lang', type=str, default='zh', help='tgt lang for bilingua; dictionary, zh')
@click.option('-enable_nsp_err', type=bool, help="enable to use dict for replace and insert errors")
@click.option('-save_name', type=str, help="specify the name of the saved file")
def main(mode, corpus_addr, num_gens, max_span, random_num_turns, num_turns, src_lang, trg_lang, enable_nsp_err, save_name):
    corpus_file = open(corpus_addr, 'r')
    dict_addr = f'{src_lang}-{trg_lang}.db'
    dict_mapping = DbDict(dict_addr, True)
    save_file = open(save_name, 'w')
    save_turns_file = open('turns_'+save_name, 'w')

    fail_cases = 0
    num_turn_choices = list(range(1, num_turns+1))
    for line in corpus_file:
        line_samples, num_turns_ls = [], []
        if mode == 'single':
            words_ls = word_tokenize(line[:-1])
            enable_nsp_err = False
        elif mode == 'double':
            cur_sen, next_sen = line.split('\t')[0], line.split('\t')[1]
            words_ls, next_ls = word_tokenize(cur_sen[:-1]), word_tokenize(next_sen[:-1])
        else:
            print('Wrong specification for mode!')
            exit(1)

        # generate different spans with different roots
        for _ in range(num_gens):
            # randomly select a turn from 1, 2, 3, 4, 5
            if random_num_turns:
                num_turns = random.choice(num_turn_choices)
            # generate one root of a path
            ops_ls, edit_locs = ssl_stratified_data_schedule(words_ls, num_turns, max_span, enable_nsp_err)
            one_path_turn_per_gen_ls, one_path_sample_per_gen_ls = [], []
            # cur_count = 5
            locs_records, replaced_contents = [], []
            # 0: delete, 1: repeat, 2: insert_nsp, 3: replace_nsp, 4: insert_dict, 5: replace_dict, 6: swap
            for ops, locs in zip(ops_ls, edit_locs):
                # len(ops) == 2 implies this is a swap operation
                if len(ops) == 2:
                    # swap
                    print("no more swap operation")
                    op_span_1, op_span_2 = locs[0], locs[1]
                    locs_records.append(op_span_1)
                    locs_records.append(op_span_2)
                    replaced_contents.append(words_ls[op_span_2[0]:op_span_2[1]])
                    replaced_contents.append(words_ls[op_span_1[0]:op_span_1[1]])
                    # new_words_ls = words_ls[:op_span_1[0]] + words_ls[op_span_2[0]:op_span_2[1]] + \
                    #  words_ls[op_span_1[1]:op_span_2[0]] + words_ls[op_span_1[0]:op_span_1[1]] + words_ls[op_span_2[1]:]
                else:
                    if ops[0] == 0:
                        # delete
                        start_pos, end_pos = locs[0][0], locs[0][1]
                        locs_records.append([start_pos, end_pos])
                        replaced_contents.append([])
                        # new_words_ls = words_ls[:start_pos]+words_ls[end_pos:]
                    elif ops[0] == 1:
                        # repeat
                        start_pos, end_pos = locs[0][0], locs[0][1]
                        num_repeats = random.choice([1, 2, 3])
                        locs_records.append([start_pos, end_pos])
                        replaced_contents.append(words_ls[start_pos:end_pos] * (num_repeats+1))
                        # new_words_ls = words_ls[:start_pos] + words_ls[start_pos:end_pos] * (num_repeats+1) + words_ls[end_pos:]
                    elif ops[0] == 2 or ops[0] == 3:
                        # nsp insert or replace
                        print("I should not sample from here!")
                        # indices_ls = list(range(len(next_ls)))
                        # edit_queue = [len(words_ls)-ele for ele in indices_ls]
                        # rand_take_pos = random.choice(indices_ls)
                        # if min(max_span, edit_queue[rand_take_pos])>1:
                        #     span_len = random.randrange(1,  min(max_span, edit_queue[rand_take_pos]))
                        # else:
                        #     span_len = 1
                        # repl_cont_ls = next_ls[rand_take_pos:rand_take_pos+span_len]

                        # if ops[0] == 2:
                        #     insert_pos = locs[0][0]
                        #     new_words_ls = words_ls[:insert_pos] + repl_cont_ls + words_ls[insert_pos:]
                        # else:
                        #     start_pos, end_pos = locs[0][0], locs[0][1]
                        #     new_words_ls = words_ls[:start_pos] + repl_cont_ls + words_ls[end_pos:]
                    elif ops[0] == 4 or ops[0] == 5:
                        # dict insert or replace
                        start_pos, end_pos = locs[0][0], locs[0][1]
                        src_cont_ls, tgt_cont_ls = [], []
                        src_trg_dir, trg_src_lang = f'{src_lang}2{trg_lang}', f'{trg_lang}2{src_lang}'

                        # first translate into target language
                        for word in words_ls[start_pos:end_pos]:
                            word = word.lower()
                            word_in_trg = dict_mapping.get(f"{src_trg_dir}{word}")
                            # print(word_in_trg)
                            # literal translation to src
                            if word_in_trg:
                                probs = list(word_in_trg.values())[:-1]
                                if len(probs) != 1:
                                    probs = [1-ele for ele in probs]
                                new_sample = random.choices(list(word_in_trg.keys())[:-1], weights=probs, k=1)
                                src_cont_ls.append(new_sample[0])

                        # translate back into src language
                        for word in src_cont_ls:
                            word_in_src = dict_mapping.get(f"{trg_src_lang}{word}")
                            # literal translation to trg
                            if word_in_src:
                                probs = list(word_in_src.values())[:-1]
                                new_sample = random.choices(list(word_in_src.keys())[:-1], weights=probs, k=1)
                                tgt_cont_ls.append(new_sample[0])

                        # verity if the content is not the same
                        if len(tgt_cont_ls) > 0 and words_ls[start_pos:end_pos] != tgt_cont_ls:
                            if ops[0] == 4:
                                print("no more dict insert operation")
                                indices_ls = list(range(len(words_ls)))
                                rand_insert_pos = random.choice(indices_ls)
                                locs_records.append([rand_insert_pos, rand_insert_pos])
                                replaced_contents.append(tgt_cont_ls)
                                # new_words_ls = words_ls[:rand_insert_pos] + tgt_cont_ls + words_ls[rand_insert_pos:]
                            else:
                                locs_records.append([start_pos, end_pos])
                                replaced_contents.append(tgt_cont_ls)
                                # new_words_ls = words_ls[:start_pos] + tgt_cont_ls + words_ls[end_pos:]
                    else:
                        print("error in the operation index")
                        exit(1)

            locs_stratified_ls, conts_stratified_ls = [], []
            # generate one path out of all paths
            if len(locs_records) > 0:
                for num_errs in range(len(locs_records), 0, -1):
                    rand_inds = random.sample(range(len(locs_records)), k=num_errs)
                    locs_records = [locs_records[ind] for ind in rand_inds]
                    replaced_contents = [replaced_contents[ind] for ind in rand_inds]
                    locs_stratified_ls.append(locs_records)
                    conts_stratified_ls.append(replaced_contents)

                for locs_records, replaced_contents in zip(locs_stratified_ls, conts_stratified_ls):
                    new_words_ls = []
                    sorted_locs_cont = sorted(list(zip(locs_records, replaced_contents)))
                    if len(sorted_locs_cont) == 1:
                        new_words_ls += words_ls[:locs_records[0][0]]+replaced_contents[0]+words_ls[locs_records[0][1]:]
                    else:
                        for cur_index, locs_content in enumerate(sorted_locs_cont):
                            locs_pair, content = locs_content[0], locs_content[1]
                            if cur_index == 0:
                                new_words_ls += words_ls[:locs_pair[0]]+content
                            elif cur_index == len(sorted_locs_cont)-1:
                                new_words_ls += words_ls[sorted_locs_cont[cur_index-1][0][1]:locs_pair[0]]+content+words_ls[locs_pair[1]:]
                            else:
                                new_words_ls += words_ls[sorted_locs_cont[cur_index-1][0][1]:locs_pair[0]]+content

                    one_path_turn_per_gen_ls += [str(len(locs_records))]
                    one_path_sample_per_gen_ls += [TreebankWordDetokenizer().detokenize(new_words_ls)]
            else:
                fail_cases += 1

            # save all negative samples in one line
            line_samples.append('\t'.join(one_path_sample_per_gen_ls))
            num_turns_ls.append('\t'.join(one_path_turn_per_gen_ls))

        save_file.write(' [SEP] '.join(line_samples)+'\n')
        save_turns_file.write(' [SEP] '.join(num_turns_ls)+'\n')

    print(f"File is generated saved at {save_name}!")

if __name__ == "__main__":
    main()
