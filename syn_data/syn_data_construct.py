import click
import json
import random
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
# 1) filter out output which is identical to reference
# 2) One-path mode: For each instance, based on number of possible choices at
# each step, randomly select one path and construct data with different number
# of errors.
# 3) All-path mode: For each instance, consider all possible choices at each step
# build the entire tree path
# 4) Construct pair-wise ranking data for each instance

"""code to run: python3 syn_data/syn_data_construct.py -lang_dir zh_en -path_option one_path -data_type train"""

@click.command()
@click.option('-lang_dir')
@click.option('-path_option', type=str)
@click.option('-data_type', help='train or dev')
def main(path_option, lang_dir, data_type):
    ori_ref_ls = open(f'news_data/{lang_dir[:2]}-{lang_dir[3:]}/news_comp_{lang_dir}.{lang_dir[3:]}_{data_type}.txt', 'r').readlines()
    ori_src_ls = open(f'news_data/{lang_dir[:2]}-{lang_dir[3:]}/news_comp_{lang_dir}.{lang_dir[:2]}_{data_type}.txt', 'r').readlines()
    ref_file = open(f'news_prepare_mask_data_syn_gen/cycle1_{lang_dir}_news_train_ref_mask.txt', 'r')
    loc_dict = json.load(open(f"news_prepare_mask_data_syn_gen/cycle1_{lang_dir}_news_train_start_end_locs.json"))

    if lang_dir == 'zh_en':
        if data_type == 'train':
            out_file = open('syn_data_gen_zh_en_train_1660639001/cycle1_infill_mt_0.15_zh_en_total.txt', 'r')
        else:
            out_file = open('syn_data_gen_zh_en_dev_1660637613/cycle1_infill_mt_0.15_zh_en_total.txt', 'r')
    elif lang_dir == 'en_de':
        if data_type == 'train':
            out_file = open('syn_data_gen_en_de_train_1660639578/cycle1_infill_mt_0.15_en_de_total.txt', 'r')
        else:
            out_file = open('syn_data_gen_en_de_dev_1660638759/cycle1_infill_mt_0.15_en_de_total.txt', 'r')
    else:
        print("We only support zh-en and en-de at the moment!")

    new_loc_dict = {}
    for ref_line, out_line in zip(ref_file, out_file):
        ref_content = ref_line.split('\t')
        if ref_content[-1] != out_line:
            sen_index, err_index = int(ref_content[0]), int(ref_content[1])
            if sen_index not in new_loc_dict:
                new_loc_dict[sen_index]={}
                new_loc_dict[sen_index]['src']=ori_src_ls[sen_index][:-1]
                new_loc_dict[sen_index]['ref']=ori_ref_ls[sen_index][:-1]
                new_loc_dict[sen_index]['outputs']=[]
            new_loc_dict[sen_index]['outputs'].append([loc_dict[sen_index][err_index], out_line[:-1]])

    if path_option == 'one_path':
        one_path_dict = {}
        with tqdm(total=len(new_loc_dict)) as pbar:
            for sen_index, cur_dict in new_loc_dict.items():
                cur_ls = cur_dict['outputs']
                overall_ls = []
                while len(cur_ls) > 0:
                    # print(cur_ls)
                    overall_ls.append(cur_ls.copy())
                    rand_index = random.randrange(len(cur_ls))
                    cur_ls.pop(rand_index)

                ref_ls = word_tokenize(cur_dict['ref'])
                one_path_dict[sen_index]={}
                one_path_dict[sen_index]['src']=cur_dict['src']
                one_path_dict[sen_index]['ref']=cur_dict['ref']
                one_path_dict[sen_index]['out']=[]
                for err_sen in overall_ls:
                    last_end_ls, reconstruct_sen_ls = [], []
                    cur_sen_reconstruct = []
                    for err_index, err in enumerate(err_sen):
                        loc, err_cont = err[0], err[1]
                        err_toks = word_tokenize(err_cont)
                        if err_index == 0:
                            cur_sen_reconstruct+=ref_ls[:loc[0]]+err_toks
                        else:
                            cur_sen_reconstruct+=ref_ls[last_end_ls[-1]:loc[0]]+err_toks
                        last_end_ls.append(loc[1])
                    cur_sen_reconstruct+=ref_ls[last_end_ls[-1]:]
                    cur_sen = TreebankWordDetokenizer().detokenize(cur_sen_reconstruct)
                    one_path_dict[sen_index]['out'].append([len(err_sen), cur_sen])
                pbar.update(1)

            with open(f"instance_{lang_dir}_stratified_{data_type}_data.json", 'w') as outfile:
                json.dump(one_path_dict, outfile)

    elif path_option == 'all_path':
        pass
    else:
        print("Your provided path is incorrect!")

if __name__ == "__main__":
    random.seed(10)
    main()
