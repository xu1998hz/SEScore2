import click
from nltk.tokenize import word_tokenize
from random import randrange
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm.auto import tqdm
import json

"""This file is able to prepare the mask data for large scale synthetic data generation. The code to run:
python3 syn_data/prepare_mask_data.py -prefix cycle1_zh_en_news_train -src_data_file news_data/zh-en/news_comp_zh_en.zh_train.txt -ref_data_file news_data/zh-en/news_comp_zh_en.en_train.txt -mask_ratio 0.15"""

def construct_mask_data(src_data, ref_data, sep_token, mask_token, mask_ratio=0.15):
    src_data, ref_data = [data[:-1] for data in src_data], [data[:-1] for data in ref_data]
    mask_inp_data_ls, mask_ref_data_ls, global_mask_loc_ls = [], [], []
    with tqdm(total=len(src_data)) as pbar:
        for sen_index, (src_sen, ref_sen) in enumerate(zip(src_data, ref_data)):
            ref_words_ls = word_tokenize(ref_sen)
            mask_len = int(len(ref_words_ls) * mask_ratio)
            pre_index_ls, start_index_ls, end_index_ls = [], [], []
            mask_inp_data = []
            for index in range(5):
                if len(end_index_ls) > 0:
                    start_index = randrange(end_index_ls[-1], len(ref_words_ls)-(5-index)*mask_len+1)
                else:
                    # only set the start index lower bound at step 0, set it to be 0
                    start_index = randrange(0, len(ref_words_ls)-(5-index)*mask_len+1)
                end_index = start_index + mask_len
                if index == 0:
                    # only set the start index lower bound at step 0, set it to be 0
                    pre_index_ls.append((0, start_index))
                else:
                    pre_index_ls.append((end_index_ls[-1], start_index))
                start_index_ls.append(start_index)
                end_index_ls.append(end_index)

            seg_mask_loc_ls = []
            for loc_index, (start_index, end_index) in enumerate(zip(start_index_ls, end_index_ls)):
                mask_ref_inp_data = ref_words_ls[:start_index] + [mask_token] + ref_words_ls[end_index:]
                mask_tar_data = ref_words_ls[start_index:end_index]
                mask_inp_data_ls.append(f'{sen_index}'+'\t'+f'{loc_index}'+'\t'+src_sen +f' {sep_token} '+ TreebankWordDetokenizer().detokenize(mask_ref_inp_data)+'\n')
                mask_ref_data_ls.append(f'{sen_index}'+'\t'+f'{loc_index}'+'\t'+TreebankWordDetokenizer().detokenize(mask_tar_data)+'\n')
                seg_mask_loc_ls.append([start_index, end_index])
            global_mask_loc_ls.append(seg_mask_loc_ls)
            pbar.update(1)
    return mask_inp_data_ls, mask_ref_data_ls, global_mask_loc_ls

@click.command()
@click.option('-prefix', type=str)
@click.option('-src_data_file', type=str)
@click.option('-ref_data_file', type=str)
@click.option('-mask_ratio', type=float, default=0.15)
def main(prefix, src_data_file, ref_data_file, mask_ratio):
    src_data = open(src_data_file, 'r').readlines()
    ref_data = open(ref_data_file, 'r').readlines()
    src_data, ref_data = [src[:-1] for src in src_data], [ref[:-1] for ref in ref_data]
    mask_inp_data_ls, mask_ref_data_ls, global_mask_loc_ls = construct_mask_data(src_data, ref_data, '<sep>', '<mask>', mask_ratio)
    save_src_mask_file = open(f'{prefix}_src_mask.txt', 'w')
    save_ref_mask_file = open(f'{prefix}_ref_mask.txt', 'w')
    # save all the prepared data
    save_src_mask_file.writelines(mask_inp_data_ls)
    save_ref_mask_file.writelines(mask_ref_data_ls)
    with open(f"{prefix}_start_end_locs.json", 'w') as outfile:
        json.dump(global_mask_loc_ls, outfile)

    print("All files are saved!")

if __name__ == "__main__":
    main()
