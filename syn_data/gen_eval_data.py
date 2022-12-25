import click
import random
from syn_data.biimt import construct_uniform_mask_data

"""python3 syn_data/gen_eval_data.py -src_lang zh -tar_lang en -file_type dev -quantity 2M -mask_ratio 0.7 -aug_factor 2"""

@click.command()
@click.option('-src_lang')
@click.option('-tar_lang')
@click.option('-file_type', help="dev or train")
@click.option('-quantity', help='2M or 4M')
@click.option('-mask_ratio', type=float)
@click.option('-aug_factor', type=int)
@click.option('-prefix', type=str, default=None)
def main(src_lang, tar_lang, file_type, quantity, mask_ratio, aug_factor, prefix):
    if not prefix:
        src_dev_data = open(f'mt_infilling_data/{src_lang}-{tar_lang}/{quantity}/{src_lang}_{file_type}_{quantity}.txt', 'r').readlines()
        ref_dev_data = open(f'mt_infilling_data/{src_lang}-{tar_lang}/{quantity}/{tar_lang}_{file_type}_{quantity}.txt', 'r').readlines()
    else:
        if prefix == 'news':
            src_dev_data = open(f'news_data/{src_lang}-{tar_lang}/news_comp_{src_lang}_{tar_lang}.{src_lang}_dev.txt')
            ref_dev_data = open(f'news_data/{src_lang}-{tar_lang}/news_comp_{src_lang}_{tar_lang}.{tar_lang}_dev.txt')
        else:
            print("We currently only supports data in news domain")
            exit(1)

    mask_inp_dev, mask_ref_dev = construct_uniform_mask_data(src_dev_data, ref_dev_data, '<mask>', '<sep>', mask_ratio, aug_factor)
    mask_inp_dev, mask_ref_dev = [sen+'\n' for sen in mask_inp_dev], [sen+'\n' for sen in mask_ref_dev]
    mask_src_file = open(f'{prefix}_{src_lang}_{tar_lang}_mask_src_{mask_ratio}_{aug_factor}.txt', 'w')
    mask_ref_file = open(f'{prefix}_{src_lang}_{tar_lang}_mask_tar_{mask_ratio}_{aug_factor}.txt', 'w')
    mask_src_file.writelines(mask_inp_dev)
    mask_ref_file.writelines(mask_ref_dev)

    print("Save masked dev data!")

if __name__ == "__main__":
    random.seed(10)
    main()
