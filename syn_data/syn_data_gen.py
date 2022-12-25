import torch
import click
import nltk
from syn_data.biimt import exp_config, preprocess_data
from transformers import (
    AutoTokenizer
)
import random
from train.cl_train import exp_config
import math

"""In the synthetic data generation, we want MT infilling model to create five
levels of outputss. Sampling by five non-overlap masks (Each span is 15% of sentences) codes:
CUDA_VISIBLE_DEVICES=0 python3 syn_data/syn_data_gen.py -model_checkpoint mt_infilling_weights_zh-en_2M_1660497817/epoch4.ckpt \
-src_file news_prepare_mask_data_syn_gen/cycle1_zh_en_news_train_src_mask.txt -ref_file news_prepare_mask_data_syn_gen/cycle1_zh_en_news_train_ref_mask.txt"""

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

@click.command()
@click.option('-gpu_index', type=int)
@click.option('-prefix', type=str)
@click.option('-batch_size', type=int, default=64)
@click.option('-model_checkpoint', type=str)
@click.option('-tok_checkpoint', type=str, default="mbart-large-50-many-to-one-mmt-tok")
@click.option('-src_file', type=str, help="addr to the src file", default='news_comp_zh_en.zh_train.txt')
@click.option('-ref_file', type=str, help="addr to the ref file", default='news_comp_zh_en.en_train.txt')
@click.option('-save_dir', type=str)
def main(gpu_index, prefix, batch_size, model_checkpoint, tok_checkpoint, src_file, ref_file, save_dir):
    torch.cuda.empty_cache()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(tok_checkpoint)
    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': ['<sep>']})
    model = torch.load(model_checkpoint).to(device)
    model.eval()
    # load in dev data
    mask_inp, mask_ref = open(src_file, 'r').readlines(), open(ref_file, 'r').readlines()
    mask_inp, mask_ref = [sen[:-1] for sen in mask_inp], [sen[:-1] for sen in mask_ref]
    # determine the start line and end line index for current gpu
    total_num_gpus = 8
    cur_gpu_size = math.ceil(len(mask_inp)/total_num_gpus)
    line_index_ls = list(range(0, len(mask_inp), cur_gpu_size))
    start_index, end_index = line_index_ls[gpu_index], min(line_index_ls[gpu_index]+cur_gpu_size, len(mask_inp))
    # rewrite the mask_inp into current gpu scope
    mask_inp, mask_ref = mask_inp[start_index:end_index], mask_ref[start_index:end_index]

    dataloader = preprocess_data(mask_inp, mask_ref, tokenizer, exp_config.max_length, batch_size, shuffle=False, sampler=False)

    gen_file = open(f'{save_dir}/{prefix}_{start_index}_{end_index}.outputs', 'w')
    with torch.no_grad():
        for data in dataloader:
            translated_tokens = model.generate(data['input_ids'].to(device))
            batch_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            for text in batch_texts:
                gen_file.write(text+'\n')
    print(f"All outputs are saved at {prefix}.outputs")

if __name__ == "__main__":
    main()
