import torch
import click
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import os

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

# nohup python3 mbart.py -src_lang zh_CN -tar_lang en_XX -src_file news_data/news_comp_zh_en.zh -batch_size 128 -gpu_index 1

@click.command()
@click.option('-src_lang', type=str)
@click.option('-tar_lang', type=str)
@click.option('-src_file', type=str)
@click.option('-batch_size', type=int)
@click.option('-gpu_index', type=int)
def main(src_lang, tar_lang, src_file, batch_size, gpu_index):
    print("Total number of GPUs: ", torch.cuda.device_count())
    print(f"Using GPU: {gpu_index}")

    # check if stored dir is there and make one if not
    if not os.path.isdir('mbart_outputs'):
        os.makedirs('mbart_outputs')

    total_num_gpus = torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu_index}") if torch.cuda.is_available() else torch.device("cpu")

    if tar_lang == 'en_XX':
        checkpoint = "facebook/mbart-large-50-many-to-one-mmt"
    elif tar_lang == 'de_DE':
        checkpoint = "facebook/mbart-large-50-one-to-many-mmt"
    else:
        print("We currently only support en_XX and de_DE for target language")

    # load in MBart model
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model = model.to(device)
    model.eval()
    # use Mbart tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.src_lang = src_lang
    src_ls = open(src_file, 'r').readlines()
    # determine the start line and end line index for current gpu
    cur_gpu_size = math.ceil(len(src_ls)/total_num_gpus)
    line_index_ls = list(range(0, len(src_ls), cur_gpu_size))
    start_index, end_index = line_index_ls[gpu_index], min(line_index_ls[gpu_index]+cur_gpu_size, len(src_ls))
    # rewrite the src_ls into current gpu scope
    src_ls = src_ls[start_index:end_index]
    # save four levels of scores
    saveFile = open(f'mbart_outputs/{src_lang}_{tar_lang}_{start_index}_{end_index}.txt', 'w')

    for batch_text in batchify(src_ls, batch_size=batch_size):
        with torch.no_grad():
            batch = tokenizer(batch_text, return_tensors="pt", max_length=256, truncation=True, padding=True)['input_ids'].to(device)
            translated = model.generate(batch, forced_bos_token_id=tokenizer.lang_code_to_id[tar_lang])
            translation = tokenizer.batch_decode(translated, skip_special_tokens=True)
            for translation_ele in ziptranslation:
                saveFile.write(translation_ele+'\n')

    print(f"Outputs are saved in mbart_outputs/{src_lang}_{tar_lang}_{start_index}_{end_index}_{num_beam}.txt !")

if __name__ == "__main__":
    main()
