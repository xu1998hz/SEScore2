import torch
import click
import nltk
from syn_data.biimt import exp_config, preprocess_data
from transformers import (
    AutoTokenizer
)
import random

""" CUDA_VISIBLE_DEVICES=0 python3 syn_data/epoch_bleu_eval.py -model_checkpoint mt_infilling_weights_first_trial/mt_infilling_weights_zh-en_2M_1660468959/epoch0.ckpt
-src_dev_file zh_en_mask_src_0.7_2.txt -ref_dev_file zh_en_mask_tar_0.7_2.txt """

@click.command()
@click.option('-batch_size', type=int, default=64)
@click.option('-model_checkpoint', type=str)
@click.option('-tok_checkpoint', type=str, default="mbart-large-50-many-to-one-mmt-tok")
@click.option('-src_dev_file', type=str, help="addr to the src dev file", default='zh_en_mask_src_0.7_2.txt')
@click.option('-ref_dev_file', type=str, help="addr to the ref dev file", default='zh_en_mask_tar_0.7_2.txt')
def main(batch_size, model_checkpoint, tok_checkpoint, src_dev_file, ref_dev_file):
    torch.cuda.empty_cache()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(tok_checkpoint)
    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': ['<sep>']})
    model = torch.load(model_checkpoint).to(device)
    model.eval()
    # load in dev data
    mask_inp_dev, mask_ref_dev = open(src_dev_file, 'r').readlines(), open(ref_dev_file, 'r').readlines()
    mask_inp_dev, mask_ref_dev = [sen[:-1] for sen in mask_inp_dev], [sen[:-1] for sen in mask_ref_dev]
    dev_dataloader = preprocess_data(mask_inp_dev, mask_ref_dev, tokenizer, exp_config.max_length, batch_size, shuffle=False, sampler=False)

    gen_dev = []
    gen_file = open(f'{model_checkpoint}.outputs', 'w')
    with torch.no_grad():
        for dev_batch in dev_dataloader:
            # print(dev_batch['input_ids'].size())
            translated_tokens = model.generate(dev_batch['input_ids'].to(device))
            batch_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            gen_dev.extend(batch_texts)

    save_gen_dev = [sen+'\n' for sen in gen_dev]
    gen_file.writelines(save_gen_dev)
    print("Files are generated which contains generated outputs!")

    total_bleu = 0
    for gen_sen, ref_sen in zip(gen_dev, mask_ref_dev):
        total_bleu+=nltk.translate.bleu_score.sentence_bleu([ref_sen], gen_sen)
    sen_bleu = total_bleu/len(mask_ref_dev)

    print(f"BLEU Score: {sen_bleu}")

if __name__ == "__main__":
    main()
