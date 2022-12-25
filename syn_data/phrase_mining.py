from transformers import (
    XLMRobertaTokenizer,
    AutoModelForMaskedLM
)
import torch
import click
import os
import numpy as np
import time
import glob

"""code to run: python3 syn_data/phrase_mining.py -file_addr wmt_news/wmt_news_zh_en.txt -batch_size 256 -start_index 0 -end_index 10000 -lang_dir zh-en"""

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

# 1) Use xlm to derive the sentence embeddings and compute L2 on pairs of sentences
# 2) Select top 1000 sentences for each instance
# 3) For each instance, construct phrase-level emebddings in 1-6 window sizes and stride size 1

def sent_emb(hidden_states, emb_type, attention_mask):
    if emb_type == 'last_layer':
        sen_embed = (hidden_states[-1]*attention_mask.unsqueeze(-1)).sum(1)/attention_mask.sum(-1).unsqueeze(-1)
    elif emb_type == 'avg_first_last':
        sen_embed = ((hidden_states[-1]+hidden_states[0])/2.0*attention_mask.unsqueeze(-1)).sum(1)/attention_mask.sum(-1).unsqueeze(-1)
    else:
        print(f"{emb_type} sentence emb type is not supported!")
        exit(1)
    return sen_embed

def pool(model, encoded_input, attention_mask, emb_type, device):
    encoded_input, attention_mask = encoded_input.to(device), attention_mask.to(device)
    outputs = model(input_ids=encoded_input, attention_mask=attention_mask, output_hidden_states=True)
    pool_embed = sent_emb(outputs.hidden_states, emb_type, attention_mask)
    return pool_embed

@click.command()
@click.option('-file_addr', type=str)
@click.option('-batch_size', type=int)
@click.option('-lang_dir', type=str)
@click.option('-start_index', type=int)
@click.option('-end_index', type=int)
@click.option('-folder_name', type=str)
def main(file_addr, batch_size, lang_dir, start_index, end_index, folder_name):
    emb_type, max_length = 'last_layer', 256
    lines = open(file_addr, 'r').readlines()[start_index:end_index]
    lines = [line[:-1] for line in lines]
    # load in xlm models and tokenizers
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large-model").to(device)
    model.eval()
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large-tok")

    save_pair_indices = [[i, min(i+batch_size, len(lines))] for i in range(0, len(lines), batch_size)]

    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    start_time = time.time()
    with torch.no_grad():
        for batch_text, save_pair_indice in zip(batchify(lines, batch_size), save_pair_indices):
            batch_dict = tokenizer(batch_text, return_tensors="pt", max_length=max_length, truncation=True, padding='max_length')
            batch_embed = pool(model, batch_dict['input_ids'], batch_dict['attention_mask'], emb_type, device)
            # save the current batch embeddings
            batch_embed = batch_embed.cpu().detach().numpy()
            save_start_index, save_end_index = save_pair_indice[0]+start_index, save_pair_indice[1]+start_index
            with open(f'{folder_name}/{save_start_index}_{save_end_index}.npy', 'wb') as f:
                np.save(f, batch_embed)
    end_time = time.time()
    print(f"Time duration: {end_time-start_time}s")

    # finish generating all sentence embeddings for the sentence
    files_ls = glob.glob(f'{folder_name}/*')
    sorted_file_ls = sorted(files_ls, key=lambda x:int(x.split('/')[1].split('_')[0]))
    total_ls =[]

    for file_name in sorted_file_ls:
        cur_ls = np.load(file_name).tolist()
        total_ls.extend(cur_ls)

    save_np_name=f"{folder_name}/{start_index}_total.npy"
    with open(save_np_name, 'wb') as f:
        np.save(f, np.array(total_ls))
    print("merge all the files!")

if __name__ == "__main__":
    main()
