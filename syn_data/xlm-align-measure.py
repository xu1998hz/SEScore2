from transformers import AutoTokenizer, XLMRobertaForMaskedLM
from nltk.tokenize import word_tokenize
import nltk
import torch.nn as nn
import torch
import click
import math
import time
import string
import re
import spacy
nltk.download('punkt')

"""CUDA_VISIBLE_DEVICES=0 python3 syn_data/xlm-align-measure.py -src_addr  sep17_zh_en_ssl_data/train.zh -ref_addr sep17_zh_en_ssl_data/train.en -opt_addr sep17_zh_en_ssl_data/0_5027365_news_index_table/mined_locs/mined_sen_op_loc_5.txt -gen_addr sep17_zh_en_ssl_data/0_5027365_news_index_table/cont_locs/english_5_mined_cont.txt -save_folder severity_measure_sep17_zh_en_news_0_5027365 -batch_size 75 -process_index 0 -tot_p 1 -stop_addr idf_weights/english_stopwords.txt"""

def detect_punkt(words_ls):
    for word in words_ls:
        if word not in string.punctuation:
            return False
    return True

def ret_weight_ls(sub_tgt_words_ls, stop_lines):
    if detect_punkt(sub_tgt_words_ls):
        return 'punkt'
    else:
        weight_ls=[]
        for word in sub_tgt_words_ls:
            if word.lower() in stop_lines:
                weight_ls.append(0)
            else:
                if word not in string.punctuation:
                    weight_ls.append(2)
        if max(weight_ls) < 1:
            return 'insig'
        else:
            return 'sig'

""" if it signifcant returns True """
def check_word_sig(word, stop_lines):
    if word.lower() in stop_lines:
        return False
    else:
        if word not in string.punctuation:
            return True
        else:
            return False

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = XLMRobertaForMaskedLM.from_pretrained("xlm-align-base-model").to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("xlm-align-base-tok", use_fast=True, add_prefix_space=True)
max_length = 512

m = nn.Softmax(dim=-1)
mask_id = tokenizer.vocab['<mask>']
ids_to_words = {val: key for key, val in tokenizer.vocab.items()}

@click.command()
@click.option('-src_addr')
@click.option('-ref_addr')
@click.option('-opt_addr')
@click.option('-gen_addr')
@click.option('-save_folder')
@click.option('-batch_size', type=int)
@click.option("-process_index", type=int)
@click.option("-tot_p", type=int)
@click.option("-stop_addr")
@click.option("-language")
def main(src_addr, ref_addr, opt_addr, gen_addr, batch_size, save_folder, process_index, tot_p, \
        stop_addr, language): # , part_index, gpu_total
    gen_texts = open(gen_addr, 'r').readlines()
    # decide the partition of the file
    file_batch_size = math.ceil(len(gen_texts)/tot_p)
    start_end_pair = [[i, min(i+file_batch_size, len(gen_texts))] for i in range(0, len(gen_texts), file_batch_size)]
    start_index, end_index = start_end_pair[process_index][0], start_end_pair[process_index][1]
    gen_texts = gen_texts[start_index:end_index] # parition content file

    src_texts = open(src_addr, 'r').readlines()# no need to partition
    tgt_texts = open(ref_addr, 'r').readlines()# no need to partition
    stop_lines = open(stop_addr, 'r').readlines() # load all the stop words for English
    stop_lines = [line[:-1] for line in stop_lines]

    ops_loc_texts = open(opt_addr, 'r').readlines()
    ops_loc_texts = ops_loc_texts[start_index:end_index] # parition loc file

    print("file is loaded!")
    save_suffix = gen_addr.split('/')[-1]
    saveFile = open(f'{save_folder}/scores_part_{process_index}_{save_suffix}', 'w')

    new_tgt_ls, locs_ls, idf_sig_eval_ls, len_sen = [], [], [], []
    filter_line_index = []

    if language == 'japanese':
        nlp = spacy.blank('ja')

    start_time = time.time()
    for line_index, (ops_loc, gen) in enumerate(zip(ops_loc_texts, gen_texts)):
        ops_ls = ops_loc[:-1].split('\t')
        sen_index, ops_type = int(ops_ls[0]), int(ops_ls[1])
        if language == 'japanese':
            tgt_words_ls = [word.text for word in nlp(tgt_texts[sen_index][:-1])]
            gen_words_ls = [word.text for word in nlp(gen[:-1])]
        else:
            tgt_words_ls = word_tokenize(tgt_texts[sen_index][:-1])
            gen_words_ls = word_tokenize(gen[:-1])

        # insert operation
        if ops_type == 1:
            start_index = int(ops_ls[2])
            new_tgt_words_ls = tgt_words_ls[:start_index] + gen_words_ls + tgt_words_ls[start_index:]
            filter_line_index.append(line_index)
            # 0 indicates the insignicance of the original content (insert has no original content)
            idf_sig_eval_ls.append([gen_words_ls, 'insig'])
        # replace operation
        elif ops_type == 2:
            start_index, ori_end_index = int(ops_ls[2].split('_')[0]), int(ops_ls[2].split('_')[1])
            # if the new content is not the substring of the old content (remove all delete cases)
            if ' '.join(gen_words_ls) not in ' '.join(tgt_words_ls[start_index:ori_end_index]):
                new_tgt_words_ls = tgt_words_ls[:start_index] + gen_words_ls + tgt_words_ls[ori_end_index:]
                filter_line_index.append(line_index)
                # calculate the weights for the original target content
                sub_tgt_words_ls = tgt_words_ls[start_index:ori_end_index]
                idf_sig_eval_ls.append([gen_words_ls, ret_weight_ls(sub_tgt_words_ls, stop_lines), sub_tgt_words_ls])
            else:
                continue
        else:
            print("Errors in op types: ", ops_type)
            exit(1)
        end_index = start_index+len(gen_words_ls)
        new_tgt_ls.append([src_texts[sen_index][:-1]]+['</s>']+new_tgt_words_ls)
        len_sen.append(len(new_tgt_words_ls))
        locs_ls.append([start_index, end_index])
    print("Finished in: ", time.time()-start_time)

    inp_id_ls, attn_ls, emb_start_end_ls, ids_ls, final_line_index, context_parts, final_idf_sig_eval_ls = [], [], [], [], [], [], []
    count, correct = 0, 0
    # store word-wise embedding locations for all instances
    emb_locs_total_ls = []
    for sen_index, (inp_text, start_end, cur_line_index, idf_sig_eval, ele_len) in enumerate(zip(new_tgt_ls, locs_ls, filter_line_index, \
        idf_sig_eval_ls, len_sen)):
        start_index, end_index = start_end[0], start_end[1]
        ind_records = tokenizer.batch_encode_plus([inp_text], return_tensors="pt", truncation=True, \
            padding='max_length', is_split_into_words=True, max_length=max_length)

        # print(ind_records)
        # print(inp_text)
        # print(len(ind_records.tokens(0)))
        # print(ind_records.tokens(0))
        # print(ind_records.word_ids(0))
        # print()
        # exit(1)

        ind_dict, emb_locs_sublist = {}, []
        for index, ind in enumerate(ind_records.word_ids(0)):
            if ind != None:
                if ind not in ind_dict:
                    ind_dict[ind] = []
                ind_dict[ind].append(index)

        # print(inp_text)
        # print(len(ind_records.tokens(0)))
        # print(ind_records.word_ids(0))
        # print()
        # print(ind_dict)
        # print(start_index+2)
        # print(end_index+2)
        # print('----------------------')
        # end index will not be reached so only needs to +1 not plus 2!
        if max(ind_dict) >= end_index+1 and start_index != end_index and len(ind_dict) == ele_len+2:
            for cur_loc in range(start_index+2, end_index+2):
                if cur_loc in ind_dict:
                    emb_locs_sublist.append(ind_dict[cur_loc])

            emb_start, emb_end = emb_locs_sublist[0][0], emb_locs_sublist[-1][-1]+1
            emb_locs_total_ls.append(emb_locs_sublist)
            ids = [ind_records['input_ids'][:, ele_pair_index[0]:ele_pair_index[-1]+1][0].clone() \
                for ele_pair_index in emb_locs_sublist]

            ind_records['input_ids'][:, emb_start:emb_end] = mask_id
            inp_id_ls.append(ind_records['input_ids'])
            attn_ls.append(ind_records['attention_mask'])
            emb_start_end_ls.append([emb_start, emb_end])
            final_line_index.append(cur_line_index)
            correct += 1
            ids_ls.append(ids)
            context_parts.append(inp_text)
            final_idf_sig_eval_ls.append(idf_sig_eval)
        else:
            # exceed max length or not have output (equivalent to delete)
            count += 1

    print("Sentences that will be kept: ", correct)

    final_score_ls = []
    count_minor_punkt, count_minor, count_major = 0, 0, 0
    with torch.no_grad():
        for inp_id_batch, attn_batch, emb_start_end_index_batch, ids_batch, context_part_batch, idf_sig_batch, ele_emb_start_end_batch in zip( \
            batchify(inp_id_ls, batch_size), batchify(attn_ls, batch_size), batchify(emb_start_end_ls, batch_size), batchify(ids_ls, batch_size), \
            batchify(context_parts, batch_size), batchify(final_idf_sig_eval_ls, batch_size), batchify(emb_locs_total_ls, batch_size)):
            # obtain the batch of model logits
            model_dict = model(input_ids=torch.cat(inp_id_batch, dim=0).to(device), attention_mask=torch.cat(attn_batch, dim=0).to(device), return_dict=True)
            # within the batch evaluate each instance's span probabilities
            for instance_logits, emb_start_end_index, ele_ids, context_part, idf_sig, ele_emb_start_end_ls in zip(model_dict["logits"], emb_start_end_index_batch, ids_batch,\
                context_part_batch, idf_sig_batch, ele_emb_start_end_batch):
                emb_start, emb_end = emb_start_end_index[0],  emb_start_end_index[1]
                span_probs_ls = []
                # print(context_part)
                # print(idf_sig)
                # print(ele_ids)
                # for ele in ele_ids:
                #     for word in ele:
                #         print(ids_to_words[word.item()])
                for id_ls, ele_emb_start_end, word_ele in zip(ele_ids, ele_emb_start_end_ls, idf_sig[0]):
                    if check_word_sig(word_ele, stop_lines):
                        ele_start_index, ele_end_index = ele_emb_start_end[0], ele_emb_start_end[-1]+1
                        word_subtok_ls = []
                        for id, emb_probs_tensor in zip(id_ls, m(instance_logits[ele_start_index:ele_end_index, :])):
                            word_subtok_ls+=[emb_probs_tensor[id].item()]
                        #     print(emb_probs_tensor[id].item())
                        # print(sum(word_subtok_ls)/len(word_subtok_ls))
                        span_probs_ls.append(sum(word_subtok_ls)/len(word_subtok_ls))
                    # else:
                    #     print("discard one word during prob computing!")

                # if len(span_probs_ls)>0:
                #     span_prob = sum(span_probs_ls)/len(span_probs_ls)
                # else:
                #     span_prob = 1.1
                # add a corrrection step for the predefined rule
                gen_cont_result = ret_weight_ls(idf_sig[0], stop_lines)
                # For insert portion
                if len(idf_sig) == 2:
                    # if insertion is punkt, assign -0.1
                    if gen_cont_result == 'punkt':
                        final_score_ls.append(-0.1)
                        count_minor_punkt+=1
                    # if insertion is insig content, assign -1
                    elif gen_cont_result == 'insig':
                        final_score_ls.append(-1)
                    # if insertion is sig content, measure minimum word prob, if min(word span) <=0.5 -> severe
                    else:
                        if min(span_probs_ls) < 0.5:
                            final_score_ls.append(-5)
                            count_major+=1
                        else:
                            final_score_ls.append(-1)
                # For replace portion
                else:
                    # for digit difference, directly assign severe score and ignore probabilities
                    if len(idf_sig[0])==1 and len(idf_sig[2])==1 and re.sub(r'[\W\s]', ' ', idf_sig[0][0]).isdigit() \
                        and re.sub(r'[\W\s]', ' ', idf_sig[2][0]).isdigit():
                        final_score_ls.append(-5)
                        count_major+=1
                    else:
                        # if both original and replaced content are punkt, assign -0.1
                        if idf_sig[1] == 'punkt' and gen_cont_result == 'punkt':
                            final_score_ls.append(-0.1)
                            count_minor_punkt+=1
                        # if both original and replaced content are not sig, assign -1
                        elif idf_sig[1] != 'sig' and gen_cont_result != 'sig':
                            final_score_ls.append(-1)
                            count_minor+=1
                        # if replace is sig content in either original or new, measure minimum word prob, if min(word span) <=0.5 -> severe
                        else:
                            if len(span_probs_ls) > 0:
                                if min(span_probs_ls) < 0.5:
                                    final_score_ls.append(-5)
                                    count_major+=1
                                else:
                                    final_score_ls.append(-1)
                                    count_minor+=1
                            else:
                                final_score_ls.append(-5)
                                count_major+=1

                # if len(span_probs_ls)>0:
                #     print("minimum prob: ", min(span_probs_ls))
                # else:
                #     print("minimum prob: ", span_prob)
                # print("total prob: ", span_prob)
                # print("final score: ", final_score_ls[-1])
                # print('---------------------------')
                # print()

    print(count_minor_punkt)
    print(count_minor)
    print(count_major)
    # save (file line index, op type, op locs, new content, prob score)
    for cur_line_index, final_score in zip(final_line_index, final_score_ls):
        saveFile.write(ops_loc_texts[cur_line_index][:-1]+'\t'+gen_texts[cur_line_index][:-1]+'\t'+str(final_score)+'\n')

if __name__ == "__main__":
    main()
