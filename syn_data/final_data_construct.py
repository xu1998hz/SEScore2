from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import click
import spacy

"""code to run:
python3 syn_data/final_data_construct.py -lang english -severe_enable True -folder_name severity_measure_sep17_zh_en_news_0_5027365 -ref_addr sep17_zh_en_ssl_data/train.en """

def file_to_dict(syn_data_dict, file_name):
    inp_file = open(file_name, 'r').readlines()
    for line in inp_file:
        line_ls = line[:-1].split('\t')
        if int(line_ls[0]) not in syn_data_dict:
            syn_data_dict[int(line_ls[0])] = {}
            syn_data_dict[int(line_ls[0])]['span'] = []
            syn_data_dict[int(line_ls[0])]['content'] = []
            syn_data_dict[int(line_ls[0])]['score'] = []
        if line_ls[1] == '1':
            syn_data_dict[int(line_ls[0])]['span'].append([int(line_ls[2]), int(line_ls[2])])
        else:
            syn_data_dict[int(line_ls[0])]['span'].append([int(line_ls[2].split('_')[0]), int(line_ls[2].split('_')[1])])
        syn_data_dict[int(line_ls[0])]['content'].append(line_ls[3])
        syn_data_dict[int(line_ls[0])]['score'].append(float(line_ls[4]))
    return syn_data_dict

def sen_construct(locs_records, replaced_contents, words_ls):
    new_words_ls = []
    sorted_locs_cont = sorted(list(zip(locs_records, replaced_contents)))
    if len(sorted_locs_cont) == 1:
        new_words_ls += words_ls[:locs_records[0][0]]+replaced_contents+words_ls[locs_records[0][1]:]
    else:
        for cur_index, locs_content in enumerate(sorted_locs_cont):
            locs_pair, content = locs_content[0], locs_content[1]
            if cur_index == 0:
                new_words_ls += words_ls[:locs_pair[0]]+[content]
            elif cur_index == len(sorted_locs_cont)-1:
                new_words_ls += words_ls[sorted_locs_cont[cur_index-1][0][1]:locs_pair[0]]+[content]+words_ls[locs_pair[1]:]
            else:
                new_words_ls += words_ls[sorted_locs_cont[cur_index-1][0][1]:locs_pair[0]]+[content]
    return new_words_ls

@click.command()
@click.option('-lang', help="english")
@click.option('-severe_enable', help="If enabled, severity measure will be added!", type=bool)
@click.option('-folder_name')
@click.option('-ref_addr')
@click.option('-del_enable', type=bool)
@click.option('-insert_replace_enable', type=bool)
def main(lang, severe_enable, folder_name, ref_addr, del_enable, insert_replace_enable):
    syn_data_dict = {}

    for i in range(1, 6):
        if del_enable:
            syn_data_dict = file_to_dict(syn_data_dict, f'{folder_name}/score_del_sen_op_loc_{i}.txt')

    print(f"len of del: {len(syn_data_dict)}")

    for i in range(1, 6):
        if insert_replace_enable:
            syn_data_dict = file_to_dict(syn_data_dict, f'{folder_name}/scores_{lang}_{i}_mined_cont_rand_None.txt')

    print(f"len of del+insert+rep: {len(syn_data_dict)}")

    if len(syn_data_dict)==0:
        print("Error in loading score files")
        exit(1)

    print(syn_data_dict[58992])

    ref_lines = open(ref_addr, 'r').readlines()
    ref_lines = [line[:-1] for line in ref_lines]
    saveFile = open(f'{lang}_{severe_enable}_severe_minor.txt', 'w')

    if lang == 'japanese':
        nlp = spacy.blank('ja')

    for sen_index in syn_data_dict:
        if lang == 'japanese':
            words_ls = [word.text for word in nlp(ref_lines[sen_index])]
            mt_out = sen_construct(syn_data_dict[sen_index]['span'], syn_data_dict[sen_index]['content'], words_ls)
        else:
            mt_out = sen_construct(syn_data_dict[sen_index]['span'], syn_data_dict[sen_index]['content'], word_tokenize(ref_lines[sen_index]))
        if lang == 'japanese':
            mt_out = ''.join(mt_out)
        else:
            mt_out = TreebankWordDetokenizer().detokenize(mt_out)
        if severe_enable:
            score = sum(syn_data_dict[sen_index]['score'])
        else:
            score = -len(syn_data_dict[sen_index]['score'])
        saveFile.write(ref_lines[sen_index]+'\t'+mt_out+'\t'+str(score)+'\n')

    print(f"Saved at {lang}_{severe_enable}_severe_minor.txt!")

if __name__ == "__main__":
    main()
