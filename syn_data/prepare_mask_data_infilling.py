import click
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

"""code to run: python3 syn_data/prepare_mask_data_infilling.py -src_addr train_400k.zh -ref_addr train_400k.en -op_addr mt_sen_op_loc_1.txt -lang_dir zh_en"""

@click.command()
@click.option('-src_addr', type=str)
@click.option('-ref_addr', type=str)
@click.option('-op_addr', type=str)
@click.option('-lang_dir', type=str)
def main(src_addr, ref_addr, op_addr, lang_dir):
    src_lines = open(src_addr, 'r').readlines()
    ref_lines = open(ref_addr, 'r').readlines()
    op_lines = open(op_addr, 'r').readlines()

    src_lines = [line[:-1] for line in src_lines]
    ref_lines = [line[:-1] for line in ref_lines]
    op_lines = [line[:-1] for line in op_lines]

    saveFile = open(lang_dir+"_"+op_addr, 'w')

    for line in op_lines:
        line_ls = line.split('\t')
        sen_index, op, edit_locs = int(line_ls[0]), int(line_ls[1]), line_ls[2]
        words_ls = word_tokenize(ref_lines[sen_index])
        if op == 1:
            edit_locs_start = int(edit_locs)
            new_words_ls = words_ls[:edit_locs_start] + ['<mask>'] + words_ls[edit_locs_start:]
        elif op == 2:
            edit_los_ls = edit_locs.split('_')
            edit_locs_start, edit_locs_end = int(edit_los_ls[0]), int(edit_los_ls[1])
            new_words_ls = words_ls[:edit_locs_start] + ['<mask>'] + words_ls[edit_locs_end:]
        new_line = src_lines[sen_index]+' <sep> '+TreebankWordDetokenizer().detokenize(new_words_ls)
        saveFile.write(new_line+'\n')

if __name__ == "__main__":
    main()
