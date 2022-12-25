import random
import click
"""This program has two purposes:
1) Simplify the task to purely edit distance ops
2) Verify if the previous pipeline is correct (compare two strings are actually similar)"""

"""python3 syn_data/select_pair.py -raw_addr train.en -raw_index_addr train_2M.en -index_addr thres_margin_sep15/thres_margin_0_5027365_value_part1.txt """

@click.command()
@click.option("-raw_addr")
@click.option("-raw_index_addr_0")
@click.option("-raw_index_addr_1")
@click.option("-raw_index_addr_2")
@click.option("-raw_index_addr_3")
@click.option("-index_addr")
# prepare two strings for each instance for edit distance ops
def main(raw_addr, raw_index_addr_0, raw_index_addr_1, raw_index_addr_2, raw_index_addr_3, index_addr):
    raw_lines = open(raw_addr, 'r').readlines()
    raw_index_lines = open(raw_index_addr_0, 'r').readlines()
    raw_index_lines_1 = open(raw_index_addr_1, 'r').readlines()
    raw_index_lines_2 = open(raw_index_addr_2, 'r').readlines()
    raw_index_lines_3 = open(raw_index_addr_3, 'r').readlines()
    raw_index_lines.extend(raw_index_lines_1)
    raw_index_lines.extend(raw_index_lines_2)
    raw_index_lines.extend(raw_index_lines_3)

    index_file = open(index_addr, 'r')
    # start_index = int(index_addr.split('.')[0].split('_')[-2])
    # end_index = int(index_addr.split('.')[0].split('_')[-1])

    # print("start index: ", start_index)
    # print("end index: ", end_index)
    start_index=0
    save_file = open(f'pair_edit_ops_data_en_ja.txt', 'w')

    for cur_index, index_line in enumerate(index_file):
        if index_line != '\n':
            index_ls = index_line[:-1].strip().split('\t')
            str1 = raw_lines[cur_index+start_index][:-1]
            tar_index = random.choice(index_ls)
            # print(index_ls)
            str2 = raw_index_lines[int(tar_index)]
            if str1.strip() == str2[:-1].strip():
                print("bugs!")
                print(str1)
                print(str2)
                exit(1)
            final_line = str(cur_index+start_index) + '\t' + str1 + '\t' + str2
            save_file.write(final_line)

    print("File is saved!")

if __name__ == "__main__":
    main()
