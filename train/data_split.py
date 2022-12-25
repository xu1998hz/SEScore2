import random
import click
import glob
import random

"""Offer the same data splits for src, ref and mt files"""
"""python3 data_split.py -file1 news_data/en-de/news_comp_en_de.en -file2 news_data/en-de/news_comp_en_de.de -file3 merge_outputs/en_XX_de_DE_total.txt"""

@click.command()
@click.option('-file1')
@click.option('-file2')
@click.option('-file3')
def main(file1, file2, file3):
    line1_ls = open(file1, 'r').readlines()
    line2_ls = open(file2, 'r').readlines()
    line3_ls = open(file3, 'r').readlines()

    ls_temp = list(zip(line1_ls, line2_ls, line3_ls))
    random.shuffle(ls_temp)

    new_line1_ls, new_line2_ls, new_line3_ls = zip(*ls_temp)
    new_line1_ls, new_line2_ls, new_line3_ls = list(new_line1_ls), list(new_line2_ls), list(new_line3_ls)

    train_file1 = open(file1+'_train.txt', 'w')
    train_file2 = open(file2+'_train.txt', 'w')
    train_file3 = open(file3+'_train.txt', 'w')

    dev_file1 = open(file1+'_dev.txt', 'w')
    dev_file2 = open(file2+'_dev.txt', 'w')
    dev_file3 = open(file3+'_dev.txt', 'w')

    split_index = int(len(new_line1_ls)*0.9)

    train_file1.writelines(new_line1_ls[:split_index])
    train_file2.writelines(new_line2_ls[:split_index])
    train_file3.writelines(new_line3_ls[:split_index])

    dev_file1.writelines(new_line1_ls[split_index:])
    dev_file2.writelines(new_line2_ls[split_index:])
    dev_file3.writelines(new_line3_ls[split_index:])

    print("Three new files are reproduced!")

if __name__ == "__main__":
    main()
