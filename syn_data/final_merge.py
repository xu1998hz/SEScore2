import glob
import numpy as np
import click
from nltk.tokenize import word_tokenize

@click.command()
@click.option('-dir_addr')
@click.option('-save_np_name')
def main(dir_addr, save_np_name):
    files_ls = glob.glob(dir_addr+'/*')
    sorted_file_ls = sorted(files_ls, key=lambda x:int(x.split('/')[1].split('.')[0]))
    total_ls =[]

    for file_name in sorted_file_ls:
        cur_ls = np.load(file_name).tolist()
        total_ls.extend(cur_ls)

    with open(save_np_name, 'wb') as f:
        np.save(f, np.array(total_ls))

if __name__ == "__main__":
    main()
