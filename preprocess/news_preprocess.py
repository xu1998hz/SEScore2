import click
import pandas as pd
import numpy as np

@click.command()
@click.option('-f')
@click.option('-src_lang')
@click.option('-tar_lang')
def main(f, src_lang, tar_lang):
    srcFile, tarFile = open(f"news_comp_{src_lang}_{tar_lang}.{src_lang}", 'w'), open(f"news_comp_{src_lang}_{tar_lang}.{tar_lang}", 'w')
    tsvFile = pd.read_csv(f, '\t', header=None, on_bad_lines='skip')
    # replace all empty locations with nans and drop all rows contain nan
    tsvFile.replace('', np.nan, inplace=True)
    tsvFile = tsvFile.dropna()
    src_ls, ref_ls = list(tsvFile[1]), list(tsvFile[0])
    src_ls = [src.replace('\n','')+'\n' for src in src_ls]
    ref_ls = [ref.replace('\n','')+'\n' for ref in ref_ls]
    assert(len(src_ls)==len(ref_ls))
    srcFile.writelines(src_ls)
    tarFile.writelines(ref_ls)
    print("files are produced!")

if __name__ == "__main__":
    main()
