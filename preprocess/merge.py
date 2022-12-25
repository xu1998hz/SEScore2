import glob
import click

@click.command()
@click.option('-dir')
@click.option('-prefix', help="zh_CN_en_XX_")
@click.option('-save')
def main(dir, prefix, save):
    file_ls = glob.glob(f'{dir}/*')
    file_ls = sorted(file_ls, key=lambda x:int(x.split(prefix)[-1].split('_')[0]))
    new_file = open(save, 'w')
    for f_name in file_ls:
        with open(f_name, 'r') as file_ele:
            for line in file_ele:
                new_file.write(line)
    print("Files are merged!")

if __name__ == "__main__":
    main()
