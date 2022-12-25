import click

@click.command()
@click.option('-file_name')
@click.option('-count')
@click.option('-type')
def main(file_name, count, type):
    lines_ls = open(file_name).readlines()
    lines_ls = [line[:-1] for line in lines_ls]
    en_file = open(f"en_{type}_{count}.txt", 'w')
    de_file = open(f"de_{type}_{count}.txt", 'w')
    for line in lines_ls:
        line_ele_ls = line.split('\t<b>\t</b>\t')
        de_line, en_line = line_ele_ls[0], line_ele_ls[1]
        en_file.write(en_line+'\n')
        de_file.write(de_line+'\n')
    print("All lines are processed!")

if __name__ == "__main__":
    main()
