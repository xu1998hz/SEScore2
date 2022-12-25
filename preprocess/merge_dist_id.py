import click

@click.command()
@click.option('-id1')
@click.option('-id2')
def main(id1, id2):
    id1_lines = open(id1, 'r')
    id2_lines = open(id2, 'r')
    start_index, end_index = id1.split('_')[2], id1.split('_')[3]

    new_id_file = open(f'merge_thres_margin_{start_index}_{end_index}.txt', 'w')

    for id1_line, id2_line in zip(id1_lines, id2_lines):
        new_id_line = id1_line[:-1]+'\t'+id2_line
        new_id_file.write(new_id_line)

    print("File is saved!")

if __name__ == "__main__":
    main()
