import click
import math

@click.command()
@click.option('-file_name')
@click.option('-num_file', type=int)
def main(file_name, num_file):
    inp_lines = open(file_name, 'r').readlines()
    seg_size = math.ceil(len(inp_lines)/num_file)
    prefix = file_name.split('.')[0]
    for i in range(0, len(inp_lines), seg_size):
        start_index, end_index = i, min(i+seg_size, len(inp_lines))
        save_file = open(f"{prefix}_{start_index}_{end_index}.txt", 'w')
        save_file.writelines(inp_lines[start_index:end_index])
        print(f"{prefix}_{start_index}_{end_index}.txt is saved!")

if __name__ == "__main__":
    main()
