import random
import click

@click.command()
@click.option('-inp_file')
def main(inp_file):
    inp_lines = open(inp_file, 'r').readlines()
    score_file = open(f'pos_neg_{inp_file}', 'w')

    pos_count, neg_count = 0, 0
    for line_index, line in enumerate(inp_lines):
        line_ls = line[:-1].split('\t')
        ref_line = line_ls[0]
        # 0: pos, original sentence; 1: easy neg: random sentence; 2: hard neg
        random_select = random.choices([0,1,2], weights=[1/16, 1/16, 14/16], k=1)[0]

        if random_select == 0:
            score_file.write(ref_line+'\t'+ref_line+'\t'+str(0)+'\n')
            pos_count+=1
        elif random_select == 1:
            random_neg_index = random.choices(range(len(inp_lines)), k=1)[0]
            selected_sen = inp_lines[random_neg_index].split('\t')[0]
            if random_neg_index!=line_index:
                score_file.write(ref_line+'\t'+selected_sen+'\t'+str(-50)+'\n')
                neg_count+=1
        score_file.write(line)

    print(pos_count)
    print(neg_count)

if __name__ == "__main__":
    main()
