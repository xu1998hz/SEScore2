from tqdm.auto import tqdm
import click

"""python3 syn_data/extract_neg.py -dists_addr sorted_dists_query_20109460_25136824_value_part2.txt -ids_addr sorted_ids_query_20109460_25136824_value_part2.txt -query_addr train.en -val_addr news_wiki.en"""

@click.command()
@click.option('-dists_addr')
@click.option('-ids_addr')
@click.option('-query_addr')
@click.option('-val_addr')
@click.option('-val_addr_1')
@click.option('-val_addr_2')
@click.option('-val_addr_3')
def main(dists_addr, ids_addr, query_addr, val_addr, val_addr_1, val_addr_2, val_addr_3):
    dist_file = open(dists_addr, 'r')
    id_file = open(ids_addr, 'r')
    query_lines = open(query_addr, 'r').readlines()

    val_lines = open(val_addr, 'r').readlines()
    val_lines_1 = open(val_addr_1, 'r').readlines()
    val_lines_2 = open(val_addr_2, 'r').readlines()
    val_lines_3 = open(val_addr_3, 'r').readlines()

    val_lines.extend(val_lines_1)
    val_lines.extend(val_lines_2)
    val_lines.extend(val_lines_3)

    start_index, end_index = int(ids_addr.split('_')[-4]), ids_addr.split('_')[-3]
    thres_id_file = open(f'thres_margin_{start_index}_{end_index}_value_part{ids_addr[-5]}.txt', 'w')
    min_samples = 128
    unqualified = 0
    correct = 0

    # with tqdm(total=5027365) as pbar:
    for cur_index, (dist_line, id_line) in enumerate(zip(dist_file, id_file)):
        dist_ls = dist_line[:-1].split('\t')
        id_ls = id_line[:-1].split('\t')
        cur_id_ls = []

        for id, dist in zip(id_ls, dist_ls):
            if float(dist) >= 1.06 and val_lines[int(id)].strip() != query_lines[start_index+cur_index].strip():
                cur_id_ls.append(id)

        thres_id_file.write('\t'.join(cur_id_ls)+'\n')
        if cur_index == int(id_ls[0]):
            correct+=1
        if len(cur_id_ls) == 0:
            unqualified+=1
        min_samples=min(min_samples, len(cur_id_ls))
            # pbar.update(1)

    print(correct)
    print(unqualified)
    print("File is generated!")

if __name__ == "__main__":
    main()
