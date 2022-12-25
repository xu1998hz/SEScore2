import click
from tqdm.auto import tqdm

"""code to run:
python3 syn_data/margin_compute.py -id_addr ids_query_5027365_10054730_value_part1.txt \
-dist_addr dists_query_5027365_10054730_value_part1.txt -sum_addr_1 sum_query_0_5027365_value_part1.txt \
-sum_addr_2 sum_query_5027365_10054730_value_part1.txt -sum_addr_3 sum_query_10054730_15082095_value_part1.txt \
-sum_addr_4 sum_query_15082095_20109460_value_part1.txt
"""

@click.command()
@click.option('-id_addr', help="ids_query_5027365_10054730_value_part1.txt")
@click.option('-dist_addr', help="dists_query_5027365_10054730_value_part1.txt")
@click.option('-sum_addr_1', help="sum_query_0_5027365_value_part1.txt")
@click.option('-sum_addr_2', help="sum_query_5027365_10054730_value_part1.txt", default=None)
@click.option('-sum_addr_3', help="sum_query_10054730_15082095_value_part1.txt", default=None)
@click.option('-sum_addr_4', help="sum_query_15082095_20109460_value_part1.txt", default=None)
@click.option('-thres', help="top k candidates are computed for margin criterion, ex: 128", type=int)
def main(id_addr, dist_addr, sum_addr_1, sum_addr_2, sum_addr_3, sum_addr_4, thres):
    ids_lines = open(id_addr, 'r')
    print("load in one file")
    dist_lines = open(dist_addr, 'r')
    print("load in two files")
    sum_lines = open(sum_addr_1, 'r').readlines()
    if sum_addr_2:
        sum_lines_2 = open(sum_addr_2, 'r').readlines()
        sum_lines.extend(sum_lines_2)
    if sum_addr_3:
        sum_lines_3 = open(sum_addr_3, 'r').readlines()
        sum_lines.extend(sum_lines_3)
    if sum_addr_4:
        sum_lines_4 = open(sum_addr_4, 'r').readlines()
        sum_lines.extend(sum_lines_4)

    print("loaded all the files")

    save_ids_file = open(f'thres_{thres}_sorted_'+id_addr.split('/')[1], 'w')
    save_dist_file = open(f'thres_{thres}_sorted_'+dist_addr.split('/')[1], 'w')

    id_sum_dict, k = {}, thres
    for index, sum_val in enumerate(sum_lines):
        id_sum_dict[index] = float(sum_val[:-1])

    with tqdm(total=5027365) as pbar:
        for sen_index, (id_ls, dist_ls) in enumerate(zip(ids_lines, dist_lines)):
            id_ls, dist_ls = id_ls[:-1].split('\t'), dist_ls[:-1].split('\t')
            id_ls = [int(id) for id in id_ls]
            dist_ls = [float(dist) for dist in dist_ls]
            forward_sum = id_sum_dict[sen_index]
            margin_ls = []
            for id, dist in zip(id_ls[:thres], dist_ls[:thres]):
                backward_sum = id_sum_dict[id]
                margin = dist/(forward_sum/(2*k)+backward_sum/(2*k))
                margin_ls.append(margin)
            sorted_pair_ls = sorted(zip(margin_ls, id_ls[:thres]), reverse=True) # dec order
            sorted_margin_ls = [str(ele[0]) for ele in sorted_pair_ls]
            sorted_id_ls = [str(ele[1]) for ele in sorted_pair_ls]

            save_ids_file.write('\t'.join(sorted_id_ls)+'\n')
            save_dist_file.write('\t'.join(sorted_margin_ls)+'\n')
            pbar.update(1)

    print("Files are saved!")

if __name__ == "__main__":
    main()
