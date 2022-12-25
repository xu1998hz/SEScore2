from tqdm.auto import tqdm

dists_file = open('thres_128_sorted_dists_query_0_5027365_value_part1.txt', 'r')
ids_file = open('thres_128_sorted_ids_query_0_5027365_value_part1.txt', 'r')

for dists, ids in zip(dists_file, ids_file):
