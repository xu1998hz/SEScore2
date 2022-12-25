import numpy as np
import faiss
import click
import time
from tqdm.auto import tqdm
import math

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

def file_to_numpy(value_file, dim):
    value = np.fromfile(value_file, dtype=np.float32, count=-1)
    value.resize(value.shape[0] // dim, dim)
    return value

"""python3 syn_data/knn_search.py"""

@click.command()
@click.option('-query_file')
@click.option('-value_file_1')
@click.option('-value_file_2')
@click.option('-value_file_3')
@click.option('-value_file_4')
@click.option('-suffix', default="part1")
def main(query_file, value_file_1, value_file_2, value_file_3, value_file_4, suffix):
    temp_ls = query_file.split('.')[0].split('_')
    query_start, query_end = temp_ls[-2], temp_ls[-1]
    ids_save_file = open(f"ids_query_{query_start}_{query_end}_value_{suffix}.txt", 'w')
    dists_save_file = open(f"dists_query_{query_start}_{query_end}_value_{suffix}.txt", 'w')
    sum_save_file = open(f"sum_query_{query_start}_{query_end}_value_{suffix}.txt", 'w')

    # map the cpu index to gpu (internal processing: only once-> cost around 9 mins)
    dim = 1024
    ngpus = faiss.get_num_gpus()
    cpu_index = faiss.IndexFlatIP(dim)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

    # load in the query matrix
    query = file_to_numpy(query_file, dim)
    # load in the value matrix
    val_1 = file_to_numpy(value_file_1, dim)
    val_ls = [val_1]
    if value_file_2 != 'None':
        val_2 = file_to_numpy(value_file_2, dim)
        val_ls.append(val_2)
    if value_file_3 != 'None':
        val_3 = file_to_numpy(value_file_3, dim)
        val_ls.append(val_3)
    if value_file_4 != 'None':
        val_4 = file_to_numpy(value_file_4, dim)
        val_ls.append(val_4)
    value = np.concatenate(val_ls, axis=0)
    value = value/np.expand_dims(np.linalg.norm(value, axis=1), axis=1) # normalize the vectors
    print("load in value matrix")

    gpu_index.add(value)
    print("load in data into index table")

    # first find 2048 k-nearest neighbors
    topk, batch_size = 128, 128
    start = time.time()
    with tqdm(total=math.ceil(query.shape[0]/batch_size)) as pbar:
        for batch_query in batchify(query, batch_size):
            dists_ls, ids_ls = gpu_index.search(x=batch_query, k=topk)
            for dists, ids in zip(dists_ls, ids_ls):
                temp_ls_ids = [str(ele) for ele in ids]
                save_ids_line = '\t'.join(temp_ls_ids)+'\n'
                temp_ls_dists = [str(ele) for ele in dists]
                save_dists_line = '\t'.join(temp_ls_dists)+'\n'
                save_sum_dists_line = str(np.sum(dists))+'\n'
                ids_save_file.write(save_ids_line)
                dists_save_file.write(save_dists_line)
                sum_save_file.write(save_sum_dists_line)
            pbar.update(1)
    end = time.time()

    print(end-start)
    print("files are saved!")

if __name__ == "__main__":
    main()
