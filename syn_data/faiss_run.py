import faiss
import numpy as np
import os

lines = open('fixed_wiki_raw_2M.en', 'r').readlines()
lines = [line[:-1] for line in lines]

D = 1024
X = np.load('final.npy')
X = np.array(X, dtype='float32')
print(X.shape)

gpu_ids = "0"  # can be e.g. "3,4" for multiple GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

# Setup
cpu_index = faiss.IndexFlatL2(D)
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
gpu_index.add(X)

# Search
topk = 4
dists, ids = gpu_index.search(x=X[4:5], k=topk)
print(lines[4])
print(dists)
print(ids)
for id in ids[0]:
    print(lines[id])
