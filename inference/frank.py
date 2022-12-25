import click
import torch
import json
from train.regression import *

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

@click.command()
@click.option('-load_benchmark', default=None, help='data for frank benchmark')
@click.option('-model_addr', default=None, help='locations of model weights')
def main(model_addr, load_benchmark):
    exp_config.device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.load(model_addr).to(exp_config.device_id)
    print("model is loaded!")
    tokenizer = AutoTokenizer.from_pretrained("rembert-tok")
    benchmark_data = json.load(open(load_benchmark))

    # collect summary_outs and refs
    summary_outs = [ele['summary'] for ele in benchmark_data]
    refs = [ele['article'] for ele in benchmark_data]
    cur_data_dict = {'pivot': refs, 'mt': summary_outs}
    batch_size=16
    cur_data_loader = preprocess_data(cur_data_dict, tokenizer, exp_config.max_length, batch_size, shuffle=False, sampler=False, mode='test')
    scores_ls = []
    for batch in cur_data_loader:
        # generate a batch of ref, mt embeddings
        score = model(batch, 'last_layer').squeeze(1).tolist()
        scores_ls.extend(score)

    save_ls = []
    for bench_ele, score in zip(benchmark_data, scores_ls):
        temp_dict = {}
        temp_dict["hash"]=bench_ele["hash"]
        temp_dict["model_name"]=bench_ele["model_name"]
        temp_dict["article"]=bench_ele["article"]
        temp_dict["summary"]=bench_ele["summary"]
        temp_dict["score"]=score
        temp_dict["split"]=bench_ele["split"]
        save_ls.append(temp_dict)

    with open('frank_results.json', 'w') as f:
        json.dump(save_ls, f)

    print("File frank_results.json is saved!")

if __name__ == "__main__":
    main()
