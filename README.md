<h1>SEScore2: Retrieval Augmented Pretraining for Text Generation Evaluation</h1>

<h3>Paper: https://arxiv.org/abs/2212.09305</h3>

<h3>Email: wendaxu@cs.ucsb.edu</h3>

<h3>Install all dependencies:</h3>

````
```
pip install -r requirement/requirements.txt

# To evaluate WMT shared metric task using official script
git clone https://github.com/google-research/mt-metrics-eval.git
cd mt-metrics-eval
pip install .

# Download evaluation data for WMT20, 21 and 22
alias mtme='python3 -m mt_metrics_eval.mtme'
mtme --download  # Puts ~1G of data into $HOME/.mt-metrics-eval.
```
````

<h3>Instructions to score sentences using SEScore2:</h3>

Download weights and data from Google Drive (https://drive.google.com/drive/folders/1I9oji2_rwvifuUSqO-59Fi_vIok_Wvq8?usp=sharing)

To run SEScore2 for text generation evaluation:

````
```
from SEScore2 import SEScore2
from train.regression import *

scorer = SEScore2('en') # load in metric with specified language, en (English), de (German), ja ('Japanese')

refs = ["SEScore is a simple but effective next generation text generation evaluation metric", "SEScore it really works"]

outs = ["SEScore is a simple effective text evaluation metric for next generation", "SEScore is not working"]

scores_ls = scorer.score(refs, outs, 1)
```
````

To reproduce our retrieval augmented data synthesis process (This code needs to be updated! Email me if you need immediate results):

````
```
# first generate laser emebddings for raw texts
scripts/run_laser_emb_gen.sh -d raw_text_file -l zh_en -t wiki # t is the domain of data, we used WMT and wiki in the paper

# build the index table after obtaining embeddings of the texts
scripts/run_sen_index_table.sh -q data_file1 -f data_file1 -s data_file2 -t data_file3 -o data_file4 -u part1 -l zh_en # We build our index table 

# extract 128 neighbors from the index table (Hard negative candidates)
python3 syn_data/extract_neg.py -dists_addr dists_query_file -ids_addr ids_query_file -query_addr query_raw_data -val_addr value_raw_data 

# obtain proposals to edits
scripts/run_edit_ops.sh -p paied_data_file -n num_CPU_processes -s addr_proposals

# edits according to proposals
python3 syn_data/ssl_data_rep_insert_del_construct.py -inp_addr raw_data
 -rep_addr addr_proposals -num_turns num_of_edits -language english

# obtain pretrainig signals through severity estimations
CUDA_VISIBLE_DEVICES=0 python3 syn_data/xlm-align-measure.py -src_addr mt_src_file -ref_addr mt_ref_file -opt_addr mined_sen_op_loc_5.txt -gen_addr english_5_mined_cont.txt -save_folder severity_measure_folder -batch_size 75 -process_index 0 -tot_p 1 -stop_addr idf_weights/english_stopwords.txt

# construct triples (raw pivot sentence, synthetic text, pseudo score)
run_severe_final_data.sh
```
````

To reproduce our baseline and SEScore2 results in machine translation(MT):

````
```
CUDA_VISIBLE_DEVICES=0 python3 inference/inference_regression.py -wmt wmt21.news -lang zh-en -model_addr -model_base xlm-roberta-large -model_addr sescore2_en.ckpt
```
````

To reproduce our baseline and SEScore2 results in Speech Translation (S2T):

````
```
CUDA_VISIBLE_DEVICES=0 python3 inference/test_all_metrics.py -metric_name sescore2  -benchmark_name st-en-ja -ckpt sescore2_ja.ckpt -model_base google/rembert -lang 'ja'
```
````

To reproduce our baseline and SEScore2 results in WebNLG (D2T):

````
```
CUDA_VISIBLE_DEVICES=0 python3 inference/test_all_metrics.py -metric_name sescore2  -benchmark_name webnlg2020 -ckpt sescore2_en.ckpt -model_base xlm-roberta-large -lang 'en'
```
````

To reproduce our baseline and SEScore2 results in Dialogue Generation (This code needs to be updated! Email me if you need immediate results):

````
```
CUDA_VISIBLE_DEVICES=0 python3 inference/test_all_metrics.py -metric_name sescore2  -benchmark_name dialogue -ckpt sescore2_en.ckpt -model_base xlm-roberta-large -lang 'en'
```
````