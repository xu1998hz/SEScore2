<h1>SESCORE2: Learning Text Generation Evaluation via Synthesizing Realistic Mistakes</h1>

<h3>Paper: https://arxiv.org/abs/2212.09305</h3>

<h3>Email: wendaxu@cs.ucsb.edu</h3>

<h3>Install all dependencies:</h3>

````
pip install -r requirement/requirements.txt
````

<h3>Instructions to score sentences using SEScoreX:</h3>

SEScoreX weights can be found in google drive: https://drive.google.com/drive/u/2/folders/1TOUXEDZOsjoq_lg616iKUyWJaK9OXhNP


To run SEScoreX for reference based text generation evaluation:


We have SEScore2 that is only pretrained on synthetic data which only supports five languages (version: pretrained)
````
from sescorex import *
scorer = sescorex(version='pretrained', rescale=False)
````


We further fine-tune the pretrained SEScore2 model using WMT17-21 DA data and WMT22 MQM data, which supports up to 100 languages. The model operates in two modes: 'seg' and 'sys'. The 'seg' mode is more effective for ranking pairs of translations, while the 'sys' mode is better suited for ranking translation systems. By default, we select the 'seg' mode.
````
from sescorex import *
scorer = sescorex(version='seg', rescale=False)
````


You can enable the 'rescale' feature to obtain interpretable scores. In this mode, a score of '0' indicates a perfect translation, '-1' corresponds to a translation with one minor error, and '-5' represents a translation with a major error. You can estimate the number of major and minor errors in the translation by counting the multiples of -5 and -1 in the score, respectively. If you prefer the raw output scores, you can disable rescaling by setting rescale=False.
````
from sescorex import *
scorer = sescorex(version='seg', rescale=True)
refs = ["SEScore is a simple but effective next generation text generation evaluation metric", "you went to hotel"]
outs = ["SEScore is a simple effective text evaluation metric for next generation", "you went to zoo"]
scores_ls = scorer.score(refs=refs, outs=outs, batch_size=32)
````

### Supported Languages
Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskrit, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.


### Table: Model Performance Comparison

| Model   | cs-uk | en-cs | en-ja | en-zh | bn-hi | hi-bn | xh-zu* | zu-xh* | en-hr | en-uk | en-af* | en-am* | en-ha* |
|---------|-------|-------|-------|-------|-------|-------|--------|--------|-------|-------|--------|--------|--------|
| XCOMET  | 0.533 | 0.499 | 0.564 | 0.566 | 0.493 | 0.521 | **0.573** | 0.623  | 0.512 | 0.493 | **0.550** | 0.568  | 0.662  |
| COMET22 | **0.550** | **0.522** | **0.580** | **0.586** | 0.503 | **0.528** | 0.564  | 0.657  | **0.551** | **0.540** | 0.548  | 0.570  | **0.693** |
| Ours    | 0.540 | 0.514 | 0.565 | 0.575 | **0.504** | 0.521 | 0.572  | **0.658** | 0.537 | 0.524 | 0.535  | **0.570** | 0.663  |


| Model   | en-ig* | en-rw* | en-lg* | en-ny* | en-om* | en-sn* | en-ss* | en-sw* | en-tn* | en-xh* | en-yo* | en-zu* | en-gu |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------|
| XCOMET  | 0.502  | 0.446  | 0.579  | 0.494  | 0.653  | 0.702  | 0.548  | 0.650  | 0.479  | 0.633  | 0.541  | 0.551  | **0.694** |
| COMET22 | **0.539** | 0.456  | 0.582  | **0.535** | 0.672  | 0.807  | 0.580  | **0.679** | **0.605** | 0.692  | 0.575  | 0.589  | 0.596 |
| Ours    | 0.538  | **0.478** | **0.603** | 0.529  | **0.697** | **0.820** | **0.598** | 0.674  | 0.585  | **0.702** | **0.591** | **0.597** | 0.607 |

| Model   | en-hi | en-ml | en-mr | en-ta |
|---------|-------|-------|-------|-------|
| XCOMET  | **0.700** | **0.713** | **0.667** | **0.663** |
| COMET22 | 0.587  | 0.617  | 0.570  | 0.626  |
| Ours    | 0.580  | 0.606  | 0.528  | 0.604  |

**Note:** * indicates African languages.

````
@inproceedings{xu-etal-2023-sescore2,
    title = "{SESCORE}2: Learning Text Generation Evaluation via Synthesizing Realistic Mistakes",
    author = "Xu, Wenda  and
      Qian, Xian  and
      Wang, Mingxuan  and
      Li, Lei  and
      Wang, William Yang",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.283/",
    doi = "10.18653/v1/2023.acl-long.283",
    pages = "5166--5183",
    abstract = "Is it possible to train a general metric for evaluating text generation quality without human-annotated ratings? Existing learned metrics either perform unsatisfactory across text generation tasks or require human ratings for training on specific tasks. In this paper, we propose SEScore2, a self-supervised approach for training a model-based metric for text generation evaluation. The key concept is to synthesize realistic model mistakes by perturbing sentences retrieved from a corpus. We evaluate SEScore2 and previous methods on four text generation tasks across three languages. SEScore2 outperforms all prior unsupervised metrics on four text generation evaluation benchmarks, with an average Kendall improvement of 0.158. Surprisingly, SEScore2 even outperforms the supervised BLEURT and COMET on multiple text generation tasks."
}
````


<h3>Old Instructions to score sentences using SEScore2:</h3>

SEScore2 pretrained weights can be found in huggingface: xu1998hz/sescore2_en_pretrained, xu1998hz/sescore2_de_pretrained, xu1998hz/sescore2_ja_pretrained.It will be downloaded automatically. (Those weights can be used to reproduce our paper results)

They ca also be found in Google drive. Download weights and data from Google Drive (https://drive.google.com/drive/folders/1I9oji2_rwvifuUSqO-59Fi_vIok_Wvq8?usp=sharing)
Pretrained weights support five languages: English, German, Spanish, Chinese and Japanese.

To run SEScore2 for text generation evaluation:

````
from SEScore2 import *

scorer = SEScore2('en', mode="pretrained") # load in metric with specified language, en (English), de (German), ja ('Japanese'),  es ('Spanish'), zh ('Chinese'). We have SEScore2 that is only pretrained on synthetic data which only supports five languages (mode: pretrained) and further finetuned on all available human rating data (supports up to 100 languages).
refs = ["SEScore is a simple but effective next generation text generation evaluation metric", "SEScore it really works"]
outs = ["SEScore is a simple effective text evaluation metric for next generation", "SEScore is not working"]
scores_ls = scorer.score(refs, outs, 1)
````

To reproduce our retrieval augmented data synthesis process (This code needs to be updated! Email me if you need immediate results):

````
# first generate laser emebddings for raw texts
scripts/run_laser_emb_gen.sh -d raw_text_file -l zh_en -t wiki # t is the domain of data, we used WMT and wiki in the paper

# build the index table after obtaining embeddings of the texts
scripts/run_sen_index_table.sh -q data_file1 -f data_file1 -s data_file2 -t data_file3 -o data_file4 -u part1 -l zh_en # We build our index table 

# compute the margin score 
python3 syn_data/margin_compute.py -id_addr ids_query.txt -dist_addr dists_query.txt -sum_addr_1 sum_query1.txt -sum_addr_2 sum_query2.txt \
-sum_addr_3 sum_query3.txt -sum_addr_4 sum_query4.txt

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
````

To reproduce our baseline and SEScore2 results in machine translation(MT):

````
CUDA_VISIBLE_DEVICES=0 python3 inference/inference_regression.py -wmt wmt21.news -lang zh-en -model_addr -model_base xlm-roberta-large -model_addr sescore2_en.ckpt
````

To reproduce our baseline and SEScore2 results in Speech Translation (S2T):

````
CUDA_VISIBLE_DEVICES=0 python3 inference/test_all_metrics.py -metric_name sescore2  -benchmark_name st-en-ja -ckpt sescore2_ja.ckpt -model_base google/rembert -lang 'ja'
````

To reproduce our baseline and SEScore2 results in WebNLG (D2T):

````
CUDA_VISIBLE_DEVICES=0 python3 inference/test_all_metrics.py -metric_name sescore2  -benchmark_name webnlg2020 -ckpt sescore2_en.ckpt -model_base xlm-roberta-large -lang 'en'
````

To reproduce our baseline and SEScore2 results in Dialogue Generation (This code needs to be updated! Email me if you need immediate results):

````
CUDA_VISIBLE_DEVICES=0 python3 inference/test_all_metrics.py -metric_name sescore2  -benchmark_name dialogue -ckpt sescore2_en.ckpt -model_base xlm-roberta-large -lang 'en'
````
