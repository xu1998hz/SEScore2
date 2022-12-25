#!/usr/bin/env bash
#!/bin/bash

# ./run_severity_measure.sh -b 75 -s oct2_en_ja_ssl_data/train_en_ja_5M.en -r oct2_en_ja_ssl_data/train_en_ja_5M.ja -o oct2_en_ja_ssl_data/mined_sen_op_loc_5.txt -g oct2_en_ja_ssl_data/japanese_5_mined_cont_rand_None.txt  -n 8 -f severity_measure_en_ja -l oct2_en_ja_ssl_data -i japanese_stopwords.txt -a japanese
# ./run_severity_measure.sh -b 75 -s sep17_zh_en_ssl_data/train.zh -r sep17_zh_en_ssl_data/train.en -o sep17_zh_en_ssl_data/0_5027365_news_index_table/mined_locs/mined_sen_op_loc_1.txt
# -g sep17_zh_en_ssl_data/0_5027365_news_index_table/cont_locs/english_1_mined_cont.txt -n 8 -f severity_measure_sep17_zh_en_news_0_5027365 -l
while getopts b:s:r:o:g:f:n:l:i: flag
do
    case "${flag}" in
        b) batch_size=${OPTARG};;
        s) src_addr=${OPTARG};;
        r) ref_addr=${OPTARG};;
        o) opt_addr=${OPTARG};;
        g) gen_addr=${OPTARG};;
        f) folder=${OPTARG};;
        n) num_processes=${OPTARG};;
        l) loc_folder=${OPTARG};;
        i) idf_weight_addr=${OPTARG};;
        a) language=${OPTARG};;
    esac
done

echo "src_addr: $src_addr"
echo "ref_addr: $ref_addr"
echo "opt_addr: $opt_addr"
echo "gen_addr: $gen_addr"
echo "batch size: $batch_size"
echo "folder: $folder"
echo "num of processes: $num_processes"
echo "loc folder: $loc_folder"
echo "idf weights: $idf_weight_addr"

mkdir "$folder"

for (( i=0; i<$num_processes; i++ ))
do
  CUDA_VISIBLE_DEVICES=$i python3 syn_data/xlm-align-measure.py -src_addr "$src_addr" -ref_addr "$ref_addr" \
    -opt_addr "$opt_addr" -gen_addr "$gen_addr" -batch_size "$batch_size" \
    -save_folder "$folder"  -process_index $i -tot_p $num_processes -stop_addr "$idf_weight_addr" -language "$language" &
  sleep 10
done

wait
echo "all processed finished at this point!"
