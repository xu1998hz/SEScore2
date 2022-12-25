#!/usr/bin/env bash
#!/bin/bash

# code: ./run_syn_data_gen.sh -l en-de -b 32 -t dev -c 1
while getopts l:b:t:c: flag
do
    case "${flag}" in
        l) lang_dir=${OPTARG};;
        b) batch_size=${OPTARG};;
        t) type=${OPTARG};;
        c) cycle=${OPTARG};;
    esac
done

cur_time=$(date +%s)
save_folder=syn_data_gen_"$lang_signal"_"$type"_"$cur_time"
mkdir "$save_folder"

for i in {0..7..1}
do
  CUDA_VISIBLE_DEVICES="$i" python3 syn_data/syn_data_gen.py -prefix cycle"$cycle"_infill_mt_0.15_"$lang_signal" \
  -model_checkpoint epoch4.ckpt -batch_size "$batch_size" -gpu_index "$i" -save_dir "$save_folder" \
  -src_file news_prepare_mask_data_syn_gen/cycle1_"$lang_signal"_news_"$type"_src_mask.txt \
  -ref_file news_prepare_mask_data_syn_gen/cycle1_"$lang_signal"_news_"$type"_ref_mask.txt &
  sleep 1
done

wait
echo "all processed finished at this point!"

