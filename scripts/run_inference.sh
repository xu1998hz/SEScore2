#!/usr/bin/env bash
#!/bin/bash

# load in the shell script arguments:
count=0;

while getopts l: flag
do
    case "${flag}" in
        l) lang_dir=${OPTARG};;
    esac
done

for DIR in new_margin_loss_weights/*; do
  CUDA_VISIBLE_DEVICES="$count" python3 inference/inference.py -lang "$lang_dir" -wmt wmt21.news -model_addr "$DIR" -emb_type last_layer -batch_size 250 -score L2 &
  sleep 1
  count=$(($count+1))
done

echo "All inference codes are running!";
