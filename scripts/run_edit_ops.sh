#!/usr/bin/env bash
#!/bin/bash

# ./run_edit_ops.sh -p sep17_zh_en_ssl_data/pair_edit_ops_data_news_0_5027365.txt -n 120 -s cont_total.txt

while getopts p:n:s:l: flag
do
    case "${flag}" in
        p) pair_addr=${OPTARG};;
        n) num_processes=${OPTARG};;
        s) save=${OPTARG};; # cont_total.txt
        l) language=${OPTARG};;
    esac
done

echo "$language"

mkdir raw_ssl_data_sep16

for (( i=0; i<$num_processes; i++ ))
do
  python3 syn_data/edit_ops_ret.py -pair_addr "$pair_addr" -process_index $i -tot_p $num_processes \
  -language "$language" &
  sleep 1
done

wait
echo "all processed finished at this point!"

python3 preprocess/merge.py -dir raw_ssl_data_sep16 -prefix cont_ -save "$save"

rm -rf raw_ssl_data_sep16
