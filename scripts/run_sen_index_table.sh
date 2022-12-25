#!/usr/bin/env bash
#!/bin/bash

# ./run_sen_index_table.sh -q news_outputs_zh_en_news_0_5027365.txt -f news_outputs_zh_en_news_0_5027365.txt -s news_outputs_zh_en_news_5027365_10054730.txt \
# -t news_outputs_zh_en_news_10054730_15082095.txt -o news_outputs_zh_en_news_15082095_20109460.txt -u part1 -l zh_en

while getopts q:f:s:t:o:u:l: flag
do
    case "${flag}" in
        q) query_file=${OPTARG};;
        f) value_file_1=${OPTARG};;
        s) value_file_2=${OPTARG};;
        t) value_file_3=${OPTARG};;
        o) value_file_4=${OPTARG};;
        u) suffix=${OPTARG};;
        l) lang_dir=${OPTARG};;
    esac
done

echo "query: $query_file"
echo "val1: $value_file_1"
echo "val2: $value_file_2"
echo "val3: $value_file_3"
echo "val4: $value_file_4"
echo "suffix: $suffix"

mkdir data
mkdir "$lang_dir"

python3 syn_data/knn_search.py -query_file data/"$query_file" -value_file_1 data/"$value_file_1" -value_file_2 "$value_file_2" \
-value_file_3 "$value_file_3" -value_file_4 "$value_file_4" -suffix "$suffix"

mv *.txt "$lang_dir"
