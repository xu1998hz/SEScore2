#!/usr/bin/env bash
#!/bin/bash

# ./run_embs_gen.sh -f fixed_wiki_raw_2M.en -s 0 -e 200000 -l zh-en
while getopts f:s:e:l:d: flag
do
    case "${flag}" in
        f) file_addr=${OPTARG};;
        s) start_index=${OPTARG};;
        e) end_index=${OPTARG};;
        l) lang_dir=${OPTARG};;
    esac
done

echo "load file addr: $file_addr"
echo "start index: $start_index"
echo "end index: $end_index"
echo "lang dir: $lang_dir"

pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

save_folder_name="$lang_dir"_"$start_index"_"$end_index"_embds
# add tok embs
python3 syn_data/phrase_mining.py -file_addr "$file_addr" -batch_size 256 -start_index "$start_index" \
 -end_index "$end_index" -lang_dir "$lang_dir" -folder_name "$save_folder_name"

