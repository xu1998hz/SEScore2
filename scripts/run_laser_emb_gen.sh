#!/usr/bin/env bash
#!/bin/bash

# -d wiki_raw_0_5010952.txt -l zh_en -t wiki

while getopts d:t:l: flag
do
    case "${flag}" in
        d) data_addr=${OPTARG};;
        l) lang_dir=${OPTARG};;
        t) data_type=${OPTARG};;
    esac
done

cd LASER/tasks/embed
chmod -R 755 LASER
./embed.sh "$data_addr" "outputs_$data_addr"
