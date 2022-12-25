#!/usr/bin/env bash
#!/bin/bash

# load in the shell script arguments: ./raw_data_process.sh -l en
while getopts l: flag
do
    case "${flag}" in
        l) lang=${OPTARG};;
    esac
done

# install some dependencies
python3 preprocess/preprocess_pretrain_data.py -lang "$lang"
