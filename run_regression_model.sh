#!/usr/bin/env bash
#!/bin/bash

#/home/tiger/.mt-metrics-eval
# load in the shell script arguments, ./run_regression_model.sh -i english_True_severe_minor_sept27.txt -l zh_en -e 160 -g False -b 32 -n 1e-5 -m rembert -h 1152 -p epoch0_best_9280_zh_en_no_rembert.ckpt
while getopts i:l:e:g:b:n:m:h:p:o:t:a: flag
do
    case "${flag}" in
        i) data_file=${OPTARG};;
        l) lang_dir=${OPTARG};;
        e) eval_step=${OPTARG};;
        g) gradual_unfrozen=${OPTARG};;
        b) batch_size=${OPTARG};;
        n) lr=${OPTARG};;
        m) model_base=${OPTARG};;
        h) hidden_size=${OPTARG};;
        p) load_ckpt=${OPTARG};;
        o) num_epoch=${OPTARG};;
        t) ted_enable=${OPTARG};;
        a) alpha=${OPTARG};;
    esac
done

echo "Data File: $data_file"
echo "Lang Dir: $lang_dir"
echo "Eval Step: $eval_step"
echo "Whether to freeze xlm: $gradual_unfrozen"
echo "Batch Size: $batch_size"
echo "lr: $lr"
echo "Model base: $model_base"
echo "Hidden size: $hidden_size"
echo "Load ckpt: $load_ckpt"
echo "Num of epoches: $num_epoch"

pip3 install mt-metrics-eval
pip3 install .
cd ..

# torchrun only supported in torch 1.10.0 and later
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
echo "installed all dependencies for mt-metrics-eval"

mtme='python3 -m mt_metrics_eval.mtme'
$mtme --download  # Puts ~1G of data into $HOME/.mt-metrics-eval.
echo "installed data for evaluations (1G)"

cur_time=$(date +%s)
save_dir_name=regression_weights_"$lr"_"$lang_dir"_"$cur_time"_"$model_base"

echo "$save_dir_name"

export MASTER_ADDR="localhost"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 --master_port=8192 train/regression.py \
-lang_dir "$lang_dir" -save_dir_name "$save_dir_name" -enable_loss_eval True -data_file "$data_file" -eval_step "$eval_step" \
-gradual_unfrozen "$gradual_unfrozen" -batch_size "$batch_size" -lr "$lr" -model_base "$model_base" -hidden_size "$hidden_size" \
-model_addr "$load_ckpt" -load_file_enable "$load_file_enable" -num_epoch "$num_epoch" -test_ted_enable "$ted_enable" -alpha "$alpha"
