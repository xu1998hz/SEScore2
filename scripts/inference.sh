#!/usr/bin/env bash
#!/bin/bash
while getopts l: flag
do
    case "${flag}" in
        l) lang_dir=${OPTARG};;
    esac
done

pip3 install mt-metrics-eval
pip3 install .
cd ..

# torchrun only supported in torch 1.10.0 and later
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
echo "installed all dependencies for mt-metrics-eval"

mtme='python3 -m mt_metrics_eval.mtme'
$mtme --download  # Puts ~1G of data into $HOME/.mt-metrics-eval.
echo "installed data for evaluations (1G)"

count=0;

for DIR in new_margin_loss_weights/*; do
  CUDA_VISIBLE_DEVICES="$count" python3 inference/inference.py -lang "$lang_dir" -wmt wmt21.news -model_addr "$DIR" -emb_type last_layer -batch_size 250 -score L2 &
  sleep 1
  count=$(($count+1))
done

echo "All inference codes are running!"
