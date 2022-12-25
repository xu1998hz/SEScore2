cp inference/data_to_text.py BARTScore/D2T
mv BARTScore/D2T/data_to_text.py BARTScore/D2T/score.py
cp inference/d2t_results.py BARTScore/

cp -r train BARTScore/D2T
cp train/regression.py BARTScore/D2T

mv xlm-roberta-large-tok BARTScore/D2T
mv epoch0_best_zhen_xlm_all.ckpt BARTScore/D2T

mv bleurt BARTScore/D2T

cd BARTScore/D2T

echo "set up is done!"
# python3 score.py --file BAGEL/data.pkl --device cuda:0 --output BAGEL/bleurt.pkl --bleurt
