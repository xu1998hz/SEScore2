# set up bleurt
pip3 install --upgrade pip  # ensures that pip is current
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip3 install .

wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip

cd ..

# setup bertscore
pip3 install bert_score
# setup comet
pip3 install unbabel-comet
# all ter, bleu, chrf
pip3 install evaluate
