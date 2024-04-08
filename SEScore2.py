from train.regression import *
from huggingface_hub import hf_hub_download

class SEScore2:
    
    def __init__(self, lang, cur_addr):
        # load in the weights of SEScore2
        exp_config.device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if lang == 'en':
            # load in xlm version
            self.tokenizer = AutoTokenizer.from_pretrained(f"xlm-roberta-large")
        elif lang == 'de':
            # load in rembert version
            self.tokenizer = AutoTokenizer.from_pretrained(f"google/rembert")
        elif lang == 'ja':
            # load in rembert version
            self.tokenizer = AutoTokenizer.from_pretrained(f"google/rembert")
        else:
            print("We currently only support three languages: en, de, ja!")
            exit(1)
        
        self.model = torch.load(cur_addr).to(exp_config.device_id)
        self.model.eval()

    def score(self, refs, outs, batch_size):
        scores_ls = []
        cur_data_dict = {'pivot': refs, 'mt': outs}
        cur_data_loader = preprocess_data(cur_data_dict, self.tokenizer, exp_config.max_length, batch_size, shuffle=False, sampler=False, mode='test')
        for batch in cur_data_loader:
            # generate a batch of ref, mt embeddings
            score = self.model(batch, 'last_layer').squeeze(1).tolist()
            scores_ls.extend(score)
        return scores_ls

# test the results 
def main():
    refs = ["SEScore is a simple but effective next generation text generation evaluation metric", "SEScore it really works"]
    outs = ["SEScore is a simple effective text evaluation metric for next generation", "SEScore is not working"]

    file_addr = hf_hub_download(repo_id="xu1998hz/sescore2_en_pretrained", filename="sescore2_en.ckpt")
    scorer = SEScore2(lang='en', cur_addr=file_addr)
    scores_ls = scorer.score(refs, outs, 1)
    print(scores_ls)

if __name__ == "__main__":
    main()