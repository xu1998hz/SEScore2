import numpy as np
from tqdm import tqdm
from train.regression import *
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import numpy as np
from scipy import interpolate

class sescorex:
    def __init__(self, version='seg', rescale=True):
        # load in the weights of SEScore2
        exp_config.device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(f"xlm-roberta-large")
        if version == 'seg':
            cur_addr = hf_hub_download(repo_id="xu1998hz/sescore2_en_pretrained", filename="sescorex_seg.ckpt")
        elif version == 'sys':
            cur_addr = hf_hub_download(repo_id="xu1998hz/sescore2_en_pretrained", filename="sescorex_sys.ckpt")
        elif version == "pretrained":
            cur_addr = hf_hub_download(repo_id="xu1998hz/sescore2_en_pretrained", filename="sescore2_en_original.ckpt")
        else:
            print("We currently support three modes: seg, sys and pretrained")
            exit(1)

        self.model = torch.load(cur_addr).to(exp_config.device_id)
        self.model.eval()
        self.rescale=rescale
        if rescale:
            self.f=self.rescale_f(version)

    def score(self, refs, outs, batch_size):
        scores_ls = []
        cur_data_dict = {'pivot': refs, 'mt': outs}
        cur_data_loader = preprocess_data(cur_data_dict, self.tokenizer, exp_config.max_length, batch_size, shuffle=False, sampler=False, mode='test')

        for batch in tqdm(cur_data_loader, desc='Processing batches'):
            # generate a batch of ref, mt embeddings
            score_batch = self.model(batch, 'last_layer').squeeze(1).tolist()
            if self.rescale:
                score_batch=[score for score in self.f(score_batch)]
            scores_ls.extend(score_batch)
        return scores_ls

    def rescale_f(self, version):
        ref_score_ls = []
        out_score_ls = []
        if version == 'pretrained':
            lang_ls = ['en']
        else:
            lang_ls = ['en', 'es', 'ru', 'zh', 'es']
            
        for lang in lang_ls:
            gt_lines = open(f'rescale_data/{version}/{lang}.mqm', 'r').readlines()
            sescorex_lines = open(f'rescale_data/{version}/{lang}.sescorex', 'r').readlines()
            for gt_l, sescorex_l in zip(gt_lines, sescorex_lines):
                gt_score = gt_l.split('\t')[-1].split('\n')[0]
                sescorex = sescorex_l.split('\t')[-1].split('\n')[0]
                if gt_score != 'None':
                    ref_score_ls+=[float(gt_score)]
                    out_score_ls+=[float(sescorex)]
        
        obs_data = np.array(ref_score_ls)
        model_data = np.array(out_score_ls)

        # Sorted observation data
        sorted_observed = np.sort(obs_data)

        # Sorted model data
        sorted_model = np.sort(model_data)

        # Interpolation function
        f = interpolate.interp1d(
            sorted_model, sorted_observed, bounds_error=False, fill_value="extrapolate"
        )
        return f

# test the results
def main():
    scorer = sescorex(version='seg', rescale=True)
    refs = ["SEScore is a simple but effective next generation text generation evaluation metric", "SEScore really works"]
    outs = ["SEScore is a simple effective text evaluation metric for next generation", "SEScore is not working"]
    scores_ls = scorer.score(refs, outs, 32)
    print(scores_ls)
        
if __name__ == "__main__":
    main()
