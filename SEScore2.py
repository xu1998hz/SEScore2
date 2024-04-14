from util.regression_xlm_roberta import *
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from scipy import stats
from scipy.interpolate import interp1d
import numpy as np
import json

def eQM_porcentual_delta_interpolate(model_present, model_future, interpolation_function):
    """
    Smoothly map the model_present distribution to the ref_dataset distribution
    using quantile mapping and interpolation.

    returns: downscaled model_present
    """
    model_present_corrected = np.zeros(model_future.size)

    # Map each value in model_present to the corresponding quantile in ref_dataset and interpolate
    for ival, model_value in enumerate(model_future):
        #model_percentile = stats.percentileofscore(model_future, model_value)
        #print(f"Before: {model_percentile}")
        model_percentile = stats.percentileofscore(model_present, model_value)
        #print(f"After: {model_percentile}")

        model_present_corrected[ival] = interpolation_function(model_percentile)

    return model_present_corrected

def load_from_json(filename):
    # Load x and y data from a JSON file
    with open(filename, 'r') as file:
        data = json.load(file)
    y_min = min(data['y_data'])
    data['y_data'].append(y_min-0.1)
    data['y_data'].remove(y_min)
    return np.array(data['x_data']), np.array(data['y_data'])


class SEScore2:

    def __init__(self, lang, mode, calibration = False):
        # model is only pretrained on synthetic data
        if mode == "pretrained":
            # load in the weights of SEScore2
            exp_config.device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            if lang == 'en':
                # load in xlm version
                cur_addr = hf_hub_download(repo_id="xu1998hz/sescore2_en_pretrained", filename="sescore2_en.ckpt")
                self.tokenizer = AutoTokenizer.from_pretrained(f"xlm-roberta-large")
                self.calibration = calibration
                if self.calibration:
                    self.model_loaded, self.mqm_loaded = load_from_json('calibration/sescore2_en.json')
                    unique_ref_values = np.unique(self.mqm_loaded)
                    ref_quantiles = np.array([stats.percentileofscore(self.mqm_loaded, v) for v in unique_ref_values])
                    # Create an interpolation function for the ref_dataset quantiles
                    self.interpolation_function = interp1d(ref_quantiles, unique_ref_values, bounds_error=False, fill_value="extrapolate")
            elif lang == 'de':
                # load in rembert version
                self.tokenizer = AutoTokenizer.from_pretrained(f"google/rembert")
                cur_addr = hf_hub_download(repo_id="xu1998hz/sescore2_de_pretrained", filename="sescore2_de.ckpt")
            elif lang == 'ja':
                # load in rembert version
                cur_addr = hf_hub_download(repo_id="xu1998hz/sescore2_ja_pretrained", filename="sescore2_ja.ckpt")
                self.tokenizer = AutoTokenizer.from_pretrained(f"google/rembert")
            elif lang == 'es':
                # load in rembert version
                cur_addr = hf_hub_download(repo_id="xu1998hz/sescore2_es_pretrained", filename="sescore2_es.ckpt")
                self.tokenizer = AutoTokenizer.from_pretrained(f"google/rembert")
            elif lang == 'zh':
                # load in rembert version
                cur_addr = hf_hub_download(repo_id="xu1998hz/sescore2_zh_pretrained", filename="sescore2_zh.ckpt")
                self.tokenizer = AutoTokenizer.from_pretrained(f"google/rembert")
            else:
                print("We currently only support three languages: en, de, ja!")
                exit(1)
        else:
            print(f"We don't support this mode {mode}")
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
            if self.calibration:
                score = eQM_porcentual_delta_interpolate(self.model_loaded,np.array(score),self.interpolation_function)
                score = score.tolist()
            scores_ls.extend(score)
        return scores_ls

# test the results
def main():
    refs = ["SEScore is a simple but effective next generation text generation evaluation metric", "SEScore it really works"]
    outs = ["SEScore is a simple effective text evaluation metric for next generation", "SEScore is not working"]

    scorer = SEScore2(lang='en', mode="pretrained")
    scores_ls = scorer.score(refs, outs, 1)
    print(scores_ls)

if __name__ == "__main__":
    main()
