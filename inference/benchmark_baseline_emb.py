import click
from mt_metrics_eval import data

human_mapping_dict = {
    "wmt21.news": {
        'en-de': ['refA', 'refD'],
        'en-ru': ['refB'],
        'zh-en': ['refA']
    },
    "wmt20": {
        'en-de': 'refb',
        'zh-en': 'refb'
    }
}

def laser_baseline():
    pass

@click.command()
@click.option('-baseline_type', help="laser1, laser2, sen_tr")
def main(baseline_type):
    evs = data.EvalSet(wmt, lang)
    srcs, mt_outs_dict, refs = evs.src, evs.sys_outputs, evs.std_ref
    if baseline_type == 'laser1':
        pass
    elif baseline_type == 'laser2':
        pass
    elif baseline_type == 'sen_tr':
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        model.encode([])
    else:
        print("Code is not currently supporting more baseline types")

if __name__ == "__main__":
    main()
