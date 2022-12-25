from comet import load_from_checkpoint
import click
import glob

"""CUDA_VISIBLE_DEVICES=0 python3 test_sescore.py -model_path sescore_ckpt/zh_en/checkpoint/sescore_english.ckpt -ref_path /opt/tiger/contra_score/wmt22-metrics-inputs-v6/metrics_inputs/txt/generaltest2022/references/generaltest2022.zh-en.ref.refA.en -out_path /opt/tiger/contra_score/wmt22-metrics-inputs-v6/metrics_inputs/txt/generaltest2022/system_outputs/zh-en/ -ref_type refA"""

@click.command()
@click.option('-model_path')
@click.option('-ref_path')
@click.option('-out_path')
@click.option('-ref_type')
def main(model_path, ref_path, out_path, ref_type):
    model = load_from_checkpoint(model_path)
    ref_lines = open(ref_path, 'r').readlines()
    ref_lines = [line[:-1] for line in ref_lines]

    for file_name in glob.glob(out_path+'/*'):
        out_lines = open(file_name, 'r').readlines()
        out_lines = [line[:-1] for line in out_lines]

        dataset = [{"src": ref, "mt": out} for ref, out in zip(ref_lines, out_lines)]
        seg_scores, _ = model.predict(dataset, batch_size=16, gpus=1)
        seg_scores = [str(score)+'\n' for score in seg_scores]

        save_path = file_name+f'.sescores_{ref_type}'
        savefile = open(save_path, 'w')
        savefile.writelines(seg_scores)

        print(f"File {save_path} is saved!")

if __name__ == "__main__":
    main()
