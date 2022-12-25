from datasets import inspect_dataset, load_dataset_builder
import datasets
import click

@click.command()
@click.option('-src_lang', type=str, help="de or zh")
@click.option('-tar_lang', type=str, help="en")
def main(src_lang, tar_lang):
    # wmt19
    inspect_dataset("wmt19", "wmt19/wmt19.py")
    if src_lang == 'de':
        builder = load_dataset_builder(
            "wmt_utils.py",
            language_pair=(src_lang, tar_lang),
            subsets={
                datasets.Split.TRAIN: ["newscommentary_v14", "newsdev2014", "newsdev2015", \
                "newsdiscussdev2015", "newsdev2016", "newsdev2017", "newsdev2018", "newsdev2019", \
                "newsdiscusstest2015", "newssyscomb2009", "newstest2008", "newstest2009", \
                "newstest2010", "newstest2011", "newstest2012", "newstest2013", "newstest2014",\
                "newstest2015", "newstest2016", "newstest2017", "newstest2018"],
            },
        )
        tar = 'de'
    elif src_lang == 'zh':
        builder = load_dataset_builder(
            "wmt_utils.py",
            language_pair=(src_lang, tar_lang),
            subsets={
                datasets.Split.TRAIN: ["newscommentary_v14", "newsdev2017", "newstest2017", "newstest2018"],
            },
        )
        tar = 'en'
    else:
        print("Current language is not supported!")
    # Standard version
    builder.download_and_prepare()
    ds = builder.as_dataset()

    save_file = open(f"wmt_news/wmt_news_{src_lang}_{tar_lang}.txt", 'w')
    for line in ds['train']:
        save_file.write(line['translation'][tar]+'\n')
    print("Lines are saved!")

if __name__ == "__main__":
    main()
