from datasets import load_dataset
import click
import os


@click.command()
@click.option('-lang')
def main(lang):
    load_dataset('mc4', lang)
    print("dataset is downloaded!")

if __name__ == "__main__":
    main()
