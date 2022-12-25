import re
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import nltk
import click
nltk.download('punkt')

lang_mapping = {
    'de': 'german',
    'en': 'english'
}

@click.command()
@click.option('-lang')
def main(lang):
    dataset = load_dataset("wikipedia", f"20220301.{lang}")
    saveFile = open(f'wiki_raw.{lang}', 'w')
    for ele in dataset['train']:
        article = ele['text']
        sentences_ls = re.split('\n\n|\n|\t\t|\t', article)
        for sentence in sentences_ls:
            if len(sentence.split())>10:
                for sent in sent_tokenize(sentence, language=lang_mapping[lang]):
                    saveFile.write(sent+'\n')
    print("All texts are processed!")

if __name__ == "__main__":
    main()
