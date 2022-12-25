import pandas as pd
import click

data_2017 = pd.read_csv('da_data/2017-da.csv')
data_2018 = pd.read_csv('da_data/2018-da.csv')
data_2019 = pd.read_csv('da_data/2019-da.csv')
data = pd.concat([data_2017, data_2018, data_2019], ignore_index=True)

@click.command()
@click.option('-lp', help="choose either en-de or zh-en")
def main(lp):
    extracted_data = data.loc[data['lp'] == lp].dropna()
    save_file = open(f'{lp}_da.txt', 'w')
    for ref, mt, score in zip(extracted_data['ref'], extracted_data['mt'], extracted_data['raw_score']):
        normalized_score = (score-100)/4
        # print(ref)
        # print(mt)
        # print(normalized_score)
        # print()
        save_file.write(ref+'\t'+mt+'\t'+str(normalized_score)+'\n')
    print(f'{lp}_da.txt is saved!')

if __name__ == "__main__":
    main()
