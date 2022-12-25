import click
import string
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def detect_punkt(words_ls):
    for word in words_ls:
        if word not in string.punctuation:
            return False
    return True

@click.command()
@click.option('-ref_addr')
@click.option('-file_name')
@click.option('-save_folder')
@click.option('-language', help="english or german")
@click.option('-thres', type=float)
def main(ref_addr, file_name, save_folder, language, thres):
    ref_lines = open(ref_addr, 'r').readlines()
    ref_lines = [ref[:-1] for ref in ref_lines]
    lines = open(file_name, 'r').readlines()
    lines = [line[:-1] for line in lines]
    saveFile = open(f'{save_folder}/score_'+file_name.split('/')[-1], 'w')

    punkt_minor, minor, severe = 0, 0, 0
    for line in lines:
        ops_ls = line[:-1].split('\t')
        sen_index, op_type = int(ops_ls[0]), int(ops_ls[1])
        if op_type == 1:
            if detect_punkt(word_tokenize(ops_ls[3], language)):
                score = -0.1
                punkt_minor+=1
            else:
                if float(line.split('\t')[-1]) >= thres:
                    score = -1
                    minor+=1
                else:
                    score = -5
                    severe+=1
        elif op_type == 2:
            start_index, end_index = int(ops_ls[2].split('_')[0]), int(ops_ls[2].split('_')[1])
            if detect_punkt(word_tokenize(ref_lines[sen_index])[start_index:end_index]) and detect_punkt(word_tokenize(ops_ls[3])):
                score = -0.1
                punkt_minor+=1
            else:
                if float(line.split('\t')[-1]) >= thres:
                    score = -1
                    minor+=1
                else:
                    score = -5
                    severe+=1
        else:
            print("wrong type!")
            exit(1)

        prefix = '\t'.join(line.split('\t')[:-1])
        new_line = prefix+'\t'+str(score)+'\n'
        saveFile.write(new_line)

    print(punkt_minor)
    print(minor)
    print(severe)

if __name__ == "__main__":
    main()
