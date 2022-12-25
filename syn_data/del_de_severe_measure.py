from nltk.tokenize import word_tokenize
import click
import string
import nltk
nltk.download('punkt')

"""python3 syn_data/del_severe_measure.py -src_addr mt_severe_data/zh_en/train_400k.zh \
-ref_addr mt_severe_data/zh_en/train_400k.en -opt_addr mt_severe_data/zh_en/del_sen_op_loc_1.txt \
-idf_addr idf_weights/wiki_tfidf_terms.csv -save_folder mt_severe_data/zh_en"""

def detect_punkt(words_ls):
    for word in words_ls:
        if word not in string.punctuation:
            return False
    return True

@click.command()
@click.option('-ref_addr')
@click.option('-opt_addr')
@click.option('-idf_addr')
@click.option('-save_folder')
def main(ref_addr, opt_addr, idf_addr, save_folder):
    tgt_texts = open(ref_addr, 'r').readlines()
    ops_loc_texts = open(opt_addr, 'r').readlines()
    df = open(idf_addr, 'r').readlines()
    df = [ele[:-1] for ele in df]
    idf_dict = {}

    for token in df:
        idf_dict[token]=0

    print("load in weights")

    saveFile = open(f'{save_folder}/score_'+opt_addr.split('/')[-1], 'w')
    punkt_minor, minor, severe = 0, 0, 0
    for ops_loc in ops_loc_texts:
        ops_ls = ops_loc[:-1].split('\t')
        sen_index, start_end_index = int(ops_ls[0]), ops_ls[2].split('_')
        tgt_words_ls = word_tokenize(tgt_texts[sen_index][:-1])
        start_index, end_index = int(start_end_index[0]), int(start_end_index[1])

        weight_ls=[]
        for word in tgt_words_ls[start_index:end_index]:
            if word.lower() in idf_dict:
                weight_ls.append(0)
            else:
                if word not in string.punctuation:
                    # print("missing word:")
                    # print(word)
                    weight_ls.append(2)


        if detect_punkt(tgt_words_ls[start_index:end_index]):
            new_line = ops_loc[:-1]+'\t'+''+'\t'+str(-0.1)+'\n'
            punkt_minor+=1
        else:
            if max(weight_ls) < 1:
                new_line = ops_loc[:-1]+'\t'+''+'\t'+str(-1)+'\n'
                minor+=1
            else:
                new_line = ops_loc[:-1]+'\t'+''+'\t'+str(-5)+'\n'
                severe+=1
        saveFile.write(new_line)
    # print(punkt_minor)
    # print(minor)
    # print(severe)

if __name__ == "__main__":
    main()
