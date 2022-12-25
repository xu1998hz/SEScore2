from nltk.tokenize import word_tokenize

# setting threshold to be 0.1
file_severe = open('severity_measure/probs_zh_en_1.outputs')
src_lines = open('mt_severe_data/zh_en/train_400k.zh', 'r').readlines()
ref_lines = open('mt_severe_data/zh_en/train_400k.en', 'r').readlines()
src_lines = [line[:-1] for line in src_lines]
ref_lines = [line[:-1] for line in ref_lines]

for line in file_severe:
    sen_ls= line[:-1].split('\t')
    sen_index, op_type, span_start_end = int(sen_ls[0]), sen_ls[1], sen_ls[2]
    rep_cont, score = sen_ls[3], sen_ls[4]
    print("src: ")
    print(src_lines[sen_index])
    print()
    print("ref: ")
    print(ref_lines[sen_index])
    print()
    cur_line = ref_lines[sen_index]
    if op_type == '1':
        print("Inserted part: ")
        start_index = int(span_start_end)
        print(word_tokenize(ref_lines[sen_index])[:start_index])
        print()
    elif op_type == '2':
        print("Replaced part: ")
        start_index, end_index = int(span_start_end.split('_')[0]), int(span_start_end.split('_')[1])
        print(word_tokenize(ref_lines[sen_index])[start_index:end_index])
        print()
    else:
        exit(1)
    print("new content: ")
    print(rep_cont)
    if op_type=='2':
        new_len, old_len = len(word_tokenize(rep_cont)), end_index-start_index
        print("length rescale: ", min(new_len, old_len)/max(new_len, old_len))
        print("raw score: ", score)
        print("rescaled score: ", min(new_len, old_len)/max(new_len, old_len)*float(score))
    else:
        print(score)

    print()
    print('------------------------------------------------------------------')
