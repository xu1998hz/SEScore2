"""This function is used to control the length of each span in the masked target sentences, currently mu is set to be 3.5"""
def poisson_dist_span(mask_len, mu=3.5):
    span_ls = []
    while mask_len > 0:
        random_span = poisson.rvs(mu, size=1, loc=0)
        if random_span > 0 and random_span < mask_len:
            r = random_span
        else:
            r = mask_len
        mask_len = mask_len - r
        span_ls.append(r.item())

    return span_ls

"""This function is used to construct masked data for target language sentences before they concatenated along with source sentences.
Mask ratio can be set to control the ratio of the mask tokens in the sentences (mask_ratio = 0.70). Return the concatenated input"""
def construct_mask_data(src_data, ref_data, mask_token, sep_token, mask_ratio=0.70):
    src_data, ref_data = [data[:-1] for data in src_data], [data[:-1] for data in ref_data]
    mask_inp_data_ls, mask_ref_data_ls = [], []
    for src_sen, ref_sen in zip(src_data, ref_data):
        ref_words_ls = word_tokenize(ref_sen)
        mask_len = int(len(ref_words_ls) * mask_ratio)
        span_ls = poisson_dist_span(mask_len, mu=3.5)
        pre_index_ls, start_index_ls, end_index_ls = [], [], []
        mask_inp_data = []
        for index, span_len in enumerate(span_ls):
            if len(end_index_ls) > 0:
                start_index = randrange(end_index_ls[-1], len(ref_words_ls)-sum(span_ls[index:])+1)
            else:
                # only set the start index lower bound at step 0, set it to be 0
                start_index = randrange(0, len(ref_words_ls)-sum(span_ls[index:])+1)
            end_index = start_index + span_len
            if index == 0:
                # only set the start index lower bound at step 0, set it to be 0
                pre_index_ls.append((0, start_index))
            else:
                pre_index_ls.append((end_index_ls[-1], start_index))
            start_index_ls.append(start_index)
            end_index_ls.append(end_index)
        for start, end in pre_index_ls:
            mask_inp_data += ref_words_ls[start:end]+[mask_token]
        # store the ground truth masked data and masked input
        for start, end in zip(start_index_ls, end_index_ls):
            mask_ref_data_ls += ref_words_ls[start:end] + ['<SEP>']
        mask_inp_data_ls.append(src_sen +f' {sep_token} '+ TreebankWordDetokenizer().detokenize(mask_inp_data))
    return mask_inp_data_ls, mask_ref_data_ls
