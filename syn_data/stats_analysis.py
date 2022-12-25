import click

"""Example code to run:
python3 syn_data/stats_analysis.py -lang_dir zh_en -out_addr syn_data_gen_zh_en_train_1660639001/cycle1_infill_mt_0.15_zh_en_total.txt"""

def empty_lines_stats(addr):
    count = 0
    inFile = open(addr, 'r')
    for line in inFile:
        if line=='\n':
            count+=1
    print(count)

def rep_dectec_rewrite(lang_dir, out_addr):
    src_file = open(f'news_prepare_mask_data_syn_gen/cycle1_{lang_dir}_news_train_src_mask.txt', 'r')
    ref_file = open(f'news_prepare_mask_data_syn_gen/cycle1_{lang_dir}_news_train_ref_mask.txt', 'r')
    out_file = open(out_addr, 'r')
    count, total, index_dict = 0, 0, {}
    for src_line, ref_line, out_line in zip(src_file, ref_file, out_file):
        ref_content = ref_line.split('\t')[-1]
        if ref_content == out_line:
            count+=1
        else:
            index = ref_line.split('\t')[0]
            if index not in index_dict:
                index_dict[index]=0
            index_dict[index]+=1
        total+=1

    index_count_1, index_count_2, index_count_3, index_count_4, index_count_5 = 0,0,0,0,0
    for index, index_count in index_dict.items():
        if index_count==1:
            index_count_1+=1
        elif index_count==2:
            index_count_2+=1
        elif index_count==3:
            index_count_3+=1
        elif index_count==4:
            index_count_4+=1
        else:
            assert(index_count==5)
            index_count_5+=1

    print("Number of lines which repeat: ", count)
    print("Number of lines which not repeat: ", total-count)
    print("Intances contain one error: ", index_count_1)
    print("Intances contain two errors: ", index_count_2)
    print("Intances contain three errors: ", index_count_3)
    print("Intances contain four error: ", index_count_4)
    print("Intances contain five error: ", index_count_5)
    total_instances = index_count_1+index_count_2+index_count_3+index_count_4+index_count_5
    print("Total number of instances: ", total_instances)

@click.command()
@click.option('-lang_dir')
@click.option('-out_addr')
def main(lang_dir, out_addr):
    rep_dectec_rewrite(lang_dir, out_addr)

if __name__ == "__main__":
    main()
