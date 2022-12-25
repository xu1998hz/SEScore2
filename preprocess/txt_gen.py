import glob
import click
import json

"""This file is used to convert all the post processed swappy files into txt and it produces Zh-En: 236,229,947 pair, En-De: 145,126,334 pairs"""
"""code to run this file: python3 txt_gen.py -dir_addr 6434175 -src_lang zh -tar_lang zh"""

@click.command()
@click.option('-dir_addr')
@click.option('-src_lang')
@click.option('-tar_lang')
@click.option('-count')
def main(dir_addr, src_lang, tar_lang, count):
    if count == '2M':
        slice_end = 2000000
    elif count == '4M':
        slice_end = 4000000
    else:
        print("Current count is not supported!")

    raw_file_ls = sorted(glob.glob(f'{dir_addr}/*'), reverse=True)
    global_count = 0
    saveSrcFile = open(f"{src_lang}_{count}.txt", 'w')
    saveRefFile = open(f"{tar_lang}_{count}.txt", 'w')

    for f_name in raw_file_ls:
        with open(f_name, 'r') as json_file:
            print(f_name)
            json_list = list(json_file)
            if global_count >= slice_end:
                break
            else:
                for json_str in json_list:
                    if json_str.strip():
                        result = json.loads(str(json_str))
                        # sanity check
                        if result['src_lang'] == src_lang and result['trg_lang'] == tar_lang:
                            saveSrcFile.write(result['src_text']+'\n')
                            saveRefFile.write(result['trg_text']+'\n')
                            global_count+=1

    print(f"All data is saved at {src_lang}_{count}.txt and {tar_lang}_{count}.txt!")

if __name__ == "__main__":
    main()
