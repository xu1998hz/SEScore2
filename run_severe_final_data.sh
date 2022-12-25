langauge="japanese"
severe_folder="severity_measure_en_ja"
ssl_folder="oct2_en_ja_ssl_data"
idf_weights_file="$langauge"_stopwords.txt
severe_enable="True"
del_enable="True"
insert_replace_enable="True"

if [ $langauge == "german" ]
then
    ref_file="train_en_de.de"
elif [ $langauge == "english" ]
then
    ref_file="train.en"
elif [ $langauge == "japanese" ]
then
    ref_file="train_en_ja_5M.ja"
else
    echo "We only support english and german"
fi

echo "Downloaded all the data!"

for (( i=1; i<6; i++ ))
do
  mkdir "$severe_folder"/"step$i"
  # organize score files
  mv "$severe_folder"/scores_part_*_"$langauge"_"$i"_mined_cont_rand_None.txt "$severe_folder"/"step$i"
  python3 preprocess/merge.py -dir "$severe_folder"/"step$i" -prefix scores_part_ \
    -save "$severe_folder"/scores_"$langauge"_"$i"_mined_cont_rand_None.txt
  rm -rf "$severe_folder"/"step$i"
  # # correct severe minor scores in insert/replace
  # python3 syn_data/fix_severe_score.py -ref_addr "$ssl_folder"/"$ref_file" -thres "$thres" \
  # -file_name "$severe_folder"/probs_"$langauge"_"$i"_mined_cont.txt -save_folder "$severe_folder" -language "$langauge"
  # echo "corrected insert/replace in step$i"
  # correct severe minor scores in delete
  if [ $langauge == "german" ]
  then
    python3 syn_data/del_de_severe_measure.py -ref_addr "$ssl_folder"/"$ref_file" -opt_addr "$ssl_folder"/del_locs/del_sen_op_loc_"$i".txt \
    -idf_addr "idf_weights/$idf_weights_file" -save_folder "$severe_folder"
  elif [ $langauge == "english" ]
  then
    python3 syn_data/del_severe_measure.py -ref_addr "$ssl_folder"/"$ref_file" -opt_addr "$ssl_folder"/del_locs/del_sen_op_loc_"$i".txt \
    -idf_addr "idf_weights/$idf_weights_file" -save_folder "$severe_folder"
  elif [ $langauge == "japanese" ]
  then
    python3 syn_data/del_severe_measure.py -ref_addr "$ssl_folder"/"$ref_file" -opt_addr "$ssl_folder"/del_sen_op_loc_"$i".txt \
    -idf_addr "idf_weights/$idf_weights_file" -save_folder "$severe_folder" -lang "$langauge"
  else
    echo "We only support english and german"
  fi
  echo "corrected delete in step$i"
done

python3 syn_data/final_data_construct.py -lang "$langauge" -severe_enable "$severe_enable" -folder_name "$severe_folder" \
  -ref_addr "$ssl_folder"/"$ref_file" -del_enable "$del_enable" -insert_replace_enable "$insert_replace_enable"

echo "Entire process is done"
