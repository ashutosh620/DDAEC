model_name=$1
model_name+=_$2
model_name+=_$3
python assess_pesq.py \
--assess_list=../../filelists/assess_list.txt \
--model_name=$model_name
