export LD_LIBRARY_PATH='/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64:/usr/local/cudnn-8.0-linux-x64-v5.1/lib64'
export CUDA_VISIBLE_DEVICES='0'
model_name=DDAEC
type=latest
model_file=../models/${model_name}/${model_name}_latest.model
echo $model_name
echo $model_file
python -u test_ddaec.py \
--test_list=../filelists/test_list.txt \
--model_file=$model_file \
--model_name=$model_name
python -u assess.py \
--assess_list=../filelists/assess_list.txt \
--model_name=$model_name
python -u assess_pesq.py \
--assess_list=../filelists/assess_list.txt \
--model_name=$model_name
