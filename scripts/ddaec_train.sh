#export LD_LIBRARY_PATH='/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64:/usr/local/cudnn-8.0-linux-x64-#v6.0/lib64'
export CUDA_VISIBLE_DEVICES='0'
model_name=DDAEC
echo $model_name
python -u train_ddaec.py \
--train_list=../filelists/train_list.txt \
--evaluate_file=../data/mixture/test/test_factory_snr-5_seen.samp \
--display_eval_steps=8000 \
--eval_plot_num=3 \
--model_name=$model_name \
--width=64 \
--batch_size=4 \
#--resume_model=../models/${model_name}/${model_name}_latest.model
