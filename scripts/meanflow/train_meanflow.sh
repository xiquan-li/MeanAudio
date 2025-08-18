export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES:-""} | tr ',' '\n' | wc -l)
btz=256
num_iterations=400_000
model=meanaudio_mf # meanaudio_mf, fluxaudio_fm
lr=1e-4

exp_id=AWV_${btz}_numgpus${NUM_GPUS}_niter${num_iterations}_T5_CLAP_${model}_fixddpbug_unfixtrajbug_scratch
text_encoder_name=t5_clap
weights=./weights/fluxaudio_fm.pth   # pre-trained weigths to be loaded for mix-field finetuning

text_c_dim=512   # 1024 + 512


OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --standalone --nproc_per_node=$NUM_GPUS \
    train.py \
    --config-name train_config_16khz.yaml \
    exp_id=$exp_id \
    compile=False \
    model=$model \
    batch_size=${btz} \
    eval_batch_size=16 \
    num_iterations=$num_iterations \
    text_encoder_name=$text_encoder_name \
    data_dim.text_c_dim=$text_c_dim \
    pin_memory=False \
    num_workers=10 \
    ac_oversample_rate=5 \
    learning_rate=$lr \
    val_interval=20_000 \
    eval_interval=20_000 \
    ++do_eval=False \
    ++use_rope=True \
    ++use_wandb=True \
    ++debug=False