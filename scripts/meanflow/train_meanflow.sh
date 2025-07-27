export CUDA_VISIBLE_DEVICES=4,5,6,7

NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES:-""} | tr ',' '\n' | wc -l)
btz=72
num_iterations=200_000
exp_id=AC_${btz}_numgpus${NUM_GPUS}_niter${num_iterations}_T5_CLAP_meanflow_improved_changecfg_seed42_flowratio0.75

text_encoder_name=t5_clap
weights=./weights/fluxaudio_fm.pth   # pre-trained weigths to be loaded for mix-field finetuning

text_c_dim=512   # 1024 + 512
model=meanaudio_mf # meanaudio_mf, fluxaudio_fm


OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --standalone --nproc_per_node=$NUM_GPUS \
    train.py \
    --config-name train_config.yaml \
    exp_id=$exp_id \
    compile=False \
    model=$model \
    batch_size=${btz} \
    eval_batch_size=32 \
    num_iterations=$num_iterations \
    text_encoder_name=$text_encoder_name \
    data_dim.text_c_dim=$text_c_dim \
    pin_memory=False \
    num_workers=10 \
    ac_oversample_rate=5 \
    weights=$weights \
    ++use_rope=True \
    ++use_wandb=True \
    ++debug=False