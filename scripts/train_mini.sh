###
# Mini training script, to check if everything runs successfully 
###

export CUDA_VISIBLE_DEVICES=4,5,6,7

NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES:-""} | tr ',' '\n' | wc -l)
btz=12

text_encoder_name=t5_clap
text_c_dim=512   # 1024 + 512

num_iterations=200
model=meanaudio_mf # meanaudio_mf, fluxaudio_fm

exp_id=debug

# Loading from pre-trained weights
pretrained_weights=./weights/flux_tta_mf.pth

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
    val_interval=100 \
    eval_interval=100 \
    save_eval_interval=100 \
    save_weights_interval=100 \
    save_checkpoint_interval=100 \
    mini_train=True \
    ema.checkpoint_every=50 \
    weights=$pretrained_weights \
    ++use_rope=True \
    ++use_wandb=False \
    ++debug=False