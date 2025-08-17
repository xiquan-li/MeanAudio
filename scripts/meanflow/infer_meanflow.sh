export CUDA_VISIBLE_DEVICES=0

output_path=./output/meanaudio_s_full/output_1nfe

prompt="Generate an audio clip that starts with people cheering, then people crying, and ends with gunshots."
model=meanaudio_mf
# ckpt_path=weights/meanaudio_mf.pth
ckpt_path=/apdcephfs_gy4/share_302507476/xiquanli/exps/MeanAudio/AWM_256_numgpus8_niter200_000_T5_CLAP_meanaudio_mf_lr1e-4_fixddpbug_scratch/AWM_256_numgpus8_niter200_000_T5_CLAP_meanaudio_mf_lr1e-4_fixddpbug_scratch_ema_final.pth
num_steps=1

python infer.py \
    --variant "meanaudio_mf" \
    --prompt "$prompt" \
    --model_path "$ckpt_path" \
    --output $output_path \
    --cfg_strength 0 \
    --encoder_name t5_clap \
    --duration 10 \
    --use_rope \
    --text_c_dim 512 \
    --num_steps $num_steps \
    --use_meanflow \
    --seed 42
