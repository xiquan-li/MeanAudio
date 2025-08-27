export CUDA_VISIBLE_DEVICES=0

output_path=./output/meanaudio_m_full/output_1nfe

cd /apdcephfs_gy4/share_302507476/xiquanli/TTA/MeanAudio/
prompt="Generate an audio clip that starts with people cheering, then people crying, and ends with gunshots."
model=meanaudio_large   
# ckpt_path=weights/meanaudio_mf.pth
ckpt_path=/apdcephfs_gy4/share_302507476/xiquanli/exps/MeanAudio/AWM_256_numgpus8_niter400_000_T5_CLAP_meanaudio_large_fixddpbug_unfixtrajbug_scratch/AWM_256_numgpus8_niter400_000_T5_CLAP_meanaudio_large_fixddpbug_unfixtrajbug_scratch_ema_final.pth 
num_steps=1

python infer.py \
    --variant "meanaudio_large" \
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
