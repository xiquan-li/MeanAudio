cd /apdcephfs_gy4/share_302507476/xiquanli/TTA/MeanAudio

export CUDA_VISIBLE_DEVICES=1

output_path=./output/fluxaudio_m_full_30_jamendo_fma_20w/output_25nfe

prompt="A basketball bounces rhythmically on a court, shoes squeak against the floor, and a referee’s whistle cuts through the air"
model=fluxaudio_m_full_30
ckpt_path=/apdcephfs_gy4/share_302507476/xiquanli/exps/MeanAudio/FMA_Jamendo_30s_bsz512_numgpus32_niter200_000_T5_CLAP_fluxaudio_m_full_30/FMA_Jamendo_30s_bsz512_numgpus32_niter200_000_T5_CLAP_fluxaudio_m_full_30_ema_final.pth

python infer.py \
    --variant $model \
    --prompt "$prompt" \
    --model_path "$ckpt_path" \
    --output $output_path \
    --num_steps 25 \
    --cfg_strength 4.5 \
    --encoder_name t5_clap \
    --duration 30 \
    --use_rope \
    --text_c_dim 512
