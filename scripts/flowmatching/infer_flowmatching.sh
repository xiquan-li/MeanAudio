cd /apdcephfs_gy4/share_302507476/xiquanli/TTA/MeanAudio

export CUDA_VISIBLE_DEVICES=1

output_path=./output/fluxaudio_m_full_4M/output_25nfe

prompt="A basketball bounces rhythmically on a court, shoes squeak against the floor, and a referee’s whistle cuts through the air"
model=fluxaudio_m_full  # fluxaudio_m_full
ckpt_path=/apdcephfs_gy4/share_302507476/xiquanli/exps/MeanAudio/AWM_4M_10s_bsz4096_numgpus64_niter100_000_T5_CLAP_fluxaudio_m_full/AWM_4M_10s_bsz4096_numgpus64_niter100_000_T5_CLAP_fluxaudio_m_full_ema_final.pth

python infer.py \
    --variant $model \
    --prompt "$prompt" \
    --model_path "$ckpt_path" \
    --output $output_path \
    --num_steps 25 \
    --cfg_strength 4.5 \
    --encoder_name t5_clap \
    --duration 10 \
    --use_rope \
    --text_c_dim 512
