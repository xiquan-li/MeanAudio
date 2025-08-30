# evaluation on audiocaps

export CUDA_VISIBLE_DEVICES=2

num_steps=25
ckpt_path=/apdcephfs_gy4/share_302507476/xiquanli/exps/MeanAudio/AWV_8kh_10s_bsz4096_numgpus64_niter50_000_T5_CLAP_fluxaudio_fm_Correct_xt/AWV_8kh_10s_bsz4096_numgpus64_niter50_000_T5_CLAP_fluxaudio_fm_Correct_xt_ema_final.pth
output_path=/apdcephfs_gy4/share_302507476/xiquanli/exps/MeanAudio/AWV_8kh_10s_bsz4096_numgpus64_niter50_000_T5_CLAP_fluxaudio_fm_Correct_xt/

python eval.py \
    --variant "fluxaudio_fm" \
    --model_path "$ckpt_path" \
    --output $output_path/audio \
    --cfg_strength 4.5 \
    --num_steps $num_steps \
    --encoder_name t5_clap \
    --duration 10 \
    --use_rope \
    --text_c_dim 512 \
    --num_steps $num_steps \
    --full_precision 


cd ./av-benchmark
gt_audio='gt_audio'  # not used if you specify gt_cache 
gt_cache='./data/audiocaps/test-features' 

pred_audio=$output_path/audio
output_metrics_dir=$output_path

python evaluate.py \
    --gt_audio $gt_audio \
    --gt_cache $gt_cache \
    --pred_audio $pred_audio \
    --pred_cache $output_metrics_dir/cache \
    --audio_length=10 \
    --recompute_pred_cache \
    --skip_video_related \
    --output_metrics_dir=$output_metrics_dir \
    # --debug