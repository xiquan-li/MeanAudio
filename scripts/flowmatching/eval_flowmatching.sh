# evaluation on audiocaps

export CUDA_VISIBLE_DEVICES=2

num_steps=25
ckpt_path=./weights/fluxaudio_s_full.pth
output_path=./exps/fluxaudio_s_full/test_${num_steps}nfe_fp32

python eval.py \
    --variant "fluxaudio_s" \
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