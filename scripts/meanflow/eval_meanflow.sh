# evaluation on audiocaps

export CUDA_VISIBLE_DEVICES=0

num_steps=1
ckpt_path=./weights/meanaudio_mf.pth
output_path=./exps/meanaudio/test_${num_steps}nfe_fp32
# python eval.py \
#     --variant "meanaudio_mf" \
#     --model_path "$ckpt_path" \
#     --output $output_path/audio \
#     --cfg_strength 0.9 \
#     --encoder_name t5_clap \
#     --duration 10 \
#     --use_rope \
#     --text_c_dim 512 \
#     --num_steps $num_steps \
#     --use_meanflow \
#     --full_precision 


gt_audio='gt_audio'  # not used if you specify gt_cache 
gt_cache='./data/audiocaps/test-features' 

pred_audio=$output_path/audio
pred_audio=/apdcephfs_gy4/share_302507476/xiquanli/exps/MeanAudio/AWM_256_numgpus8_niter200_000_T5_CLAP_meanaudio_mf_lr1e-4_fixddpbug_scratch/test-sampled/

output_metrics_dir=/apdcephfs_gy4/share_302507476/xiquanli/exps/MeanAudio/AWM_256_numgpus8_niter200_000_T5_CLAP_meanaudio_mf_lr1e-4_fixddpbug_scratch/
# output_metrics_dir=$output_path

python av-benchmark/evaluate.py \
    --gt_audio $gt_audio \
    --gt_cache $gt_cache \
    --pred_audio $pred_audio \
    --pred_cache $output_metrics_dir/cache \
    --audio_length=10 \
    --recompute_pred_cache \
    --skip_video_related \
    --output_metrics_dir=$output_metrics_dir \
    # --debug