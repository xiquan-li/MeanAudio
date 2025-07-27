# evaluation on audiocaps

export CUDA_VISIBLE_DEVICES=0

num_steps=25
ckpt_path=./weights/meanaudio_mf.pth
ckpt_path=/apdcephfs_gy4/share_302507476/xiquanli/exps/MMAudio/meanflow/AC_72_numgpus4_niter200_000_T5_CLAP_meanflow_improved_changecfg_seed42_flowratio0.5/AC_72_numgpus4_niter200_000_T5_CLAP_meanflow_improved_changecfg_seed42_flowratio0.5_ema_final.pth
output_path=./exps/meanaudio/test_${num_steps}nfe_bf16
output_path=/apdcephfs_gy4/share_302507476/xiquanli/exps/MMAudio/meanflow/AC_72_numgpus4_niter200_000_T5_CLAP_meanflow_improved_changecfg_seed42_flowratio0.5/test_25nfe

python eval.py \
    --variant "meanaudio_mf" \
    --model_path "$ckpt_path" \
    --output $output_path/audio \
    --cfg_strength 0.9 \
    --encoder_name t5_clap \
    --duration 10 \
    --use_rope \
    --text_c_dim 512 \
    --num_steps $num_steps \
    --use_meanflow \
    --full_precision 



cd av-benchmark
gt_audio='gt_audio'  # not used if you specify gt_cache 
gt_cache='../../../data/AudioCaps/test-features' 

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