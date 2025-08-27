export CUDA_VISIBLE_DEVICES=0

output_path=./exps/meanaudio_l_full/output_1nfe

prompt="A man is chopping vegetables"
ckpt_path=weights/meanaudio_l_full.pth
num_steps=1

python infer.py \
    --variant "meanaudio_l" \
    --prompt "$prompt" \
    --model_path "$ckpt_path" \
    --output $output_path \
    --cfg_strength 0 \
    --encoder_name t5_clap \
    --duration 10 \
    --use_rope \
    --text_c_dim 512 \
    --num_steps $num_steps \
    --use_meanflow
