export CUDA_VISIBLE_DEVICES=0

output_path=./exps/meanaudio/output_1nfe

prompt="A basketball bounces rhythmically on a court, shoes squeak against the floor, and a referee’s whistle cuts through the air"
model=meanaudio_mf
ckpt_path=weights/meanaudio_mf.pth
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
    --use_meanflow
