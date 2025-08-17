export CUDA_VISIBLE_DEVICES=1

output_path=./exps/fluxaudio_s_full/output_25nfe

prompt="A basketball bounces rhythmically on a court, shoes squeak against the floor, and a referee’s whistle cuts through the air"
ckpt_path=weights/fluxaudio_s_full.pth

python infer.py \
    --variant "fluxaudio_s" \
    --prompt "$prompt" \
    --model_path "$ckpt_path" \
    --output $output_path \
    --num_steps 25 \
    --cfg_strength 4.5 \
    --encoder_name t5_clap \
    --duration 10 \
    --use_rope \
    --text_c_dim 512
