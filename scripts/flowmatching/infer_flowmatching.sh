export CUDA_VISIBLE_DEVICES=1

output_path=./exps/fluxaudio/output_25nfe

prompt="A basketball bounces rhythmically on a court, shoes squeak against the floor, and a referee’s whistle cuts through the air"
model=fluxaudio_fm
ckpt_path=weights/fluxaudio_fm.pth

python infer.py \
    --variant "fluxaudio_fm" \
    --prompt "$prompt" \
    --model_path "$ckpt_path" \
    --output $output_path \
    --num_steps 25 \
    --cfg_strength 4.5 \
    --encoder_name t5_clap \
    --duration 10 \
    --use_rope \
    --text_c_dim 512
