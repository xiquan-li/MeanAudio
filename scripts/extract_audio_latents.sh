echo "CODE DIR: $CODE_DIR"
cd $CODE_DIR



## split audio clips
PATH_TO_AUDIO_DIR=
OUTPUT_PARTITION_FILE=

python training/partition_clips.py \
    --data_dir $PATH_TO_AUDIO_DIR \
    --output_dir $OUTPUT_PARTITION_FILE


## extract audio latents
export CUDA_VISIBLE_DEVICES=0

CAPTIONS_TSV=./sets/audiocaps-test.tsv
OUTPUT_LATENT_DIR=
OUTPUT_NPZ_DIR=

torchrun --standalone --nproc_per_node=1 training/extract_audio_latents.py \
    --captions_tsv $CAPTIONS_TSV \
    --data_dir $PATH_TO_AUDIO_DIR \
    --clips_tsv $OUTPUT_PARTITION_FILE \
    --latent_dir $OUTPUT_LATENT_DIR \
    --output_dir $OUTPUT_NPZ_DIR \
    --text_encoder='t5_clap'  # ['clip', 't5', 't5_clap']