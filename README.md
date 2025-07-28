<div align="center">
<p align="center">
  <h1>MeanAudio: Fast and Faithful Text-to-Audio Generation with Mean Flows</h1>
  <a href=>Paper</a> | <a href="https://meanaudio.github.io/">Webpage</a> 
</p>
</div>


## Overview 
MeanAudio is a novel MeanFlow-based model tailored for fast and faithful text-to-audio generation. It can synthesize realistic sound in a single step, achieving a real-time factor (RTF) of 0.013 on a single NVIDIA 3090 GPU. Moreover, it also demonstrates strong performance in multi-step generation.

<div align="center">
  <img src="sets/performance.png" alt="" width="450">
</div>


## Environmental Setup

**1. Create a new conda environment:**

```bash
conda create -n meanaudio python=3.11

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
```
<!-- ```
conda install -c conda-forge 'ffmpeg<7
```
(Optional, if you use miniforge and don't already have the appropriate ffmpeg) -->

**2. Install with pip:**

```bash
git clone https://github.com/xiquan-li/MeanAudio.git

cd MeanAudio
pip install -e .
```

<!-- (If you encounter the File "setup.py" not found error, upgrade your pip with pip install --upgrade pip) -->


## Quick Start

<!-- **1. Download pre-trained models:** -->

Firstly, download pre-trained models from this [Folder](https://drive.google.com/drive/folders/1nbIsVjl4pqLaAnqj-M8UPkahu28S59Kj?usp=sharing), and put them into `MeanAudio/weights/`. Then, run: 

```bash 
bash scripts/meanflow/infer_meanflow.sh
``` 
You can change `prompt` and `num_steps` in the script to generate diverse sounds. 
Here is a detailed explanation of the downloaded checkpoints: 

1. [fluxaudio_fm.pth](https://drive.google.com/file/d/1PAJ7Asx_3e9HiaUoGIfSXI3K7BqgBR9x/view?usp=sharing): The Flux-style flow transformer trained on WavCaps, AudioCaps and Clotho dataset with the **standard flow matching objective**. It is capable of generating audio with multiple ($\geq 25$) sampling steps. You can run `scripts/flowmatching/infer_flowmatching.sh` to generate sound with this model.

2. [meanaudio_mf.pth](https://drive.google.com/file/d/1BFWiHVJwdyXihE14znDYiAWF0-mnEtA7/view?usp=sharing): The Flux-style flow transformer fine-tuned on AudioCaps with the **Mean Flow Objective**, supporting both single-step and multi-step audio generation.


3. Others: [best_netG.pt](https://drive.google.com/file/d/1PAJ7Asx_3e9HiaUoGIfSXI3K7BqgBR9x/view?usp=sharing): The [BigVGAN Vocoder](https://github.com/NVIDIA/BigVGAN). [v1-16.pth](https://drive.google.com/file/d/1bJlNhGGjmDBKjz04bpOi-UjfuJILSiGU/view?usp=sharing):  The 1D VAE. 
[music_speech_audioset_epoch_15_esc_89.98.pt](https://drive.google.com/file/d/1KGQ5Q8xHOoItPDdJAB8ry6kKJ5HkMyo9/view?usp=share_link): The [CLAP](https://github.com/LAION-AI/CLAP) Encoder. 

## Training

### 1. Latent & Text Feature Extraction: 
We first extract VAE latents & text encoder embeddings to enable fast and efficient training. `scripts/extract_audio_latents.sh` provides a detailed guide for it. The pipeline includes two steps: a) partition audios into 10s clips. b) extract latents & embeddings. 

**To avoid the laborious data pre-processing step, we have uploaded an extracted version of [AudioCaps](https://audiocaps.github.io). Feel free to download it from this [link](https://drive.google.com/file/d/1C_P3ZQQWxUgMuCw-qvYj2C2r0iM35Sfy/view?usp=share_link), unzip it and put it under `MeanAudio/data/`. Then you can directly jump to the second step 😊.**

However, if you want to train the model on other datasets besides AudioCaps, you should still run `scripts/extract_audio_latents.sh` to do feature extraction. 
Remember to adjust `config/data/t5_clap.yaml` for correct metadata paths. 
### 2. Install Validation Packages: 
We rely on [av-benchmark](https://github.com/hkchengrex/av-benchmark) for validation & evaluation. Please install it first before training.

### 3. Train with MeanFlow objective: 
Use the script below to train a MeanAudio model. By default, this will initialize the flow transformer from the pretrained ckpt `fluxaudio_fm.pth` and do MeanFlow fine-tuning. 
```bash
bash scripts/meanflow/train_meanflow.sh
```

### 4. (Optional) Pre-training with Standard Flow Matching: 
Use the script below to train a Flux-style transformer using the conditional flow matching objective: 
```bash 
bash scripts/flowmatching/train_flowmatching.sh
```
The obtained model can serve as a strong initialization for the mixed-flow fine-tuning. 

## Evaluation

Use the script below to do evaluation, before this, please first install [av-benchmark](https://github.com/hkchengrex/av-benchmark) for metrics calculation. You can specify `num_steps` and `ckpt_path` to evaluate different models with different sampling steps. 
```bash
bash scripts/meanflow/eval_meanflow.sh 
```

## Citation

```bibtex
TODO
```



## Acknowledgement

Many thanks to:
- [MMAudio](https://github.com/hkchengrex/MMAudio) for the MMDiT code and training & inference structure
- [MeanFlow-pytorch](https://github.com/haidog-yaqub/MeanFlow) and [MeanFlow-official](https://github.com/Gsunshine/meanflow) for the mean flow implementation
- [Make-An-Audio 2](https://github.com/bytedance/Make-An-Audio-2) BigVGAN Vocoder and the VAE
- [av-benchmark](https://github.com/hkchengrex/av-benchmark) for benchmarking results
