<div align="center">
<p align="center">
  <h2>MeanAudio: Fast and Faithful Text-to-Audio Generation with Mean Flows</h2>
  <a href=>Paper</a> | <a href="https://meanaudio.github.io/">Webpage</a> 
</p>
</div>


## Environment Setup

### Prerequisites

**1. Create a new conda environment:**

```bash
conda create -n meanaudio python=3.9

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
```
<!-- ```
conda install -c conda-forge 'ffmpeg<7
```
(Optional, if you use miniforge and don't already have the appropriate ffmpeg) -->

**2. Install with pip :**

```bash
git clone 

cd MMAudio
pip install -e .
```

<!-- (If you encounter the File "setup.py" not found error, upgrade your pip with pip install --upgrade pip) -->


## Quick Start

By default, these scripts use the `large_44k_v2` model. 
In our experiments, inference only takes around 6GB of GPU memory (in 16-bit mode) which should fit in most modern GPUs.

### Command-line interface

With `demo.py`

```bash
python demo.py --duration=8 --video=<path to video> --prompt "your prompt" 
```

The output (audio in `.flac` format, and video in `.mp4` format) will be saved in `./output`.
See the file for more options.
Simply omit the `--video` option for text-to-audio synthesis.
The default output (and training) duration is 8 seconds. Longer/shorter durations could also work, but a large deviation from the training duration may result in a lower quality.

### Gradio interface

Supports video-to-audio and text-to-audio synthesis.
You can also try experimental image-to-audio synthesis which duplicates the input image to a video for processing. This might be interesting to some but it is not something MMAudio has been trained for.
Use [port forwarding](https://unix.stackexchange.com/questions/115897/whats-ssh-port-forwarding-and-whats-the-difference-between-ssh-local-and-remot) (e.g., `ssh -L 7860:localhost:7860 server`) if necessary. The default port is `7860` which you can specify with `--port`.

```bash
python gradio_demo.py
```

## Training

See [TRAINING.md](docs/TRAINING.md).

## Evaluation

See [EVAL.md](docs/EVAL.md).

## Training Datasets

MMAudio was trained on several datasets, including [AudioSet](https://research.google.com/audioset/), [Freesound](https://github.com/LAION-AI/audio-dataset/blob/main/laion-audio-630k/README.md), [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/), [AudioCaps](https://audiocaps.github.io/), and [WavCaps](https://github.com/XinhaoMei/WavCaps). These datasets are subject to specific licenses, which can be accessed on their respective websites. We do not guarantee that the pre-trained models are suitable for commercial use. Please use them at your own risk.


## Citation

```bibtex

```

## Relevant Repositories

- [av-benchmark](https://github.com/hkchengrex/av-benchmark) for benchmarking results.


## Acknowledgement

Many thanks to:
- [Make-An-Audio 2](https://github.com/bytedance/Make-An-Audio-2) for the 16kHz BigVGAN pretrained model and the VAE architecture
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [Synchformer](https://github.com/v-iashin/Synchformer) 
- [EDM2](https://github.com/NVlabs/edm2) for the magnitude-preserving VAE network architecture
