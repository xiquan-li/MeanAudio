import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
from pathlib import Path
import torch
import torchaudio
from meanaudio.eval_utils import (ModelConfig, all_model_cfg, generate_mf, generate_fm, setup_eval_logging)
from meanaudio.model.flow_matching import FlowMatching
from meanaudio.model.mean_flow import MeanFlow
from meanaudio.model.networks import MeanAudio, get_mean_audio
from meanaudio.model.utils.features_utils import FeaturesUtils
from huggingface_hub import snapshot_download
import argparse

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
log = logging.getLogger()


@torch.inference_mode()
def MeanAudioDemoInfer(
    prompt='',
    negative_prompt='',
    encoder_name='t5_clap',
    variant='meanaudio_s',
    duration=10,
    cfg_strength=0,  # for meanflow, cfg_strength is integrated in training, and thus don't need to be specified here
    num_steps=1,
    output='./output',
    seed=42,
    full_precision=False,
    use_rope=True,
    text_c_dim=512,
    use_meanflow=False
):
    '''
    prompt (str): 
        The text description guiding the audio generation (e.g., "a dog is barking").
    negative_prompt (str): 
        A text description for sounds that should be avoided in the generated audio.
    model_path (str): 
        Path to the model weights file. If empty, it defaults to ./weights/{variant}.pth.
    encoder_name (str): 
        Specifies the text encoder to use (default: 't5_clap').
    variant (str): 
        Specifies the model variant to load (default: 'meanaudio_mf'). Must be a key in all_model_cfg.
    duration (int): 
        The desired duration of the generated audio in seconds (default: 10).
    cfg_strength (float): 
        Classifier-Free Guidance strength. Ignored if use_meanflow is True or variant is 'meanaudio_mf' (default: 4.5).
    num_steps (int): 
        Number of steps for the generation process (default: 1).
    output (str): 
        Directory path where the generated audio file will be saved (default: './output').
    seed (int): 
        Random seed for generation reproducibility (default: 42).
    full_precision (bool): 
        If True, uses torch.float32 precision; otherwise, uses torch.bfloat16 (default: False).
    use_rope (bool): 
        Whether to use Rotary Position Embedding in the model (default: True).
    text_c_dim (int): 
        Dimension of the text context vector (default: 512).
    use_meanflow (bool): 
        If True, uses the MeanFlow generation method; otherwise, uses FlowMatching. If variant is 'meanaudio_mf', this is automatically set to True (default: False).    
    '''
    setup_eval_logging()
    output_dir = Path(output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32 if full_precision else torch.bfloat16
    
    if duration <= 0 or num_steps <= 0:
        raise ValueError("Duration and number of steps must be positive.")
    if variant not in all_model_cfg:
        raise ValueError(f"Unknown model variant: {variant}. Available: {list(all_model_cfg.keys())}")

    model_path = all_model_cfg[variant].model_path  # by default, this will use meanaudio_s_full.pth or fluxaudio_s_full.pth
    if not model_path.exists():
        log.info(f'Model not found at {model_path}')
        log.info('Downloading models to "./weights/"...')
        try:
            weights_dir = Path('./weights')
            weights_dir.mkdir(exist_ok=True)
            snapshot_download(repo_id="AndreasXi/MeanAudio", local_dir="./weights" )
        except Exception as e:
            log.error(f"Failed to download model: {e}")
            raise FileNotFoundError(f"Model file not found and download failed: {model_path}, you may need to download the model manually.")
    
    model = all_model_cfg[variant]
    seq_cfg = model.seq_cfg
    seq_cfg.duration = duration
    
    net = get_mean_audio(model.model_name, use_rope=use_rope, text_c_dim=text_c_dim)
    net = net.to(device, dtype).eval()
    net.load_weights(torch.load(model_path, map_location=device, weights_only=True))
    net.update_seq_lengths(seq_cfg.latent_seq_len)
    
    if variant == 'meanaudio_s' or variant == 'meanaudio_l': 
        use_meanflow=True 
    if use_meanflow:
        generation_func = MeanFlow(steps=num_steps)
        cfg_strength=0  # for meanflow, cfg_strength is integrated in training, and thus don't need to be specified here
    else:
        generation_func = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)
    
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        enable_conditions=True,
        encoder_name=encoder_name,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False
    )
    feature_utils = feature_utils.to(device, dtype).eval()
    
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    
    generate_fn = generate_mf if use_meanflow else generate_fm
    kwargs = {
        'negative_text': [negative_prompt],
        'feature_utils': feature_utils,
        'net': net,
        'rng': rng,
        'cfg_strength': cfg_strength
    }
    
    if use_meanflow:
        kwargs['mf'] = generation_func
    else:
        kwargs['fm'] = generation_func
    
    audios = generate_fn([prompt], **kwargs)
    audio = audios.float().cpu()[0]
    safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
    save_path = output_dir / f'{safe_filename}--numsteps{num_steps}--seed{seed}.wav'
    torchaudio.save(save_path, audio, seq_cfg.sampling_rate)
    log.info(f'Audio saved to {save_path}')
    log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))
    return save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, help='Input prompt', default='A dog is barking')
    parser.add_argument('--variant', type=str, help='Model variant', choices=['meanaudio_s', 'meanaudio_l', 'fluxaudio_s'], default='meanaudio_s')
    parser.add_argument('--num_steps', type=int, help='Number of steps', default=1)
    args = parser.parse_args()

    audio_path = MeanAudioDemoInfer(prompt=args.prompt, 
                                    variant=args.variant, 
                                    num_steps=args.num_steps)
    log.info('Inference completed')
