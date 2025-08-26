import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
from argparse import ArgumentParser
from pathlib import Path
import torch
import torchaudio
from meanaudio.eval_utils import (ModelConfig, all_model_cfg, generate_mf, generate_fm, setup_eval_logging)
from meanaudio.model.flow_matching import FlowMatching
from meanaudio.model.mean_flow import MeanFlow
from meanaudio.model.networks import MeanAudio, get_mean_audio
from meanaudio.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
log = logging.getLogger()


@torch.inference_mode()
def main():
    setup_eval_logging()

    parser = ArgumentParser()
    parser.add_argument('--variant',
                        type=str,
                        default='small_16k_mf',
                        help='small_16k_mf, small_16k_fm')
    
    parser.add_argument('--prompt', type=str, help='Input prompt', default='')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt', default='')
    parser.add_argument('--duration', type=float, default=9.975)  # for 312 latents, seq_config should has a duration of 9.975s 
    parser.add_argument('--cfg_strength', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=25)

    parser.add_argument('--output', type=Path, help='Output directory', default='./output')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--full_precision', action='store_true')
    parser.add_argument('--model_path', type=str, help='Ckpt path of trained model')
    parser.add_argument('--encoder_name', choices=['clip', 't5', 't5_clap'], type=str, help='text encoder name')
    parser.add_argument('--use_rope', action='store_true', help='Whether or not use position embedding for model')
    parser.add_argument('--text_c_dim', type=int, default=512, 
                        help='Dim of the text_features_c, 1024 for pooled T5 and 512 for CLAP')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_meanflow', action='store_true', help='Whether or not use mean flow for inference')
    args = parser.parse_args()

    if args.debug: 
        import debugpy
        debugpy.listen(6666) 
        print("Waiting for debugger attach (rank 0)...")
        debugpy.wait_for_client()  
    
    if args.variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {args.variant}')
    model: ModelConfig = all_model_cfg[args.variant]  # model is just the model config
    seq_cfg = model.seq_cfg  

    negative_prompt: str = args.negative_prompt
    output_dir: str = args.output.expanduser()
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    cfg_strength: float = args.cfg_strength

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    output_dir.mkdir(parents=True, exist_ok=True)
    # load a pretrained model
    net: MeanAudio = get_mean_audio(model.model_name, 
                                    use_rope=args.use_rope, 
                                    text_c_dim=args.text_c_dim).to(device, dtype).eval() 
    net.load_weights(torch.load(args.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {args.model_path}')

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    if args.use_meanflow:
        mf = MeanFlow(steps=num_steps)
    else:
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  enable_conditions=True,
                                  encoder_name=args.encoder_name, 
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len)
    prompts: str = [args.prompt]

    # prompts_gpt_music = [
    #     "A bright acoustic guitar melody with gentle piano accents, evoking sunlight filtering through forest leaves.",
    #     "Heavy bass and driving drums that feel like speeding through the city at night.",
    #     "Soft strings intertwined with airy vocals, creating a dreamlike atmosphere.",
    #     "Pulsating electronic synths with shifting tones, like walking through a neon-lit future city.",
    #     "Slow piano and cello interplay, deep and emotional, telling a forgotten story.",
    #     "Lively brass and Latin percussion, bursting with carnival energy.",
    #     "Minimal ambient textures with long sustained notes, like wandering alone across a snowy plain.",
    #     "Explosive guitar riffs with raw, shouted vocals, full of unrestrained power.",
    #     "Layered choir harmonies over an electronic beat, like drifting weightlessly through space.",
    #     "Crisp marimba and warm bass grooves, perfect for a lazy afternoon.",
    #     "Hypnotic tribal drums layered with chanting voices, evoking an ancient ritual.",
    #     "Smooth jazz saxophone over mellow bass, glowing with late-night warmth.",
    #     "Rapid violin runs over a fast-paced orchestral background, brimming with tension.",
    #     "Deep house beats and shimmering synth pads, ideal for a midnight dance floor.",
    #     "Gentle harp glissandos under a soft flute melody, like a spring morning in a garden.",
    #     "Powerful orchestral brass with rolling timpani, evoking a heroic battle scene.",
    #     "Lo-fi beats with vinyl crackle, giving a cozy bedroom vibe.",
    #     "Melancholic piano chords layered with distant rain sounds, evoking quiet introspection.",
    #     "Upbeat funk guitar riffs and tight snare hits, impossible not to dance to.",
    #     "Swelling cinematic strings with distant choirs, evoking a sense of awe and wonder.",
    #     "Minimal techno with hypnotic repetition, perfect for deep focus.",
    #     "Warm Rhodes piano chords and slow bass, dripping with soulful elegance.",
    #     "Aggressive double-kick drumming under distorted guitars, pure metal energy.",
    #     "Soft ukulele strums with playful whistles, evoking a sunny beachside.",
    #     "Experimental glitch beats with fragmented vocal samples, unpredictable and futuristic.",
    #     "Emotional ballad vocals soaring over grand piano, heartfelt and intimate.",
    #     "Dark ambient drones with eerie metallic textures, evoking an abandoned industrial site.",
    #     "Fast bluegrass banjo picking with lively fiddle, radiating rustic charm.",
    #     "Sparkling synthesizers over a steady dance beat, full of optimism.",
    #     "Slow gospel choir harmonies rising into a powerful climax, filled with spiritual strength."
    # ]
    prompts_fma = [
        "A lively and energetic 90s hip-hop rap song celebrating carefree living.",
        "Upbeat and empowering hip-hop celebrating black culture and resilience in America.",
        "A fast-paced rap/hip-hop track from the 90s reflecting the struggles and lifestyle of a rapper.",
        "This energetic and youthful pop song incorporates elements of electronic and synthpop, evoking nostalgia for 80s and 90s pop culture with catchy melodies and synthesized sounds.",
        "This energetic and upbeat track blends elements of rock, alternative, and indie music reminiscent of the 2000s.",
        "A lively blend of jazz and funk with an upbeat tempo, perfect for dancing and reminiscent of the 70s and 80s, offering a nostalgic yet fresh sound.",
        "A mellow mix of rock and indie with British influences and thoughtful lyrics sung by a male vocalist, ideal for unwinding.",
        "A melancholic blend of indie, folk, and singer-songwriter styles with introspective lyrics on themes of love and loss.",
        "A soothing blend of rock and blues with a mellow vibe, centered around themes of love and heartbreak.",
        "A fast-tempo, energetic hip-hop song from the 90s with a fusion of funk, featuring lyrics about love and relationships.",
        "Experimental, avant-garde, and noisy music with a slow tempo and a dark, unsettling mood, featuring a unique and unconventional sound blending Loud-Rock and Psych-Rock elements.",
        "A hypnotic and immersive blend of ambient, experimental, and noise genres with a slow tempo and avant-garde style.",
        "A lively and energetic cover of a classic rock song from the 1970s.",
        "A melancholic Baroque classical piece featuring intricate harpsichord melodies and harmonies, evoking grandeur and majesty.",
        "This 1960s music combines rock and roll, blues, and classic rock with a guitar-heavy sound, featuring lyrics about rebellion and freedom.",
        "A lively and catchy fusion of rock, pop, and alternative with a youthful, energetic vibe, perfect for dancing.",
        "This melancholic folk song explores themes of love, loss, and introspection.",
        "A fast-paced instrumental piece with a classical vibe featuring stringed instruments, evoking an energetic and uplifting mood.",
        "An energetic blend of jazz, funk, and ska with a driving rhythm and catchy melodies, perfect for dancing.",
        "A lively fusion of jazz, funk, and avant-garde with unconventional instruments and experimental flair.",
        "A unique and captivating blend of jazz, funk, and avant-garde styles with complex harmonies and intricate rhythms.",
        "The music combines jazz, funk, and ska with a lively and energetic tempo reminiscent of the 80s and 90s.",
        "A high-energy blend of hard rock and heavy metal with rebellious themes, ideal for action scenes in movies or video games.",
        "The music is a dark ambient piece with haunting soundscapes that evoke unease and tension."
    ]

    prompts_music = [
        'Guitar and piano playing a warm music, with a soft and gentle melody, perfect for a romantic evening.',
        'A lively blend of pop and funk with bright synths and a steady dance beat.',
        "A lively blend of pop and funk with bright synths and a steady dance beat.",
        "Fast-paced electronic track with driving basslines and shimmering melodies.",
        "Upbeat indie rock with jangly guitars and catchy, sing-along choruses.",
        "A slow, ethereal soundscape filled with reverb-soaked synths and distant echoes.",
        "Minimalist piano patterns layered over soft drones, evoking a sense of calm.",
        "Dreamlike textures with shimmering pads, airy vocals, and subtle field recordings.",
        "A grand orchestral score with sweeping strings and powerful brass.",
        "A delicate chamber piece featuring solo violin and piano accompaniment.",
        "An emotional symphonic composition that builds toward a triumphant finale."
    ]

    # prompts = [args.prompt]
    prompts = [
        'Guitar and piano playing a warm music, with a soft and gentle melody, perfect for a romantic evening.',
        'A lively and energetic 90s hip-hop rap song celebrating carefree living.',
        'Generate an audio clip that starts with people cheering, then people crying, and ends with gunshots',
        'Battlefield scene, continuous roar of artillery and gunfire, high fidelity, the sharp crack of bullets, the thundering explosions of bombs, and the screams of wounded soldier.', 
        'Pop music that upbeat, catchy, and easy to listen, high fidelity, with simple melodies, electronic instruments and polished production.	', 
        'The steady crashing of waves against the shore,high fidelity, the whooshing sound of water receding back into the ocean, the sound of seagulls and other coastal birds, and the distant sound of ships or boats.',
        'Two space shuttles are fighting in the space.'
    ]
    
    prompts = prompts_fma
    if args.use_meanflow:
        for prompt in tqdm(prompts): 
            log.info(f'Prompt: {prompt}')
            log.info(f'Negative prompt: {negative_prompt}')
            audios = generate_mf([prompt],
                                  negative_text=[negative_prompt],
                                  feature_utils=feature_utils,
                                  net=net,
                                  mf=mf,
                                  rng=rng,
                                  cfg_strength=cfg_strength)
            audio = audios.float().cpu()[0]
            safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
            save_path = output_dir / f'{safe_filename}--numsteps{num_steps}--seed{args.seed}.wav'
            torchaudio.save( save_path, audio, seq_cfg.sampling_rate)
            log.info(f'Audio saved to {save_path}')
        log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))
    else:
        for prompt in tqdm(prompts): 
            log.info(f'Prompt: {prompt}')
            log.info(f'Negative prompt: {negative_prompt}')
            audios = generate_fm([prompt],
                                  negative_text=[negative_prompt],
                                  feature_utils=feature_utils,
                                  net=net,
                                  fm=fm,
                                  rng=rng,
                                  cfg_strength=cfg_strength)
            audio = audios.float().cpu()[0]
            safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
            save_path = output_dir / f'{safe_filename}--numsteps{num_steps}--seed{args.seed}.wav'
            torchaudio.save(save_path, audio, seq_cfg.sampling_rate)

            log.info(f'Audio saved to {save_path}')
        log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))


if __name__ == '__main__':
    main()
