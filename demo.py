import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
from argparse import ArgumentParser
from pathlib import Path
import torch
import torchaudio
import gradio as gr
from transformers import AutoModel
from meanaudio.eval_utils import (
    ModelConfig,
    all_model_cfg,
    generate_mf,
    generate_fm,
    setup_eval_logging,
)
from meanaudio.model.flow_matching import FlowMatching
from meanaudio.model.mean_flow import MeanFlow
from meanaudio.model.networks import MeanAudio, get_mean_audio
from meanaudio.model.utils.features_utils import FeaturesUtils
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import gc
from datetime import datetime
from huggingface_hub import snapshot_download
log = logging.getLogger()
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
setup_eval_logging()
OUTPUT_DIR = Path("./output/gradio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#snapshot_download(repo_id="junxiliu/Meanaudio", local_dir="./weights",allow_patterns=["*.pt", "*.pth"] )

current_model_states = {

}

def load_model_if_needed(
    variant, model_path, encoder_name, use_rope, text_c_dim
):
    global current_model_states
    dtype = torch.float32
    existing_state = current_model_states.get(variant)
    needs_reload = (
        existing_state is None
        or existing_state["args"].variant != variant
        or existing_state["args"].model_path != model_path
        or existing_state["args"].encoder_name != encoder_name
        or existing_state["args"].use_rope != use_rope
        or existing_state["args"].text_c_dim != text_c_dim
    )
    if needs_reload:
        log.info(f"Loading/reloading model '{variant}'.")
        if variant not in all_model_cfg:
            raise ValueError(f"Unknown model variant: {variant}")
        model: ModelConfig = all_model_cfg[variant]
        seq_cfg = model.seq_cfg

        class MockArgs:
            pass
        mock_args = MockArgs()
        mock_args.variant = variant
        mock_args.model_path = model_path
        mock_args.encoder_name = encoder_name
        mock_args.use_rope = use_rope
        mock_args.text_c_dim = text_c_dim

        net: MeanAudio = (
            get_mean_audio(
                model.model_name,
                use_rope=mock_args.use_rope,
                text_c_dim=mock_args.text_c_dim,
            )
            .to(device, dtype)
            .eval()
        )
        net.load_weights(
            torch.load(
                mock_args.model_path, map_location=device, weights_only=True
            )
        )
        log.info(f"Loaded weights from {mock_args.model_path}")

        feature_utils = FeaturesUtils(
            tod_vae_ckpt=model.vae_path,
            enable_conditions=True,
            encoder_name=mock_args.encoder_name,
            mode=model.mode,
            bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
            need_vae_encoder=False,
        )
        feature_utils = feature_utils.to(device, dtype).eval()

        current_model_states[variant] = {
            "net": net,
            "feature_utils": feature_utils,
            "seq_cfg": seq_cfg,
            "args": mock_args,
        }
        log.info(f"Model '{variant}' loaded successfully.")

        return net, feature_utils, seq_cfg, mock_args
    else:
        log.info(f"Model '{variant}' already loaded with current settings. Skipping reload.")

        return existing_state["net"], existing_state["feature_utils"], existing_state["seq_cfg"], existing_state["args"]

def initialize_all_default_models():
    log.info("Initializing default models...")
    default_models = ['meanaudio_mf', 'fluxaudio_fm']
    common_params = {
        "encoder_name": "t5_clap",
        "use_rope": True,
        "text_c_dim": 512,

    }
    for variant in default_models:
        model_path = f"./weights/{variant}.pth"

        try:
            load_model_if_needed(
                variant, model_path, **common_params
            )
            log.info(f"Default model '{variant}' initialized successfully.")
        except Exception as e:
            log.error(f"Failed to initialize default model '{variant}': {e}")

#initialize_all_default_models()


@torch.inference_mode()
def generate_audio_gradio(
    prompt,
    negative_prompt,
    duration,
    cfg_strength,
    num_steps,
    seed,
    variant,
):
    global current_model_states

    model_path = f"./weights/{variant}.pth"
    encoder_name = "t5_clap"
    use_rope = True
    text_c_dim = 512

    model_state = current_model_states.get(variant)
    if model_state is None:
        error_msg = f"Error: Model '{variant}' is not available. It may not have been loaded correctly during startup."
        log.error(error_msg)
        return error_msg, None

    net = model_state["net"]
    feature_utils = model_state["feature_utils"]
    seq_cfg = model_state["seq_cfg"]

    args = model_state["args"]
    dtype = torch.float32

    temp_seq_cfg = type(seq_cfg)(**seq_cfg.__dict__)
    temp_seq_cfg.duration = duration

    net.update_seq_lengths(temp_seq_cfg.latent_seq_len)

    rng = torch.Generator(device=device)
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()

    use_meanflow = variant == "meanaudio_mf"
    if use_meanflow:
        sampler = MeanFlow(steps=num_steps)
        log.info("Using MeanFlow for generation.")
        generation_func = generate_mf
        sampler_arg_name = "mf"
        cfg_strength = 3
    else:
        sampler = FlowMatching(
            min_sigma=0, inference_mode="euler", num_steps=num_steps
        )
        log.info("Using FlowMatching for generation.")
        generation_func = generate_fm
        sampler_arg_name = "fm"

    prompts = [prompt]
    audios = generation_func(
        prompts,
        negative_text=[negative_prompt],
        feature_utils=feature_utils,
        net=net,
        rng=rng,
        cfg_strength=cfg_strength,
        **{sampler_arg_name: sampler},
    )
    audio = audios.float().cpu()[0]
    safe_prompt = (
        "".join(c for c in prompt if c.isalnum() or c in (" ", "_"))
        .rstrip()
        .replace(" ", "_")[:50]
    )
    current_time_string = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{safe_prompt}_{current_time_string}.flac"
    save_path = OUTPUT_DIR / filename
    torchaudio.save(str(save_path), audio, temp_seq_cfg.sampling_rate)
    log.info(f"Audio saved to {save_path}")

    gc.collect()

    return (
        f"Generated audio for prompt: '{prompt}' using {'MeanFlow' if use_meanflow else 'FlowMatching'}",
        str(save_path),
    )

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    text_size="sm",
    spacing_size="sm",
).set(
    background_fill_primary="*neutral_50",
    background_fill_secondary="*background_fill_primary",
    block_background_fill="*background_fill_primary",
    block_border_width="0px",
    panel_background_fill="*neutral_50",
    panel_border_width="0px",
    input_background_fill="*neutral_100",
    input_border_color="*neutral_200",
    button_primary_background_fill="*primary_300",
    button_primary_background_fill_hover="*primary_400",
    button_secondary_background_fill="*neutral_200",
    button_secondary_background_fill_hover="*neutral_300",
)
custom_css = """
#main-headertitle {
    text-align: center;
    margin-top: 15px;
    margin-bottom: 10px;
    color: var(--neutral-600);
    font-weight: 600;
}
#main-header {
    text-align: center;
    margin-top: 5px;
    margin-bottom: 10px;
    color: var(--neutral-600);
    font-weight: 600;
}
#model-settings-header, #generation-settings-header {
    color: var(--neutral-600);
    margin-top: 8px;
    margin-bottom: 8px;
    font-weight: 500;
    font-size: 1.1em;
}
.setting-section {
    padding: 10px 12px;
    border-radius: 6px;
    background-color: var(--neutral-50);
    margin-bottom: 10px;
    border: 1px solid var(--neutral-100);
}
hr {
    border: none;
    height: 1px;
    background-color: var(--neutral-200);
    margin: 8px 0;
}
#generate-btn {
    width: 100%;
    max-width: 250px;
    margin: 10px auto;
    display: block;
    padding: 10px 15px;
    font-size: 16px;
    border-radius: 5px;
}
#status-box {
    min-height: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 8px;
    border-radius: 5px;
    border: 1px solid var(--neutral-200);
    color: var(--neutral-700);
}
#project-badges {
    text-align: center;
    margin-top: 30px;
    margin-bottom: 20px;
}
#project-badges #badge-container {
    display: flex;
    gap: 10px;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
}
#project-badges img {
    border-radius: 5px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    height: 20px;
    transition: transform 0.1s ease, box-shadow 0.1s ease;
}
#project-badges a:hover img {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}
#audio-output {
    height: 200px;
    border-radius: 5px;
    border: 1px solid var(--neutral-200);
}
.gradio-dropdown label, .gradio-checkbox label, .gradio-number label, .gradio-textbox label {
    font-weight: 500;
    color: var(--neutral-700);
    font-size: 0.9em;
}
.gradio-row {
   gap: 8px;
}
.gradio-block {
   margin-bottom: 8px;
}
.setting-section .gradio-block {
    margin-bottom: 6px;
}
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-track {
  background: var(--neutral-100);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb {
  background: var(--neutral-300);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background: var(--neutral-400);
}
* {
  scrollbar-width: thin;
  scrollbar-color: var(--neutral-300) var(--neutral-100);
}
"""
with gr.Blocks(title="MeanAudio Generator", theme=theme, css=custom_css) as demo:
    gr.Markdown("# MeanAudio:Fast and Faithful Text-to-Audio Generation with Mean Flows", elem_id="main-header")
    with gr.Column(elem_classes="setting-section"):
        with gr.Row():
            available_variants = (
                list(all_model_cfg.keys()) if all_model_cfg else []
            )
            default_variant = (
                'meanaudio_mf'
            )
            variant = gr.Dropdown(
                label="Model Variant",
                choices=available_variants,
                value=default_variant,
                interactive=True,
                scale=3,
            )

    with gr.Column(elem_classes="setting-section"):
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the sound you want to generate...",
                scale=1,
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="Describe sounds you want to avoid...",
                value="",
                scale=1,
            )
        with gr.Row():
            duration = gr.Number(
                label="Duration (sec)", value=10.0, minimum=0.1, scale=1
            )
            cfg_strength = gr.Number(
                label="CFG (Meanflow forced to 3)", value=3, minimum=0.0, scale=1
            )
        with gr.Row():
            seed = gr.Number(
                label="Seed (-1 for random)", value=42, precision=0, scale=1
            )
            num_steps = gr.Number(
                label="Number of Steps",
                value=1,
                precision=0,
                minimum=1,
                scale=1,
            )
    generate_button = gr.Button("Generate", variant="primary", elem_id="generate-btn")
    generate_output_text = gr.Textbox(
        label="Result Status", interactive=False, elem_id="status-box"
    )
    audio_output = gr.Audio(
        label="Generated Audio", type="filepath", elem_id="audio-output"
    )
    generate_button.click(
        fn=generate_audio_gradio,
        inputs=[
            prompt,
            negative_prompt,
            duration,
            cfg_strength,
            num_steps,
            seed,
            variant,
        ],
        outputs=[generate_output_text, audio_output],
    )
    audio_examples = [
        ["A speech and gunfire followed by a gun being loaded", "", 10.0, 3, 1, 42, "meanaudio_mf"],
        ["Typing on a keyboard", "", 10.0, 3, 1, 42, "meanaudio_mf"],
        ["A man speaks followed by a popping noise and laughter", "", 10.0, 3, 2, 42, "meanaudio_mf"],
        ["Some humming followed by a toilet flushing", "", 10.0, 3, 2, 42, "meanaudio_mf"],
        ["Rain falling on a hard surface as thunder roars in the distance", "", 10.0, 3, 5, 42, "meanaudio_mf"],
        ["Food sizzling and oil popping", "", 10.0, 3, 25, 42, "meanaudio_mf"],
        ["Pots and dishes clanking as a man talks followed by liquid pouring into a container", "", 8.0, 3, 2, 42, "meanaudio_mf"],
        ["A few seconds of silence then a rasping sound against wood", "", 12.0, 3, 2, 42, "meanaudio_mf"],
        ["A man speaks as he gives a speech and then the crowd cheers", "", 10.0, 3, 25, 42, "fluxaudio_fm"],
        ["A goat bleating repeatedly", "", 10.0, 3, 50, 123, "fluxaudio_fm"],
        ["Tires squealing followed by an engine revving", "", 12.0, 4, 25, 456, "fluxaudio_fm"],
        ["Hammer slowly hitting the wooden table", "", 10.0, 3.5, 25, 42, "fluxaudio_fm"],
        ["Dog barking excitedly and man shouting as race car engine roars past", "", 10.0, 3, 1, 42, "meanaudio_mf"],
        ["A dog barking and a cat mewing and a racing car passes by", "", 12.0, 3, 5, -1, "meanaudio_mf"],
        ["Whistling with birds chirping", "", 10.0, 4, 50, 42, "fluxaudio_fm"],
    ]
    gr.Examples(
        examples=audio_examples,
        inputs=[prompt, negative_prompt, duration, cfg_strength, num_steps, seed, variant],
        #outputs=[generate_output_text, audio_output],
        #fn=generate_audio_gradio,
        examples_per_page=5,
        label="Example Prompts",
    )

if __name__ == "__main__":
    demo.launch()
