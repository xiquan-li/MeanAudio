import dataclasses
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from colorlog import ColoredFormatter
from PIL import Image
from torchvision.transforms import v2

from meanaudio.data.av_utils import ImageInfo, VideoInfo, read_frames, reencode_with_audio
from meanaudio.model.flow_matching import FlowMatching
from meanaudio.model.mean_flow import MeanFlow
from meanaudio.model.networks import MeanAudio, FluxAudio
from meanaudio.model.sequence_config import CONFIG_16K, CONFIG_44K, SequenceConfig
from meanaudio.model.utils.features_utils import FeaturesUtils
from meanaudio.utils.download_utils import download_model_if_needed

log = logging.getLogger()


@dataclasses.dataclass
class ModelConfig:
    model_name: str
    model_path: Path
    vae_path: Path
    bigvgan_16k_path: Optional[Path]
    mode: str

    @property
    def seq_cfg(self) -> SequenceConfig:
        if self.mode == '16k':
            return CONFIG_16K  # get sequence config when calling cfg.seq_cfgs
        elif self.mode == '44k':
            return CONFIG_44K

    def download_if_needed(self):
        raise NotImplementedError("Downloading models is not supported")
        download_model_if_needed(self.model_path)
        download_model_if_needed(self.vae_path)
        if self.bigvgan_16k_path is not None:
            download_model_if_needed(self.bigvgan_16k_path)


fluxaudio_fm = ModelConfig(model_name='fluxaudio_fm', 
                           model_path=Path('./weights/fluxaudio_fm.pth'),
                           vae_path=Path('./weights/v1-16.pth'),
                           bigvgan_16k_path=Path('./weights/best_netG.pt'),
                           mode='16k')
meanaudio_mf = ModelConfig(model_name='meanaudio_mf', 
                           model_path=Path('./weights/meanaudio_mf.pth'),
                           vae_path=Path('./weights/v1-16.pth'),
                           bigvgan_16k_path=Path('./weights/best_netG.pt'),
                           mode='16k')

all_model_cfg: dict[str, ModelConfig] = {
    'fluxaudio_fm': fluxaudio_fm, 
    'meanaudio_mf': meanaudio_mf, 
}


def generate_fm(
    text: Optional[list[str]],
    *,
    negative_text: Optional[list[str]] = None,
    feature_utils: FeaturesUtils,
    net: FluxAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
) -> torch.Tensor:
    # generate audio with vanilla flow matching

    device = feature_utils.device
    dtype = feature_utils.dtype

    bs = len(text)

    if text is not None:
        text_features, text_features_c = feature_utils.encode_text(text)
    else:
        text_features, text_features_c = net.get_empty_string_sequence(bs)

    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0 = torch.randn(bs,
                     net.latent_seq_len,
                     net.latent_dim,
                     device=device,
                     dtype=dtype,
                     generator=rng)
    preprocessed_conditions = net.preprocess_conditions(text_features, text_features_c)
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    cfg_ode_wrapper = lambda t, x: net.ode_wrapper(t, x, preprocessed_conditions, empty_conditions,
                                                   cfg_strength)
    x1 = fm.to_data(cfg_ode_wrapper, x0)
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


def generate_mf(
    text: Optional[list[str]],
    *,
    negative_text: Optional[list[str]] = None,
    feature_utils: FeaturesUtils,
    net: MeanFlow,
    mf: MeanFlow,
    rng: torch.Generator,
    cfg_strength: float,
) -> torch.Tensor:
    # generate audio with mean flow
    device = feature_utils.device
    dtype = feature_utils.dtype

    bs = len(text)

    if text is not None:
        text_features, text_features_c = feature_utils.encode_text(text)
    else:
        text_features, text_features_c = net.get_empty_string_sequence(bs)

    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0 = torch.randn(bs,
                     net.latent_seq_len,
                     net.latent_dim,
                     device=device,
                     dtype=dtype,
                     generator=rng)
    preprocessed_conditions = net.preprocess_conditions(text_features, text_features_c)
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    cfg_ode_wrapper = lambda t, r, x: net.ode_wrapper(t, r, x, preprocessed_conditions, empty_conditions,
                                                      cfg_strength)
    x1 = mf.to_data(cfg_ode_wrapper, x0)
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


LOGFORMAT = "[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s"


def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level) # set up root logger <=> logging.getLogger().setLevel(log_level) 
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()  # to Console
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger() 
    log.setLevel(log_level)
    log.addHandler(stream)