from typing import Literal, Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from open_clip import create_model_from_pretrained
from torchvision.transforms import Normalize
from transformers import T5EncoderModel, AutoTokenizer

from meanaudio.ext.autoencoder import AutoEncoderModule
from meanaudio.ext.mel_converter import get_mel_converter
from meanaudio.model.utils.distributions import DiagonalGaussianDistribution
import laion_clap
import logging


def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model


class FeaturesUtils(nn.Module):

    def __init__(
        self,
        *,
        tod_vae_ckpt: Optional[str] = None,
        bigvgan_vocoder_ckpt: Optional[str] = None,
        enable_conditions: bool = True,
        encoder_name=Literal['clip', 't5', 't5_clap', 't5_clap_cat'], 
        mode=Literal['16k', '44k'],
        need_vae_encoder: bool = True,
    ):
        super().__init__()
        
        if enable_conditions:
            self.encoder_name = encoder_name
            if encoder_name == 'clip': 
                self.text_encoder = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384',
                                                            return_transform=False)
                self.clip_preprocess = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
                self.text_encoder = patch_clip(self.text_encoder)

                self.tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')  # same as 'ViT-H-14'
            elif encoder_name == 't5': 
                logging.info('FeatureUtils: Loading google/flan-t5-large ... ')   # root logger 
                self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
                self.text_encoder = T5EncoderModel.from_pretrained('google/flan-t5-large').eval()

            elif encoder_name == 't5_clap' or encoder_name == 't5_clap_cat':
                self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
                self.text_encoder = T5EncoderModel.from_pretrained('google/flan-t5-large').eval()
                self.laion_clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base').eval()
                self._clap_ckpt_path = "./weights/music_speech_audioset_epoch_15_esc_89.98.pt"  
                self.laion_clap_model.load_ckpt(self._clap_ckpt_path, verbose=False)

            else: 
                raise ValueError(f"Encoder {encoder_name} is not allowed, select from ['clip', 't5']")

        else:
            self.text_encoder = None
            self.tokenizer = None

        if tod_vae_ckpt is not None:
            self.mel_converter = get_mel_converter(mode)
            self.tod = AutoEncoderModule(vae_ckpt_path=tod_vae_ckpt,
                                         vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                                         mode=mode,
                                         need_vae_encoder=need_vae_encoder)
        else:
            self.tod = None

    def compile(self):
        if self.text_encoder is not None:
            self.text_encoder.encode_text = torch.compile(self.text_encoder.encode_text)  # ONLY for CLIP text encoder
        self.decode = torch.compile(self.decode)
        self.vocode = torch.compile(self.vocode)

    def train(self, mode: bool) -> None:
        return super().train(False)

    @torch.inference_mode()
    def encode_text(self, text: list[str]) -> torch.Tensor:
        assert self.text_encoder is not None, 'Text encoder is not loaded'
        assert self.tokenizer is not None, 'Tokenizer is not loaded'
        # x: (B, L)
        if self.encoder_name == 'clip': 
            tokens = self.tokenizer(text).to(self.device)
            text_features = self.text_encoder.encode_text(tokens, normalize=True)
        elif self.encoder_name == 't5': 
            tokens = self.tokenizer(
                text, 
                max_length=77, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            input_ids, attention_mask = tokens.input_ids.cuda(), tokens.attention_mask.cuda()
            text_features = self.text_encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )[0]
            text_features_c = text_features.mean(dim=1)
        elif self.encoder_name == 't5_clap' or self.encoder_name == 't5_clap_cat': 
            tokens = self.tokenizer(
                text, 
                max_length=77, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            input_ids, attention_mask = tokens.input_ids.cuda(), tokens.attention_mask.cuda()
            text_features = self.text_encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )[0]
            text_features_c = self.laion_clap_model.get_text_embedding(text, use_tensor=True)
            
            if self.encoder_name == 't5_clap_cat': 
                text_features_c = torch.cat([text_features.mean(dim=-2), text_features_c], dim=-1)
        return text_features, text_features_c

    @torch.inference_mode()
    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        assert self.tod is not None, 'VAE is not loaded'
        # x: (B * L)
        mel = self.mel_converter(x)
        dist = self.tod.encode(mel)

        return dist

    @torch.inference_mode()
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.vocode(mel)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.decode(z.transpose(1, 2))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


if __name__ == '__main__': 
    # features = FeaturesUtilsAT(
    #     tod_vae_ckpt='./ext_weights/v1-16.pth',
    #     bigvgan_vocoder_ckpt='./ext_weights/best_netG.pt',
    #     mode='16k',
    #     encoder_name='t5'
    # )
    # print(features)

    clap_ckpt = "./weights/music_speech_audioset_epoch_15_esc_89.98.pt"
    weights = torch.load(clap_ckpt, weights_only=False)
    print(weights.keys())
