import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from meanaudio.ext.rotary_embeddings import compute_rope_rotations
from meanaudio.model.embeddings import TimestepEmbedder
from meanaudio.model.low_level import MLP, ChannelLastConv1d, ConvMLP
from meanaudio.model.transformer_layers import (FinalBlock, JointBlock, MMDitSingleBlock)

log = logging.getLogger()


@dataclass
class PreprocessedConditions:
    text_f: torch.Tensor
    text_f_c: torch.Tensor


class FluxAudio(nn.Module):
    # Flux style latent transformer for TTA, single time step embedding

    def __init__(self,
                 *,
                 latent_dim: int,
                 text_dim: int,
                 text_c_dim: int, 
                 hidden_dim: int,
                 depth: int,
                 fused_depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 latent_seq_len: int,
                 text_seq_len: int = 77,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None,
                 empty_string_feat: Optional[torch.Tensor] = None,
                 empty_string_feat_c: Optional[torch.Tensor] = None,
                 use_rope: bool = False) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len
        self._text_seq_len = text_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.mm_depth = depth - fused_depth

        self.audio_input_proj = nn.Sequential(
            ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
            nn.SELU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
        )

        self.text_input_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            MLP(hidden_dim, hidden_dim * 4),
        )

        self.text_cond_proj = nn.Sequential(
            nn.Linear(text_c_dim, hidden_dim),
            MLP(hidden_dim, hidden_dim*4)
        )

        self.final_layer = FinalBlock(hidden_dim, latent_dim)  

        self.t_embed = TimestepEmbedder(hidden_dim,
                                        frequency_embedding_size=256,
                                        max_period=10000)
        
        self.joint_blocks = nn.ModuleList([
            JointBlock(hidden_dim,
                         num_heads,
                         mlp_ratio=mlp_ratio,
                         pre_only=(i == depth - fused_depth - 1)) for i in range(depth - fused_depth)  # last layer is pre-only (only appllied to text and vision)
        ])

        self.fused_blocks = nn.ModuleList([
            MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=3, padding=1)
            for i in range(fused_depth)
        ])

        if latent_mean is None:
            # these values are not meant to be used
            # if you don't provide mean/std here, we should load them later from a checkpoint
            assert latent_std is None
            latent_mean = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
            latent_std = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
        else:
            assert latent_std is not None
            assert latent_mean.numel() == latent_dim, f'{latent_mean.numel()=} != {latent_dim=}'

        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))
        if empty_string_feat_c is None: 
            empty_string_feat_c = torch.zeros((text_c_dim))

        assert empty_string_feat.shape[-1] == text_dim, f'{empty_string_feat.shape[-1]} == {text_dim}'
        assert empty_string_feat_c.shape[-1] == text_c_dim, f'{empty_string_feat_c.shape[-1]} == {text_c_dim}'

        self.latent_mean = nn.Parameter(latent_mean.view(1, 1, -1), requires_grad=False)  # (1, 1, d)
        self.latent_std = nn.Parameter(latent_std.view(1, 1, -1), requires_grad=False)   # (1, 1, d)

        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False) 
        self.empty_string_feat_c = nn.Parameter(empty_string_feat_c, requires_grad=False)


        self.initialize_weights()
        if self.use_rope: 
            log.info("Network: Enabling RoPE embeddings")
            self.initialize_rotations()
        else: 
            log.info("Network: RoPE embedding disabled")
            self.latent_rot = None
            self.text_rot = None  

    def initialize_rotations(self):
        base_freq = 1.0
        latent_rot = compute_rope_rotations(self._latent_seq_len,
                                            self.hidden_dim // self.num_heads,
                                            10000,
                                            freq_scaling=base_freq,
                                            device=self.device)
        text_rot = compute_rope_rotations(self._text_seq_len,
                                          self.hidden_dim // self.num_heads,
                                          10000,
                                          freq_scaling=base_freq, 
                                          device=self.device)

        self.latent_rot = nn.Buffer(latent_rot, persistent=False)  # will not be saved into state dict
        self.text_rot = nn.Buffer(text_rot, persistent=False)

    def update_seq_lengths(self, latent_seq_len: int) -> None: 
        self._latent_seq_len = latent_seq_len
        if self.use_rope: 
            self.initialize_rotations()   # after changing seq_len we need to re-initialize RoPE to match new seq_len

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].weight, 0)  # the linear layer -> 6 coefficients
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # return (x - self.latent_mean) / self.latent_std
        return x.sub_(self.latent_mean).div_(self.latent_std)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        # return x * self.latent_std + self.latent_mean
        return x.mul_(self.latent_std).add_(self.latent_mean)

    def preprocess_conditions(self, text_f: torch.Tensor, text_f_c: torch.Tensor) -> PreprocessedConditions:  
        """
        cache computations that do not depend on the latent/time step
        i.e., the features are reused over steps during inference
        """
        assert text_f.shape[1] == self._text_seq_len, f'{text_f.shape=} {self._text_seq_len=}'

        bs = text_f.shape[0]

        # get global and local text features
        # NOTE here the order of projection has been changed so global and local features are projected seperately 
        text_f_c = self.text_cond_proj(text_f_c)  # (B, D)
        text_f = self.text_input_proj(text_f)  # (B, VN, D)

        return PreprocessedConditions(text_f=text_f,
                                        text_f_c=text_f_c)

    def predict_flow(self, latent: torch.Tensor, t: torch.Tensor,
                     conditions: PreprocessedConditions) -> torch.Tensor:
        """
        for non-cacheable computations
        """
        assert latent.shape[1] == self._latent_seq_len, f'{latent.shape=} {self._latent_seq_len=}'

        text_f = conditions.text_f
        text_f_c = conditions.text_f_c

        latent = self.audio_input_proj(latent)  # (B, N, D)

        global_c = self.t_embed(t).unsqueeze(1) + text_f_c.unsqueeze(1)  # (B, 1, D)

        extended_c = global_c  # extended_c: Latent_c, global_c: Text_c

        for block in self.joint_blocks:
            latent, text_f = block(latent, text_f, global_c, extended_c, self.latent_rot, self.text_rot)  # (B, N, D)

        for block in self.fused_blocks:
            latent = block(latent, extended_c, self.latent_rot)

        flow = self.final_layer(latent, extended_c)  # (B, N, out_dim), remove t
        return flow

    def forward(self, latent: torch.Tensor, text_f: torch.Tensor, text_f_c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, N, C) 
        text_f: (B, T, D)
        t: (B,)
        """
        conditions = self.preprocess_conditions(text_f, text_f_c)  # cachable operations 
        flow = self.predict_flow(latent, t, conditions)  # non-cachable operations
        return flow

    def get_empty_string_sequence(self, bs: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1), \
            self.empty_string_feat_c.unsqueeze(0).expand(bs, -1)  # (b, d)

    def get_empty_conditions(
            self,
            bs: int,
            *,
            negative_text_features: Optional[torch.Tensor] = None) -> PreprocessedConditions:
        if negative_text_features is not None:  
            empty_string_feat, empty_string_feat_c = negative_text_features  
        else:
            empty_string_feat, empty_string_feat_c = self.get_empty_string_sequence(1)

        conditions = self.preprocess_conditions(empty_string_feat,
                                                empty_string_feat_c)  # use encoder's empty features
        
        if negative_text_features is None:
            conditions.text_f = conditions.text_f.expand(bs, -1, -1)
            
            conditions.text_f_c = conditions.text_f_c.expand(bs, -1)

        return conditions

    def ode_wrapper(self, t: torch.Tensor, latent: torch.Tensor, conditions: PreprocessedConditions,
                    empty_conditions: PreprocessedConditions, cfg_strength: float) -> torch.Tensor:
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)

        if cfg_strength < 1.0:
            return self.predict_flow(latent, t, conditions)
        else:
            return (cfg_strength * self.predict_flow(latent, t, conditions) +
                    (1 - cfg_strength) * self.predict_flow(latent, t, empty_conditions))

    def load_weights(self, src_dict) -> None:
        if 't_embed.freqs' in src_dict:
            del src_dict['t_embed.freqs']
        if 'latent_rot' in src_dict:
            del src_dict['latent_rot']
        if 'text_rot' in src_dict:
            del src_dict['text_rot']

        if 'empty_string_feat_c' not in src_dict.keys():  # FIXME: issue of version mismatch here
            src_dict['empty_string_feat_c'] = src_dict['empty_string_feat'].mean(dim=0)
        self.load_state_dict(src_dict, strict=True)

    @property
    def device(self) -> torch.device:
        return self.latent_mean.device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len

        
class MeanAudio(nn.Module):
    # Flux style latent transformer for TTA, dual time step embedding

    def __init__(self,
                 *,
                 latent_dim: int,
                 text_dim: int,
                 text_c_dim: int, 
                 hidden_dim: int,
                 depth: int,
                 fused_depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 latent_seq_len: int,
                 text_seq_len: int = 77,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None,
                 empty_string_feat: Optional[torch.Tensor] = None,
                 empty_string_feat_c: Optional[torch.Tensor] = None,
                 use_rope: bool = False) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len
        self._text_seq_len = text_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.mm_depth = depth - fused_depth

        self.audio_input_proj = nn.Sequential(
            ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
            nn.SELU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
        )

        self.text_input_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            MLP(hidden_dim, hidden_dim * 4),
        )

        self.text_cond_proj = nn.Sequential(
            nn.Linear(text_c_dim, hidden_dim),
            MLP(hidden_dim, hidden_dim*4)
        )

        self.final_layer = FinalBlock(hidden_dim, latent_dim)  

        self.t_embed = TimestepEmbedder(hidden_dim,
                                        frequency_embedding_size=256,
                                        max_period=10000)
        #add
        self.r_embed = TimestepEmbedder(hidden_dim,
                                        frequency_embedding_size=256,
                                        max_period=10000)
        self.joint_blocks = nn.ModuleList([
            JointBlock(hidden_dim,
                         num_heads,
                         mlp_ratio=mlp_ratio,
                         pre_only=(i == depth - fused_depth - 1)) for i in range(depth - fused_depth)  # last layer is pre-only (only appllied to text and vision)
        ])

        self.fused_blocks = nn.ModuleList([
            MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=3, padding=1)
            for i in range(fused_depth)
        ])

        if latent_mean is None:
            # these values are not meant to be used
            # if you don't provide mean/std here, we should load them later from a checkpoint
            assert latent_std is None
            latent_mean = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
            latent_std = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
        else:
            assert latent_std is not None
            assert latent_mean.numel() == latent_dim, f'{latent_mean.numel()=} != {latent_dim=}'

        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))
        if empty_string_feat_c is None: 
            empty_string_feat_c = torch.zeros((text_c_dim))

        assert empty_string_feat.shape[-1] == text_dim, f'{empty_string_feat.shape[-1]} == {text_dim}'
        assert empty_string_feat_c.shape[-1] == text_c_dim, f'{empty_string_feat_c.shape[-1]} == {text_c_dim}'

        self.latent_mean = nn.Parameter(latent_mean.view(1, 1, -1), requires_grad=False)  # (1, 1, d)
        self.latent_std = nn.Parameter(latent_std.view(1, 1, -1), requires_grad=False)   # (1, 1, d)

        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False) 
        self.empty_string_feat_c = nn.Parameter(empty_string_feat_c, requires_grad=False)


        self.initialize_weights()
        if self.use_rope: 
            log.info("Network: Enabling RoPE embeddings")
            self.initialize_rotations()
        else: 
            log.info("Network: RoPE embedding disabled")
            self.latent_rot = None
            self.text_rot = None  

    def initialize_rotations(self):
        base_freq = 1.0
        latent_rot = compute_rope_rotations(self._latent_seq_len,
                                            self.hidden_dim // self.num_heads,
                                            10000,
                                            freq_scaling=base_freq,
                                            device=self.device)
        text_rot = compute_rope_rotations(self._text_seq_len,
                                          self.hidden_dim // self.num_heads,
                                          10000,
                                          freq_scaling=base_freq, 
                                          device=self.device)

        self.latent_rot = nn.Buffer(latent_rot, persistent=False)  # will not be saved into state dict
        self.text_rot = nn.Buffer(text_rot, persistent=False)

    def update_seq_lengths(self, latent_seq_len: int) -> None: 
        self._latent_seq_len = latent_seq_len
        if self.use_rope: 
            self.initialize_rotations()   # after changing seq_len we need to re-initialize RoPE to match new seq_len

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].weight, 0)  # the linear layer -> 6 coefficients
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # return (x - self.latent_mean) / self.latent_std
        return x.sub_(self.latent_mean).div_(self.latent_std)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        # return x * self.latent_std + self.latent_mean
        return x.mul_(self.latent_std).add_(self.latent_mean)

    def preprocess_conditions(self, text_f: torch.Tensor, text_f_c: torch.Tensor) -> PreprocessedConditions:  
        """
        cache computations that do not depend on the latent/time step
        i.e., the features are reused over steps during inference
        """
        assert text_f.shape[1] == self._text_seq_len, f'{text_f.shape=} {self._text_seq_len=}'

        bs = text_f.shape[0]

        # get global and local text features
        # NOTE here the order of projection has been changed so global and local features are projected seperately 
        text_f_c = self.text_cond_proj(text_f_c)  # (B, D)
        text_f = self.text_input_proj(text_f)  # (B, VN, D)

        return PreprocessedConditions(text_f=text_f,
                                        text_f_c=text_f_c)

    def predict_flow(self, latent: torch.Tensor, t: torch.Tensor,r: torch.Tensor,#need r<t
                     conditions: PreprocessedConditions) -> torch.Tensor:
        """
        for non-cacheable computations
        """
        #assert r<=t,"r should smaller than t"
       
        assert latent.shape[1] == self._latent_seq_len, f'{latent.shape=} {self._latent_seq_len=}'

        text_f = conditions.text_f
        text_f_c = conditions.text_f_c

        latent = self.audio_input_proj(latent)  # (B, N, D)
        #easy try:same embed
        global_c = self.t_embed(t).unsqueeze(1) + self.r_embed(r).unsqueeze(1) + text_f_c.unsqueeze(1)  # (B, 1, D)

        extended_c = global_c  # !TODO add fine-grained control

        for block in self.joint_blocks:
            latent, text_f = block(latent, text_f, global_c, extended_c, self.latent_rot, self.text_rot)  # (B, N, D)

        for block in self.fused_blocks:
            latent = block(latent, extended_c, self.latent_rot)

        flow = self.final_layer(latent, extended_c)  # (B, N, out_dim), remove t
        return flow

    def forward(self, latent: torch.Tensor, text_f: torch.Tensor, text_f_c: torch.Tensor, r: torch.Tensor,t: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, N, C) 
        text_f: (B, T, D)
        text_f_c
        r: (B,)
        t:(B,)
        """
        #print("2")
        
        conditions = self.preprocess_conditions(text_f, text_f_c)  # cachable operations 
        #print(conditions)
        flow = self.predict_flow(latent, t,r, conditions)  # non-cachable operations
        return flow

    def get_empty_string_sequence(self, bs: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1), \
            self.empty_string_feat_c.unsqueeze(0).expand(bs, -1)  # (b, d)

    def get_empty_conditions(
            self,
            bs: int,
            *,
            negative_text_features: Optional[torch.Tensor] = None) -> PreprocessedConditions:
        if negative_text_features is not None:  
            empty_string_feat, empty_string_feat_c = negative_text_features  
        else:
            empty_string_feat, empty_string_feat_c = self.get_empty_string_sequence(1)

        conditions = self.preprocess_conditions(empty_string_feat,
                                                empty_string_feat_c)  # use encoder's empty features
        if negative_text_features is None:
            conditions.text_f = conditions.text_f.expand(bs, -1, -1)
            
            conditions.text_f_c = conditions.text_f_c.expand(bs, -1)

        return conditions

    def ode_wrapper(self, t: torch.Tensor, r: torch.Tensor, latent: torch.Tensor, conditions: PreprocessedConditions,
                    empty_conditions: PreprocessedConditions, cfg_strength: float) -> torch.Tensor:
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        r = r * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        #(r)
        if cfg_strength < 1.0:
            return self.predict_flow(latent, t,r, conditions)
        else:
            return (cfg_strength * self.predict_flow(latent, t,r, conditions) +
                    (1 - cfg_strength) * self.predict_flow(latent, t,r, empty_conditions))
    

    def load_weights(self, src_dict) -> None:
        def remove_prefix(storage):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in storage.items():
                name = k.replace("ema_model.", "")
                new_state_dict[name] = v

            return new_state_dict
 
        src_dict=remove_prefix(src_dict)
        if 't_embed.freqs' in src_dict:
            del src_dict['t_embed.freqs']
        if 'r_embed.freqs' in src_dict:
            del src_dict['r_embed.freqs']
        if 'latent_rot' in src_dict:
            del src_dict['latent_rot']
        if 'text_rot' in src_dict:
            del src_dict['text_rot']

        if 'empty_string_feat_c' not in src_dict.keys():  # FIXME: issue of version mismatch here
            src_dict['empty_string_feat_c'] = src_dict['empty_string_feat'].mean(dim=0)
        if '_extra_state' in src_dict:
            del src_dict['_extra_state']
        self.load_state_dict(src_dict, strict=True)

    @property
    def device(self) -> torch.device:
        return self.latent_mean.device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len
    

def fluxaudio_fm(**kwargs) -> FluxAudio: 
    num_heads = 7
    return FluxAudio(latent_dim=20,
                     text_dim=1024,
                     hidden_dim=64 * num_heads,
                     depth=12,
                     fused_depth=8,
                     num_heads=num_heads,
                     latent_seq_len=312,  # for 10s audio
                     **kwargs)


def fluxaudio_s_full(**kwargs) -> FluxAudio: 
    num_heads = 7
    return FluxAudio(latent_dim=40,
                     text_dim=1024,
                     hidden_dim=64 * num_heads,
                     depth=12,
                     fused_depth=8,
                     num_heads=num_heads,
                     latent_seq_len=430,  # for 10s audio
                     **kwargs)


def fluxaudio_m_full(**kwargs) -> FluxAudio: 
    num_heads = 14
    return FluxAudio(latent_dim=40,
                     text_dim=1024,
                     hidden_dim=64 * num_heads,
                     depth=12,
                     fused_depth=8,
                     num_heads=num_heads,
                     latent_seq_len=430,  # for 10s audio
                     **kwargs)


def fluxaudio_m_full_30(**kwargs) -> FluxAudio: 
    num_heads = 14
    return FluxAudio(latent_dim=40,
                     text_dim=1024,
                     hidden_dim=64 * num_heads,
                     depth=12,
                     fused_depth=8,
                     num_heads=num_heads,
                     latent_seq_len=1291,  # for 30s audio
                     **kwargs)


def meanaudio_mf(**kwargs) -> MeanAudio:  
    num_heads = 7
    return MeanAudio(latent_dim=20,
                     text_dim=1024,
                     hidden_dim=64 * num_heads,
                     depth=12,
                     fused_depth=8,
                     num_heads=num_heads,
                     latent_seq_len=312,  # for 10s audio
                     **kwargs)

def meanaudio_large(**kwargs) -> MeanAudio:  
    num_heads = 14
    return MeanAudio(latent_dim=20,
                     text_dim=1024,
                     hidden_dim=64 * num_heads,
                     depth=12,
                     fused_depth=8,
                     num_heads=num_heads,
                     latent_seq_len=312,  # for 10s audio
                     **kwargs)

def meanaudio_m_full(**kwargs) -> MeanAudio: 
    num_heads = 14
    return MeanAudio(latent_dim=40,
                     text_dim=1024,
                     hidden_dim=64 * num_heads,
                     depth=12,
                     fused_depth=8,
                     num_heads=num_heads,
                     latent_seq_len=430,  # for 10s audio
                     **kwargs)


def get_mean_audio(name: str, **kwargs) -> MeanAudio:
    if name == 'meanaudio_mf':  # !TODO change name here for 16khz models
        return meanaudio_mf(**kwargs)
    if name == 'meanaudio_large': 
        return meanaudio_large(**kwargs)
    if name == 'fluxaudio_fm': 
        return fluxaudio_fm(**kwargs)
    if name == 'fluxaudio_m_full': 
        return fluxaudio_m_full(**kwargs)
    if name == 'fluxaudio_m_full_30': 
        return fluxaudio_m_full_30(**kwargs)
    if name == 'fluxaudio_s_full': 
        return fluxaudio_s_full(**kwargs)
    if name == 'meanaudio_m_full': 
        return meanaudio_m_full(**kwargs)

    raise ValueError(f'Unknown model name: {name}')


if __name__ == '__main__':
    from meanaudio.model.utils.sample_utils import log_normal_sample

    logging.basicConfig(
        level=logging.INFO,  
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # logging.FileHandler("main.log"), 
            logging.StreamHandler()          
        ]
    )

    network: MeanAudio = get_mean_audio('meanaudio_mf', 
                                        use_rope=False, 
                                        text_c_dim=512)

    x = torch.randn(256, 312, 20)
    print(x.shape)
    print('Finish')

