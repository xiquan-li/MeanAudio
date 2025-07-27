from typing import Optional

from nitrous_ema import PostHocEMA
from omegaconf import DictConfig

from meanaudio.model.networks import get_mean_audio


def synthesize_ema(cfg: DictConfig, sigma: float, step: Optional[int]):
    if not cfg.use_repa: 
        # !NOTE here we need to re-define model so be careful of passed arguments (need to be coherent with before)
        vae = get_mean_audio(cfg.model, text_c_dim=cfg.data_dim.text_c_dim)  
    else: 
        vae = get_mean_audio(cfg.model, text_c_dim=cfg.data_dim.text_c_dim, 
                             repa_layer=cfg.repa_layer,   # repa config
                             z_dim=cfg.z_dim,
                             z_len=cfg.z_len, 
                             ufo_objective=cfg.ufo_objective,
                             proj_version=cfg.repa_version)
    emas = PostHocEMA(vae,
                      sigma_rels=cfg.ema.sigma_rels,
                      update_every=cfg.ema.update_every,
                      checkpoint_every_num_steps=cfg.ema.checkpoint_every,
                      checkpoint_folder=cfg.ema.checkpoint_folder)

    synthesized_ema = emas.synthesize_ema_model(sigma_rel=sigma, step=step, device='cpu')
    state_dict = synthesized_ema.ema_model.state_dict()
    return state_dict
