import os

import torch
import torch.distributed
from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed
import torch.optim as optim
from av_bench.evaluate import evaluate
from av_bench.extract import extract
from nitrous_ema import PostHocEMA
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from meanaudio.model.flow_matching import FlowMatching
from meanaudio.model.mean_flow import MeanFlow
from meanaudio.model.networks import get_mean_audio
from meanaudio.model.sequence_config import CONFIG_16K, CONFIG_44K
from meanaudio.model.utils.features_utils import FeaturesUtils
from meanaudio.model.utils.parameter_groups import get_parameter_groups
from meanaudio.utils.dist_utils import (info_if_rank_zero, local_rank, string_if_rank_zero)
from meanaudio.utils.log_integrator import Integrator
from meanaudio.utils.logger import TensorboardLogger
from meanaudio.utils.time_estimator import PartialTimeEstimator, TimeEstimator
import wandb

class RunnerMeanFlow:

    def __init__(self,
                 cfg: DictConfig,
                 log: TensorboardLogger,
                 run_path: Union[str, Path],
                 for_training: bool = True,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None):
        self.exp_id = cfg.exp_id
        self.use_amp = cfg.amp
        self.enable_grad_scaler = cfg.enable_grad_scaler
        self.for_training = for_training
        self.cfg = cfg
        self.use_wandb = cfg.get("use_wandb", False)

        if self.use_wandb and local_rank == 0: 
            wandb.init(
                project = "MeanAudio", 
                name = cfg.exp_id, 
                settings=wandb.Settings(init_timeout=120),
                # config = cfg
            )

        # sequence config
        self.seq_cfg = CONFIG_16K  
        mode = '16k'

        self.sample_rate = self.seq_cfg.sampling_rate
        self.duration_sec = self.seq_cfg.duration

        # model
        if cfg['text_encoder_name'] == 'clip': 
            empty_string_feat = torch.load('./weights/empty_string.pth', weights_only=True)[0] 
            log.info('Loading empty string feature from ./weights/empty_string.pth for CLIP ...')
        elif cfg['text_encoder_name'] == 't5': 
            empty_string_feat = torch.load('./weights/empty_string_t5.pth', weights_only=True)[0] 
            empty_string_feat_c = torch.load('./weights/empty_string_t5_c.pth',  weights_only=True)[0]
            log.info('Loading empty string feature from ./weights/empty_string_t5.pth and ./weights/empty_string_t5_c.pth for T5')
        elif cfg['text_encoder_name'] == 't5_clap': 
            empty_string_feat = torch.load('./weights/empty_string_t5.pth', weights_only=True)[0]  # abandon the first (btz) dim. 
            empty_string_feat_c = torch.load('./weights/empty_string_clap_c.pth',  weights_only=True)[0]
            log.info('Loading empty string feature from ./weights/empty_string_t5.pth and ./weights/empty_string_clap_c.pth for T5 and CLAP')
        elif cfg['text_encoder_name'] == 't5_clap_cat':
            empty_string_feat = torch.load('./weights/empty_string_t5.pth', weights_only=True)[0]  # abandon the first (btz) dim. 
            empty_string_feat_c = torch.load('./weights/empty_string_clap_c.pth',  weights_only=True)[0]
            empty_string_feat_c = torch.cat([empty_string_feat.mean(dim=-2), empty_string_feat_c], dim=-1)
            log.info('Loading empty string feature from ./weights/empty_string_t5.pth and ./weights/empty_string_clap_c.pth for T5 and CLAP, concating condition features ... ')
        else: 
            raise NotImplementedError(f'Encoder {cfg["text_encoder_name"]} not implemented')
        self.network = DDP(get_mean_audio(cfg.model,  # get the model based on base_config.yaml
                                          latent_mean=latent_mean,  # mean and std calculated from the dataset 
                                          latent_std=latent_std,
                                          empty_string_feat=empty_string_feat,
                                          empty_string_feat_c=empty_string_feat_c,  
                                          use_rope=cfg.use_rope,
                                          text_c_dim=cfg.data_dim.text_c_dim).cuda(),
                           device_ids=[local_rank],
                           broadcast_buffers=False)
        if cfg.compile:
            self.train_fn = torch.compile(self.train_fn)
            self.val_fn = torch.compile(self.val_fn)

        self.mf = MeanFlow()
        
        # ema profile
        if for_training and cfg.ema.enable and local_rank == 0:
            self.ema = PostHocEMA(self.network.module,
                                  sigma_rels=cfg.ema.sigma_rels,
                                  update_every=cfg.ema.update_every,
                                  checkpoint_every_num_steps=cfg.ema.checkpoint_every,
                                  checkpoint_folder=cfg.ema.checkpoint_folder,
                                  step_size_correction=True).cuda()
            self.ema_start = cfg.ema.start
        else:
            self.ema = None

        self.rng = torch.Generator(device='cuda')
        self.rng.manual_seed(cfg['seed'] + local_rank)

        # setting up feature extractors and VAEs
        text_encoder_name = cfg['text_encoder_name']
        if mode == '16k':  
            self.features = FeaturesUtils(
                tod_vae_ckpt=cfg['vae_16k_ckpt'],
                bigvgan_vocoder_ckpt=cfg['bigvgan_vocoder_ckpt'],
                encoder_name=text_encoder_name,
                enable_conditions=True,
                mode=mode,
                need_vae_encoder=False,
            )
        elif mode == '44k':
            self.features = FeaturesUtils(
                tod_vae_ckpt=cfg['vae_44k_ckpt'],
                encoder_name=text_encoder_name, 
                enable_conditions=True,
                mode=mode,
                need_vae_encoder=False,
            )
        self.features = self.features.cuda().eval()

        if cfg.compile:
            self.features.compile()

        # TODO: change these parameters compatible with meanflow
        self.log_normal_sampling_mean = cfg.sampling.mean
        self.log_normal_sampling_scale = cfg.sampling.scale
        self.null_condition_probability = cfg.null_condition_probability
        self.cfg_strength = cfg.cfg_strength

        # setting up logging
        self.log = log
        self.run_path = Path(run_path)
        string_if_rank_zero(self.log, 'model_size',
                            f'{sum([param.nelement() for param in self.network.parameters()])}')
        string_if_rank_zero(
            self.log, 'number_of_parameters_that_require_gradient: ',
            str(
                sum([
                    param.nelement()
                    for param in filter(lambda p: p.requires_grad, self.network.parameters())
                ])))
        info_if_rank_zero(self.log, 'torch version: ' + torch.__version__)
        self.train_integrator = Integrator(self.log, distributed=True)
        self.val_integrator = Integrator(self.log, distributed=True)

        # setting up optimizer and loss
        if for_training:
            self.enter_train()
            parameter_groups = get_parameter_groups(self.network, cfg, print_log=(local_rank == 0))
            self.optimizer = optim.AdamW(parameter_groups,
                                         lr=cfg['learning_rate'],
                                         weight_decay=cfg['weight_decay'],
                                         betas=[0.9, 0.95],
                                         eps=1e-6 if self.use_amp else 1e-8,
                                         fused=True)
            if self.enable_grad_scaler:
                self.scaler = torch.amp.GradScaler(init_scale=2048)
            self.clip_grad_norm = cfg['clip_grad_norm']

            # linearly warmup learning rate
            linear_warmup_steps = cfg['linear_warmup_steps']

            def warmup(currrent_step: int):
                return (currrent_step + 1) / (linear_warmup_steps + 1)

            warmup_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup)

            # setting up learning rate scheduler
            if cfg['lr_schedule'] == 'constant':
                next_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1)
            elif cfg['lr_schedule'] == 'poly':
                total_num_iter = cfg['iterations']
                next_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                             lr_lambda=lambda x:
                                                             (1 - (x / total_num_iter))**0.9)
            elif cfg['lr_schedule'] == 'step':
                total_num_iter = cfg['num_iterations']
                lr_schedule_steps = [int(0.8 * total_num_iter), int(0.9 * total_num_iter)]
                log.info(f'Assigning lr steps: {lr_schedule_steps}')
                next_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                lr_schedule_steps,
                                                                cfg['lr_schedule_gamma'])
            else:
                raise NotImplementedError

            self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer,
                                                             [warmup_scheduler, next_scheduler],
                                                             [linear_warmup_steps])

            # Logging info
            self.log_text_interval = cfg['log_text_interval']
            self.log_extra_interval = cfg['log_extra_interval']
            self.save_weights_interval = cfg['save_weights_interval']
            self.save_checkpoint_interval = cfg['save_checkpoint_interval']
            self.save_copy_iterations = cfg['save_copy_iterations']
            self.num_iterations = cfg['num_iterations']

            # update() is called when we log metrics, within the logger
            self.log.batch_timer = TimeEstimator(self.num_iterations, self.log_text_interval)
            # update() is called every iteration, in this script
            self.log.data_timer = PartialTimeEstimator(self.num_iterations, 1, ema_alpha=0.9)
        else:
            self.enter_val()

    def train_fn( 
        self,
        text_f: torch.Tensor,
        text_f_c: torch.Tensor, 
        a_mean: torch.Tensor,
        a_std: torch.Tensor,
        # it: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample
        a_randn = torch.empty_like(a_mean).normal_(generator=self.rng)
        x1 = a_mean + a_std * a_randn
        bs = x1.shape[0]  # batch_size * seq_len * num_channels

        x1 = self.network.module.normalize(x1)

        text_f_undrop = text_f
        text_f_c_undrop = text_f_c
        samples = torch.rand(bs, device=x1.device, generator=self.rng)
        null_text = (samples < self.null_condition_probability)
        text_f[null_text] = self.network.module.empty_string_feat

        null_text_c = (samples < self.null_condition_probability)  # here we do null condition together
        text_f_c[null_text_c] = self.network.module.empty_string_feat_c  
        loss, r, t = self.mf.loss(self.network,
                                  x1,
                                  text_f,
                                  text_f_c,
                                  text_f_undrop,
                                  text_f_c_undrop,
                                  self.network.module.empty_string_feat,
                                  self.network.module.empty_string_feat_c)
        mean_loss = loss.mean()
        return x1, loss, mean_loss, t, r

    def val_fn(
        self,
        text_f: torch.Tensor,
        text_f_c: torch.Tensor, 
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = x1.shape[0]  # batch_size * seq_len * num_channels
        # normalize the latents
        x1 = self.network.module.normalize(x1)

        text_f_undrop = text_f
        text_f_c_undrop = text_f_c
        samples = torch.rand(bs, device=x1.device, generator=self.rng)
        null_text = (samples < self.null_condition_probability)
        text_f[null_text] = self.network.module.empty_string_feat

        null_text_c = (samples < self.null_condition_probability)
        text_f_c[null_text_c] = self.network.module.empty_string_feat_c
        
        loss, r, t = self.mf.loss(self.network,
                                  x1,
                                  text_f,
                                  text_f_c,
                                  text_f_undrop,
                                  text_f_c_undrop,
                                  self.network.module.empty_string_feat,
                                  self.network.module.empty_string_feat_c)
        mean_loss = loss.mean()
        return loss, mean_loss, t, r

    def train_pass(self, data, it: int = 0):

        if not self.for_training:
            raise ValueError('train_pass() should not be called when not training.')

        self.enter_train()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            text_f = data['text_features'].cuda(non_blocking=True)
            text_f_c = data['text_features_c'].cuda(non_blocking=True)
            a_mean = data['a_mean'].cuda(non_blocking=True)
            a_std = data['a_std'].cuda(non_blocking=True)
            
            self.log.data_timer.end()
            if it % self.log_extra_interval == 0:
                unmasked_text_f = text_f.clone()
                unmasked_text_f_c = text_f_c.clone()
            #with torch.amp.autocast('cuda', enabled=False):
            x1, loss, mean_loss, t,r = self.train_fn(text_f, text_f_c, a_mean, a_std)
           
            self.train_integrator.add_dict({'loss': mean_loss})

        if it % self.log_text_interval == 0 and it != 0:
            lr = self.scheduler.get_last_lr()[0]
            self.train_integrator.add_scalar('lr', lr)
            self.train_integrator.add_binned_tensor('binned_loss', loss, t)
            self.train_integrator.finalize('train', it)
            self.train_integrator.reset_except_hooks()

            if self.use_wandb and local_rank == 0: 
                wandb.log(
                    {
                        "lr": lr,
                        "train/loss": mean_loss.detach().float()
                    },
                    step=it  # explicitly x-axis it
                )

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.enable_grad_scaler:
            self.scaler.scale(mean_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                       self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            mean_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                       self.clip_grad_norm)
            self.optimizer.step()

        if self.ema is not None and it >= self.ema_start:
            self.ema.update()
        self.scheduler.step()
        self.integrator.add_scalar('grad_norm', grad_norm)

        self.enter_val()
        with torch.amp.autocast('cuda', enabled=self.use_amp,
                                dtype=torch.bfloat16), torch.inference_mode():
            try:
                if it % self.log_extra_interval == 0:
                    # save GT audio
                    # unnormalize the latents
                    x1 = self.network.module.unnormalize(x1[0:1])
                    mel = self.features.decode(x1)
                    audio = self.features.vocode(mel).cpu()[0]  # 1 * num_samples
                    self.log.log_spectrogram('train', f'spec-gt-r{local_rank}', mel.cpu()[0], it)
                    self.log.log_audio('train',
                                       f'audio-gt-r{local_rank}',
                                       audio,
                                       it,
                                       sample_rate=self.sample_rate)

                    # save audio from sampling
                    x0 = torch.empty_like(x1[0:1]).normal_(generator=self.rng)
                    text_f = unmasked_text_f[0:1]
                    text_f_c = unmasked_text_f_c[0:1]  # the first element with same sequence
                    conditions = self.network.module.preprocess_conditions(text_f, text_f_c)
                    empty_conditions = self.network.module.get_empty_conditions(x0.shape[0])
                    cfg_ode_wrapper = lambda t,r,x: self.network.module.ode_wrapper(
                        t,r,x, conditions, empty_conditions, self.cfg_strength)
                    x1_hat = self.mf.to_data(cfg_ode_wrapper, x0)
                    x1_hat = self.network.module.unnormalize(x1_hat)
                    mel = self.features.decode(x1_hat)
                    audio = self.features.vocode(mel).cpu()[0]
                    self.log.log_spectrogram('train', f'spec-r{local_rank}', mel.cpu()[0], it)
                    self.log.log_audio('train',
                                       f'audio-r{local_rank}',
                                       audio,
                                       it,
                                       sample_rate=self.sample_rate)
            except Exception as e:
                self.log.warning(f'Error in extra logging: {e}')
                if self.cfg.debug:
                    raise

        # Save network weights and checkpoint if needed
        save_copy = it in self.save_copy_iterations

        if (it % self.save_weights_interval == 0 and it != 0) or save_copy:
            self.save_weights(it)

        if it % self.save_checkpoint_interval == 0 and it != 0:
            self.save_checkpoint(it, save_copy=save_copy)

        self.log.data_timer.start()

    @torch.inference_mode()
    def validation_pass(self, data, it: int = 0):
        self.enter_val()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            text_f = data['text_features'].cuda(non_blocking=True)
            text_f_c = data['text_features_c'].cuda(non_blocking=True)
            a_mean = data['a_mean'].cuda(non_blocking=True)
            a_std = data['a_std'].cuda(non_blocking=True)  
            a_randn = torch.empty_like(a_mean).normal_(generator=self.rng)
            x1 = a_mean + a_std * a_randn  # differs from train_pass is that validation_pass pass x1 into val_fn
        
            self.log.data_timer.end()
            # with torch.amp.autocast('cuda', enabled=False):
            loss, mean_loss, t, r = self.val_fn(text_f.clone(), text_f_c.clone(), x1)

            self.val_integrator.add_binned_tensor('binned_loss', loss, t)
            self.val_integrator.add_dict({'loss': mean_loss})

        self.log.data_timer.start()
        return mean_loss.detach().float()

    @torch.inference_mode()
    def inference_pass(self,
                       data,   # batch data
                       it: int,
                       data_cfg: DictConfig,
                       *,
                       save_eval: bool = True) -> Path:
        self.enter_val()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            text_f = data['text_features'].cuda(non_blocking=True)
            text_f_c = data['text_features_c'].cuda(non_blocking=True)
            a_mean = data['a_mean'].cuda(non_blocking=True)  # for the shape only

            # sample
            x0 = torch.empty_like(a_mean).normal_(generator=self.rng)
            conditions = self.network.module.preprocess_conditions(text_f, text_f_c)
            empty_conditions = self.network.module.get_empty_conditions(x0.shape[0])
            cfg_ode_wrapper = lambda t, r, x: self.network.module.ode_wrapper(
                t, r, x, conditions, empty_conditions, self.cfg_strength)
            x1_hat = self.mf.to_data(cfg_ode_wrapper, x0)
            x1_hat = self.network.module.unnormalize(x1_hat)
            mel = self.features.decode(x1_hat)
            audio = self.features.vocode(mel).cpu()  # (btz, n_samples)
            for i in range(audio.shape[0]):
                audio_id = data['id'][i]

                if data_cfg.output_subdir is not None:
                    # validation
                    if save_eval:
                        iter_naming = f'{it:09d}'
                    else:
                        iter_naming = 'val-cache'
                    audio_dir = self.log.log_audio(iter_naming,  # write audios
                                                   f'{audio_id}',
                                                   audio[i],
                                                   it=None,
                                                   sample_rate=self.sample_rate,
                                                   subdir=Path(data_cfg.output_subdir)) 
                else:
                    # full test set, usually
                    audio_dir = self.log.log_audio(f'{data_cfg.tag}-sampled',
                                                   f'{audio_id}',
                                                   audio[i],
                                                   it=None,
                                                   sample_rate=self.sample_rate)  
        del text_f, text_f_c, a_mean
        torch.cuda.empty_cache()

        return Path(audio_dir)

    @torch.inference_mode()
    def eval(self, audio_dir: Path, it: int, data_cfg: DictConfig) -> dict[str, float]:
        with torch.amp.autocast('cuda', enabled=False):
            if local_rank == 0:
                extract(audio_path=audio_dir,
                        output_path=audio_dir / 'cache',
                        device='cuda',
                        batch_size=16,  # btz=16: avoid OOM
                        skip_video_related=True,  # avoid extracting video related features 
                        audio_length=10) 
                output_metrics = evaluate(gt_audio_cache=Path(data_cfg.gt_cache),
                                          skip_video_related=True, 
                                          pred_audio_cache=audio_dir / 'cache')
                for k, v in output_metrics.items():
                    # pad k to 10 characters
                    # pad v to 10 decimal places
                    self.log.log_scalar(f'{data_cfg.tag}/{k}', v, it)
                    self.log.info(f'{data_cfg.tag}/{k:<10}: {v:.10f}')
                    if k in ["FD-VGG", "FD-PASST", "FD-PANN", "MS-CLAP-Score",
                              "LAION-CLAP-Score", "ISC-PANNS-mean", "KL-PANNS-softmax"]: 
                        if self.use_wandb and local_rank == 0: 
                            wandb.log({f'{data_cfg.tag}/{k}': v}, step=it)
                    
            else:
                output_metrics = None

        return output_metrics

    def save_weights(self, it, save_copy=False):  # only save net's weights
        if local_rank != 0:
            return

        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = self.run_path / f'{self.exp_id}_{it}.pth'
            torch.save(self.network.module.state_dict(), model_path)
            self.log.info(f'Network weights saved to {model_path}.')

        # if last exists, move it to a shadow copy
        model_path = self.run_path / f'{self.exp_id}_last.pth'
        if model_path.exists():
            shadow_path = model_path.with_name(model_path.name.replace('last', 'shadow'))
            model_path.replace(shadow_path)
            self.log.info(f'Network weights shadowed to {shadow_path}.')

        torch.save(self.network.module.state_dict(), model_path)
        self.log.info(f'Network weights saved to {model_path}.')

    def save_checkpoint(self, it, save_copy=False):  # save it, optim, net together
        if local_rank != 0:
            return

        checkpoint = {
            'it': it,
            'weights': self.network.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema': self.ema.state_dict() if self.ema is not None else None,
        }

        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = self.run_path / f'{self.exp_id}_ckpt_{it}.pth'
            torch.save(checkpoint, model_path)
            self.log.info(f'Checkpoint saved to {model_path}.')

        # if ckpt_last exists, move it to a shadow copy
        model_path = self.run_path / f'{self.exp_id}_ckpt_last.pth'
        if model_path.exists():
            shadow_path = model_path.with_name(model_path.name.replace('last', 'shadow'))
            model_path.replace(shadow_path)  # moves the file
            self.log.info(f'Checkpoint shadowed to {shadow_path}.')

        torch.save(checkpoint, model_path)
        self.log.info(f'Checkpoint saved to {model_path}.')

    def get_latest_checkpoint_path(self):
        ckpt_path = self.run_path / f'{self.exp_id}_ckpt_last.pth'
        if not ckpt_path.exists():
            info_if_rank_zero(self.log, f'No checkpoint found at {ckpt_path}.')
            return None
        return ckpt_path

    def get_latest_weight_path(self):
        weight_path = self.run_path / f'{self.exp_id}_last.pth'
        if not weight_path.exists():
            self.log.info(f'No weight found at {weight_path}.')
            return None
        return weight_path

    def get_final_ema_weight_path(self):  # for sample (final testing)
        weight_path = self.run_path / f'{self.exp_id}_ema_final.pth'
        if not weight_path.exists():
            self.log.info(f'No weight found at {weight_path}.')
            return None
        return weight_path

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location}, weights_only=True)

        it = checkpoint['it']
        weights = checkpoint['weights']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        if self.ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
            self.log.info(f'EMA states loaded from step {self.ema.step}')

        map_location = 'cuda:%d' % local_rank
        self.network.module.load_state_dict(weights)   # directly load weights to model
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        self.log.info(f'Global iteration {it} loaded.')
        self.log.info('Network weights, optimizer states, and scheduler states loaded.')

        return it

    def load_weights_in_memory(self, src_dict):
        self.network.module.load_weights(src_dict)
        self.log.info('Network weights loaded from memory.')

    def load_weights(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location}, weights_only=True)

        fallback_mapping = {
            "r_embed.mlp.0.weight": "t_embed.mlp.0.weight", 
            "r_embed.mlp.0.bias": "t_embed.mlp.0.bias", 
            "r_embed.mlp.2.weight": "t_embed.mlp.2.weight", 
            "r_embed.mlp.2.bias": "t_embed.mlp.2.bias" 
        }
            
        for target_prefix, source_prefix in fallback_mapping.items():
            if target_prefix not in src_dict.keys(): 
                self.log.info(f"Copying from {source_prefix} to {target_prefix}")
                src_dict[target_prefix] = src_dict[source_prefix].clone()

        self.log.info(f'Importing network weights from {path}...')
        self.load_weights_in_memory(src_dict)

    def weights(self):
        return self.network.module.state_dict()

    def enter_train(self):
        self.integrator = self.train_integrator
        self.network.train()
        return self

    def enter_val(self):
        self.network.eval()
        return self

