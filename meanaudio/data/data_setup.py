import logging
import random

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from meanaudio.data.extracted_audio import ExtractedAudio
from meanaudio.data.mm_dataset import MultiModalDataset
from meanaudio.utils.dist_utils import local_rank

log = logging.getLogger()


# Re-seed randomness every time we start a worker
def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    log.debug(f'Worker {worker_id} re-seeded with seed {worker_seed} in rank {local_rank}')


def load_audio_data(cfg: DictConfig, data_cfg: DictConfig) -> Dataset: 
    dataset = ExtractedAudio(tsv_path=data_cfg.tsv, 
                            concat_text_fc=cfg.concat_text_fc,   # FIX here we determine usage of concat based on global config
                            data_dim=cfg.data_dim, 
                            npz_dir=data_cfg.npz_dir, 
                            repa_npz_dir=data_cfg.repa_npz_dir,
                            exclude_cls=cfg.get('exclude_cls', False), 
                            repa_version=cfg.get('repa_version', 1))
    return dataset


def setup_training_datasets(cfg: DictConfig) -> tuple[Dataset, DistributedSampler, DataLoader]:

    if cfg.mini_train: 
        audiocaps_mini = load_audio_data(cfg, cfg.data.AudioCaps_val_npz)  # use val set as the miniset
        dataset = MultiModalDataset([],
                                    [audiocaps_mini])
            
    else: 

        audiocaps_npz = load_audio_data(cfg, cfg.data.AudioCaps_npz) 
        # !TODO: think of a better way to handle different datasets
        
        # freesound1_npz = load_audio_data_npz(cfg, cfg.data.FreeSound1_npz)
        # freesound2_npz = load_audio_data_npz(cfg, cfg.data.FreeSound2_npz)
        # freesound3_npz = load_audio_data_npz(cfg, cfg.data.FreeSound3_npz)

        # audioset_sl_npz = load_audio_data_npz(cfg, cfg.data.AudioSetSL_npz)
        # bbcsound_npz = load_audio_data_npz(cfg, cfg.data.BBCSound_npz)
        # clotho_npz = load_audio_data_npz(cfg, cfg.data.Clotho_npz)

        dataset = MultiModalDataset([], [audiocaps_npz]) 
        # dataset = MultiModalDataset([], [audiocaps_npz]*cfg.ac_oversample_rate + [audioset_sl_npz, bbcsound_npz, clotho_npz,  
        #                                                                         freesound1_npz, freesound2_npz, freesound3_npz])
        
        
    batch_size = cfg.batch_size  # per-gpu batch size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory
    sampler, loader = construct_loader(dataset,
                                       batch_size,
                                       num_workers,
                                       shuffle=True,
                                       drop_last=True,
                                       pin_memory=pin_memory)

    return dataset, sampler, loader


def setup_test_datasets(cfg):  # used in sample
    dataset = load_audio_data(cfg, cfg.data.AudioCaps_test_npz)  # ALL with NPZ format

    batch_size = cfg.eval_batch_size  # FIX: from train config
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory
    sampler, loader = construct_loader(dataset,
                                       batch_size,
                                       num_workers,
                                       shuffle=False,
                                       drop_last=False,
                                       pin_memory=pin_memory)

    return dataset, sampler, loader


def setup_val_datasets(cfg: DictConfig) -> tuple[Dataset, DataLoader, DataLoader]:
    dataset = load_audio_data(cfg, cfg.data.AudioCaps_val_npz)

    val_batch_size = cfg.batch_size
    val_eval_batch_size = cfg.eval_batch_size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory
    _, val_loader = construct_loader(dataset,
                                     val_batch_size,
                                     num_workers,
                                     shuffle=False,
                                     drop_last=False,
                                     pin_memory=pin_memory)
    _, eval_loader = construct_loader(dataset,
                                      val_eval_batch_size,
                                      num_workers,
                                      shuffle=False,
                                      drop_last=False,
                                      pin_memory=pin_memory)

    return dataset, val_loader, eval_loader


def error_avoidance_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))   # batch = [x for x in batch if x is not None]
    return default_collate(batch)


def construct_loader(dataset: Dataset,
                     batch_size: int,
                     num_workers: int,
                     *,
                     shuffle: bool = True,
                     drop_last: bool = True,
                     pin_memory: bool = False,
                     error_avoidance: bool = False) -> tuple[DistributedSampler, DataLoader]:
    train_sampler = DistributedSampler(dataset, rank=local_rank, shuffle=shuffle)
    train_loader = DataLoader(dataset,
                              batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              worker_init_fn=worker_init_fn,
                              drop_last=drop_last,
                              persistent_workers=num_workers > 0,
                              pin_memory=pin_memory,
                              collate_fn=error_avoidance_collate if error_avoidance else None)
    return train_sampler, train_loader
