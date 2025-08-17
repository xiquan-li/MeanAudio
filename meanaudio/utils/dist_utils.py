import os
from logging import Logger
import torch.distributed as dist
from meanaudio.utils.logger import TensorboardLogger

local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1

def get_global_rank():
    """Get global rank across all nodes"""
    return dist.get_rank() if dist.is_initialized() else 0

def info_if_rank_zero(logger: Logger, msg: str):
    # Use global rank 0 instead of local rank 0 for multi-node compatibility
    if get_global_rank() == 0:
        logger.info(f'================================================ {msg} ===========================================================')

def string_if_rank_zero(logger: TensorboardLogger, tag: str, msg: str):
    # Use global rank 0 instead of local rank 0 for multi-node compatibility
    if get_global_rank() == 0:
        logger.log_string(tag, msg)
