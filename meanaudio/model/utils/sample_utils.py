from typing import Optional

import torch


def log_normal_sample(x: torch.Tensor,
                      generator: Optional[torch.Generator] = None,
                      m: float = 0.0,
                      s: float = 1.0) -> torch.Tensor:
    bs = x.shape[0]
    s = torch.randn(bs, device=x.device, generator=generator) * s + m
    return torch.sigmoid(s)
import torch
from typing import Optional, Tuple

def log_normal_sample_r_t(
    x: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    m: float = 0.0,
    s: float = 1.0,
    epsilon: float = 1.0  # 控制第二个张量的最小增量
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成两个张量，确保第二个张量的每个元素都大于第一个张量。
    
    参数:
    x (torch.Tensor): 输入张量（用于确定 batch_size 和设备）
    generator (torch.Generator, optional): 随机数生成器
    m (float): 正态分布的均值（默认为 0）
    s (float): 正态分布的标准差（默认为 1）
    epsilon (float): 控制第二个张量的最小增量（默认为 1）
    
    返回:
    Tuple[torch.Tensor, torch.Tensor]: 两个经过 sigmoid 处理的张量，第二个的每个元素均大于第一个
    """
    bs = x.shape[0]
    device = x.device
    
    # 生成第一个张量的原始值
    s1 = torch.randn(bs, device=device, generator=generator) * s + m
    
    # 生成第二个张量，确保每个元素比第一个大：
    # 使用绝对值正态分布作为增量，保证非负性
    increment = torch.abs(torch.randn(bs, device=device, generator=generator)) * epsilon
    s2 = s1 + increment
    
    # 应用 sigmoid 并返回
    #第二个比第一个大
    return torch.sigmoid(s1), torch.sigmoid(s2)