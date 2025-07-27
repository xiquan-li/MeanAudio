import torch
from torch import nn
from torch.nn import functional as F


class ChannelLastConv1d(nn.Conv1d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, seq, D
        x = x.permute(0, 2, 1)  # B, D, seq
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


# https://github.com/Stability-AI/sd3-ref
class MLP(nn.Module):  # gated FFN

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ConvMLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w2 = ChannelLastConv1d(hidden_dim,
                                    dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w3 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

if __name__ == "__main__": 
    latent_dim = 20
    hidden_dim = 64 * 7
    conv1d = ChannelLastConv1d(
        in_channels = latent_dim, 
        out_channels = hidden_dim, 
        kernel_size = 7, 
        padding = 3
    )
    num_params = sum([p.numel() for p in conv1d.parameters()])
    print(conv1d)
    print(f"Num params for conv1d: {num_params}")

    B, T, D = 128, 250, 20
    x = torch.randn(B, T, D)
    h = conv1d(x)

    conv_mlp = ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3)
    num_params = sum([p.numel() for p in conv_mlp.parameters()])
    print(conv_mlp)
    print(f"Nim params for convmlp: {num_params}")
    y = conv_mlp(h)
    print(y.shape)