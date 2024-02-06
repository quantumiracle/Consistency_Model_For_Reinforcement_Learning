import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.helpers import SinusoidalPosEmb


class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):
        if len(time.shape) > 1:
            time = time.squeeze(1)  # added for shaping t from (batch_size, 1) to (batch_size,)
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)

class ResNetBlock(nn.Module):
    def __init__(self, in_features, hidden_dim, dropout_rate=0.1):
        super(ResNetBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def forward(self, x):
        identity = x
        out = self.layer(x)
        out += identity
        return out

class LN_Resnet(nn.Module):
    def __init__(self, state_dim, action_dim, device, t_dim=16, hidden_size=256, dropout_rate=0.1):
        super(LN_Resnet, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        input_dim = state_dim + action_dim + t_dim

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
        )
        self.resnet_block1 = ResNetBlock(hidden_size, hidden_size, dropout_rate)
        self.resnet_block2 = ResNetBlock(hidden_size, hidden_size, dropout_rate)
        self.resnet_block3 = ResNetBlock(hidden_size, hidden_size, dropout_rate)
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x, time, state):
        if len(time.shape) > 1:
            time = time.squeeze(1)  # added for shaping t from (batch_size, 1) to (batch_size,)
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.input_layer(x)
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.output_layer(x)
        return x



blk = lambda ic, oc: nn.Sequential(
    nn.GroupNorm(32, num_channels=ic),
    nn.SiLU(),
    nn.Conv2d(ic, oc, 3, padding=1),
    nn.GroupNorm(32, num_channels=oc),
    nn.SiLU(),
    nn.Conv2d(oc, oc, 3, padding=1),
)

class Unet(nn.Module):
    def __init__(self, 
        n_channel: int,
        D: int = 128,
        device: torch.device = torch.device("cpu"),
        ) -> None:
        super(Unet, self).__init__()
        self.device = device

        self.freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=D, dtype=torch.float32) / D
        )

        self.down = nn.Sequential(
            *[
                nn.Conv2d(n_channel, D, 3, padding=1),
                blk(D, D),
                blk(D, 2 * D),
                blk(2 * D, 2 * D),
            ]
        )

        self.time_downs = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.Linear(2 * D, D),
            nn.Linear(2 * D, 2 * D),
            nn.Linear(2 * D, 2 * D),
        )

        self.mid = blk(2 * D, 2 * D)

        self.up = nn.Sequential(
            *[
                blk(2 * D, 2 * D),
                blk(2 * 2 * D, D),
                blk(D, D),
                nn.Conv2d(2 * D, 2 * D, 3, padding=1),
            ]
        )
        self.last = nn.Conv2d(2 * D + n_channel, n_channel, 3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        # time embedding
        args = t.float() * self.freqs[None].to(t.device)
        t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(x.device)

        x_ori = x

        # perform F(x, t)
        hs = []
        for idx, layer in enumerate(self.down):
            if idx % 2 == 1:
                x = layer(x) + x
            else:
                x = layer(x)
                x = F.interpolate(x, scale_factor=0.5)
                hs.append(x)

            x = x + self.time_downs[idx](t_emb)[:, :, None, None]

        x = self.mid(x)

        for idx, layer in enumerate(self.up):
            if idx % 2 == 0:
                x = layer(x) + x
            else:
                x = torch.cat([x, hs.pop()], dim=1)
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = layer(x)

        x = self.last(torch.cat([x, x_ori], dim=1))

        return x

