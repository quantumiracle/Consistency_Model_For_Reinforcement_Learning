# Implementation of Consistency Model
# https://arxiv.org/pdf/2303.01469.pdf

import math
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.helpers import Losses

class Consistency(nn.Module):
    """
    no c_in (EDM preconditioning); no scaled t; no adaptive ema schedule
    """

    def __init__(self, state_dim, action_dim, model, max_action, 
                 n_timesteps=100,
                 loss_type='l2', clip_denoised=True, action_norm=False,
                 eps: float = 0.002, D: int = 128) -> None:
        super(Consistency, self).__init__()

        self.eps = eps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model
        self.clip_denoised = clip_denoised
        self.action_norm = action_norm

        self.min_T = 2.0  # tau_{n-1}
        self.max_T = 80.0  # 80.0
        self.t_seq = np.linspace(self.min_T, self.max_T, n_timesteps)

        self.loss_fn = Losses[loss_type]()

    def predict_consistency(self, state, action, t) -> torch.Tensor:
        if isinstance(t, float):
            t = (
                torch.tensor([t] * action.shape[0], dtype=torch.float32)
                .to(action.device)
                .unsqueeze(1)
            )  # (batch, 1)

        action_ori = action  # (batch, action_dim)
        action = self.model(action, t, state)  # be careful of the order

        # sigma_data = 0.5
        t_ = t - self.eps
        c_skip_t = 0.25 / (t_.pow(2) + 0.25) # (batch, 1)
        c_out_t = 0.5 * t_ / (t.pow(2) + 0.25).pow(0.5)
        output = c_skip_t * action_ori + c_out_t * action
        if self.action_norm:
            output = self.max_action * torch.tanh(output)  # normalization
        return output

    def loss(self, state, action, z, t1, t2, ema_model=None, weights=torch.tensor(1.0)):
        x2 = action + z * t2  # x2: (batch, action_dim), t2: (batch, 1)
        if self.action_norm:
            x2 = self.max_action * torch.tanh(x2)
        x2 = self.predict_consistency(state, x2, t2)

        with torch.no_grad():
            x1 = action + z * t1
            if self.action_norm:
                x1 = self.max_action * torch.tanh(x1)
            if ema_model is None:
                x1 = self.predict_consistency(state, x1, t1)
            else:
                x1 = ema_model.predict_consistency(state, x1, t1)

        loss = self.loss_fn(x2, x1, weights, take_mean=False)  # prediction, target
        return loss

    # @torch.no_grad()
    def sample(self, state):
        """ this function needs to preserve the gradient for policy gradient to go through"""
        ts = list(reversed(self.t_seq))
        action_shape = list(state.shape)
        action_shape[-1] = self.action_dim
        action = torch.randn(action_shape).to(device=state.device) * self.max_T 
        if self.action_norm:
            action = self.max_action * torch.tanh(action)

        action = self.predict_consistency(state, action, ts[0])

        for t in ts[1:]:
            z = torch.randn_like(action)
            action = action + math.sqrt(t**2 - self.eps**2) * z
            if self.action_norm:
                action = self.max_action * torch.tanh(action)
            action = self.predict_consistency(state, action, t)

        action.clamp_(-self.max_action, self.max_action)
        return action

    def forward(self, state) -> torch.Tensor:
        # Sample steps
        pre_action = self.sample(
            state,
        )
        return pre_action
