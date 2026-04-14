#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Simple MLP policy network for Robot Vacuum.
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def _make_fc(in_dim, out_dim, gain=1.41421):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _make_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=1.41421):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum"
        self.device = device

        local_view_dim, global_feature_dim, legal_action_dim = Config.FEATURES
        act_num = Config.ACTION_NUM
        structured_dim = global_feature_dim + legal_action_dim

        self.local_view_dim = local_view_dim
        self.global_feature_dim = global_feature_dim
        self.legal_action_dim = legal_action_dim
        self.local_view_side = int(local_view_dim**0.5)

        if self.local_view_side * self.local_view_side != local_view_dim:
            raise ValueError(f"LOCAL_VIEW_SIZE must be a square number, got {local_view_dim}")

        self.local_encoder = nn.Sequential(
            _make_conv(1, 16),
            nn.ReLU(),
            _make_conv(16, 32),
            nn.ReLU(),
            _make_conv(32, 64, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_dim = 64 * ((self.local_view_side + 1) // 2) * ((self.local_view_side + 1) // 2)
        self.local_proj = nn.Sequential(
            _make_fc(conv_out_dim, 192),
            nn.ReLU(),
            _make_fc(192, 128),
            nn.ReLU(),
        )

        self.structured_encoder = nn.Sequential(
            _make_fc(structured_dim, 128),
            nn.ReLU(),
            _make_fc(128, 128),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            _make_fc(256, 256),
            nn.ReLU(),
            _make_fc(256, 128),
            nn.ReLU(),
        )

        self.actor_tower = nn.Sequential(
            _make_fc(128, 256),
            nn.ReLU(),
            _make_fc(256, 128),
            nn.ReLU(),
        )
        self.critic_tower = nn.Sequential(
            _make_fc(128, 256),
            nn.ReLU(),
            _make_fc(256, 128),
            nn.ReLU(),
        )

        self.actor_head = _make_fc(128, act_num, gain=0.01)
        self.critic_head = _make_fc(128, 1, gain=0.01)

    def forward(self, s, inference=False):
        x = s.to(torch.float32)

        local_view = x[:, : self.local_view_dim]
        global_start = self.local_view_dim
        global_end = global_start + self.global_feature_dim
        global_state = x[:, global_start:global_end]
        legal_action = x[:, global_end : global_end + self.legal_action_dim]

        local_view = local_view.view(-1, 1, self.local_view_side, self.local_view_side)
        local_embed = self.local_proj(self.local_encoder(local_view))

        structured = torch.cat([global_state, legal_action], dim=1)
        structured_embed = self.structured_encoder(structured)

        fused = torch.cat([local_embed, structured_embed], dim=1)
        fused = self.fusion(fused)

        actor_hidden = self.actor_tower(fused)
        critic_hidden = self.critic_tower(fused)

        logits = self.actor_head(actor_hidden)
        value = self.critic_head(critic_hidden)
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
