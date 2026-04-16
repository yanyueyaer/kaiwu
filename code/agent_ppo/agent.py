#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Robot Vacuum Agent.
"""

import os

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from kaiwudrl.interface.agent import BaseAgent


class Agent(BaseAgent):
    """
    PPO 智能体封装，负责把观测变成动作，并把样本送给 learner。

    当前版本刻意保持为“策略网络直接决策”的 baseline：
    1. `Preprocessor` 负责状态解析、特征提取和 reward shaping。
    2. `Agent` 只负责前向推理、合法动作掩码和 PPO 所需的数据组织。
    3. 关于回充的规则信息只作为特征/奖励/调试信息存在，不再在这里硬改最终动作。
    """

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        """
        初始化模型、优化器、算法封装和预处理器。

        这里保留一个最简 baseline：
        1. 观测先交给 `Preprocessor` 做特征提取和 reward shaping。
        2. 动作只由策略网络在合法动作集合上采样。
        3. 不再额外叠加 NPC/探索/回充规则去接管最终动作。

        这样拆分的目的，是把“充电逻辑”更多放到状态建模和奖励设计层，
        而不是在 agent 侧用 if-else 直接接管动作，便于后续判断模型是否
        真的学会了低电量主动回桩。
        """
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.logger = logger
        self.monitor = monitor
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.last_reward = 0.0
        self.last_action_debug = self._empty_action_debug()

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs):
        """
        在新 episode 开始时重置运行时状态。

        每一局都重新创建 `Preprocessor`，避免上局的地图记忆、访问计数、回充状态等信息串到新局。
        """
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.last_reward = 0.0
        self.last_action_debug = self._empty_action_debug()

    def _empty_action_debug(self):
        """构造默认动作调试信息，便于训练日志记录当前决策来源。"""
        return {
            "source": "policy_only",
            "used_charge_rule": False,
            "deterministic_action": None,
            "sampled_action": None,
            "greedy_action": None,
            "selected_action": None,
            "is_stochastic": True,
        }

    def observation_process(self, env_obs):
        """
        将环境原始观测转换成模型输入。

        这个函数是 agent 与 preprocessor 的衔接点：
        1. `feature_process` 负责更新内部状态、提取特征、计算 shaped reward。
        2. 这里把结果包装成 `ObsData`，供 `predict` 使用。
        3. `self.last_reward` 会缓存当前 step 的 shaped reward，训练主循环随后会把它写进样本。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        self.last_reward = reward

        # `ObsData` 只保留推理真正需要的最小字段。
        # 其余调试信息都继续保留在 preprocessor 内部，避免 agent 侧重复维护状态。
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {}
        return obs_data, remain_info

    def action_process(self, act_data, is_stochastic=True):
        """
        从 `ActData` 中取出本次真正要提交给环境的动作。

        训练阶段默认用采样动作，评估/部署阶段可以切到贪心动作。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        self.last_action_debug["selected_action"] = self.last_action
        self.last_action_debug["is_stochastic"] = bool(is_stochastic)
        return self.last_action

    def predict(self, list_obs_data):
        """
        对当前观测执行一次前向推理并输出动作分布。

        当前版本刻意保持为最简 baseline：
        1. 网络输出动作 logits。
        2. 只施加合法动作掩码。
        3. 直接从策略分布中采样，不再用规则层强行改写。

        这样做的目的，是让“低电量时是否主动回充”主要由 reward 驱动学习，而不是由规则硬控。

        返回时同时给出两套动作：
        1. `action` 是按概率采样得到的行为动作，训练阶段默认使用它。
        2. `d_action` 是贪心动作，评估或部署阶段可以直接切换到它。
        """
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        legal_action = obs_data.legal_action

        # 先拿到原始 logits，再只在合法动作集合上归一化成概率分布。
        logits, value = self._run_model(feature)
        legal_arr = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits, legal_arr)

        # 训练用采样动作，评估可切到贪心动作。
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)
        # 调试字段显式保留 sampled / greedy / selected 三层动作，
        # 方便在终局日志里区分“策略本来想怎么走”和“最后真正提交了什么动作”。
        self.last_action_debug = {
            "source": "policy_only",
            "used_charge_rule": False,
            "deterministic_action": None,
            "sampled_action": int(action),
            "greedy_action": int(d_action),
            "selected_action": None,
            "is_stochastic": True,
        }

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """使用贪心动作做一次推理，常用于评估或部署阶段。"""
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])[0]
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data):
        """把采样得到的一批训练样本交给 PPO 算法更新模型。"""
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """保存当前模型参数。"""
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """
        从 checkpoint 加载模型参数。

        如果最新 checkpoint 与当前 baseline 结构不兼容，则跳过加载，避免训练直接中断。
        """
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        if not path or not os.path.exists(model_file_path):
            self.logger.warning(f"skip loading model {model_file_path}, checkpoint not found")
            return

        try:
            state_dict = torch.load(model_file_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.logger.info(f"load model {model_file_path} successfully")
        except Exception as err:
            self.logger.warning(
                f"skip loading model {model_file_path}, incompatible checkpoint with current baseline: {err}"
            )

    def _run_model(self, feature):
        """
        执行一次模型前向推理。

        返回值包含：
        1. 动作 logits，用来构造策略分布。
        2. 价值估计，用于 PPO 的 value/advantage 计算。

        这里强制走 eval + no_grad，是因为这个函数只服务于 actor 侧在线采样。
        真正需要梯度的训练更新发生在 learner 的 `learn` 流程里。
        """
        self.model.set_eval_mode()
        obs_tensor = (
            torch.tensor(np.array([feature], dtype=np.float32)).view(1, Config.DIM_OF_OBSERVATION).to(self.device)
        )
        with torch.no_grad():
            rst = self.model(obs_tensor, inference=True)
        logits = rst[0].cpu().numpy()[0]
        value = rst[1].cpu().numpy()[0]
        return logits, value

    def _legal_soft_max(self, logits, legal_action):
        """
        对 logits 做带合法动作掩码的 softmax。

        非法动作会被压到极小概率，最终分布只在合法动作上归一化。
        这样既能保持策略网络输出维度固定，又能避免采样到环境不接受的动作。
        """
        large_weight, epsilon = 1e20, 1e-5
        tmp = logits - large_weight * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -large_weight, 1)
        tmp = (np.exp(tmp) + epsilon) * legal_action
        return self._sanitize_prob(tmp, legal_action, fallback=legal_action)

    def _sanitize_prob(self, prob, legal_action, fallback=None):
        """
        清洗并归一化动作概率分布。

        这个函数是概率安全兜底：
        1. 先裁掉负数和非法动作残留概率。
        2. 如果总和为 0，则退回 `fallback`。
        3. 如果依然无可用动作，则在合法动作上均匀分布。

        这里存在的意义，是把数值稳定性问题都集中消化在一处，避免上游
        logits 极端值、掩码异常或浮点误差直接把采样流程打崩。
        """
        legal_arr = np.asarray(legal_action, dtype=np.float64)
        prob_arr = np.asarray(prob, dtype=np.float64)
        prob_arr = np.clip(prob_arr, 0.0, None) * legal_arr

        total = float(np.sum(prob_arr))
        if total <= 1e-12 and fallback is not None:
            prob_arr = np.asarray(fallback, dtype=np.float64)
            prob_arr = np.clip(prob_arr, 0.0, None) * legal_arr
            total = float(np.sum(prob_arr))

        if total <= 1e-12:
            prob_arr = np.zeros_like(legal_arr, dtype=np.float64)
            valid_idx = np.flatnonzero(legal_arr > 0)
            if valid_idx.size == 0:
                return prob_arr.astype(np.float32)
            prob_arr[valid_idx] = 1.0 / valid_idx.size
            return prob_arr.astype(np.float32)

        prob_arr /= total

        valid_idx = np.flatnonzero(legal_arr > 0)
        if valid_idx.size > 0:
            last_idx = int(valid_idx[-1])
            prefix_sum = float(np.sum(prob_arr) - prob_arr[last_idx])
            prob_arr[last_idx] = max(0.0, 1.0 - prefix_sum)
            total = float(np.sum(prob_arr))
            if total > 1e-12:
                prob_arr /= total

        return prob_arr.astype(np.float32)

    def get_action_debug_snapshot(self):
        """返回上一帧决策调试信息的副本，供训练日志使用。"""
        return dict(self.last_action_debug)

    def _legal_sample(self, probs, use_max=False):
        """
        从动作概率分布中选出动作。

        `use_max=True` 时返回贪心动作；
        否则按概率分布随机采样，作为 PPO 训练时的行为策略动作。

        采样前再次调用 `_sanitize_prob`，是为了保证即使上游传进来的概率
        向量已经被二次修改，也不会因为数值异常导致越界或空分布问题。
        """
        probs = self._sanitize_prob(probs, np.ones_like(probs, dtype=np.float32), fallback=probs)
        if use_max:
            return int(np.argmax(probs))

        cdf = np.cumsum(probs, dtype=np.float64)
        rand_v = float(np.random.random())
        action = int(np.searchsorted(cdf, rand_v, side="right"))
        if action >= len(probs):
            action = len(probs) - 1
        return action
