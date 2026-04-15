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
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
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

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs):
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.last_reward = 0.0

    def observation_process(self, env_obs):
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        self.last_reward = reward

        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {}
        return obs_data, remain_info

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    def predict(self, list_obs_data):
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        legal_action = obs_data.legal_action

        logits, value = self._run_model(feature)
        legal_arr = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits, legal_arr)
        prob = self._apply_npc_guidance(prob, legal_arr)
        prob = self._apply_explore_guidance(prob, legal_arr)

        rule_charge_control = self._build_rule_charge_control(prob, legal_arr)
        if rule_charge_control is not None:
            rule_prob = rule_charge_control["prob"]
            deterministic_action = rule_charge_control.get("deterministic_action")
            behavior_prob = rule_prob
            if deterministic_action is not None and 0 <= int(deterministic_action) < len(rule_prob):
                deterministic_action = int(deterministic_action)
                behavior_prob = np.zeros_like(rule_prob, dtype=np.float32)
                behavior_prob[deterministic_action] = 1.0
                rule_action = deterministic_action
                rule_d_action = deterministic_action
            else:
                rule_action = self._legal_sample(rule_prob, use_max=False)
                rule_d_action = self._legal_sample(rule_prob, use_max=True)
            return [
                ActData(
                    action=[rule_action],
                    d_action=[rule_d_action],
                    prob=list(behavior_prob),
                    value=value,
                )
            ]

        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])[0]
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
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
        large_weight, epsilon = 1e20, 1e-5
        tmp = logits - large_weight * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -large_weight, 1)
        tmp = (np.exp(tmp) + epsilon) * legal_action
        return self._sanitize_prob(tmp, legal_action, fallback=legal_action)

    def _sanitize_prob(self, prob, legal_action, fallback=None):
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

    def _normalize_prob(self, prob, legal_action, fallback):
        return self._sanitize_prob(prob, legal_action, fallback=fallback)

    def _apply_npc_guidance(self, prob, legal_action):
        guidance = self.preprocessor.get_npc_guidance()
        action_weights = guidance.get("action_weights") or []
        if len(action_weights) != len(prob):
            return prob

        weights = np.array(action_weights, dtype=np.float32)
        adjusted_prob = self._normalize_prob(prob * weights, legal_action, prob)

        target_action = guidance.get("target_action")
        if not guidance.get("should_evade") or target_action is None:
            return adjusted_prob

        if target_action >= len(legal_action) or legal_action[target_action] <= 0:
            return adjusted_prob

        heuristic_prob = np.zeros_like(prob, dtype=np.float32)
        heuristic_prob[target_action] = 1.0

        danger_level = float(np.clip(guidance.get("danger_level", 0.0), 0.0, 1.0))
        blend_ratio = 0.25 + 0.55 * danger_level
        blended_prob = (1.0 - blend_ratio) * adjusted_prob + blend_ratio * heuristic_prob
        return self._normalize_prob(blended_prob, legal_action, adjusted_prob)

    def _apply_explore_guidance(self, prob, legal_action):
        guidance = self.preprocessor.get_explore_guidance()
        if not guidance.get("active"):
            return prob

        target_action = guidance.get("target_action")
        force_action = guidance.get("force_action")
        mode_name = guidance.get("mode")
        intensity = float(np.clip(guidance.get("intensity", 0.0), 0.0, 1.0))
        hold_active = bool(guidance.get("hold_active"))
        post_charge_expand = bool(guidance.get("post_charge_expand"))

        heuristic_prob = np.zeros_like(prob, dtype=np.float32)
        valid_force_action = None
        if force_action is not None:
            force_action = int(force_action)
            if 0 <= force_action < len(legal_action) and legal_action[force_action] > 0:
                valid_force_action = force_action
                heuristic_prob[force_action] += 0.70 if mode_name == "post_charge_release" else 1.0

        if target_action is not None:
            target_action = int(target_action)
            if 0 <= target_action < len(legal_action) and legal_action[target_action] > 0:
                target_weight = 1.0
                if (
                    valid_force_action is not None
                    and mode_name == "post_charge_release"
                    and target_action != valid_force_action
                ):
                    target_weight = 0.45
                heuristic_prob[target_action] += target_weight

        if not np.any(heuristic_prob > 0.0):
            return prob

        heuristic_prob = self._normalize_prob(heuristic_prob, legal_action, prob)
        if mode_name == "post_charge_release":
            blend_ratio = 0.22 + 0.16 * intensity
            if hold_active:
                blend_ratio += 0.08
            if post_charge_expand:
                blend_ratio += 0.06
            blend_ratio = min(blend_ratio, 0.52)
        elif mode_name == "expand_frontier":
            blend_ratio = 0.10 + 0.16 * intensity
            if hold_active:
                blend_ratio += 0.06
            if post_charge_expand:
                blend_ratio += 0.08
            blend_ratio = min(blend_ratio, 0.45)
        elif mode_name == "recenter":
            blend_ratio = 0.06 + 0.10 * intensity
        else:
            blend_ratio = 0.08 + 0.12 * intensity
        blended_prob = (1.0 - blend_ratio) * prob + blend_ratio * heuristic_prob
        return self._normalize_prob(blended_prob, legal_action, prob)

    def _build_rule_charge_control(self, base_prob, legal_action):
        guidance = self.preprocessor.get_charge_guidance()
        if not guidance.get("should_return"):
            return None

        def _is_valid_action(action):
            return (
                action is not None
                and 0 <= int(action) < len(legal_action)
                and legal_action[int(action)] > 0
            )

        target_action = guidance.get("target_action")
        force_action = guidance.get("force_action")
        controller_mode = guidance.get("controller_mode", "")
        path_found = bool(guidance.get("path_found"))
        dock_mode = bool(guidance.get("dock_mode"))
        first_charge_phase = bool(guidance.get("first_charge_phase"))
        first_charge_stage = guidance.get("first_charge_stage", "")
        target_dist = guidance.get("target_dist")
        stall_steps = int(guidance.get("charge_stall_steps", 0))
        urgency = float(np.clip(guidance.get("urgency", 0.0), 0.0, 1.0))
        route_reliable = bool(guidance.get("route_reliable"))

        ranked_actions = []
        for action in guidance.get("control_actions") or []:
            if not _is_valid_action(action):
                continue
            action = int(action)
            if action not in ranked_actions:
                ranked_actions.append(action)

        preferred_action = None
        if _is_valid_action(force_action):
            preferred_action = int(force_action)
        elif _is_valid_action(target_action):
            preferred_action = int(target_action)

        if preferred_action is not None:
            if preferred_action in ranked_actions:
                ranked_actions.remove(preferred_action)
            if path_found or controller_mode in ("near_dock", "final_dock"):
                # If a charger-directed action is already known, keep it at the front
                # once the controller has entered the docking corridor.
                ranked_actions.insert(0, preferred_action)
            else:
                ranked_actions.append(preferred_action)

        if not ranked_actions:
            fallback_prob = self._normalize_prob(np.asarray(base_prob, dtype=np.float32), legal_action, legal_action)
            deterministic_action = None
            # Once the robot is already inside the charger area, use a wider hard-commit
            # window so the docking action is not diluted by policy exploration.
            commit_dist = 3 if first_charge_phase else 2
            near_commit_dist = 5 if first_charge_phase else 3
            if (
                preferred_action is not None
                and controller_mode == "final_dock"
                and target_dist is not None
                and int(target_dist) <= commit_dist
            ):
                deterministic_action = preferred_action
            elif (
                preferred_action is not None
                and controller_mode == "near_dock"
                and target_dist is not None
                and int(target_dist) <= near_commit_dist
                and (first_charge_phase or stall_steps >= 2 or (not route_reliable))
            ):
                deterministic_action = preferred_action
            return {
                "prob": fallback_prob,
                "deterministic_action": deterministic_action,
            }

        mask = np.zeros_like(base_prob, dtype=np.float32)
        score_arr = np.zeros_like(base_prob, dtype=np.float32)
        decay = 1.0
        if controller_mode == "final_dock":
            decay_factor = 0.52
            min_score = 0.26
        elif controller_mode == "near_dock" or dock_mode:
            decay_factor = 0.58
            min_score = 0.18
        elif controller_mode == "return_recovery":
            decay_factor = 0.74
            min_score = 0.08
        else:
            decay_factor = 0.68
            min_score = 0.10

        for action in ranked_actions:
            mask[action] = 1.0
            score_arr[action] = decay
            decay = max(decay * decay_factor, min_score)

        # Restrict the policy to charger-related candidates first, then blend it with
        # a hand-ranked docking prior.
        heuristic_prob = self._normalize_prob(score_arr, legal_action, mask)
        base_focus_prob = self._normalize_prob(np.asarray(base_prob, dtype=np.float32) * mask, legal_action, heuristic_prob)

        blend_ratio = 0.52 + 0.14 * urgency
        if path_found:
            blend_ratio += 0.06
        if stall_steps >= 2:
            blend_ratio += 0.04

        # Near the dock, let rule control dominate so the model does not waste the
        # remaining battery on small detours or oscillations.
        if controller_mode == "final_dock":
            blend_ratio = max(blend_ratio, 0.92 if route_reliable else 0.88)
        elif controller_mode == "near_dock":
            near_dock_floor = 0.82 if route_reliable else 0.78
            if first_charge_phase:
                near_dock_floor += 0.10
            if stall_steps >= 2:
                near_dock_floor += 0.06
            blend_ratio = max(blend_ratio, near_dock_floor)
        elif controller_mode == "return_recovery":
            blend_ratio = max(blend_ratio, 0.60 if route_reliable else 0.54)
        elif first_charge_phase:
            if first_charge_stage == "dock":
                blend_ratio = max(blend_ratio, 0.80 if route_reliable else 0.76)
            else:
                blend_ratio = max(blend_ratio, 0.72 if route_reliable else 0.68)
        else:
            blend_ratio = max(blend_ratio, 0.58 if route_reliable else 0.54)
        blend_ratio = float(np.clip(blend_ratio, 0.50, 0.94))

        rule_prob = (1.0 - blend_ratio) * base_focus_prob + blend_ratio * heuristic_prob
        rule_prob = self._normalize_prob(rule_prob, legal_action, heuristic_prob)

        deterministic_action = None
        commit_dist = 3 if first_charge_phase else 2
        near_commit_dist = 5 if first_charge_phase else 3
        if (
            preferred_action is not None
            and controller_mode == "final_dock"
            and target_dist is not None
            and int(target_dist) <= commit_dist
        ):
            deterministic_action = preferred_action
        elif (
            preferred_action is not None
            and controller_mode == "near_dock"
            and target_dist is not None
            and int(target_dist) <= near_commit_dist
            and (first_charge_phase or stall_steps >= 2 or (not route_reliable))
        ):
            deterministic_action = preferred_action

        return {
            "prob": rule_prob,
            "deterministic_action": deterministic_action,
        }

    def _legal_sample(self, probs, use_max=False):
        probs = self._sanitize_prob(probs, np.ones_like(probs, dtype=np.float32), fallback=probs)
        if use_max:
            return int(np.argmax(probs))

        cdf = np.cumsum(probs, dtype=np.float64)
        rand_v = float(np.random.random())
        action = int(np.searchsorted(cdf, rand_v, side="right"))
        if action >= len(probs):
            action = len(probs) - 1
        return action
