#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Robot Vacuum.
"""

import os
import time

import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


def _extract_extra_info(env_obs):
    observation = env_obs.get("observation", {})
    return env_obs.get("extra_info") or observation.get("extra_info") or {}


def _extract_result_details(env_obs, fm, truncated, step):
    env_info = env_obs.get("observation", {}).get("env_info", {})
    extra_info = _extract_extra_info(env_obs)
    snapshot = fm.get_debug_snapshot()

    result_code = extra_info.get("result_code")
    result_message = extra_info.get("result_message")
    total_score = int(env_info.get("total_score", 0))
    clean_score = int(env_info.get("clean_score", total_score))
    remaining_charge = int(env_info.get("remaining_charge", snapshot["remaining_charge"]))
    charge_count = int(env_info.get("charge_count", snapshot["charge_count"]))
    max_step = max(int(env_info.get("max_step", fm.max_step)), 1)
    frame_no = int(env_obs.get("frame_no", step))
    completed_by_max_step = bool(
        truncated
        and result_code in (None, 0)
        and not result_message
        and max(step, frame_no) >= max_step
    )

    if completed_by_max_step:
        fail_reason = "completed_max_step"
    elif truncated:
        fail_reason = "abnormal_truncated"
    elif result_message:
        fail_reason = str(result_message)
    elif result_code not in (None, 0):
        fail_reason = f"result_code_{result_code}"
    elif remaining_charge <= 0:
        fail_reason = "battery_depleted"
    elif snapshot["nearest_npc_dist"] is not None and snapshot["nearest_npc_dist"] <= 1:
        fail_reason = "npc_collision"
    else:
        fail_reason = "unknown_failure"

    return {
        "result_code": result_code,
        "result_message": result_message,
        "fail_reason": fail_reason,
        "total_score": total_score,
        "clean_score": clean_score,
        "remaining_charge": remaining_charge,
        "charge_count": charge_count,
        "frame_no": frame_no,
        "max_step": max_step,
        "is_completed": completed_by_max_step,
        "snapshot": snapshot,
    }


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics: {training_metrics}")

            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, _ = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0

            self.logger.info(
                f"Episode {self.episode_cnt} start, feature_dim={Config.DIM_OF_OBSERVATION}, "
                f"local_view={Config.LOCAL_VIEW_SIZE}, global_state={Config.GLOBAL_FEATURE_SIZE}"
            )

            while not done:
                act_data_list = self.agent.predict([obs_data])
                if not act_data_list:
                    self.logger.error(
                        f"Episode {self.episode_cnt} predict returned no action data at step {step}, abort episode"
                    )
                    break
                act_data = act_data_list[0]
                act = self.agent.action_process(act_data)

                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                frame_no = env_obs["frame_no"]
                step += 1
                done = terminated or truncated

                next_obs_data, _ = self.agent.observation_process(env_obs)
                next_obs_data.frame_no = frame_no

                reward_scalar = float(self.agent.last_reward)
                total_reward += reward_scalar

                final_reward = 0.0
                if done:
                    fm = self.agent.preprocessor
                    result_details = _extract_result_details(env_obs, fm, truncated, step)
                    if result_details["is_completed"]:
                        cleaning_ratio = fm.dirt_cleaned / max(fm.total_dirt, 1)
                        final_reward = 5.0 + 5.0 * cleaning_ratio
                        result_str = "WIN"
                    else:
                        fail_reason = result_details["fail_reason"]
                        snapshot = result_details["snapshot"]
                        if fail_reason == "battery_depleted":
                            final_reward = -3.6 if result_details["charge_count"] <= 0 else -2.8
                            nearest_charger = snapshot["nearest_charger_dist"]
                            if nearest_charger is not None and nearest_charger >= 24:
                                final_reward -= 0.8
                            elif nearest_charger is not None and nearest_charger >= 12:
                                final_reward -= 0.4
                            elif nearest_charger is not None and nearest_charger <= 6:
                                final_reward -= 0.3
                            if snapshot["dock_mode"]:
                                final_reward -= 0.4
                            elif snapshot["return_mode"]:
                                final_reward -= 0.2
                            if snapshot["first_charge_phase"]:
                                final_reward -= 0.8
                                if snapshot["first_charge_stage"] == "dock":
                                    final_reward -= 0.25
                                elif snapshot["first_charge_stage"] == "return":
                                    final_reward -= 0.15
                            if snapshot["charge_recovery_mode"] == "reroute":
                                final_reward -= 0.08
                            if snapshot["charge_stall_steps"] >= 4:
                                final_reward -= 0.10
                        elif fail_reason == "npc_collision":
                            final_reward = -2.4
                            if snapshot["first_charge_phase"]:
                                final_reward -= 0.2
                        else:
                            final_reward = -2.0
                        result_str = "FAIL"

                    snapshot = result_details["snapshot"]
                    self.logger.info(
                        f"[GAMEOVER] ep:{self.episode_cnt} steps:{step} result:{result_str} "
                        f"reason:{result_details['fail_reason']} final_bonus:{final_reward:.2f} "
                        f"total_reward:{total_reward:.3f} total_score:{result_details['total_score']} "
                        f"clean_score:{result_details['clean_score']} dirt_cleaned:{fm.dirt_cleaned}/{fm.total_dirt} "
                        f"remaining_charge:{result_details['remaining_charge']} charge_count:{result_details['charge_count']} "
                        f"explored_ratio:{snapshot['explored_ratio']:.4f} pos:{snapshot['pos']} "
                        f"visit:{snapshot['current_visit']} nearest_charger:{snapshot['nearest_charger_dist']} "
                        f"nearest_npc:{snapshot['nearest_npc_dist']} return_mode:{snapshot['return_mode']} "
                        f"return_reason:{snapshot['return_reason']} charge_action:{snapshot['charge_action']} "
                        f"charge_urgency:{snapshot['charge_urgency']:.2f} battery_margin:{snapshot['battery_margin']} "
                        f"route_found:{snapshot['charge_route_found']} route_source:{snapshot['charge_route_source']} "
                        f"route_reliable:{snapshot['charge_route_reliable']} "
                        f"charge_stall:{snapshot['charge_stall_steps']} "
                        f"charge_return_state:{snapshot['charge_return_mode']} "
                        f"charge_recovery:{snapshot['charge_recovery_mode']} "
                        f"charge_controller:{snapshot['charge_controller_mode']} "
                        f"charge_allowed:{snapshot['charge_allowed_count']} "
                        f"soft_radius:{snapshot['soft_clean_radius']} "
                        f"hard_radius:{snapshot['hard_clean_radius']} "
                        f"strategy_mode:{snapshot['strategy_mode']} strategy_intensity:{snapshot['strategy_intensity']:.2f} "
                        f"dock_mode:{snapshot['dock_mode']} "
                        f"dock_radius:{snapshot['dock_radius']} first_charge_phase:{snapshot['first_charge_phase']} "
                        f"first_charge_stage:{snapshot['first_charge_stage']} "
                        f"first_charge_budget_buffer:{snapshot['first_charge_budget_buffer']} "
                        f"first_charge_budget_margin:{snapshot['first_charge_budget_margin']} "
                        f"first_charge_step_limit:{snapshot['first_charge_step_limit']} "
                        f"first_charge_out_radius:{snapshot['first_charge_out_radius']} "
                        f"charge_rule_control:{snapshot['charge_rule_control']} "
                        f"explore_mode:{snapshot['explore_mode']} "
                        f"explore_name:{snapshot['explore_mode_name']} "
                        f"explore_reason:{snapshot['explore_reason']} explore_action:{snapshot['explore_action']} "
                        f"explore_target_dist:{snapshot['explore_target_dist']} "
                        f"explore_desired_radius:{snapshot['explore_desired_radius']} "
                        f"explore_route_found:{snapshot['explore_route_found']} "
                        f"explore_route_source:{snapshot['explore_route_source']} "
                        f"explore_hold:{snapshot['explore_hold_active']} "
                        f"explore_hold_left:{snapshot['explore_hold_left']} "
                        f"post_charge_expand:{snapshot['post_charge_expand']} "
                        f"cycle_explore_gain:{snapshot['prev_charge_cycle_explore_gain']} "
                        f"cycle_clean_gain:{snapshot['prev_charge_cycle_clean_gain']} "
                        f"npc_mode:{snapshot['npc_mode']} npc_reason:{snapshot['npc_reason']} "
                        f"npc_action:{snapshot['npc_action']} npc_safe_dist:{snapshot['npc_safe_dist']} "
                        f"result_code:{result_details['result_code']} "
                        f"result_message:{result_details['result_message']}"
                    )

                reward_arr = np.array([reward_scalar], dtype=np.float32)
                value_arr = act_data.value.flatten()[: Config.VALUE_NUM]

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array(act_data.action),
                    reward=reward_arr,
                    done=np.array([float(done)]),
                    reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    value=value_arr,
                    next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    collector[-1].reward = collector[-1].reward + np.array([final_reward], dtype=np.float32)

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        self.monitor.put_data(
                            {
                                os.getpid(): {
                                    "reward": total_reward + final_reward,
                                    "episode_cnt": self.episode_cnt,
                                }
                            }
                        )
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = next_obs_data
