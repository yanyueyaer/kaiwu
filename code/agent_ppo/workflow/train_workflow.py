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
    """
    训练流程入口。

    这里负责做训练侧的最外层编排：
    1. 读取环境配置。
    2. 构造 `EpisodeRunner`。
    3. 持续消费 runner 产出的样本并发送给 learner。
    4. 按固定时间间隔自动保存模型。
    """
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
    """
    兼容不同观测结构，提取环境返回的 `extra_info`。

    有些环境字段直接挂在最外层，有些包在 `observation` 下面，这里统一做一次兜底。
    """
    observation = env_obs.get("observation", {})
    return env_obs.get("extra_info") or observation.get("extra_info") or {}


def _extract_result_details(env_obs, fm, truncated, step):
    """
    从终局观测中提取结果标签和核心统计。

    这里不再强调规则排障细节，而是更关注当前 baseline 是否学会：
    1. 存活到终局。
    2. 成功完成至少一次充电。
    3. 在保证存活的前提下尽可能提高清扫比例。

    返回结果会同时保留：
    1. 环境直接给出的结果字段，如 `result_code`、`result_message`。
    2. 训练关心的行为统计，如 `charge_count`、`clean_ratio`。
    3. `Preprocessor` 导出的调试快照，便于 GAMEOVER 时定位问题到底出在回桩、贴桩还是撞 NPC。
    """
    env_info = env_obs.get("observation", {}).get("env_info", {})
    extra_info = _extract_extra_info(env_obs)
    snapshot = fm.get_debug_snapshot()

    # 先统一收集环境原始统计，避免后面的 fail_reason 判断混进太多取字段逻辑。
    result_code = extra_info.get("result_code")
    result_message = extra_info.get("result_message")
    total_score = int(env_info.get("total_score", 0))
    clean_score = int(env_info.get("clean_score", total_score))
    remaining_charge = int(env_info.get("remaining_charge", snapshot["remaining_charge"]))
    charge_count = int(env_info.get("charge_count", snapshot["charge_count"]))
    max_step = max(int(env_info.get("max_step", fm.max_step)), 1)
    frame_no = int(env_obs.get("frame_no", step))
    clean_ratio = float(fm.dirt_cleaned) / float(max(fm.total_dirt, 1))

    completed_by_max_step = bool(
        truncated
        and result_code in (None, 0)
        and not result_message
        and max(step, frame_no) >= max_step
    )

    # fail_reason 是训练侧自己的统一终局标签。
    # 它的目的不是完全复刻环境结果码，而是把奖励、监控和排查真正关心的失败类型压成少数几类。
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
        "clean_ratio": clean_ratio,
        "first_charge_success": float(charge_count > 0),
        "battery_depleted": float(fail_reason == "battery_depleted"),
        "snapshot": snapshot,
    }


def _compute_final_reward(result_details):
    """
    计算 episode 终局奖励。

    设计原则：
    1. 存活并完成清扫是主目标。
    2. 电量耗尽要有显著惩罚，避免策略把回充当成可有可无。
    3. 终局奖励尽量简单，主要学习压力交给 step-level reward shaping。

    也就是说，这里只负责在 episode 结束时做一次“方向性校正”：
    完成任务给正奖励，明显失败给负奖励；真正细粒度的回充学习信号仍来自 step reward。
    """
    if result_details["is_completed"]:
        # 正常跑满并完成清扫时，清扫比例是主项，充电次数只给很轻的附加项，
        # 避免策略为了刷充电次数而牺牲主任务。
        final_reward = 4.0 + 6.0 * result_details["clean_ratio"] + 0.4 * min(result_details["charge_count"], 3)
        return final_reward, "WIN"

    if result_details["fail_reason"] == "battery_depleted":
        # 电量耗尽是当前阶段最希望优先压下去的失败类型，所以这里惩罚最重。
        final_reward = -4.0
        if result_details["charge_count"] <= 0:
            final_reward -= 1.0
        nearest_charger = result_details["snapshot"]["nearest_charger_dist"]
        if nearest_charger is not None and nearest_charger <= 3:
            final_reward -= 0.4
        return final_reward, "FAIL"

    if result_details["fail_reason"] == "npc_collision":
        return -2.5, "FAIL"

    return -2.0, "FAIL"


def _build_episode_metrics(result_details):
    """
    整理一组更适合观察回充学习效果的核心指标。

    这些指标主要用于监控和日志，不直接参与训练计算。重点是把“是否学会首充、
    是否还会电量耗尽、最终清扫得怎么样”这几个维度稳定暴露出来。
    """
    snapshot = result_details["snapshot"]
    low_battery_route_progress_mean = snapshot.get("low_battery_route_progress_mean")
    return {
        "clean_ratio": float(result_details["clean_ratio"]),
        "charge_count": int(result_details["charge_count"]),
        "first_charge_success": float(result_details["first_charge_success"]),
        "battery_depleted": float(result_details["battery_depleted"]),
        "remaining_charge": int(result_details["remaining_charge"]),
        "completed": float(result_details["is_completed"]),
        "explored_ratio": float(snapshot["explored_ratio"]),
        "nearest_charger_dist": (
            -1 if snapshot["nearest_charger_dist"] is None else int(snapshot["nearest_charger_dist"])
        ),
        "first_charge_step": -1 if snapshot["first_charge_step"] is None else int(snapshot["first_charge_step"]),
        "return_trigger_step": -1 if snapshot["return_trigger_step"] is None else int(snapshot["return_trigger_step"]),
        "return_trigger_margin": (
            -999 if snapshot["return_trigger_margin"] is None else int(snapshot["return_trigger_margin"])
        ),
        "first_return_trigger_step": (
            -1 if snapshot["first_return_trigger_step"] is None else int(snapshot["first_return_trigger_step"])
        ),
        "min_battery_margin": (
            -999 if snapshot["min_battery_margin"] is None else int(snapshot["min_battery_margin"])
        ),
        "battery_margin_at_first_charge": (
            -999
            if snapshot["battery_margin_at_first_charge"] is None
            else int(snapshot["battery_margin_at_first_charge"])
        ),
        "low_battery_steps": int(snapshot["low_battery_steps"]),
        "low_battery_route_progress_mean": (
            0.0 if low_battery_route_progress_mean is None else float(low_battery_route_progress_mean)
        ),
        "charge_route_found_rate": float(snapshot["charge_route_found_rate"]),
        "route_stall_steps_total": int(snapshot["route_stall_steps_total"]),
        "max_charge_stall_steps": int(snapshot["max_charge_stall_steps"]),
        "dock_contact_without_charge": int(snapshot["dock_contact_without_charge"]),
        "charge_guidance_steps": int(snapshot["charge_guidance_steps"]),
        "charge_route_found_steps": int(snapshot["charge_route_found_steps"]),
    }


class EpisodeRunner:
    """按 episode 驱动环境交互、样本收集、终局奖励和监控上报。"""

    def __init__(self, env, agent, usr_conf, logger, monitor):
        """
        保存训练循环需要的环境、agent、配置和监控对象。

        `EpisodeRunner` 本身不做策略学习，它只负责跑环境、收样本、补终局奖励
        并把关键指标上报给日志和监控系统。
        """
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        """
        持续运行 episode，并在每局结束后产出一批训练样本。

        这是训练主循环：
        1. 重置环境，载入最新模型。
        2. 持续执行 `obs -> act -> env.step -> next_obs`。
        3. 收集 shaped reward 和 value/prob 等训练字段。
        4. 在终局时补发 final reward，并输出核心监控指标。

        这里会把 step reward 和 final reward 分开处理：
        - step reward 来自 `Preprocessor.reward_process()`，负责逐步塑造行为。
        - final reward 只在终局加到最后一帧，负责把整局成败明确反馈给 PPO。
        """
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

            # 每局开始都重新加载一次最新模型，确保 actor 侧尽快跟上 learner 已更新的参数。
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
                # 先基于当前观测产出动作，再推动环境执行一步。
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

                # 训练写样本时使用的是 preprocessor 给出的 shaped reward，而不是环境原始 reward。
                # 这样奖励设计可以完全围绕“清扫-探索-回充”这套学习目标来做。
                reward_scalar = float(self.agent.last_reward)
                total_reward += reward_scalar

                final_reward = 0.0
                if done:
                    fm = self.agent.preprocessor
                    result_details = _extract_result_details(env_obs, fm, truncated, step)
                    final_reward, result_str = _compute_final_reward(result_details)
                    metrics = _build_episode_metrics(result_details)
                    snapshot = result_details["snapshot"]
                    action_debug = self.agent.get_action_debug_snapshot()

                    charge_progress_delta = snapshot.get("charge_route_progress_delta")
                    charge_progress_delta_str = (
                        "None" if charge_progress_delta is None else f"{float(charge_progress_delta):.2f}"
                    )
                    low_battery_route_progress_mean = snapshot.get("low_battery_route_progress_mean")
                    low_battery_route_progress_mean_str = (
                        "None"
                        if low_battery_route_progress_mean is None
                        else f"{float(low_battery_route_progress_mean):.3f}"
                    )

                    self.logger.info(
                        f"[GAMEOVER] ep:{self.episode_cnt} steps:{step} result:{result_str} "
                        f"reason:{result_details['fail_reason']} final_bonus:{final_reward:.2f} "
                        f"total_reward:{total_reward:.3f} total_score:{result_details['total_score']} "
                        f"clean_score:{result_details['clean_score']} clean_ratio:{metrics['clean_ratio']:.4f} "
                        f"charge_count:{metrics['charge_count']} "
                        f"first_charge_success:{int(metrics['first_charge_success'])} "
                        f"battery_depleted:{int(metrics['battery_depleted'])} "
                        f"remaining_charge:{metrics['remaining_charge']} "
                        f"explored_ratio:{metrics['explored_ratio']:.4f} "
                        f"pos:{snapshot['pos']} visit:{snapshot['current_visit']} "
                        f"nearest_charger:{snapshot['nearest_charger_dist']} "
                        f"nearest_npc:{snapshot['nearest_npc_dist']} "
                        f"return_mode:{snapshot['return_mode']} "
                        f"return_reason:{snapshot['return_reason']} "
                        f"return_trigger_step:{snapshot['return_trigger_step']} "
                        f"return_trigger_margin:{snapshot['return_trigger_margin']} "
                        f"first_return_trigger_step:{snapshot['first_return_trigger_step']} "
                        f"first_charge_step:{snapshot['first_charge_step']} "
                        f"first_charge_stage:{snapshot['first_charge_stage']} "
                        f"battery_margin:{snapshot['battery_margin']} "
                        f"min_battery_margin:{snapshot['min_battery_margin']} "
                        f"battery_margin_at_first_charge:{snapshot['battery_margin_at_first_charge']} "
                        f"low_battery_steps:{snapshot['low_battery_steps']} "
                        f"low_battery_route_progress_mean:{low_battery_route_progress_mean_str} "
                        f"route_found:{snapshot['charge_route_found']} "
                        f"route_found_rate:{snapshot['charge_route_found_rate']:.3f} "
                        f"charge_guidance_steps:{snapshot['charge_guidance_steps']} "
                        f"charge_route_found_steps:{snapshot['charge_route_found_steps']} "
                        f"charge_progress_delta:{charge_progress_delta_str} "
                        f"route_stall_steps_total:{snapshot['route_stall_steps_total']} "
                        f"max_charge_stall_steps:{snapshot['max_charge_stall_steps']} "
                        f"dock_contact_without_charge:{snapshot['dock_contact_without_charge']} "
                        f"action_source:{action_debug['source']} "
                        f"action_sampled:{action_debug['sampled_action']} "
                        f"action_selected:{action_debug['selected_action']} "
                        f"result_code:{result_details['result_code']} "
                        f"result_message:{result_details['result_message']}"
                    )

                # 单步样本先按 step reward 落盘；若终局，再把 final reward 叠到最后一帧。
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
                        # 监控侧只上报高价值的整局指标，避免面板里被大量中间字段淹没。
                        self.monitor.put_data(
                            {
                                os.getpid(): {
                                    "reward": total_reward + final_reward,
                                    "episode_cnt": self.episode_cnt,
                                    "clean_ratio": metrics["clean_ratio"],
                                    "charge_count": metrics["charge_count"],
                                    "first_charge_success": metrics["first_charge_success"],
                                    "battery_depleted": metrics["battery_depleted"],
                                    "remaining_charge": metrics["remaining_charge"],
                                    "completed": metrics["completed"],
                                    "explored_ratio": metrics["explored_ratio"],
                                    "nearest_charger_dist": metrics["nearest_charger_dist"],
                                    "first_charge_step": metrics["first_charge_step"],
                                    "return_trigger_step": metrics["return_trigger_step"],
                                    "return_trigger_margin": metrics["return_trigger_margin"],
                                    "first_return_trigger_step": metrics["first_return_trigger_step"],
                                    "min_battery_margin": metrics["min_battery_margin"],
                                    "battery_margin_at_first_charge": metrics["battery_margin_at_first_charge"],
                                    "low_battery_steps": metrics["low_battery_steps"],
                                    "low_battery_route_progress_mean": metrics["low_battery_route_progress_mean"],
                                    "charge_route_found_rate": metrics["charge_route_found_rate"],
                                    "route_stall_steps_total": metrics["route_stall_steps_total"],
                                    "max_charge_stall_steps": metrics["max_charge_stall_steps"],
                                    "dock_contact_without_charge": metrics["dock_contact_without_charge"],
                                    "charge_guidance_steps": metrics["charge_guidance_steps"],
                                    "charge_route_found_steps": metrics["charge_route_found_steps"],
                                }
                            }
                        )
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = next_obs_data
