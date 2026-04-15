#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
"""

import heapq

import numpy as np


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


def _signed_norm(v, scale):
    scale = max(float(scale), 1.0)
    return float(np.clip(float(v) / scale, -1.0, 1.0))


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _first_or_default(value):
    items = _as_list(value)
    return items[0] if items else {}


def _as_point(value):
    if not isinstance(value, dict):
        return None
    x = value.get("x")
    z = value.get("z")
    if x is None or z is None:
        return None
    return int(x), int(z)


class Preprocessor:
    GRID_SIZE = 128
    VIEW_SIZE = 21
    VIEW_HALF = VIEW_SIZE // 2
    MAX_VISIT_COUNT = 20
    BLOCKED_CELL = 0
    UNKNOWN_CELL = -1
    ACTION_DELTAS = (
        (1, 0),   # 0 right
        (1, -1),  # 1 up-right
        (0, -1),  # 2 up
        (-1, -1), # 3 up-left
        (-1, 0),  # 4 left
        (-1, 1),  # 5 down-left
        (0, 1),   # 6 down
        (1, 1),   # 7 down-right
    )
    OPPOSITE_ACTIONS = (4, 5, 6, 7, 0, 1, 2, 3)
    DELTA_TO_ACTION = {delta: idx for idx, delta in enumerate(ACTION_DELTAS)}

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 2000

        self.battery = 200
        self.battery_max = 200
        self.remaining_charge = 200
        self.prev_remaining_charge = 200

        self.cur_pos = (0, 0)
        self.last_action = -1
        self.prev_action = -1

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1
        self.step_cleaned_count = 0

        self.total_charger = 0
        self.charge_count = 0
        self.prev_charge_count = 0
        self.npc_count = 0
        self.total_map = 1
        self.map_random = 0

        self.explored_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        self.memory_map = np.full((self.GRID_SIZE, self.GRID_SIZE), -1, dtype=np.int8)
        self.visit_count = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int16)

        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0
        self.nearest_npc_dist = 200.0
        self.last_nearest_npc_dist = 200.0
        self.charger_route_dist = 200.0
        self.last_charger_route_dist = 200.0
        self.explore_route_dist = 200.0
        self.last_explore_route_dist = 200.0
        self.explored_cells = 0
        self.prev_explored_cells = 0
        self.expand_hold_until_step = -1
        self.post_charge_expand_until_step = -1
        self.post_charge_release_until_step = -1
        self._expand_focus_target = None
        self._expand_focus_reason = ""
        self.prev_charge_cycle_explore_gain = 0
        self.prev_charge_cycle_clean_gain = 0
        self.prev_charge_cycle_step_span = 0
        self.charge_cycle_explore_start = 0
        self.charge_cycle_clean_start = 0
        self.charge_cycle_start_step = 0

        self._view_map = np.zeros((self.VIEW_SIZE, self.VIEW_SIZE), dtype=np.float32)
        self._legal_act = [1] * 8
        self._npc_positions = []
        self._charger_positions = []

        self.return_to_charge_mode = False
        self._charge_guidance = self._empty_charge_guidance()
        self._explore_guidance = self._empty_explore_guidance()
        self._npc_guidance = self._empty_npc_guidance()
        self._strategy_state = self._empty_strategy_state()
        self._charge_route_cache = None
        self._explore_route_cache = None
        self._charge_progress_stall_steps = 0
        self._last_charge_target_pos = None
        self._charge_target_lock_pos = None
        self._charge_target_lock_until_step = -1
        self._charge_route_commit_target_pos = None
        self._charge_route_commit_until_step = -1

    def _empty_charge_guidance(self):
        return {
            "should_return": False,
            "reason": "",
            "target_pos": None,
            "target_action": None,
            "target_dist": None,
            "battery_margin": None,
            "activation_buffer": 0,
            "release_buffer": 0,
            "urgency": 0.0,
            "path_found": False,
            "soft_radius": 0,
            "hard_radius": 0,
            "dock_mode": False,
            "dock_radius": 0,
            "return_mode": "",
            "recovery_mode": "",
            "controller_mode": "",
            "force_action": None,
            "control_actions": [],
            "route_source": "",
            "route_reliable": False,
            "first_charge_phase": False,
            "first_charge_stage": "",
            "first_charge_budget_buffer": 0,
            "first_charge_budget_margin": None,
            "first_charge_step_limit": 0,
            "first_charge_out_radius": 0,
            "charge_rule_control": False,
            "charge_stall_steps": 0,
        }

    def _empty_explore_guidance(self):
        return {
            "active": False,
            "should_pull_back": False,
            "mode": "",
            "reason": "",
            "target_pos": None,
            "target_action": None,
            "target_dist": None,
            "target_dx": 0.0,
            "target_dz": 0.0,
            "force_action": None,
            "soft_radius": 0,
            "hard_radius": 0,
            "desired_radius": 0,
            "intensity": 0.0,
            "path_found": False,
            "route_source": "",
            "hold_active": False,
            "hold_steps_left": 0,
            "post_charge_expand": False,
        }

    def _empty_npc_guidance(self):
        return {
            "should_evade": False,
            "reason": "",
            "target_action": None,
            "nearest_dist": None,
            "safest_next_dist": None,
            "danger_level": 0.0,
            "action_weights": [1.0] * len(self.ACTION_DELTAS),
        }

    def _empty_strategy_state(self):
        return {
            "mode_name": "clean",
            "mode_intensity": 0.0,
            "clean_mode": True,
            "expand_mode": False,
            "return_mode": False,
            "dock_mode": False,
        }

    def _activate_expand_hold(self, hold_steps):
        hold_steps = max(int(hold_steps), 0)
        if hold_steps <= 0:
            return
        self.expand_hold_until_step = max(self.expand_hold_until_step, self.step_no + hold_steps - 1)

    def _activate_post_charge_expand(self, hold_steps):
        hold_steps = max(int(hold_steps), 0)
        if hold_steps <= 0:
            return
        self.post_charge_expand_until_step = max(self.post_charge_expand_until_step, self.step_no + hold_steps - 1)

    def _activate_post_charge_release(self, hold_steps):
        hold_steps = max(int(hold_steps), 0)
        if hold_steps <= 0:
            return
        self.post_charge_release_until_step = max(
            self.post_charge_release_until_step,
            self.step_no + hold_steps - 1,
        )

    def _clear_post_charge_release(self):
        self.post_charge_release_until_step = -1

    def _lock_charge_target(self, target_pos, hold_steps):
        if target_pos is None:
            return
        self._charge_target_lock_pos = tuple(target_pos)
        hold_steps = max(int(hold_steps), 0)
        if hold_steps > 0:
            self._charge_target_lock_until_step = max(
                self._charge_target_lock_until_step,
                self.step_no + hold_steps - 1,
            )

    def _clear_charge_target_lock(self):
        self._charge_target_lock_pos = None
        self._charge_target_lock_until_step = -1

    def _get_charge_target_lock(self):
        if self._charge_target_lock_pos is None:
            return None
        if self.step_no > self._charge_target_lock_until_step:
            self._clear_charge_target_lock()
            return None
        if self._charge_target_lock_pos not in self._charger_positions:
            self._clear_charge_target_lock()
            return None
        return self._charge_target_lock_pos

    def _commit_charge_route(self, target_pos, hold_steps):
        if target_pos is None:
            return
        self._charge_route_commit_target_pos = tuple(target_pos)
        hold_steps = max(int(hold_steps), 0)
        if hold_steps > 0:
            self._charge_route_commit_until_step = max(
                self._charge_route_commit_until_step,
                self.step_no + hold_steps - 1,
            )

    def _clear_charge_route_commit(self):
        self._charge_route_commit_target_pos = None
        self._charge_route_commit_until_step = -1

    def _is_charge_route_committed(self, target_pos):
        if target_pos is None or self._charge_route_commit_target_pos is None:
            return False
        if tuple(target_pos) != self._charge_route_commit_target_pos:
            return False
        if self.step_no > self._charge_route_commit_until_step:
            self._clear_charge_route_commit()
            return False
        return True

    def _has_expand_hold(self):
        return bool(self.step_no <= self.expand_hold_until_step)

    def _has_post_charge_expand(self):
        return bool(self.step_no <= self.post_charge_expand_until_step)

    def _has_post_charge_release(self):
        return bool(self.step_no <= self.post_charge_release_until_step)

    def _in_first_charge_phase(self):
        return bool(self.charge_count <= 0)

    def _is_reverse_action(self, action):
        if not (0 <= int(action) < len(self.ACTION_DELTAS)):
            return False
        if not (0 <= int(self.last_action) < len(self.OPPOSITE_ACTIONS)):
            return False
        return int(action) == int(self.OPPOSITE_ACTIONS[int(self.last_action)])

    def _is_ping_pong_action(self, action):
        if not (0 <= int(action) < len(self.ACTION_DELTAS)):
            return False
        if not (0 <= int(self.prev_action) < len(self.OPPOSITE_ACTIONS)):
            return False
        if not (0 <= int(self.last_action) < len(self.OPPOSITE_ACTIONS)):
            return False
        return (
            int(self.last_action) == int(self.OPPOSITE_ACTIONS[int(self.prev_action)])
            and int(action) == int(self.prev_action)
        )

    def _clear_expand_focus(self):
        self._expand_focus_target = None
        self._expand_focus_reason = ""

    def _did_charge_this_step(self, target_dist=None):
        if self.charge_count > self.prev_charge_count:
            return True
        if target_dist is None or int(target_dist) > 1:
            return False
        full_charge_jump = max(30, int(self.battery_max * 0.18))
        return bool(
            self.prev_remaining_charge <= self.battery_max - full_charge_jump
            and self.remaining_charge >= self.battery_max - 1
            and (self.remaining_charge - self.prev_remaining_charge) >= full_charge_jump
        )

    def _handle_charge_success(self):
        self.prev_charge_cycle_explore_gain = max(0, self.explored_cells - self.charge_cycle_explore_start)
        self.prev_charge_cycle_clean_gain = max(0, self.dirt_cleaned - self.charge_cycle_clean_start)
        self.prev_charge_cycle_step_span = max(1, self.step_no - self.charge_cycle_start_step)
        self.charge_cycle_explore_start = self.explored_cells
        self.charge_cycle_clean_start = self.dirt_cleaned
        self.charge_cycle_start_step = self.step_no
        self._apply_post_charge_sequence()
        self._clear_charge_target_lock()
        self._clear_charge_route_commit()
        self.return_to_charge_mode = False
        self._charge_progress_stall_steps = 0
        self._last_charge_target_pos = None

    def _apply_post_charge_sequence(self):
        inefficient_cycle = bool(
            self.charge_count >= 4
            and self.prev_charge_cycle_explore_gain <= 10
            and self.prev_charge_cycle_clean_gain <= 12
        )
        release_steps = 4 if self.charge_count <= 2 else 3
        expand_steps = 14 if self.charge_count <= 2 else 10
        hold_steps = 10 if self.charge_count <= 2 else 8
        if inefficient_cycle:
            release_steps = 0
            expand_steps = min(expand_steps, 6)
            hold_steps = min(hold_steps, 4)

        if release_steps > 0:
            self._activate_post_charge_release(release_steps)
        else:
            self._clear_post_charge_release()
        self._activate_post_charge_expand(expand_steps)
        self._activate_expand_hold(hold_steps)
        self._clear_expand_focus()

    def pb2struct(self, env_obs, last_action):
        observation = env_obs.get("observation", env_obs)
        frame_state = observation.get("frame_state") or {}
        env_info = observation.get("env_info") or {}

        hero = _first_or_default(frame_state.get("heroes"))
        hero_pos = _as_point(hero.get("pos", {}))
        if hero_pos is None:
            hero_pos = _as_point(env_info.get("pos", {})) or self.cur_pos

        self.step_no = int(observation.get("step_no", env_info.get("step_no", self.step_no)))
        self.max_step = max(int(env_info.get("max_step", self.max_step)), 1)
        self.cur_pos = hero_pos
        self.prev_action = self.last_action
        self.last_action = last_action

        self.prev_remaining_charge = self.remaining_charge
        self.prev_charge_count = self.charge_count

        self.battery = int(hero.get("battery", env_info.get("remaining_charge", self.battery)))
        self.remaining_charge = int(env_info.get("remaining_charge", self.battery))
        self.battery_max = max(int(hero.get("battery_max", env_info.get("battery_max", self.battery_max))), 1)

        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero.get("dirt_cleaned", hero.get("score", self.dirt_cleaned)))
        self.total_dirt = max(int(env_info.get("total_dirt", self.total_dirt)), 1)

        self.total_charger = int(env_info.get("total_charger", self.total_charger))
        self.charge_count = int(env_info.get("charge_count", self.charge_count))
        self.npc_count = int(env_info.get("npc_count", self.npc_count))
        self.total_map = max(int(env_info.get("total_map", self.total_map)), 1)
        self.map_random = int(env_info.get("map_random", self.map_random))
        self.step_cleaned_count = len(env_info.get("step_cleaned_cells") or [])

        legal_action = observation.get("legal_action")
        if legal_action is None:
            legal_action = observation.get("legal_act")
        if legal_action is None:
            legal_action = env_obs.get("legal_action")
        if legal_action is None:
            legal_action = env_obs.get("legal_act")
        self._legal_act = [int(x) for x in (legal_action or [1] * 8)]

        self._npc_positions = []
        for npc in _as_list(frame_state.get("npcs")):
            pos = _as_point((npc or {}).get("pos", {}))
            if pos is not None:
                self._npc_positions.append(pos)
        self.npc_count = max(self.npc_count, len(self._npc_positions))

        self._charger_positions = []
        for organ in _as_list(frame_state.get("organs")):
            if not isinstance(organ, dict):
                continue
            sub_type = organ.get("sub_type")
            if sub_type not in (None, 1):
                continue
            pos = _as_point(organ.get("pos", {}))
            if pos is not None:
                self._charger_positions.append(pos)
        self.total_charger = max(self.total_charger, len(self._charger_positions))

        map_info = observation.get("map_info")
        self.prev_explored_cells = self.explored_cells
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            self._update_memory()

        self._mark_visit()
        self._update_cleaned_memory(env_info.get("step_cleaned_cells") or [])
        self.explored_cells = int(self.explored_map.sum())
        self.last_charger_route_dist = self.charger_route_dist
        self.last_explore_route_dist = self.explore_route_dist
        self.last_nearest_npc_dist = self.nearest_npc_dist
        _, npc_dist = self._nearest_point(self._npc_positions)
        self.nearest_npc_dist = 200.0 if npc_dist is None else float(npc_dist)
        self._charge_guidance = self._build_charge_guidance()
        self.charger_route_dist = (
            200.0 if self._charge_guidance["target_dist"] is None else float(self._charge_guidance["target_dist"])
        )
        self._update_charge_progress_state()
        self._explore_guidance = self._build_explore_guidance()
        self.explore_route_dist = (
            200.0 if self._explore_guidance["target_dist"] is None else float(self._explore_guidance["target_dist"])
        )
        self._npc_guidance = self._build_npc_guidance()
        self._strategy_state = self._build_strategy_state()

    def _update_charge_progress_state(self):
        guidance = self._charge_guidance
        target_pos = guidance.get("target_pos")
        target_dist = guidance.get("target_dist")
        should_return = bool(guidance.get("should_return"))

        if not should_return or target_pos is None or target_dist is None:
            self._charge_progress_stall_steps = 0
            self._last_charge_target_pos = target_pos
            self._clear_charge_route_commit()
            return

        route_reliable = bool(guidance.get("route_reliable"))
        same_target = bool(target_pos == self._last_charge_target_pos)
        if not same_target:
            self._clear_charge_route_commit()

        progress = None
        if self.last_charger_route_dist < 200.0:
            progress = float(self.last_charger_route_dist) - float(target_dist)

        if progress is None or not same_target:
            self._charge_progress_stall_steps = 0
        elif progress >= 1.0:
            self._charge_progress_stall_steps = 0
        elif progress > 0.0:
            decay = 2 if int(target_dist) <= 5 else 1
            self._charge_progress_stall_steps = max(self._charge_progress_stall_steps - decay, 0)
        elif progress == 0.0:
            stall_inc = 2 if int(target_dist) <= 5 else 1
            self._charge_progress_stall_steps = min(self._charge_progress_stall_steps + stall_inc, 12)
        else:
            if int(target_dist) <= 2:
                stall_inc = 4
            elif int(target_dist) <= 5:
                stall_inc = 3
            else:
                stall_inc = 2 if route_reliable else 3
            self._charge_progress_stall_steps = min(self._charge_progress_stall_steps + stall_inc, 12)

        stall_clear_threshold = 5 if route_reliable else 4
        if int(target_dist) > 2 and self._charge_progress_stall_steps >= stall_clear_threshold:
            self._charge_route_cache = None
            self._clear_charge_route_commit()

        self._last_charge_target_pos = target_pos

    def _update_memory(self):
        hx, hz = self.cur_pos
        for row in range(self.VIEW_SIZE):
            for col in range(self.VIEW_SIZE):
                gx = hx - self.VIEW_HALF + row
                gz = hz - self.VIEW_HALF + col
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    cell = int(self._view_map[row, col])
                    self.explored_map[gx, gz] = 1
                    self.memory_map[gx, gz] = cell

    def _mark_visit(self):
        hx, hz = self.cur_pos
        if 0 <= hx < self.GRID_SIZE and 0 <= hz < self.GRID_SIZE:
            self.visit_count[hx, hz] = min(self.visit_count[hx, hz] + 1, 32767)

    def _update_cleaned_memory(self, cleaned_cells):
        for cell in cleaned_cells:
            point = _as_point(cell)
            if point is None:
                continue
            x, z = point
            if 0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE:
                self.explored_map[x, z] = 1
                self.memory_map[x, z] = 1

    def _get_local_view_feature(self):
        return (self._view_map / 2.0).astype(np.float32).flatten()

    def _nearest_point(self, positions):
        if not positions:
            return None, None

        hx, hz = self.cur_pos
        best_point = None
        best_dist = None
        for x, z in positions:
            dist = max(abs(x - hx), abs(z - hz))
            if best_dist is None or dist < best_dist:
                best_point = (x, z)
                best_dist = dist
        return best_point, best_dist

    def _calc_nearest_dirt_dist(self):
        dirt_coords = np.argwhere(self._view_map == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def _get_visit_penalty(self, pos):
        x, z = pos
        if 0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE:
            return min(int(self.visit_count[x, z]), self.MAX_VISIT_COUNT)
        return self.MAX_VISIT_COUNT

    def _npc_risk_at(self, pos):
        if not self._npc_positions:
            return 0.0, None

        px, pz = pos
        nearest_dist = None
        for x, z in self._npc_positions:
            dist = max(abs(x - px), abs(z - pz))
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist

        if nearest_dist is None:
            return 0.0, None
        if nearest_dist <= 1:
            return 6.0, nearest_dist
        if nearest_dist == 2:
            return 2.0, nearest_dist
        if nearest_dist == 3:
            return 0.8, nearest_dist
        if nearest_dist == 4:
            return 0.25, nearest_dist
        return 0.0, nearest_dist

    def _is_blocked(self, pos, allow_unknown=True):
        x, z = pos
        if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
            return True
        if pos in self._charger_positions:
            return False
        cell = int(self.memory_map[x, z])
        if cell == self.UNKNOWN_CELL:
            return not allow_unknown
        return cell == self.BLOCKED_CELL

    def _chebyshev_dist(self, start_pos, target_pos):
        return max(abs(target_pos[0] - start_pos[0]), abs(target_pos[1] - start_pos[1]))

    def _route_step_penalty(self, pos):
        x, z = pos
        cost = 0.0

        cell = int(self.memory_map[x, z])
        if cell == self.UNKNOWN_CELL:
            cost += 0.35

        visit = self._get_visit_penalty(pos)
        cost += min(visit, self.MAX_VISIT_COUNT) * 0.02

        npc_risk, _ = self._npc_risk_at(pos)
        cost += npc_risk * 0.75
        return cost

    def _get_route_cache(self, cache_name):
        if cache_name == "explore":
            return self._explore_route_cache
        return self._charge_route_cache

    def _set_route_cache(self, cache_name, cache):
        if cache_name == "explore":
            self._explore_route_cache = cache
        else:
            self._charge_route_cache = cache

    def _trim_cached_route(self, target_pos, cache_name="charge", allow_unknown=True):
        cache = self._get_route_cache(cache_name)
        if not cache or cache.get("target_pos") != target_pos:
            return None
        if bool(cache.get("allow_unknown", True)) != bool(allow_unknown):
            return None

        path = list(cache.get("path") or [])
        actions = list(cache.get("actions") or [])
        if not path:
            return None

        try:
            idx = path.index(self.cur_pos)
        except ValueError:
            return None

        trimmed_path = path[idx:]
        trimmed_actions = actions[idx:]
        route = {
            "target_pos": target_pos,
            "path": trimmed_path,
            "actions": trimmed_actions,
            "first_action": trimmed_actions[0] if trimmed_actions else None,
            "path_steps": max(len(trimmed_path) - 1, 0),
            "path_found": True,
        }

        next_action = route["first_action"]
        if next_action is not None and (next_action >= len(self._legal_act) or not self._legal_act[next_action]):
            return None

        self._set_route_cache(
            cache_name,
            {
            "target_pos": target_pos,
            "path": trimmed_path,
            "actions": trimmed_actions,
            "allow_unknown": bool(allow_unknown),
            },
        )
        return route

    def _plan_path_to_target(self, target_pos, allow_unknown=True):
        if target_pos is None:
            return None

        start_pos = self.cur_pos
        if start_pos == target_pos:
            return {
                "target_pos": target_pos,
                "path": [start_pos],
                "actions": [],
                "first_action": None,
                "path_steps": 0,
                "path_found": True,
            }

        base_dist = self._chebyshev_dist(start_pos, target_pos)
        if base_dist <= 18:
            margin = 12
        elif base_dist <= 36:
            margin = 16
        else:
            margin = 20

        min_x = max(0, min(start_pos[0], target_pos[0]) - margin)
        max_x = min(self.GRID_SIZE - 1, max(start_pos[0], target_pos[0]) + margin)
        min_z = max(0, min(start_pos[1], target_pos[1]) - margin)
        max_z = min(self.GRID_SIZE - 1, max(start_pos[1], target_pos[1]) + margin)

        frontier = [(float(base_dist), 0, 0.0, start_pos)]
        best_cost = {start_pos: 0.0}
        parents = {start_pos: (None, None)}
        max_expansions = max(3000, (max_x - min_x + 1) * (max_z - min_z + 1))
        expansions = 0

        while frontier and expansions < max_expansions:
            _, cur_steps, cur_cost, cur_pos = heapq.heappop(frontier)
            if cur_cost > best_cost.get(cur_pos, float("inf")) + 1e-9:
                continue
            if cur_pos == target_pos:
                break

            expansions += 1
            cx, cz = cur_pos
            for action, (dx, dz) in enumerate(self.ACTION_DELTAS):
                nx = cx + dx
                nz = cz + dz
                if not (min_x <= nx <= max_x and min_z <= nz <= max_z):
                    continue

                next_pos = (nx, nz)
                if self._is_blocked(next_pos, allow_unknown=allow_unknown):
                    continue

                if dx != 0 and dz != 0:
                    side_pos_1 = (cx + dx, cz)
                    side_pos_2 = (cx, cz + dz)
                    if self._is_blocked(side_pos_1, allow_unknown=allow_unknown) and self._is_blocked(
                        side_pos_2, allow_unknown=allow_unknown
                    ):
                        continue

                next_cost = cur_cost + 1.0 + self._route_step_penalty(next_pos)
                prev_cost = best_cost.get(next_pos)
                if prev_cost is not None and next_cost >= prev_cost - 1e-9:
                    continue

                best_cost[next_pos] = next_cost
                parents[next_pos] = (cur_pos, action)
                next_steps = cur_steps + 1
                heuristic = float(self._chebyshev_dist(next_pos, target_pos))
                heapq.heappush(frontier, (next_cost + heuristic, next_steps, next_cost, next_pos))

        if target_pos not in parents:
            return None

        path = [target_pos]
        actions = []
        cur_pos = target_pos
        while True:
            prev_pos, action = parents[cur_pos]
            if prev_pos is None:
                break
            actions.append(action)
            path.append(prev_pos)
            cur_pos = prev_pos

        path.reverse()
        actions.reverse()
        return {
            "target_pos": target_pos,
            "path": path,
            "actions": actions,
            "first_action": actions[0] if actions else None,
            "path_steps": len(actions),
            "path_found": True,
        }

    def _get_route_to_target(self, target_pos, cache_name="charge", allow_unknown=True):
        cached_route = self._trim_cached_route(target_pos, cache_name=cache_name, allow_unknown=allow_unknown)
        if cached_route is not None:
            return cached_route

        route = self._plan_path_to_target(target_pos, allow_unknown=allow_unknown)
        if route is None:
            self._set_route_cache(cache_name, None)
            return None

        self._set_route_cache(
            cache_name,
            {
            "target_pos": target_pos,
            "path": list(route["path"]),
            "actions": list(route["actions"]),
            "allow_unknown": bool(allow_unknown),
            },
        )
        return route

    def _get_best_route_to_charger(self, allow_unknown=True, preferred_target=None):
        if not self._charger_positions:
            self._charge_route_cache = None
            self._clear_charge_route_commit()
            return None

        if preferred_target is not None:
            preferred_target = tuple(preferred_target)
            if preferred_target not in self._charger_positions:
                return None

            cached_route = self._trim_cached_route(
                preferred_target, cache_name="charge", allow_unknown=allow_unknown
            )
            if cached_route is not None:
                return cached_route

            preferred_route = self._plan_path_to_target(preferred_target, allow_unknown=allow_unknown)
            if preferred_route is None:
                self._charge_route_cache = None
                self._clear_charge_route_commit()
                return None

            self._set_route_cache(
                "charge",
                {
                    "target_pos": preferred_route["target_pos"],
                    "path": list(preferred_route["path"]),
                    "actions": list(preferred_route["actions"]),
                    "allow_unknown": bool(allow_unknown),
                },
            )
            self._commit_charge_route(preferred_route["target_pos"], hold_steps=6)
            return preferred_route

        best_route = None
        cache = self._get_route_cache("charge")
        if cache and cache.get("target_pos") in self._charger_positions:
            cached_route = self._trim_cached_route(
                cache["target_pos"], cache_name="charge", allow_unknown=allow_unknown
            )
            if cached_route is not None:
                best_route = cached_route

        sorted_targets = sorted(self._charger_positions, key=lambda pos: self._chebyshev_dist(self.cur_pos, pos))
        for target_pos in sorted_targets:
            if best_route is not None and target_pos == best_route["target_pos"]:
                continue

            route = self._plan_path_to_target(target_pos, allow_unknown=allow_unknown)
            if route is None:
                continue

            if best_route is None:
                best_route = route
                continue

            best_steps = int(best_route["path_steps"])
            route_steps = int(route["path_steps"])
            if route_steps < best_steps:
                best_route = route
                continue

            if route_steps == best_steps:
                if self._chebyshev_dist(self.cur_pos, target_pos) < self._chebyshev_dist(self.cur_pos, best_route["target_pos"]):
                    best_route = route

        if best_route is None:
            self._charge_route_cache = None
            self._clear_charge_route_commit()
            return None

        self._set_route_cache(
            "charge",
            {
            "target_pos": best_route["target_pos"],
            "path": list(best_route["path"]),
            "actions": list(best_route["actions"]),
            "allow_unknown": bool(allow_unknown),
            },
        )
        self._commit_charge_route(best_route["target_pos"], hold_steps=6)
        return best_route

    def _count_neighbor_cells(self, pos, radius, cell_value):
        x, z = pos
        count = 0
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dz == 0:
                    continue
                nx = x + dx
                nz = z + dz
                if 0 <= nx < self.GRID_SIZE and 0 <= nz < self.GRID_SIZE:
                    if int(self.memory_map[nx, nz]) == cell_value:
                        count += 1
        return count

    def _select_explore_target(self, charger_pos, desired_radius, soft_radius, hard_radius, budget_limit=None):
        if charger_pos is None:
            self._explore_route_cache = None
            return None

        min_radius = max(3, desired_radius - 6)
        max_radius = min(hard_radius, desired_radius + 6)
        search_radius = max_radius + 2
        min_x = max(0, charger_pos[0] - search_radius)
        max_x = min(self.GRID_SIZE - 1, charger_pos[0] + search_radius)
        min_z = max(0, charger_pos[1] - search_radius)
        max_z = min(self.GRID_SIZE - 1, charger_pos[1] + search_radius)

        candidates = []
        for x in range(min_x, max_x + 1):
            for z in range(min_z, max_z + 1):
                pos = (x, z)
                cell = int(self.memory_map[x, z])
                if cell in (self.BLOCKED_CELL, self.UNKNOWN_CELL):
                    continue

                charger_dist = self._chebyshev_dist(charger_pos, pos)
                if charger_dist < min_radius or charger_dist > max_radius:
                    continue

                frontier_count = self._count_neighbor_cells(pos, radius=1, cell_value=self.UNKNOWN_CELL)
                nearby_dirt = self._count_neighbor_cells(pos, radius=2, cell_value=2)
                if frontier_count <= 0 and nearby_dirt <= 0 and cell != 2:
                    continue

                visit_penalty = self._get_visit_penalty(pos)
                npc_risk, _ = self._npc_risk_at(pos)
                route_hint = self._chebyshev_dist(self.cur_pos, pos)
                ring_fit = 1.0 - min(abs(charger_dist - desired_radius) / max(max_radius - min_radius, 1), 1.0)

                score = 0.0
                score += 0.70 * float(frontier_count)
                score += 1.15 * float(nearby_dirt)
                if cell == 2:
                    score += 2.2
                score += 2.4 * ring_fit
                score -= 0.16 * min(visit_penalty, self.MAX_VISIT_COUNT)
                score -= 0.05 * route_hint
                score -= 1.30 * npc_risk
                if pos == self.cur_pos and cell != 2:
                    score -= 0.6
                candidates.append((score, pos))

        if not candidates:
            self._explore_route_cache = None
            return None

        candidates.sort(reverse=True)
        best_target = None
        for base_score, pos in candidates[:12]:
            route = self._get_route_to_target(pos, cache_name="explore")
            if route is not None:
                route_steps = int(route["path_steps"])
                if budget_limit is not None and (route_steps + charger_dist) > budget_limit:
                    continue
                total_score = base_score - 0.06 * route_steps
                candidate = {
                    "score": total_score,
                    "target_pos": pos,
                    "target_action": route["first_action"],
                    "target_dist": route_steps,
                    "path_found": True,
                    "route_source": "planned",
                    "reason": "frontier_expand",
                }
            else:
                target_action = self._choose_action_towards(pos)
                if target_action is None:
                    continue
                route_steps = self._chebyshev_dist(self.cur_pos, pos)
                if budget_limit is not None and (route_steps + charger_dist) > budget_limit:
                    continue
                total_score = base_score - 0.10 * route_steps
                candidate = {
                    "score": total_score,
                    "target_pos": pos,
                    "target_action": target_action,
                    "target_dist": route_steps,
                    "path_found": False,
                    "route_source": "greedy",
                    "reason": "frontier_expand_greedy",
                }

            if best_target is None or candidate["score"] > best_target["score"]:
                best_target = candidate

        if best_target is None:
            self._explore_route_cache = None
        return best_target

    def _get_expand_focus_candidate(self, charger_pos, desired_radius, hard_radius, budget_limit=None):
        target_pos = self._expand_focus_target
        if target_pos is None or charger_pos is None:
            return None

        x, z = target_pos
        if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
            return None
        if self._is_blocked(target_pos):
            return None

        charger_dist = self._chebyshev_dist(charger_pos, target_pos)
        if charger_dist < max(3, desired_radius - 10) or charger_dist > min(hard_radius, desired_radius + 8):
            return None

        cell = int(self.memory_map[x, z])
        frontier_count = self._count_neighbor_cells(target_pos, radius=1, cell_value=self.UNKNOWN_CELL)
        nearby_dirt = self._count_neighbor_cells(target_pos, radius=2, cell_value=2)
        if cell != 2 and frontier_count <= 0 and nearby_dirt <= 0:
            return None

        npc_risk, _ = self._npc_risk_at(target_pos)
        if npc_risk >= 2.0:
            return None

        route = self._get_route_to_target(target_pos, cache_name="explore")
        if route is not None:
            if budget_limit is not None and (int(route["path_steps"]) + charger_dist) > budget_limit:
                return None
            return {
                "score": 0.0,
                "target_pos": target_pos,
                "target_action": route["first_action"],
                "target_dist": int(route["path_steps"]),
                "path_found": True,
                "route_source": "planned_hold",
                "reason": self._expand_focus_reason or "expand_hold",
            }

        target_action = self._choose_action_towards(target_pos)
        if target_action is None:
            return None

        if budget_limit is not None and (int(self._chebyshev_dist(self.cur_pos, target_pos)) + charger_dist) > budget_limit:
            return None

        return {
            "score": 0.0,
            "target_pos": target_pos,
            "target_action": target_action,
            "target_dist": int(self._chebyshev_dist(self.cur_pos, target_pos)),
            "path_found": False,
            "route_source": "greedy_hold",
            "reason": self._expand_focus_reason or "expand_hold_greedy",
        }

    def _build_strategy_state(self):
        charge = self._charge_guidance
        explore = self._explore_guidance

        state = self._empty_strategy_state()
        if charge["should_return"]:
            state["mode_name"] = "dock" if charge["dock_mode"] else "return"
            state["mode_intensity"] = float(charge["urgency"])
            state["clean_mode"] = False
            state["return_mode"] = True
            state["dock_mode"] = bool(charge["dock_mode"])
            return state

        if explore["mode"] in ("expand_frontier", "post_charge_release", "recenter"):
            state["mode_name"] = "expand"
            state["clean_mode"] = False
            state["expand_mode"] = True
            state["mode_intensity"] = float(explore["intensity"])
        else:
            local_dirt_ratio = float(np.count_nonzero(self._view_map == 2)) / float(max(self._view_map.size, 1))
            state["mode_intensity"] = float(np.clip(local_dirt_ratio * 3.0, 0.0, 1.0))
        return state

    def _get_cleaning_radius_limits(self):
        reserve = max(22, int(self.battery_max * 0.10))
        base_soft = max(18, int(self.battery_max * 0.13))
        base_hard = max(base_soft + 8, int(self.battery_max * 0.20))

        free_energy = max(0, self.remaining_charge - reserve)
        energy_soft = max(12, int(free_energy * 0.50))
        energy_hard = max(18, int(free_energy * 0.68))

        soft_radius = int(np.clip(min(base_soft, energy_soft), 12, 36))
        hard_candidate = min(base_hard, energy_hard)
        hard_radius = int(np.clip(max(soft_radius + 6, hard_candidate), soft_radius + 6, 44))
        return soft_radius, hard_radius

    def _rank_charge_actions(self, target_pos, target_action=None, dock_mode=False, strict_progress=False):
        if target_pos is None:
            return []

        hx, hz = self.cur_pos
        tx, tz = target_pos
        current_dist = max(abs(tx - hx), abs(tz - hz))
        current_l1 = abs(tx - hx) + abs(tz - hz)
        ranked_actions = []

        for action, (dx, dz) in enumerate(self.ACTION_DELTAS):
            if action >= len(self._legal_act) or not self._legal_act[action]:
                continue

            nx = hx + dx
            nz = hz + dz
            next_pos = (nx, nz)
            next_dist = max(abs(tx - nx), abs(tz - nz))
            progress = current_dist - next_dist
            next_l1 = abs(tx - nx) + abs(tz - nz)
            l1_progress = current_l1 - next_l1
            visit_penalty = self._get_visit_penalty(next_pos)
            npc_risk, next_npc_dist = self._npc_risk_at(next_pos)
            preferred_action = target_action is not None and action == int(target_action)

            if next_npc_dist is not None and next_npc_dist <= 1:
                continue
            if dock_mode and not preferred_action:
                if next_dist > current_dist:
                    continue
                if current_dist <= 3 and next_dist >= current_dist and next_pos != target_pos:
                    continue
                if current_dist <= 2 and next_l1 >= current_l1 and next_pos != target_pos:
                    continue
            if strict_progress and not preferred_action:
                if progress < 0:
                    continue
                if progress == 0 and l1_progress <= 0 and next_pos != target_pos:
                    continue

            score = 0.0
            score += (100.0 if dock_mode else 72.0) * progress
            score += (14.0 if dock_mode else 8.0) * l1_progress
            score -= (2.5 if dock_mode else 1.2) * next_dist
            score -= (0.60 if dock_mode else 0.35) * visit_penalty
            score -= (18.0 if dock_mode else 14.0) * npc_risk
            if strict_progress and (not preferred_action) and progress <= 0 and next_pos != target_pos:
                score -= 12.0 if dock_mode else 7.0

            if next_pos == target_pos:
                score += 80.0 if dock_mode else 30.0
            elif dock_mode and next_dist <= 1:
                score += 24.0

            if target_action is not None and action == int(target_action):
                score += 16.0 if dock_mode else 10.0
            if self._is_reverse_action(action):
                score -= 10.0 if (dock_mode or strict_progress) else 4.0
            if self._is_ping_pong_action(action):
                score -= 14.0 if (dock_mode or strict_progress) else 5.0
            if action == self.last_action and progress >= 0:
                score += 1.0

            ranked_actions.append((score, progress, l1_progress, -next_dist, action))

        ranked_actions.sort(reverse=True)
        return [int(action) for _, _, _, _, action in ranked_actions]

    def _choose_action_towards(self, target_pos):
        if target_pos is None:
            return None

        hx, hz = self.cur_pos
        tx, tz = target_pos
        current_dist = max(abs(tx - hx), abs(tz - hz))
        current_l1 = abs(tx - hx) + abs(tz - hz)
        _, current_npc_dist = self._npc_risk_at((hx, hz))

        best_action = None
        best_score = None
        for action, (dx, dz) in enumerate(self.ACTION_DELTAS):
            if action >= len(self._legal_act) or not self._legal_act[action]:
                continue

            nx = hx + dx
            nz = hz + dz
            next_dist = max(abs(tx - nx), abs(tz - nz))
            progress = current_dist - next_dist
            next_l1 = abs(tx - nx) + abs(tz - nz)
            l1_progress = current_l1 - next_l1
            next_visit = self._get_visit_penalty((nx, nz))
            npc_risk, next_npc_dist = self._npc_risk_at((nx, nz))

            score = progress * 100.0 + l1_progress * 5.0 - next_dist
            score -= next_visit * 0.35
            score -= npc_risk * 18.0
            if current_npc_dist is not None and next_npc_dist is not None:
                score += (next_npc_dist - current_npc_dist) * 8.0
            if action == self.last_action:
                score += 0.05

            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _choose_action_away_from(self, target_pos):
        if target_pos is None:
            return None

        hx, hz = self.cur_pos
        tx, tz = target_pos
        current_dist = max(abs(tx - hx), abs(tz - hz))

        best_action = None
        best_score = None
        for action, (dx, dz) in enumerate(self.ACTION_DELTAS):
            if action >= len(self._legal_act) or not self._legal_act[action]:
                continue

            nx = hx + dx
            nz = hz + dz
            next_pos = (nx, nz)
            next_dist = max(abs(tx - nx), abs(tz - nz))
            outward = next_dist - current_dist
            visit_penalty = self._get_visit_penalty(next_pos)
            npc_risk, next_npc_dist = self._npc_risk_at(next_pos)
            if next_npc_dist is not None and next_npc_dist <= 1:
                continue

            score = outward * 100.0 + next_dist * 2.0
            score -= visit_penalty * 0.45
            score -= npc_risk * 18.0
            if action == self.last_action:
                score += 0.05

            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _is_valid_legal_action(self, action):
        return (
            action is not None
            and 0 <= int(action) < len(self._legal_act)
            and self._legal_act[int(action)]
        )

    def _estimate_charge_trip(self, quick_dist, target_dist, path_found, route_reliable, first_charge_phase):
        target_dist = int(target_dist if target_dist is not None else quick_dist)
        # First charge failures are still the dominant loss mode, so estimate the return
        # trip with a larger safety margin and let the agent turn back earlier.
        trip_dist = float(target_dist) + (7.0 if first_charge_phase else 4.0)
        if first_charge_phase and target_dist >= 6:
            trip_dist += 2.0
        if not path_found:
            trip_dist += float(
                np.clip(
                    quick_dist * (0.84 if first_charge_phase else 0.60) + (14.0 if first_charge_phase else 6.0),
                    16.0 if first_charge_phase else 10.0,
                    36.0 if first_charge_phase else 18.0,
                )
            )
        elif not route_reliable:
            trip_dist += float(
                np.clip(
                    quick_dist * (0.56 if first_charge_phase else 0.35) + (8.0 if first_charge_phase else 4.0),
                    12.0 if first_charge_phase else 8.0,
                    24.0 if first_charge_phase else 16.0,
                )
            )
        elif target_dist >= 10:
            trip_dist += 2.0
        if first_charge_phase and target_dist >= 12:
            trip_dist += 3.0
        return trip_dist

    def _resolve_charge_plan(self, target_pos, quick_dist, preferred_target=None, allow_unknown_fallback=False):
        target_dist = int(quick_dist)
        route_source = "quick"
        target_action = None
        path_found = False
        route_reliable = False

        route = self._get_best_route_to_charger(allow_unknown=False, preferred_target=preferred_target)
        if route is None and preferred_target is not None:
            route = self._get_best_route_to_charger(allow_unknown=False)
        if route is not None:
            target_pos = route["target_pos"]
            target_dist = int(route["path_steps"])
            target_action = route["first_action"]
            path_found = True
            route_reliable = True
            route_source = "planned_safe"
        elif allow_unknown_fallback:
            route = self._get_best_route_to_charger(allow_unknown=True, preferred_target=preferred_target)
            if route is None and preferred_target is not None:
                route = self._get_best_route_to_charger(allow_unknown=True)
            if route is not None:
                target_pos = route["target_pos"]
                target_dist = int(route["path_steps"])
                target_action = route["first_action"]
                path_found = True
                route_reliable = False
                route_source = "planned_unknown"

        if target_action is None:
            target_action = self._choose_action_towards(target_pos)
            if target_action is not None and not path_found:
                route_source = "greedy"

        return {
            "target_pos": target_pos,
            "target_action": target_action,
            "target_dist": int(target_dist),
            "path_found": bool(path_found),
            "route_reliable": bool(route_reliable),
            "route_source": route_source,
        }

    def _get_charge_activation_buffer(
        self,
        target_dist,
        path_found,
        route_reliable,
        first_charge_phase,
        current_visit,
        step_limit=0,
    ):
        buffer_value = max(28, int(self.battery_max * 0.16))
        buffer_value += min(16, max(0, int(target_dist)) // 2)
        if first_charge_phase:
            # Reserve extra battery for the first docking attempt, because a miss here
            # usually collapses the whole episode score.
            buffer_value += 14
            if step_limit > 0 and self.step_no >= max(step_limit - 20, 44):
                buffer_value += 8
        else:
            if self.charge_count >= 2:
                buffer_value += min(6, int(self.charge_count))
            if self.step_no >= 700:
                buffer_value += 5
            elif self.step_no >= 500:
                buffer_value += 3
            elif self.step_no >= 320:
                buffer_value += 1

        if current_visit >= 10:
            buffer_value += 2
        if self.nearest_npc_dist <= 2:
            buffer_value += 6
        elif self.nearest_npc_dist <= 4:
            buffer_value += 3
        if not path_found:
            buffer_value += 12
        elif not route_reliable:
            buffer_value += 7
        return int(min(self.battery_max - 1, buffer_value))

    def _build_charge_guidance_core(
        self,
        first_charge_phase,
        target_pos,
        quick_dist,
        soft_radius,
        hard_radius,
        current_visit,
        locked_target=None,
    ):
        first_charge_out_radius = 0
        first_charge_step_limit = 0
        if first_charge_phase:
            # The first charging cycle is the hardest constraint, so cap outward range
            # and step budget more aggressively than later cycles.
            first_charge_out_radius = int(np.clip(max(7, soft_radius - 7), 7, 9))
            first_charge_step_limit = 74
            if quick_dist >= 24:
                first_charge_step_limit = min(first_charge_step_limit, 68)
            elif quick_dist >= 16:
                first_charge_step_limit = min(first_charge_step_limit, 70)
            if self.nearest_npc_dist <= 2:
                first_charge_step_limit = min(first_charge_step_limit, 68)
            elif current_visit >= 10:
                first_charge_step_limit = min(first_charge_step_limit, 70)

        allow_unknown_fallback = False
        if first_charge_phase:
            # When the safe path is missing, switch to broader charger search earlier
            # instead of waiting until the battery margin is already exhausted.
            allow_unknown_fallback = bool(
                self.return_to_charge_mode
                or self.step_no >= max(first_charge_step_limit - 24, 42)
                or quick_dist <= first_charge_out_radius + 8
                or self.remaining_charge <= max(72, int(self.battery_max * 0.40))
            )
        else:
            allow_unknown_fallback = True

        plan = self._resolve_charge_plan(
            target_pos,
            quick_dist,
            preferred_target=locked_target if first_charge_phase else None,
            allow_unknown_fallback=allow_unknown_fallback,
        )
        target_pos = plan["target_pos"]
        target_action = plan["target_action"]
        target_dist = int(plan["target_dist"])
        path_found = bool(plan["path_found"])
        route_reliable = bool(plan["route_reliable"])
        route_source = plan["route_source"]
        recovery_mode = ""

        reroute_stall_threshold = 5 if route_reliable else 4
        last_progress = None
        if self.last_charger_route_dist < 200.0:
            last_progress = float(self.last_charger_route_dist) - float(target_dist)
        committed_route = bool(self._is_charge_route_committed(target_pos))
        allow_reroute = (
            (not committed_route)
            or (last_progress is not None and last_progress <= -1.0)
            or self._charge_progress_stall_steps >= reroute_stall_threshold + 2
        )
        if (
            self.return_to_charge_mode
            and target_pos is not None
            and target_dist > 2
            and self._charge_progress_stall_steps >= reroute_stall_threshold
            and allow_reroute
        ):
            self._charge_route_cache = None
            self._clear_charge_route_commit()
            reroute = self._resolve_charge_plan(
                target_pos,
                quick_dist,
                preferred_target=locked_target if first_charge_phase else None,
                allow_unknown_fallback=True,
            )
            if reroute["target_action"] is not None:
                target_pos = reroute["target_pos"]
                target_action = reroute["target_action"]
                target_dist = int(reroute["target_dist"])
                path_found = bool(reroute["path_found"])
                route_reliable = bool(reroute["route_reliable"])
                route_source = reroute["route_source"]
                recovery_mode = "reroute"

        reroute_stall_threshold = 5 if route_reliable else 4
        first_charge_route_risky = bool(
            first_charge_phase
            and ((not path_found) or (not route_reliable) or route_source in ("planned_unknown", "greedy"))
        )
        dock_radius = int(np.clip(max(4, soft_radius // 3 + (1 if path_found else 0)), 4, 6))
        if first_charge_phase:
            dock_radius = int(np.clip(max(dock_radius, 6), 5, 6))
        near_dock_threshold = max(dock_radius, 5 if route_reliable else 6)
        if first_charge_phase:
            # Enter charger-zone control earlier in the first cycle so the policy stops
            # spending the last safe battery on exploration noise.
            near_dock_threshold = max(near_dock_threshold, 8 if first_charge_route_risky else 7)
        final_dock_threshold = 2
        if first_charge_phase:
            final_dock_threshold = 5 if first_charge_route_risky else 4
        elif (not route_reliable) or self._charge_progress_stall_steps >= 2:
            final_dock_threshold = 3
        final_dock_threshold = min(final_dock_threshold, near_dock_threshold)
        activation_buffer = self._get_charge_activation_buffer(
            target_dist,
            path_found,
            route_reliable,
            first_charge_phase,
            current_visit,
            step_limit=first_charge_step_limit,
        )
        if route_source == "planned_unknown":
            activation_buffer += 6 if first_charge_phase else 6
        elif route_source == "greedy":
            activation_buffer += 8 if first_charge_phase else 8
        if first_charge_phase:
            # Inflate the activation budget during the first cycle so return mode starts
            # while there is still enough energy for reroute plus final docking jitter.
            if not path_found:
                activation_buffer += 14
            elif not route_reliable:
                activation_buffer += 10
            if route_source == "greedy":
                activation_buffer += 6
            if self.step_no >= max(first_charge_step_limit - 28, 40):
                activation_buffer += 8
        activation_buffer = int(min(self.battery_max - 1, activation_buffer))
        release_buffer = int(min(self.battery_max - 1, max(activation_buffer + 18, int(self.battery_max * 0.40))))
        battery_trip = self._estimate_charge_trip(
            quick_dist,
            target_dist,
            path_found,
            route_reliable,
            first_charge_phase,
        )
        if route_source == "planned_unknown":
            battery_trip += 5.0 if first_charge_phase else 4.0
        elif route_source == "greedy":
            battery_trip += 6.0 if first_charge_phase else 6.0
        if first_charge_phase:
            # Penalize weak route quality more strongly here because first-charge misses
            # are much more expensive than giving up a few exploration steps.
            if not path_found:
                battery_trip += 12.0
            elif not route_reliable:
                battery_trip += 8.0
            if route_source == "greedy":
                battery_trip += 4.0
            if target_dist >= max(first_charge_out_radius - 1, 6):
                battery_trip += 4.0
        battery_margin = int(self.remaining_charge - battery_trip)
        first_charge_budget_buffer = 0
        first_charge_budget_margin = None
        if first_charge_phase:
            first_charge_budget_buffer = int(min(self.battery_max - 1, activation_buffer + 12))
            first_charge_budget_margin = int(battery_margin - first_charge_budget_buffer)

        just_recharged = self._did_charge_this_step(target_dist)
        if just_recharged:
            self._handle_charge_success()

        charge_ready_release = bool(
            (not first_charge_phase)
            and self.return_to_charge_mode
            and self.charge_count > 0
            and target_dist <= near_dock_threshold + 1
            and battery_margin >= max(release_buffer, activation_buffer + 12)
            and (
                self.remaining_charge >= self.battery_max - 1
                or target_dist <= final_dock_threshold
                or self._charge_progress_stall_steps >= 2
            )
        )

        should_return = False
        reason = ""
        if just_recharged:
            should_return = False
            reason = "charge_success_release"
        elif charge_ready_release:
            should_return = False
            reason = "charge_buffer_release"
            self._apply_post_charge_sequence()
        elif self.return_to_charge_mode:
            should_return = True
            reason = "first_charge_hold_return" if first_charge_phase else "hold_return"
        elif first_charge_phase:
            # During the first cycle, any sign of risky routing should flip into return
            # mode early; the goal is "charge first, optimize score later".
            risky_return_step = max(first_charge_step_limit - (28 if first_charge_route_risky else 24), 42)
            risky_out_radius = max(first_charge_out_radius - (1 if first_charge_route_risky else 0), 6)
            if battery_margin <= activation_buffer:
                should_return = True
                reason = "first_charge_budget_return"
            elif (
                first_charge_route_risky
                and target_dist >= max(risky_out_radius - 1, 5)
                and battery_margin <= activation_buffer + (14 if not path_found else 12)
            ):
                should_return = True
                reason = "first_charge_risky_route_return"
            elif (
                (not path_found)
                and self.step_no >= risky_return_step
                and target_dist >= max(risky_out_radius - 2, 5)
            ):
                should_return = True
                reason = "first_charge_pathless_return"
            elif (
                route_source == "greedy"
                and target_dist >= max(risky_out_radius - 1, 5)
                and battery_margin <= activation_buffer + 16
            ):
                should_return = True
                reason = "first_charge_greedy_return"
            elif (
                self.step_no >= risky_return_step
                and target_dist >= risky_out_radius
                and (first_charge_route_risky or battery_margin <= activation_buffer + 12)
            ):
                should_return = True
                reason = "first_charge_route_return"
            elif self.step_no >= first_charge_step_limit and target_dist >= max(4, first_charge_out_radius - 2):
                should_return = True
                reason = "first_charge_step_limit"
            elif target_dist >= risky_out_radius and battery_margin <= activation_buffer + (
                10 if first_charge_route_risky else 12
            ):
                should_return = True
                reason = "first_charge_radius_return"
        else:
            if target_dist > hard_radius:
                should_return = True
                reason = "radius_return"
            elif not path_found and target_dist > soft_radius:
                should_return = True
                reason = "route_uncertain_return"
            elif not route_reliable and battery_margin <= activation_buffer + (10 if route_source == "greedy" else 8):
                should_return = True
                reason = "risky_budget_return"
            elif battery_margin <= activation_buffer:
                should_return = True
                reason = "budget_return"
            elif self.charge_count >= 2 and self.step_no >= 520 and battery_margin <= activation_buffer + 6:
                should_return = True
                reason = "late_cycle_return"

        self.return_to_charge_mode = bool(should_return)
        if self.return_to_charge_mode:
            self.post_charge_expand_until_step = -1
            self._clear_post_charge_release()
            self._clear_expand_focus()
            if target_pos is not None and (
                first_charge_phase or (not route_reliable) or target_dist <= near_dock_threshold + 1
            ):
                if first_charge_phase:
                    lock_steps = 10 if (target_dist <= near_dock_threshold + 1 or first_charge_route_risky) else 6
                else:
                    lock_steps = 8 if ((not route_reliable) or target_dist <= near_dock_threshold + 1) else 5
                self._lock_charge_target(target_pos, hold_steps=lock_steps)
        else:
            self._clear_charge_route_commit()
            if not first_charge_phase:
                self._clear_charge_target_lock()

        dock_mode = bool(self.return_to_charge_mode and target_dist <= near_dock_threshold)
        final_dock_mode = bool(self.return_to_charge_mode and target_dist <= final_dock_threshold)
        force_action = None
        control_actions = []
        controller_mode = ""
        if self.return_to_charge_mode:
            dock_action = None
            if target_pos is not None and target_dist <= near_dock_threshold + (1 if first_charge_route_risky else 0) and (
                target_dist <= final_dock_threshold or (not path_found) or (not route_reliable)
            ):
                dock_action = self._choose_action_towards(target_pos)
                if self._is_valid_legal_action(dock_action):
                    target_action = int(dock_action)
                    if recovery_mode == "" and target_dist > final_dock_threshold:
                        recovery_mode = "approach"

            preferred_action = int(target_action) if self._is_valid_legal_action(target_action) else None
            strict_progress = bool(final_dock_mode or (dock_mode and self._charge_progress_stall_steps >= 3))
            ranked_actions = self._rank_charge_actions(
                target_pos,
                target_action=preferred_action,
                dock_mode=dock_mode,
                strict_progress=strict_progress,
            )
            if not ranked_actions:
                ranked_actions = self._rank_charge_actions(
                    target_pos,
                    target_action=preferred_action,
                    dock_mode=dock_mode,
                    strict_progress=False,
                )
            if preferred_action is not None and preferred_action not in ranked_actions:
                ranked_actions = [preferred_action] + ranked_actions

            if final_dock_mode:
                controller_mode = "final_dock"
                limit = 3
                commit_dist = 3 if first_charge_phase else 2
                if target_dist <= commit_dist:
                    if self._is_valid_legal_action(dock_action):
                        force_action = int(dock_action)
                    elif ranked_actions:
                        force_action = int(ranked_actions[0])
            elif dock_mode:
                controller_mode = "near_dock"
                limit = 3
                # Near the charger, promote the direct docking action so the final steps
                # are mostly rule-driven instead of policy-driven.
                if self._is_valid_legal_action(dock_action) and (
                    first_charge_phase
                    or (not route_reliable)
                    or self._charge_progress_stall_steps >= 1
                    or target_dist <= max(final_dock_threshold + 1, 3)
                ):
                    force_action = int(dock_action)
                    if first_charge_phase or self._charge_progress_stall_steps >= 1:
                        limit = 1 if target_dist <= max(final_dock_threshold + 1, 3) else 2
                    else:
                        limit = 2
            elif recovery_mode == "reroute" or self._charge_progress_stall_steps >= reroute_stall_threshold:
                controller_mode = "return_recovery"
                limit = 4 if route_reliable else 5
            else:
                controller_mode = "return"
                limit = 4 if route_reliable else 5

            control_actions = ranked_actions[:limit]
            if force_action is not None and force_action not in control_actions:
                control_actions = [int(force_action)] + [action for action in control_actions if action != int(force_action)]
                control_actions = control_actions[:limit]

        urgency = 0.0
        if self.return_to_charge_mode:
            shortfall = max(0.0, activation_buffer - battery_margin) / max(float(activation_buffer), 1.0)
            stall_pressure = min(float(self._charge_progress_stall_steps) / 6.0, 1.0)
            dock_pressure = 1.0 - min(float(target_dist) / max(float(near_dock_threshold + 1), 1.0), 1.0)
            urgency = float(np.clip(max(shortfall, stall_pressure * 0.6, dock_pressure * 0.7), 0.25, 1.0))
            if controller_mode == "near_dock":
                urgency = max(urgency, 0.62 if route_reliable else 0.58)
            elif controller_mode == "final_dock":
                urgency = max(urgency, 0.78 if route_reliable else 0.72)
            if first_charge_phase and controller_mode == "near_dock":
                urgency = max(urgency, 0.70 if first_charge_route_risky else 0.64)
            elif first_charge_phase and controller_mode == "final_dock":
                urgency = max(urgency, 0.84 if first_charge_route_risky else 0.80)

        first_charge_stage = "explore"
        if self.return_to_charge_mode:
            first_charge_stage = "dock" if dock_mode else "return"

        return {
            "should_return": self.return_to_charge_mode,
            "reason": reason,
            "target_pos": target_pos,
            "target_action": target_action,
            "target_dist": target_dist,
            "battery_margin": battery_margin,
            "activation_buffer": activation_buffer,
            "release_buffer": release_buffer,
            "urgency": urgency,
            "path_found": path_found,
            "soft_radius": soft_radius,
            "hard_radius": hard_radius,
            "dock_mode": dock_mode,
            "dock_radius": dock_radius,
            "return_mode": "dock" if dock_mode else ("return" if self.return_to_charge_mode else ""),
            "recovery_mode": recovery_mode,
            "controller_mode": controller_mode,
            "force_action": force_action,
            "control_actions": control_actions,
            "route_source": route_source,
            "route_reliable": route_reliable,
            "first_charge_phase": bool(first_charge_phase),
            "first_charge_stage": first_charge_stage,
            "first_charge_budget_buffer": int(first_charge_budget_buffer),
            "first_charge_budget_margin": (
                None if first_charge_budget_margin is None else int(first_charge_budget_margin)
            ),
            "first_charge_step_limit": int(first_charge_step_limit),
            "first_charge_out_radius": int(first_charge_out_radius),
            "charge_rule_control": bool(self.return_to_charge_mode),
            "charge_stall_steps": int(self._charge_progress_stall_steps),
        }

    def _make_explore_guidance(
        self,
        *,
        active=False,
        mode="",
        reason="",
        target_pos=None,
        target_action=None,
        target_dist=None,
        desired_radius=0,
        intensity=0.0,
        path_found=False,
        route_source="",
        hold_active=False,
        hold_steps_left=0,
        post_charge_expand=False,
        should_pull_back=False,
        force_action=None,
    ):
        guidance = self._empty_explore_guidance()
        guidance.update(
            {
                "active": bool(active),
                "should_pull_back": bool(should_pull_back),
                "mode": mode,
                "reason": reason,
                "target_pos": target_pos,
                "target_action": target_action,
                "target_dist": target_dist,
                "force_action": force_action,
                "soft_radius": 0,
                "hard_radius": 0,
                "desired_radius": int(desired_radius),
                "intensity": float(intensity),
                "path_found": bool(path_found),
                "route_source": route_source,
                "hold_active": bool(hold_active),
                "hold_steps_left": int(hold_steps_left),
                "post_charge_expand": bool(post_charge_expand),
            }
        )
        if target_pos is not None:
            guidance["target_dx"] = _signed_norm(target_pos[0] - self.cur_pos[0], self.GRID_SIZE)
            guidance["target_dz"] = _signed_norm(target_pos[1] - self.cur_pos[1], self.GRID_SIZE)
        if target_dist is not None:
            guidance["target_dist"] = int(target_dist)
        if target_action is not None:
            guidance["target_action"] = int(target_action)
        if force_action is not None:
            guidance["force_action"] = int(force_action)
        return guidance

    def _build_first_charge_guidance(self, target_pos, quick_dist, soft_radius, hard_radius, current_visit, locked_target):
        return self._build_charge_guidance_core(
            first_charge_phase=True,
            target_pos=target_pos,
            quick_dist=quick_dist,
            soft_radius=soft_radius,
            hard_radius=hard_radius,
            current_visit=current_visit,
            locked_target=locked_target,
        )

    def _build_charge_guidance(self):
        target_pos, quick_dist = self._nearest_point(self._charger_positions)
        if target_pos is None:
            self.return_to_charge_mode = False
            self._charge_route_cache = None
            self._clear_charge_route_commit()
            self._clear_charge_target_lock()
            return self._empty_charge_guidance()

        first_charge_phase = self._in_first_charge_phase()
        if not first_charge_phase:
            self._clear_charge_target_lock()
        locked_target = self._get_charge_target_lock() if first_charge_phase else None
        if locked_target is not None:
            target_pos = locked_target
            quick_dist = self._chebyshev_dist(self.cur_pos, locked_target)
        soft_radius, hard_radius = self._get_cleaning_radius_limits()

        hx, hz = self.cur_pos
        current_visit = 0
        if 0 <= hx < self.GRID_SIZE and 0 <= hz < self.GRID_SIZE:
            current_visit = int(self.visit_count[hx, hz])

        if first_charge_phase:
            return self._build_first_charge_guidance(
                target_pos=target_pos,
                quick_dist=quick_dist,
                soft_radius=soft_radius,
                hard_radius=hard_radius,
                current_visit=current_visit,
                locked_target=locked_target,
            )
        return self._build_charge_guidance_core(
            first_charge_phase=False,
            target_pos=target_pos,
            quick_dist=quick_dist,
            soft_radius=soft_radius,
            hard_radius=hard_radius,
            current_visit=current_visit,
            locked_target=None,
        )

    def get_charge_guidance(self):
        return dict(self._charge_guidance)

    def _build_explore_guidance(self):
        guidance = self._charge_guidance
        if guidance["target_pos"] is None or guidance["target_dist"] is None:
            self._explore_route_cache = None
            self._clear_expand_focus()
            return self._empty_explore_guidance()
        if guidance["should_return"]:
            self._explore_route_cache = None
            self._clear_expand_focus()
            return self._empty_explore_guidance()
        target_dist = int(guidance["target_dist"])
        soft_radius = int(guidance["soft_radius"])
        hard_radius = int(guidance["hard_radius"])
        dock_radius = int(guidance["dock_radius"])
        battery_margin = int(guidance["battery_margin"])
        activation_buffer = int(guidance["activation_buffer"])
        release_buffer = int(guidance["release_buffer"])
        first_charge_phase = bool(guidance.get("first_charge_phase"))

        hx, hz = self.cur_pos
        current_visit = 0
        if 0 <= hx < self.GRID_SIZE and 0 <= hz < self.GRID_SIZE:
            current_visit = int(self.visit_count[hx, hz])

        if first_charge_phase:
            desired_radius = int(
                np.clip(
                    max(6, int(guidance.get("first_charge_out_radius", max(8, soft_radius - 6))) - 1),
                    6,
                    max(6, hard_radius - 2),
                )
            )
        else:
            energy_headroom = max(0.0, float(battery_margin - activation_buffer))
            energy_span = max(1.0, float(release_buffer - activation_buffer))
            energy_ratio = float(np.clip(energy_headroom / energy_span, 0.0, 1.0))
            desired_radius = int(
                np.clip(
                    soft_radius + round((hard_radius - soft_radius - 1) * (0.30 + 0.45 * energy_ratio)),
                    max(8, soft_radius - 2),
                    max(soft_radius + 2, hard_radius - 1),
                )
            )

        post_charge_expand = self._has_post_charge_expand() and battery_margin >= max(activation_buffer + 6, 18)
        if post_charge_expand:
            desired_radius = int(
                np.clip(
                    max(desired_radius, soft_radius + 4),
                    max(8, soft_radius),
                    max(soft_radius + 2, hard_radius - 1),
                )
            )

        release_exit_radius = max(dock_radius + 1, 4)
        post_charge_release = bool(
            self._has_post_charge_release()
            and battery_margin >= max(activation_buffer + 8, 18)
        )
        if post_charge_release and target_dist > release_exit_radius:
            self._clear_post_charge_release()
            post_charge_release = False

        local_dirt_count = int(np.count_nonzero(self._view_map == 2))
        hold_active = self._has_expand_hold()
        budget_limit = int(max(desired_radius + 2, self.remaining_charge - max(activation_buffer, 12)))

        explore_target = None
        if hold_active or post_charge_expand:
            explore_target = self._get_expand_focus_candidate(
                charger_pos=guidance["target_pos"],
                desired_radius=desired_radius,
                hard_radius=hard_radius,
                budget_limit=budget_limit,
            )
            if explore_target is None and self._expand_focus_target is not None:
                self._clear_expand_focus()

        if explore_target is None:
            explore_target = self._select_explore_target(
                charger_pos=guidance["target_pos"],
                desired_radius=desired_radius,
                soft_radius=soft_radius,
                hard_radius=hard_radius,
                budget_limit=budget_limit,
            )

        if post_charge_release:
            force_action = self._choose_action_away_from(guidance["target_pos"])
            if self._is_valid_legal_action(force_action):
                hold_steps_left = max(0, self.post_charge_release_until_step - self.step_no + 1)
                release_target_pos = None
                release_target_action = int(force_action)
                release_target_dist = max(0, desired_radius - target_dist)
                release_reason = "post_charge_release"
                release_path_found = False
                release_route_source = "local_outward"

                if explore_target is not None:
                    release_target_pos = explore_target["target_pos"]
                    if explore_target["target_action"] is not None:
                        release_target_action = int(explore_target["target_action"])
                    release_target_dist = int(explore_target["target_dist"])
                    release_reason = explore_target["reason"]
                    release_path_found = bool(explore_target["path_found"])
                    release_route_source = "release_{}".format(explore_target["route_source"] or "greedy")
                    if release_target_pos is not None:
                        self._expand_focus_target = tuple(release_target_pos)
                        self._expand_focus_reason = "post_charge_release"
                else:
                    self._explore_route_cache = None
                    self._clear_expand_focus()

                explore_guidance = self._make_explore_guidance(
                    active=True,
                    mode="post_charge_release",
                    reason=release_reason,
                    target_pos=release_target_pos,
                    target_action=release_target_action,
                    target_dist=release_target_dist,
                    desired_radius=desired_radius,
                    intensity=0.58 if explore_target is not None else 0.66,
                    path_found=release_path_found,
                    route_source=release_route_source,
                    hold_active=True,
                    hold_steps_left=hold_steps_left,
                    post_charge_expand=True,
                    force_action=int(force_action),
                )
                explore_guidance["soft_radius"] = soft_radius
                explore_guidance["hard_radius"] = hard_radius
                return explore_guidance

            self._clear_post_charge_release()
            post_charge_release = False

        if explore_target is None:
            should_recenter = bool(
                (not first_charge_phase)
                and target_dist <= max(dock_radius + 1, 4)
                and battery_margin >= max(activation_buffer + 10, 18)
                and (post_charge_expand or current_visit >= 6 or self.charge_count >= 2)
            )
            recenter_action = self._choose_action_away_from(guidance["target_pos"]) if should_recenter else None
            if self._is_valid_legal_action(recenter_action):
                fallback = self._make_explore_guidance(
                    active=True,
                    mode="recenter",
                    reason="leave_charger_zone",
                    target_action=int(recenter_action),
                    target_dist=max(1, desired_radius - target_dist),
                    desired_radius=desired_radius,
                    intensity=0.52 if post_charge_expand else 0.42,
                    hold_active=hold_active,
                    hold_steps_left=max(0, self.expand_hold_until_step - self.step_no + 1),
                    post_charge_expand=post_charge_expand,
                    force_action=int(recenter_action),
                )
                fallback["soft_radius"] = soft_radius
                fallback["hard_radius"] = hard_radius
                return fallback

            self._explore_route_cache = None
            self._clear_expand_focus()
            fallback = self._make_explore_guidance(
                active=False,
                mode="clean_local" if local_dirt_count > 0 else "",
                reason="local_clean" if local_dirt_count > 0 else "safe_zone",
                desired_radius=desired_radius,
            )
            fallback["soft_radius"] = soft_radius
            fallback["hard_radius"] = hard_radius
            return fallback

        new_focus_target = tuple(explore_target["target_pos"]) if explore_target["target_pos"] is not None else None
        hold_steps = 10 if first_charge_phase else (14 if post_charge_expand else 12)
        if (not hold_active) or (new_focus_target is not None and new_focus_target != self._expand_focus_target):
            self._activate_expand_hold(hold_steps)
        if new_focus_target is not None:
            self._expand_focus_target = new_focus_target
            self._expand_focus_reason = "post_charge_expand" if post_charge_expand else "expand_frontier"

        expand_gap = max(0, desired_radius - target_dist)
        intensity = float(
            np.clip(
                0.22
                + 0.30 * (expand_gap / max(desired_radius, 1))
                + (0.12 if post_charge_expand else 0.0)
                + (0.06 if local_dirt_count == 0 else 0.0)
                + (0.04 if current_visit >= 8 else 0.0),
                0.18,
                0.72,
            )
        )
        hold_steps_left = max(0, self.expand_hold_until_step - self.step_no + 1)
        explore_guidance = self._make_explore_guidance(
            active=explore_target["target_action"] is not None,
            mode="expand_frontier",
            reason=explore_target["reason"],
            target_pos=explore_target["target_pos"],
            target_action=explore_target["target_action"],
            target_dist=int(explore_target["target_dist"]),
            desired_radius=desired_radius,
            intensity=intensity,
            path_found=bool(explore_target["path_found"]),
            route_source=explore_target["route_source"],
            hold_active=True,
            hold_steps_left=hold_steps_left,
            post_charge_expand=post_charge_expand,
        )
        explore_guidance["soft_radius"] = soft_radius
        explore_guidance["hard_radius"] = hard_radius
        return explore_guidance

    def get_explore_guidance(self):
        return dict(self._explore_guidance)

    def _build_npc_guidance(self):
        npc_point, npc_dist = self._nearest_point(self._npc_positions)
        if npc_point is None or npc_dist is None:
            return self._empty_npc_guidance()

        hx, hz = self.cur_pos
        action_weights = [0.0] * len(self.ACTION_DELTAS)
        best_action = None
        best_score = None
        safest_next_dist = None

        for action, (dx, dz) in enumerate(self.ACTION_DELTAS):
            if action >= len(self._legal_act) or not self._legal_act[action]:
                continue

            nx = hx + dx
            nz = hz + dz
            next_visit = self._get_visit_penalty((nx, nz))
            _, next_npc_dist = self._npc_risk_at((nx, nz))
            if next_npc_dist is None:
                next_npc_dist = self.GRID_SIZE

            if next_npc_dist <= 1:
                weight = 0.01
            elif next_npc_dist == 2:
                weight = 0.10
            elif next_npc_dist == 3:
                weight = 0.40
            elif next_npc_dist == 4:
                weight = 0.75
            else:
                weight = 1.00

            if next_npc_dist > npc_dist:
                weight *= 1.35
            elif next_npc_dist < npc_dist:
                weight *= 0.55

            weight /= 1.0 + next_visit * 0.05
            weight = float(np.clip(weight, 0.01, 2.0))
            action_weights[action] = weight

            score = weight * 100.0 + next_npc_dist - next_visit * 0.2
            if best_score is None or score > best_score:
                best_score = score
                best_action = action
                safest_next_dist = next_npc_dist

        should_evade = npc_dist <= 4
        if npc_dist <= 1:
            reason = "npc_imminent"
            danger_level = 1.0
        elif npc_dist == 2:
            reason = "npc_high_risk"
            danger_level = 0.8
        elif npc_dist <= 4:
            reason = "npc_nearby"
            danger_level = 0.45
        else:
            reason = ""
            danger_level = 0.0

        return {
            "should_evade": should_evade,
            "reason": reason,
            "target_action": best_action if should_evade else None,
            "nearest_dist": int(npc_dist),
            "safest_next_dist": None if safest_next_dist is None else int(safest_next_dist),
            "danger_level": danger_level,
            "action_weights": action_weights,
        }

    def get_npc_guidance(self):
        guidance = dict(self._npc_guidance)
        guidance["action_weights"] = list(self._npc_guidance.get("action_weights", []))
        return guidance

    def _get_global_state_feature(self):
        hx, hz = self.cur_pos
        local_total = float(self._view_map.size)

        local_dirt_ratio = float(np.count_nonzero(self._view_map == 2)) / local_total
        local_clean_ratio = float(np.count_nonzero(self._view_map == 1)) / local_total
        local_obstacle_ratio = float(np.count_nonzero(self._view_map == 0)) / local_total

        explored_cells = int(self.explored_map.sum())
        explored_ratio = float(explored_cells) / float(self.GRID_SIZE * self.GRID_SIZE)
        known_dirt_ratio = float(np.count_nonzero(self.memory_map == 2)) / float(max(explored_cells, 1))

        current_visit = 0
        if 0 <= hx < self.GRID_SIZE and 0 <= hz < self.GRID_SIZE:
            current_visit = int(self.visit_count[hx, hz])

        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0

        guidance = self._charge_guidance
        explore_guidance = self._explore_guidance
        strategy = self._strategy_state
        charger_point = guidance["target_pos"]
        charger_dist = guidance["target_dist"]
        if charger_point is None or charger_dist is None:
            charger_dx = 0.0
            charger_dz = 0.0
            charger_dist_norm = 1.0
            in_charger_zone = 0.0
            safe_return_margin = -1.0
            low_battery_alert = 0.0
        else:
            charger_dx = _signed_norm(charger_point[0] - hx, self.GRID_SIZE)
            charger_dz = _signed_norm(charger_point[1] - hz, self.GRID_SIZE)
            charger_dist_norm = _norm(charger_dist, self.GRID_SIZE)
            in_charger_zone = 1.0 if charger_dist <= 1 else 0.0
            safe_return_margin = _signed_norm(guidance["battery_margin"], self.battery_max)
            low_battery_alert = float(guidance["urgency"]) if guidance["should_return"] else 0.0

        explore_target_pos = explore_guidance["target_pos"]
        explore_target_dist = explore_guidance["target_dist"]
        if explore_target_pos is None or explore_target_dist is None:
            explore_dx = 0.0
            explore_dz = 0.0
            explore_target_dist_norm = 1.0
        else:
            explore_dx = float(explore_guidance["target_dx"])
            explore_dz = float(explore_guidance["target_dz"])
            explore_target_dist_norm = _norm(explore_target_dist, self.GRID_SIZE)
        desired_radius_norm = _norm(explore_guidance["desired_radius"], self.GRID_SIZE)
        explore_path_found = 1.0 if explore_guidance["path_found"] else 0.0

        npc_point, npc_dist = self._nearest_point(self._npc_positions)
        if npc_point is None:
            npc_dx = 0.0
            npc_dz = 0.0
            npc_dist_norm = 1.0
            npc_danger = 0.0
        else:
            npc_dx = _signed_norm(npc_point[0] - hx, self.GRID_SIZE)
            npc_dz = _signed_norm(npc_point[1] - hz, self.GRID_SIZE)
            npc_dist_norm = _norm(npc_dist, self.GRID_SIZE)
            if npc_dist <= 1:
                npc_danger = 1.0
            elif npc_dist == 2:
                npc_danger = 0.75
            elif npc_dist <= 4:
                npc_danger = 0.35
            else:
                npc_danger = 0.0

        return np.array(
            [
                _norm(self.step_no, self.max_step),
                _norm(self.battery, self.battery_max),
                _norm(self.dirt_cleaned, self.total_dirt),
                1.0 - _norm(self.dirt_cleaned, self.total_dirt),
                _norm(hx, self.GRID_SIZE - 1),
                _norm(hz, self.GRID_SIZE - 1),
                local_dirt_ratio,
                local_clean_ratio,
                local_obstacle_ratio,
                explored_ratio,
                _norm(current_visit, self.MAX_VISIT_COUNT),
                known_dirt_ratio,
                _norm(self.nearest_dirt_dist, self.GRID_SIZE),
                dirt_delta,
                _norm(self.charge_count, 10),
                _norm(self.total_charger, 4),
                _norm(self.npc_count, 4),
                _norm(self.step_cleaned_count, 10),
                charger_dist_norm,
                charger_dx,
                charger_dz,
                in_charger_zone,
                safe_return_margin,
                low_battery_alert,
                1.0 if strategy["clean_mode"] else 0.0,
                1.0 if strategy["expand_mode"] else 0.0,
                1.0 if strategy["return_mode"] else 0.0,
                1.0 if strategy["dock_mode"] else 0.0,
                float(strategy["mode_intensity"]),
                desired_radius_norm,
                explore_target_dist_norm,
                explore_path_found,
                explore_dx,
                explore_dz,
                npc_dist_norm,
                npc_dx,
                npc_dz,
                npc_danger,
            ],
            dtype=np.float32,
        )

    def get_legal_action(self):
        return list(self._legal_act)

    def get_debug_snapshot(self):
        _, npc_dist = self._nearest_point(self._npc_positions)
        hx, hz = self.cur_pos
        current_visit = 0
        if 0 <= hx < self.GRID_SIZE and 0 <= hz < self.GRID_SIZE:
            current_visit = int(self.visit_count[hx, hz])

        guidance = self._charge_guidance
        npc_guidance = self._npc_guidance
        explore_guidance = self._explore_guidance
        strategy = self._strategy_state
        return {
            "pos": self.cur_pos,
            "remaining_charge": int(self.remaining_charge),
            "charge_count": int(self.charge_count),
            "npc_count": int(self.npc_count),
            "current_visit": current_visit,
            "explored_ratio": float(self.explored_map.sum()) / float(self.GRID_SIZE * self.GRID_SIZE),
            "nearest_charger_dist": None if guidance["target_dist"] is None else int(guidance["target_dist"]),
            "nearest_npc_dist": None if npc_dist is None else int(npc_dist),
            "return_mode": bool(guidance["should_return"]),
            "return_reason": guidance["reason"],
            "charge_action": guidance["target_action"],
            "battery_margin": None if guidance["battery_margin"] is None else int(guidance["battery_margin"]),
            "charge_urgency": float(guidance["urgency"]),
            "charge_route_found": bool(guidance["path_found"]),
            "charge_route_source": guidance["route_source"],
            "charge_route_reliable": bool(guidance.get("route_reliable")),
            "charge_stall_steps": int(guidance.get("charge_stall_steps", 0)),
            "charge_return_mode": guidance.get("return_mode", ""),
            "charge_recovery_mode": guidance.get("recovery_mode", ""),
            "charge_controller_mode": guidance.get("controller_mode", ""),
            "charge_allowed_count": len(guidance.get("control_actions") or []),
            "soft_clean_radius": int(guidance["soft_radius"]),
            "hard_clean_radius": int(guidance["hard_radius"]),
            "dock_mode": bool(guidance["dock_mode"]),
            "dock_radius": int(guidance["dock_radius"]),
            "first_charge_phase": bool(guidance.get("first_charge_phase")),
            "first_charge_stage": guidance.get("first_charge_stage", ""),
            "first_charge_budget_buffer": int(guidance.get("first_charge_budget_buffer", 0)),
            "first_charge_budget_margin": (
                None if guidance.get("first_charge_budget_margin") is None else int(guidance.get("first_charge_budget_margin"))
            ),
            "first_charge_step_limit": int(guidance.get("first_charge_step_limit", 0)),
            "first_charge_out_radius": int(guidance.get("first_charge_out_radius", 0)),
            "charge_rule_control": bool(guidance.get("charge_rule_control")),
            "strategy_mode": strategy["mode_name"],
            "strategy_intensity": float(strategy["mode_intensity"]),
            "explore_mode": bool(explore_guidance["active"]),
            "explore_mode_name": explore_guidance["mode"],
            "explore_reason": explore_guidance["reason"],
            "explore_action": explore_guidance["target_action"],
            "explore_target_dist": None if explore_guidance["target_dist"] is None else int(explore_guidance["target_dist"]),
            "explore_desired_radius": int(explore_guidance["desired_radius"]),
            "explore_route_found": bool(explore_guidance["path_found"]),
            "explore_route_source": explore_guidance["route_source"],
            "explore_hold_active": bool(explore_guidance["hold_active"]),
            "explore_hold_left": int(explore_guidance["hold_steps_left"]),
            "post_charge_expand": bool(explore_guidance["post_charge_expand"]),
            "prev_charge_cycle_explore_gain": int(self.prev_charge_cycle_explore_gain),
            "prev_charge_cycle_clean_gain": int(self.prev_charge_cycle_clean_gain),
            "npc_mode": bool(npc_guidance["should_evade"]),
            "npc_reason": npc_guidance["reason"],
            "npc_action": npc_guidance["target_action"],
            "npc_safe_dist": npc_guidance["safest_next_dist"],
        }

    def feature_process(self, env_obs, last_action):
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()
        global_state = self._get_global_state_feature()
        legal_action = self.get_legal_action()

        feature = np.concatenate([local_view, global_state, np.array(legal_action, dtype=np.float32)])
        reward = self.reward_process()
        return feature, legal_action, reward

    def reward_process(self):
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        guidance = self._charge_guidance
        first_charge_phase = bool(guidance.get("first_charge_phase"))
        first_charge_budget_margin = guidance.get("first_charge_budget_margin")
        if first_charge_budget_margin is not None:
            first_charge_budget_margin = int(first_charge_budget_margin)
        new_explored_cells = max(0, self.explored_cells - self.prev_explored_cells)
        charged_this_step = bool(self.charge_count > self.prev_charge_count)
        first_charge_success = bool(charged_this_step and self.prev_charge_count <= 0)
        hx, hz = self.cur_pos
        current_visit = 0
        if 0 <= hx < self.GRID_SIZE and 0 <= hz < self.GRID_SIZE:
            current_visit = int(self.visit_count[hx, hz])

        reward = -0.0025

        cleaning_reward = 0.075 * cleaned_this_step
        if guidance["should_return"]:
            cleaning_reward *= 0.18 if guidance["dock_mode"] else 0.35
        reward += cleaning_reward

        if not guidance["should_return"] and new_explored_cells > 0:
            reward += 0.0040 * min(new_explored_cells, 6)

        if charged_this_step:
            reward += 0.18
            if first_charge_success:
                reward += 0.95
                if self.step_no <= 220:
                    reward += 0.10
                elif self.step_no <= 320:
                    reward += 0.05

            cycle_explore_gain = int(self.prev_charge_cycle_explore_gain)
            cycle_clean_gain = int(self.prev_charge_cycle_clean_gain)
            if cycle_explore_gain <= 2 and cycle_clean_gain <= 4:
                reward -= 0.28
            elif cycle_explore_gain <= 6 and cycle_clean_gain <= 10:
                reward -= 0.12
            elif cycle_explore_gain >= 24 or cycle_clean_gain >= 30:
                reward += 0.08

        if not guidance["should_return"]:
            charger_dist = guidance.get("target_dist")
            battery_margin = guidance.get("battery_margin")
            activation_buffer = int(guidance.get("activation_buffer", 0))

            if cleaned_this_step == 0 and new_explored_cells == 0 and current_visit >= 6:
                reward -= 0.004 * min(current_visit - 5, 6)

            if (
                charger_dist is not None
                and battery_margin is not None
                and int(battery_margin) >= activation_buffer + 10
                and int(charger_dist) <= int(guidance.get("dock_radius", 0)) + 1
            ):
                reward -= 0.018
                if current_visit >= 4:
                    reward -= 0.003 * min(current_visit - 3, 5)

        route_progress = None
        if self.last_charger_route_dist < 200.0 and self.charger_route_dist < 200.0:
            route_progress = self.last_charger_route_dist - self.charger_route_dist

        if guidance["should_return"]:
            if route_progress is not None:
                if route_progress > 0:
                    progress_scale = 0.028 if guidance["dock_mode"] else 0.022
                    reward += progress_scale * min(route_progress, 2.0)
                elif route_progress < 0:
                    regress_scale = 0.042 if guidance["dock_mode"] else 0.032
                    reward += regress_scale * max(route_progress, -2.0)

            battery_margin = guidance.get("battery_margin")
            if battery_margin is not None and int(battery_margin) < 0:
                reward -= 0.030 if guidance["dock_mode"] else 0.024
            if not guidance["path_found"]:
                reward -= 0.004

            stall_steps = int(guidance.get("charge_stall_steps", 0))
            if stall_steps >= 2:
                reward -= 0.008 * min(stall_steps, 6)
                if guidance["dock_mode"]:
                    reward -= 0.004 * min(stall_steps, 4)

            target_dist = guidance.get("target_dist")
            if guidance["dock_mode"] and target_dist is not None and int(target_dist) <= 1 and not charged_this_step:
                reward -= 0.035

        if first_charge_phase:
            if first_charge_budget_margin is not None:
                if not guidance["should_return"] and first_charge_budget_margin <= 0:
                    reward -= 0.032
                elif not guidance["should_return"] and first_charge_budget_margin <= 6:
                    reward -= 0.014
                elif guidance["should_return"] and first_charge_budget_margin < -6:
                    reward -= 0.010

            if self.step_no >= 160:
                reward -= 0.004
            if self.step_no >= 220:
                reward -= 0.008
            if self.step_no >= 300:
                reward -= 0.012

        if self.nearest_npc_dist <= 1:
            reward -= 0.04
        elif self.nearest_npc_dist == 2:
            reward -= 0.015
        if self.last_nearest_npc_dist <= 4:
            if self.nearest_npc_dist > self.last_nearest_npc_dist:
                reward += 0.006
            elif self.nearest_npc_dist < self.last_nearest_npc_dist:
                reward -= 0.006

        return reward
