"""
3D drone simulation environment with TSA-style course geometry, curriculum randomization,
wind, turbulence, battery sag, drag, sensor noise, latency, dropout, and scoring.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
import math
from pathlib import Path
from typing import Any, Deque, Iterable, Mapping

import cv2
import numpy as np

from src.tsa_rules import TSAChallengeProfile, create_default_hs_2026_profile


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return vec * 0.0
    return vec / norm


def _rotation_yaw(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


@dataclass(frozen=True)
class CurriculumConfig:
    """Episode scheduling and difficulty control."""

    easy_until: int = 12
    medium_until: int = 30

    def level_for_episode(self, episode_index: int) -> str:
        if episode_index < self.easy_until:
            return "easy"
        if episode_index < self.medium_until:
            return "medium"
        return "hard"


@dataclass(frozen=True)
class RandomizationConfig:
    """Domain randomization knobs used to model imperfect real-world flight."""

    wind_base_mps: float = 1.0
    wind_gust_probability: float = 0.18
    wind_turbulence_sigma: float = 0.35
    drag_coefficient_range: tuple[float, float] = (0.06, 0.16)
    mass_multiplier_range: tuple[float, float] = (0.90, 1.15)
    battery_capacity_wh_range: tuple[float, float] = (18.0, 26.0)
    battery_sag_rate: float = 0.004
    response_decay_rate: float = 0.0025
    sensor_position_noise_m: float = 0.04
    sensor_velocity_noise_mps: float = 0.03
    sensor_attitude_noise_rad: float = 0.01
    sensor_latency_steps: int = 2
    sensor_dropout_probability: float = 0.03


@dataclass(frozen=True)
class Gate:
    """Course gate geometry."""

    center: np.ndarray
    yaw_rad: float
    inner_width_m: float
    inner_height_m: float
    frame_thickness_m: float

    @property
    def half_width(self) -> float:
        return self.inner_width_m / 2.0

    @property
    def half_height(self) -> float:
        return self.inner_height_m / 2.0


@dataclass
class DroneState:
    """Continuous state of the drone inside the simulator."""

    position_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity_mps: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    attitude_rad: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    angular_velocity_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    battery_wh: float = 24.0
    battery_capacity_wh: float = 24.0
    time_s: float = 0.0
    gate_index: int = 0
    collided: bool = False
    off_course_events: int = 0
    gate_precision_total: float = 0.0
    smoothness_total: float = 0.0
    safety_total: float = 0.0
    energy_used_wh: float = 0.0


@dataclass
class EpisodeMetrics:
    """Rich episode report returned by the simulator."""

    score: float = 0.0
    reward: float = 0.0
    gates_cleared: int = 0
    collisions: int = 0
    gate_precision: float = 0.0
    efficiency: float = 0.0
    smoothness: float = 0.0
    safety: float = 0.0
    off_course_events: int = 0
    time_elapsed_s: float = 0.0
    energy_used_wh: float = 0.0
    timed_out: bool = False
    success: bool = False
    curriculum_level: str = "easy"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Drone3DSimEnv:
    """Physics-lite 3D drone sim with TSA-style gated course objectives."""

    observation_dim = 26
    action_dim = 7

    def __init__(
        self,
        profile: TSAChallengeProfile | None = None,
        randomization: RandomizationConfig | None = None,
        curriculum: CurriculumConfig | None = None,
        seed: int = 2026,
    ) -> None:
        self.profile = profile or create_default_hs_2026_profile()
        self.randomization = randomization or RandomizationConfig()
        self.curriculum = curriculum or CurriculumConfig()
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.state = DroneState()
        self.gates: list[Gate] = []
        self._latent_observation_queue: Deque[np.ndarray] = deque(maxlen=max(1, self.randomization.sensor_latency_steps + 1))
        self._previous_action = np.zeros(self.action_dim, dtype=np.float32)
        self._difficulty = "easy"
        self._episode_index = 0
        self._step_limit = int(self.profile.max_episode_time_s / 0.05)
        self._step_count = 0
        self._wind_vector = np.zeros(3, dtype=np.float32)
        self._wind_gust = np.zeros(3, dtype=np.float32)
        self._mass_kg = 1.6
        self._drag_coefficient = 0.1
        self._response_factor = 1.0
        self._current_observation = np.zeros(self.observation_dim, dtype=np.float32)
        self.last_metrics: EpisodeMetrics | None = None
        self._last_reward = 0.0
        self._last_state_for_render: np.ndarray | None = None

    def reset(self, seed: int | None = None, episode_index: int = 0, difficulty: str | None = None) -> np.ndarray:
        """Reset the simulator and return the first observation."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(self.base_seed + episode_index)

        self._episode_index = episode_index
        self._difficulty = difficulty or self.curriculum.level_for_episode(episode_index)
        self._step_count = 0
        self._previous_action = np.zeros(self.action_dim, dtype=np.float32)
        self._latent_observation_queue.clear()
        self.last_metrics = None
        self._last_reward = 0.0

        difficulty_scale = {"easy": 0.55, "medium": 0.85, "hard": 1.15}[self._difficulty]
        gate_count = self.profile.gate_count + (0 if self._difficulty == "easy" else 1 if self._difficulty == "medium" else 2)
        gate_spacing = self.profile.gate_spacing_m * difficulty_scale

        self.gates = self._build_course(gate_count=gate_count, gate_spacing=gate_spacing, difficulty_scale=difficulty_scale)

        launch_height = _clamp(1.2 + self.rng.normal(0.0, 0.08), self.profile.min_altitude_m, self.profile.max_altitude_m - 0.25)
        self.state = DroneState(
            position_m=np.array([0.0, 0.0, launch_height], dtype=np.float32),
            velocity_mps=np.zeros(3, dtype=np.float32),
            attitude_rad=np.zeros(3, dtype=np.float32),
            angular_velocity_rad_s=np.zeros(3, dtype=np.float32),
            battery_capacity_wh=float(self.rng.uniform(*self.randomization.battery_capacity_wh_range)),
            battery_wh=float(self.rng.uniform(*self.randomization.battery_capacity_wh_range)),
            time_s=0.0,
            gate_index=0,
            collided=False,
            off_course_events=0,
        )

        self._mass_kg = float(self.rng.uniform(*self.randomization.mass_multiplier_range)) * 1.55
        self._drag_coefficient = float(self.rng.uniform(*self.randomization.drag_coefficient_range)) * difficulty_scale
        self._response_factor = 1.0 + (0.08 if self._difficulty == "easy" else 0.0) - (0.07 if self._difficulty == "hard" else 0.0)
        self._wind_vector = self._sample_wind(difficulty_scale)
        self._wind_gust = np.zeros(3, dtype=np.float32)

        observation = self._sensor_reading(self.state)
        self._latent_observation_queue.append(observation.copy())
        self._current_observation = observation
        return observation

    def _build_course(self, gate_count: int, gate_spacing: float, difficulty_scale: float) -> list[Gate]:
        gates: list[Gate] = []
        base_width = self.profile.gate.inner_width_m * (1.2 if self._difficulty == "easy" else 1.0 if self._difficulty == "medium" else 0.85)
        base_height = self.profile.gate.inner_height_m * (1.2 if self._difficulty == "easy" else 1.0 if self._difficulty == "medium" else 0.85)
        for gate_index in range(gate_count):
            x = min(self.profile.field_length_m - 1.5, 2.0 + gate_index * gate_spacing)
            y = float(self.rng.normal(0.0, self.profile.field_width_m * 0.08 * difficulty_scale))
            z = float(_clamp(1.1 + self.rng.normal(0.0, 0.18 * difficulty_scale), self.profile.min_altitude_m + 0.1, self.profile.max_altitude_m - 0.1))
            yaw = float(self.rng.normal(0.0, math.radians(6.0 if self._difficulty == "easy" else 12.0 if self._difficulty == "medium" else 18.0)))
            gates.append(
                Gate(
                    center=np.array([x, y, z], dtype=np.float32),
                    yaw_rad=yaw,
                    inner_width_m=base_width,
                    inner_height_m=base_height,
                    frame_thickness_m=self.profile.gate.frame_thickness_m,
                )
            )
        return gates

    def _sample_wind(self, difficulty_scale: float) -> np.ndarray:
        base_speed = self.randomization.wind_base_mps * difficulty_scale
        direction = _normalize(self.rng.normal(0.0, 1.0, size=3).astype(np.float32))
        direction[2] = 0.0
        if np.linalg.norm(direction) <= 0.0:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        speed = float(self.rng.uniform(0.4 * base_speed, 1.6 * base_speed))
        return direction * speed

    def _gate_local_coordinates(self, gate: Gate, position: np.ndarray) -> np.ndarray:
        relative = position - gate.center
        return _rotation_yaw(-gate.yaw_rad) @ relative

    def _current_gate(self) -> Gate:
        return self.gates[min(self.state.gate_index, len(self.gates) - 1)]

    def _expert_core_action(self) -> np.ndarray:
        gate = self._current_gate()
        relative = self._gate_local_coordinates(gate, self.state.position_m)
        velocity_local = _rotation_yaw(-gate.yaw_rad) @ self.state.velocity_mps
        roll = _clamp(-relative[1] / max(gate.half_width, 1e-3) - 0.2 * velocity_local[1], -1.0, 1.0)
        pitch = _clamp(relative[0] / max(1.5 * self.profile.gate_spacing_m, 1e-3) - 0.2 * velocity_local[0], -1.0, 1.0)
        throttle = _clamp((gate.center[2] - self.state.position_m[2]) / max(gate.half_height, 1e-3) - 0.15 * velocity_local[2], -1.0, 1.0)
        yaw_error = ((gate.yaw_rad - self.state.attitude_rad[2] + math.pi) % (2.0 * math.pi)) - math.pi
        yaw_rate = _clamp(yaw_error / math.radians(35.0), -1.0, 1.0)
        return np.array([roll, pitch, throttle, yaw_rate], dtype=np.float32)

    def expert_action(self) -> np.ndarray:
        """Return a heuristic teacher action for training and smoke validation."""

        core = self._expert_core_action()
        braking = _clamp(np.linalg.norm(self.state.velocity_mps) / max(self.profile.max_speed_mps, 1e-3), 0.0, 1.0)
        return np.array([core[0], core[1], core[2], core[3], braking, 0.0, 0.0], dtype=np.float32)

    def _sensor_reading(self, true_state: DroneState) -> np.ndarray:
        position = true_state.position_m + self.rng.normal(0.0, self.randomization.sensor_position_noise_m, size=3).astype(np.float32)
        velocity = true_state.velocity_mps + self.rng.normal(0.0, self.randomization.sensor_velocity_noise_mps, size=3).astype(np.float32)
        attitude = true_state.attitude_rad + self.rng.normal(0.0, self.randomization.sensor_attitude_noise_rad, size=3).astype(np.float32)

        if self.rng.random() < self.randomization.sensor_dropout_probability:
            return np.zeros(self.observation_dim, dtype=np.float32)

        gate = self._current_gate()
        gate_relative = self._gate_local_coordinates(gate, true_state.position_m)
        progress = float(true_state.gate_index / max(len(self.gates), 1))
        time_fraction = float(true_state.time_s / max(self.profile.max_episode_time_s, 1e-3))
        difficulty_onehot = np.array([
            1.0 if self._difficulty == "easy" else 0.0,
            1.0 if self._difficulty == "medium" else 0.0,
            1.0 if self._difficulty == "hard" else 0.0,
        ], dtype=np.float32)
        speed_norm = np.array([np.linalg.norm(velocity) / max(self.profile.max_speed_mps, 1e-3)], dtype=np.float32)
        gate_index_norm = np.array([true_state.gate_index / max(len(self.gates), 1)], dtype=np.float32)
        battery_norm = np.array([true_state.battery_wh / max(true_state.battery_capacity_wh, 1e-3)], dtype=np.float32)
        gate_size = np.array([gate.inner_width_m, gate.inner_height_m], dtype=np.float32)
        observation = np.concatenate([
            position,
            velocity,
            attitude,
            self._wind_vector,
            battery_norm,
            gate_relative,
            gate_size,
            np.array([progress], dtype=np.float32),
            np.array([time_fraction], dtype=np.float32),
            difficulty_onehot,
            np.array([float(true_state.collided)], dtype=np.float32),
            gate_index_norm,
            speed_norm,
        ])
        return observation.astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        if not self._latent_observation_queue:
            return self._current_observation.copy()
        if len(self._latent_observation_queue) <= self.randomization.sensor_latency_steps:
            return self._latent_observation_queue[0].copy()
        return self._latent_observation_queue[-(self.randomization.sensor_latency_steps + 1)].copy()

    def _compute_energy_draw(self, control_force_norm: float, speed_norm: float) -> float:
        return 0.0035 * control_force_norm + 0.0012 * speed_norm

    def step(self, action: Iterable[float], dt: float = 0.05) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance the environment by a single time step."""

        action_array = np.asarray(list(action), dtype=np.float32).flatten()
        if action_array.size < self.action_dim:
            action_array = np.pad(action_array, (0, self.action_dim - action_array.size))
        action_array = np.clip(action_array[: self.action_dim], -1.0, 1.0)

        previous_position = self.state.position_m.copy()
        previous_velocity = self.state.velocity_mps.copy()
        previous_action = self._previous_action.copy()
        self._previous_action = action_array.copy()

        gate = self._current_gate()
        core = action_array[:4]
        roll_cmd, pitch_cmd, throttle_cmd, yaw_rate_cmd = [float(value) for value in core]

        body_accel = np.array([
            pitch_cmd * 5.5,
            roll_cmd * 5.5,
            throttle_cmd * 4.0,
        ], dtype=np.float32)
        world_accel = _rotation_yaw(self.state.attitude_rad[2]) @ body_accel

        wind_gust = np.zeros(3, dtype=np.float32)
        if self.rng.random() < self.randomization.wind_gust_probability:
            wind_gust = self.rng.normal(0.0, self.randomization.wind_turbulence_sigma * 2.0, size=3).astype(np.float32)
            wind_gust[2] = 0.0
        self._wind_gust = wind_gust
        turbulence = self.rng.normal(0.0, self.randomization.wind_turbulence_sigma, size=3).astype(np.float32)
        turbulence[2] *= 0.5

        air_relative_velocity = previous_velocity - (self._wind_vector + wind_gust)
        drag_force = -self._drag_coefficient * air_relative_velocity * np.abs(air_relative_velocity)

        battery_fraction = _clamp(self.state.battery_wh / max(self.state.battery_capacity_wh, 1e-3), 0.05, 1.0)
        response_factor = self._response_factor * (0.55 + 0.45 * battery_fraction)
        response_factor = max(0.2, response_factor - self.randomization.response_decay_rate * self.state.time_s)

        accel = response_factor * world_accel + drag_force / max(self._mass_kg, 1e-3) + 0.45 * turbulence
        accel[2] += -0.3 * (1.0 - battery_fraction)
        self.state.velocity_mps = previous_velocity + accel * dt

        speed = float(np.linalg.norm(self.state.velocity_mps))
        if speed > self.profile.max_speed_mps:
            self.state.velocity_mps = self.state.velocity_mps / speed * self.profile.max_speed_mps

        self.state.position_m = previous_position + self.state.velocity_mps * dt
        self.state.attitude_rad += np.array([roll_cmd, pitch_cmd, yaw_rate_cmd], dtype=np.float32) * dt * 0.4
        self.state.attitude_rad = np.clip(self.state.attitude_rad, np.array([-math.radians(self.profile.max_roll_deg)] * 1 + [-math.radians(self.profile.max_pitch_deg)] * 1 + [-math.pi], dtype=np.float32), np.array([math.radians(self.profile.max_roll_deg), math.radians(self.profile.max_pitch_deg), math.pi], dtype=np.float32))

        power_draw = self._compute_energy_draw(float(np.linalg.norm(body_accel)), speed)
        self.state.battery_wh = max(0.0, self.state.battery_wh - power_draw * dt * self.state.battery_capacity_wh)
        self.state.energy_used_wh += power_draw * dt * self.state.battery_capacity_wh
        self.state.time_s += dt
        self._step_count += 1

        reward = 0.0
        reward += 0.35 * (self.state.position_m[0] - previous_position[0])
        reward -= 0.02 * float(np.linalg.norm(self.state.position_m[1:]))
        reward -= 0.01 * float(np.linalg.norm(self.state.velocity_mps - previous_velocity))
        reward -= 0.0015 * self.state.energy_used_wh

        gate_passed = False
        collision = False
        precision_score = 0.0
        if self.state.gate_index < len(self.gates):
            gate_local_prev = self._gate_local_coordinates(gate, previous_position)
            gate_local_now = self._gate_local_coordinates(gate, self.state.position_m)
            crossed_gate_plane = gate_local_prev[0] <= 0.0 < gate_local_now[0]
            within_opening = abs(gate_local_now[1]) <= gate.half_width and abs(gate_local_now[2]) <= gate.half_height
            near_plane = abs(gate_local_now[0]) <= gate.frame_thickness_m
            if crossed_gate_plane and within_opening:
                gate_passed = True
                precision_error = float(abs(gate_local_now[1]) / max(gate.half_width, 1e-3) + abs(gate_local_now[2]) / max(gate.half_height, 1e-3))
                precision_score = max(0.0, 1.0 - 0.5 * precision_error)
                reward += 12.0 + 4.0 * precision_score
                self.state.gate_precision_total += precision_score
                self.state.gate_index += 1
                if self.state.gate_index >= len(self.gates):
                    reward += 20.0
            elif near_plane and not within_opening:
                collision = True
                self.state.collided = True
                reward -= 8.0
                self.state.off_course_events += 1
                self.state.gate_index = min(self.state.gate_index, len(self.gates) - 1)

        if self._out_of_bounds(self.state.position_m):
            collision = True
            self.state.collided = True
            reward -= 10.0
            self.state.off_course_events += 1

        if self.state.collided:
            reward -= 2.0

        smoothness = max(0.0, 1.0 - float(np.mean(np.abs(action_array - previous_action))))
        safety = self._safety_score()
        reward += 0.35 * smoothness + 0.5 * safety

        self.state.smoothness_total += smoothness
        self.state.safety_total += safety
        self._last_reward = reward

        self._latent_observation_queue.append(self._sensor_reading(self.state))
        self._current_observation = self._get_observation()

        done = self._check_done()
        metrics = self._build_metrics(reward=reward, timed_out=self.state.time_s >= self.profile.max_episode_time_s)
        self.last_metrics = metrics
        info = {
            "metrics": metrics.to_dict(),
            "gate_passed": gate_passed,
            "collision": collision,
            "difficulty": self._difficulty,
        }
        return self._current_observation.copy(), float(reward), done, info

    def _out_of_bounds(self, position: np.ndarray) -> bool:
        return bool(
            position[0] < -self.profile.course_margin_m
            or position[0] > self.profile.field_length_m + self.profile.course_margin_m
            or abs(position[1]) > self.profile.field_width_m / 2.0 + self.profile.course_margin_m
            or position[2] < self.profile.min_altitude_m - 0.4
            or position[2] > self.profile.max_altitude_m + 0.4
        )

    def _safety_score(self) -> float:
        roll_ok = abs(self.state.attitude_rad[0]) <= math.radians(self.profile.max_roll_deg)
        pitch_ok = abs(self.state.attitude_rad[1]) <= math.radians(self.profile.max_pitch_deg)
        speed_ok = np.linalg.norm(self.state.velocity_mps) <= self.profile.max_speed_mps
        altitude_ok = self.profile.min_altitude_m <= self.state.position_m[2] <= self.profile.max_altitude_m
        battery_ok = self.state.battery_wh > 0.0
        return float(sum([roll_ok, pitch_ok, speed_ok, altitude_ok, battery_ok]) / 5.0)

    def _check_done(self) -> bool:
        if self.state.collided:
            return True
        if self.state.gate_index >= len(self.gates):
            return True
        if self.state.time_s >= self.profile.max_episode_time_s:
            return True
        if self.state.battery_wh <= 0.0:
            return True
        if self._out_of_bounds(self.state.position_m):
            return True
        return False

    def _build_metrics(self, reward: float, timed_out: bool) -> EpisodeMetrics:
        gates_cleared = min(self.state.gate_index, len(self.gates))
        efficiency = float(gates_cleared / max(1.0, self.state.time_s / max(self.profile.max_episode_time_s, 1e-3)))
        gate_precision = float(self.state.gate_precision_total / max(1, gates_cleared))
        smoothness = float(self.state.smoothness_total / max(1.0, self._step_count))
        safety = float(self.state.safety_total / max(1.0, self._step_count))
        metrics = EpisodeMetrics(
            reward=float(reward),
            gates_cleared=int(gates_cleared),
            collisions=int(self.state.collided),
            gate_precision=gate_precision,
            efficiency=efficiency,
            smoothness=smoothness,
            safety=safety,
            off_course_events=int(self.state.off_course_events),
            time_elapsed_s=float(self.state.time_s),
            energy_used_wh=float(self.state.energy_used_wh),
            timed_out=timed_out,
            success=bool(gates_cleared >= len(self.gates) and not self.state.collided),
            curriculum_level=self._difficulty,
        )
        metrics.score = self.profile.score_episode(metrics.to_dict())
        return metrics

    def observation_to_tensor(self) -> np.ndarray:
        return self._current_observation.copy()

    def render(self, width: int = 1280, height: int = 720) -> np.ndarray:
        """Render a high-level 3D perspective view using OpenCV primitives."""

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._draw_background(canvas)
        camera_pos = self.state.position_m + np.array([-3.0, -3.5, 2.2], dtype=np.float32)
        target = self.state.position_m + np.array([6.0, 0.0, 0.1], dtype=np.float32)
        self._draw_ground_grid(canvas, camera_pos, target)
        for index, gate in enumerate(self.gates):
            self._draw_gate(canvas, camera_pos, target, gate, index=index)
        self._draw_drone(canvas, camera_pos, target)
        self._draw_hud(canvas)
        return canvas

    def _draw_background(self, canvas: np.ndarray) -> None:
        height = canvas.shape[0]
        for y in range(height):
            blend = y / max(1, height - 1)
            sky = np.array([18 + 20 * blend, 34 + 26 * blend, 54 + 48 * blend], dtype=np.uint8)
            canvas[y, :] = sky
        cv2.rectangle(canvas, (0, int(height * 0.63)), (canvas.shape[1], height), (26, 54, 28), -1)
        cv2.rectangle(canvas, (0, int(height * 0.63)), (canvas.shape[1], int(height * 0.67)), (40, 92, 44), -1)

    def _camera_basis(self, camera_pos: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = _normalize(target - camera_pos)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = _normalize(np.cross(forward, world_up))
        up = _normalize(np.cross(right, forward))
        return forward, right, up

    def _project(self, point: np.ndarray, camera_pos: np.ndarray, target: np.ndarray, width: int, height: int) -> tuple[int, int] | None:
        forward, right, up = self._camera_basis(camera_pos, target)
        relative = point - camera_pos
        x = float(np.dot(relative, right))
        y = float(np.dot(relative, up))
        z = float(np.dot(relative, forward))
        if z <= 0.1:
            return None
        focal = width * 0.85
        screen_x = int(width * 0.5 + (x / z) * focal)
        screen_y = int(height * 0.58 - (y / z) * focal)
        return screen_x, screen_y

    def _draw_ground_grid(self, canvas: np.ndarray, camera_pos: np.ndarray, target: np.ndarray) -> None:
        width, height = canvas.shape[1], canvas.shape[0]
        for x in np.linspace(0.0, self.profile.field_length_m, 16):
            start = np.array([x, -self.profile.field_width_m * 0.5, 0.0], dtype=np.float32)
            end = np.array([x, self.profile.field_width_m * 0.5, 0.0], dtype=np.float32)
            p1 = self._project(start, camera_pos, target, width, height)
            p2 = self._project(end, camera_pos, target, width, height)
            if p1 and p2:
                cv2.line(canvas, p1, p2, (50, 70, 50), 1, cv2.LINE_AA)
        for y in np.linspace(-self.profile.field_width_m * 0.5, self.profile.field_width_m * 0.5, 10):
            start = np.array([0.0, y, 0.0], dtype=np.float32)
            end = np.array([self.profile.field_length_m, y, 0.0], dtype=np.float32)
            p1 = self._project(start, camera_pos, target, width, height)
            p2 = self._project(end, camera_pos, target, width, height)
            if p1 and p2:
                cv2.line(canvas, p1, p2, (60, 82, 60), 1, cv2.LINE_AA)

    def _gate_corners(self, gate: Gate) -> list[np.ndarray]:
        local = np.array([
            [-gate.frame_thickness_m, -gate.half_width, -gate.half_height],
            [gate.frame_thickness_m, -gate.half_width, -gate.half_height],
            [gate.frame_thickness_m, gate.half_width, -gate.half_height],
            [-gate.frame_thickness_m, gate.half_width, -gate.half_height],
            [-gate.frame_thickness_m, -gate.half_width, gate.half_height],
            [gate.frame_thickness_m, -gate.half_width, gate.half_height],
            [gate.frame_thickness_m, gate.half_width, gate.half_height],
            [-gate.frame_thickness_m, gate.half_width, gate.half_height],
        ], dtype=np.float32)
        rotation = _rotation_yaw(gate.yaw_rad)
        return [gate.center + rotation @ corner for corner in local]

    def _draw_gate(self, canvas: np.ndarray, camera_pos: np.ndarray, target: np.ndarray, gate: Gate, index: int) -> None:
        width, height = canvas.shape[1], canvas.shape[0]
        corners = self._gate_corners(gate)
        color = (90, 220, 255) if index >= self.state.gate_index else (84, 180, 96)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        projected = [self._project(corner, camera_pos, target, width, height) for corner in corners]
        for a, b in edges:
            if projected[a] and projected[b]:
                cv2.line(canvas, projected[a], projected[b], color, 3, cv2.LINE_AA)
        label = self._project(gate.center, camera_pos, target, width, height)
        if label:
            cv2.putText(canvas, f"G{index + 1}", (label[0] - 18, label[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)

    def _draw_drone(self, canvas: np.ndarray, camera_pos: np.ndarray, target: np.ndarray) -> None:
        width, height = canvas.shape[1], canvas.shape[0]
        screen = self._project(self.state.position_m, camera_pos, target, width, height)
        if screen is None:
            return
        body_color = (0, 215, 255) if not self.state.collided else (40, 80, 255)
        cv2.circle(canvas, screen, 14, body_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, screen, 24, (255, 255, 255), 2, cv2.LINE_AA)
        vel = self.state.position_m + _normalize(self.state.velocity_mps + np.array([0.1, 0.0, 0.0], dtype=np.float32)) * 1.1
        vel_screen = self._project(vel, camera_pos, target, width, height)
        if vel_screen:
            cv2.arrowedLine(canvas, screen, vel_screen, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)

    def _draw_hud(self, canvas: np.ndarray) -> None:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (20, 20), (430, 190), (15, 18, 22), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.putText(canvas, "TSA HS 2026 3D SIM", (34, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Difficulty: {self._difficulty.upper()}", (34, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Gate: {min(self.state.gate_index + 1, len(self.gates))}/{len(self.gates)}", (34, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Battery: {self.state.battery_wh:.1f} Wh", (34, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Speed: {np.linalg.norm(self.state.velocity_mps):.2f} m/s", (34, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Score: {0.0 if self.last_metrics is None else self.last_metrics.score:.2f}", (34, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 160), 1, cv2.LINE_AA)
        if self.last_metrics is not None:
            cv2.putText(canvas, f"Collisions: {self.last_metrics.collisions}  Time: {self.last_metrics.time_elapsed_s:.1f}s", (460, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    def _build_metrics(self, reward: float, timed_out: bool) -> EpisodeMetrics:
        gates_cleared = min(self.state.gate_index, len(self.gates))
        efficiency = float(gates_cleared / max(1.0, self.state.time_s / max(self.profile.max_episode_time_s, 1e-3)))
        gate_precision = float(self.state.gate_precision_total / max(1, gates_cleared))
        smoothness = float(self.state.smoothness_total / max(1.0, self._step_count))
        safety = float(self.state.safety_total / max(1.0, self._step_count))
        metrics = EpisodeMetrics(
            reward=float(reward),
            gates_cleared=int(gates_cleared),
            collisions=int(self.state.collided),
            gate_precision=gate_precision,
            efficiency=efficiency,
            smoothness=smoothness,
            safety=safety,
            off_course_events=int(self.state.off_course_events),
            time_elapsed_s=float(self.state.time_s),
            energy_used_wh=float(self.state.energy_used_wh),
            timed_out=timed_out,
            success=bool(gates_cleared >= len(self.gates) and not self.state.collided),
            curriculum_level=self._difficulty,
        )
        metrics.score = self.profile.score_episode(metrics.to_dict())
        return metrics

    def run_episode(self, policy_fn: Any | None = None, max_steps: int | None = None) -> EpisodeMetrics:
        """Convenience helper for headless rollouts."""

        observation = self.reset()
        done = False
        step_limit = max_steps or self._step_limit
        step_count = 0
        while not done and step_count < step_limit:
            action = self.expert_action() if policy_fn is None else policy_fn(observation)
            observation, _, done, _ = self.step(action)
            step_count += 1
        return self.last_metrics or self._build_metrics(0.0, timed_out=True)