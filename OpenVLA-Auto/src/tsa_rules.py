"""
Competition profile and TSA-style scoring rules for the HS 2026 drone course.

The profile is intentionally configurable so it can track a chapter's exact rulebook
overlay without changing the rest of the training and evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GateSpec:
    """Geometry of an individual course gate."""

    inner_width_m: float
    inner_height_m: float
    frame_thickness_m: float = 0.08


@dataclass(frozen=True)
class ChallengeScoring:
    """Scoring weights used to compute a TSA-style episode score."""

    gate_clear_bonus: float = 14.0
    precision_bonus: float = 5.0
    efficiency_bonus: float = 4.0
    smoothness_bonus: float = 3.0
    safety_bonus: float = 4.0
    collision_penalty: float = 12.0
    timeout_penalty: float = 10.0
    off_course_penalty: float = 8.0


@dataclass(frozen=True)
class TSAChallengeProfile:
    """Configurable high-school TSA drone challenge profile."""

    event_name: str
    season: str
    objective: str
    field_length_m: float
    field_width_m: float
    field_height_m: float
    gate_count: int
    gate_spacing_m: float
    gate: GateSpec
    scoring: ChallengeScoring
    max_episode_time_s: float
    max_speed_mps: float
    max_vertical_speed_mps: float
    max_roll_deg: float
    max_pitch_deg: float
    max_yaw_rate_deg_s: float
    min_altitude_m: float
    max_altitude_m: float
    launch_zone_radius_m: float
    landing_zone_radius_m: float
    course_margin_m: float = 0.6
    notes: str = ""

    def validate(self) -> None:
        """Raise an error if the profile dimensions are internally inconsistent."""

        if self.gate_count <= 0:
            raise ValueError("gate_count must be positive")
        if self.field_length_m <= 0 or self.field_width_m <= 0 or self.field_height_m <= 0:
            raise ValueError("Field dimensions must be positive")
        if self.gate.inner_width_m >= self.field_width_m:
            raise ValueError("Gate width must be smaller than field width")
        if self.gate.inner_height_m >= self.field_height_m:
            raise ValueError("Gate height must be smaller than field height")
        if self.max_altitude_m <= self.min_altitude_m:
            raise ValueError("max_altitude_m must exceed min_altitude_m")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def score_episode(self, metrics: Mapping[str, Any]) -> float:
        """Convert raw episode metrics into a TSA-style score."""

        gates_cleared = float(metrics.get("gates_cleared", 0.0))
        collisions = float(metrics.get("collisions", 0.0))
        gate_precision = float(metrics.get("gate_precision", 0.0))
        efficiency = float(metrics.get("efficiency", 0.0))
        smoothness = float(metrics.get("smoothness", 0.0))
        safety = float(metrics.get("safety", 0.0))
        off_course = float(metrics.get("off_course_events", 0.0))
        timed_out = bool(metrics.get("timed_out", False))

        score = 0.0
        score += gates_cleared * self.scoring.gate_clear_bonus
        score += gate_precision * self.scoring.precision_bonus
        score += efficiency * self.scoring.efficiency_bonus
        score += smoothness * self.scoring.smoothness_bonus
        score += safety * self.scoring.safety_bonus
        score -= collisions * self.scoring.collision_penalty
        score -= off_course * self.scoring.off_course_penalty
        if timed_out:
            score -= self.scoring.timeout_penalty
        return score


def create_default_hs_2026_profile() -> TSAChallengeProfile:
    """Create a configurable baseline profile for the HS 2026 TSA course.

    The values are intentionally adjustable so local chapter overlays or later rulebook
    updates can be mirrored without changing the rest of the simulation stack.
    """

    profile = TSAChallengeProfile(
        event_name="TSA HS Drone Challenge",
        season="2026",
        objective="Autonomously complete a gated aerial course with precision, safety, and efficiency.",
        field_length_m=18.0,
        field_width_m=12.0,
        field_height_m=4.5,
        gate_count=6,
        gate_spacing_m=2.5,
        gate=GateSpec(inner_width_m=1.2, inner_height_m=1.2, frame_thickness_m=0.08),
        scoring=ChallengeScoring(),
        max_episode_time_s=180.0,
        max_speed_mps=8.0,
        max_vertical_speed_mps=3.0,
        max_roll_deg=35.0,
        max_pitch_deg=35.0,
        max_yaw_rate_deg_s=120.0,
        min_altitude_m=0.25,
        max_altitude_m=3.6,
        launch_zone_radius_m=0.75,
        landing_zone_radius_m=0.85,
        course_margin_m=0.7,
        notes="Baseline TSA-style profile; tune dimensions and scoring weights to the official rulebook if your chapter uses local overlays.",
    )
    profile.validate()
    return profile