"""
Deployment adapter contracts for future real-drone integration.

This file intentionally stops at interfaces and abstract contracts so the simulation and
training pipeline can target a stable command API without implementing any radio or
flight-controller communication yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class DroneControlCommand:
    """Normalized flight command produced by a policy or safety layer."""

    roll: float
    pitch: float
    throttle: float
    yaw_rate: float
    payload_release: float = 0.0
    auxiliary_1: float = 0.0
    auxiliary_2: float = 0.0


@dataclass(frozen=True)
class DroneTelemetry:
    """Minimal telemetry bundle for future hardware adapters."""

    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    battery_fraction: float
    status: str


@runtime_checkable
class DroneAdapter(Protocol):
    """Interface for a future radio / FC adapter."""

    def connect(self) -> None:
        """Establish a hardware connection."""

    def send_command(self, command: DroneControlCommand) -> None:
        """Send a single command to the vehicle."""

    def get_telemetry(self) -> DroneTelemetry:
        """Return the most recent telemetry snapshot."""

    def disconnect(self) -> None:
        """Close the hardware connection."""


class BaseDroneAdapter:
    """Abstract base that makes the deferred integration boundary explicit."""

    def connect(self) -> None:
        raise NotImplementedError("Hardware communication is deferred until the drone stack is built.")

    def send_command(self, command: DroneControlCommand) -> None:
        raise NotImplementedError("Hardware communication is deferred until the drone stack is built.")

    def get_telemetry(self) -> DroneTelemetry:
        raise NotImplementedError("Hardware communication is deferred until the drone stack is built.")

    def disconnect(self) -> None:
        raise NotImplementedError("Hardware communication is deferred until the drone stack is built.")