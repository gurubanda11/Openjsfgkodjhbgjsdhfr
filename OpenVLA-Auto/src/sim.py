"""
Virtual robot/drone simulation renderer driven by OpenVLA action vectors.
"""

from dataclasses import dataclass
import math
from typing import List

import cv2
import numpy as np


@dataclass
class SimState:
    mode: str = "robot"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


class VirtualSim:
    def __init__(self, width: int = 480, height: int = 540) -> None:
        self.width = width
        self.height = height
        self.state = SimState()

    def set_mode(self, mode: str) -> None:
        if mode in {"robot", "drone"}:
            self.state.mode = mode

    def step(self, action_vector: List[float], dt: float) -> None:
        ax = action_vector + [0.0] * (7 - len(action_vector))
        dx, dy, dz, droll, dpitch, dyaw, grip = ax[:7]

        scale_xy = 120.0
        scale_z = 1.5
        self.state.vx = float(dx) * scale_xy
        self.state.vy = float(dy) * scale_xy
        self.state.vz = float(dz) * scale_z

        self.state.x += self.state.vx * dt
        self.state.y += self.state.vy * dt
        self.state.z = max(-2.0, min(2.0, self.state.z + self.state.vz * dt))
        self.state.yaw += float(dyaw) * 2.0 * dt

        self.state.x = max(-200.0, min(200.0, self.state.x))
        self.state.y = max(-200.0, min(200.0, self.state.y))

    def _draw_grid(self, img: np.ndarray) -> None:
        step = 40
        for x in range(0, self.width, step):
            cv2.line(img, (x, 0), (x, self.height), (35, 35, 35), 1)
        for y in range(0, self.height, step):
            cv2.line(img, (0, y), (self.width, y), (35, 35, 35), 1)

    def _world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        cx = self.width // 2
        cy = self.height // 2
        sx = int(cx + x)
        sy = int(cy + y)
        return sx, sy

    def _draw_robot(self, img: np.ndarray, sx: int, sy: int, yaw: float) -> None:
        cv2.circle(img, (sx, sy), 18, (0, 220, 255), -1)
        tip = (int(sx + 28 * math.cos(yaw)), int(sy + 28 * math.sin(yaw)))
        cv2.line(img, (sx, sy), tip, (255, 255, 255), 3)

    def _draw_drone(self, img: np.ndarray, sx: int, sy: int, yaw: float, z: float) -> None:
        altitude_color = (80, 255, 80) if z >= 0 else (80, 80, 255)
        r = int(16 + max(0, z) * 3)
        cv2.circle(img, (sx, sy), r, altitude_color, 2)
        cv2.line(img, (sx - 20, sy), (sx + 20, sy), (255, 255, 255), 2)
        cv2.line(img, (sx, sy - 20), (sx, sy + 20), (255, 255, 255), 2)
        tip = (int(sx + 30 * math.cos(yaw)), int(sy + 30 * math.sin(yaw)))
        cv2.arrowedLine(img, (sx, sy), tip, (0, 220, 255), 2, tipLength=0.25)

    def render(self, fps: float, latency_ms: float, device: str, action_text: str) -> np.ndarray:
        panel = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        panel[:] = (20, 20, 24)
        self._draw_grid(panel)

        sx, sy = self._world_to_screen(self.state.x, self.state.y)
        if self.state.mode == "robot":
            self._draw_robot(panel, sx, sy, self.state.yaw)
        else:
            self._draw_drone(panel, sx, sy, self.state.yaw, self.state.z)

        cv2.putText(panel, "OPENVLA VIRTUAL SIM", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"Mode: {self.state.mode.upper()}  (1=ROBOT, 2=DRONE)", (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Pos: x={self.state.x:.1f}, y={self.state.y:.1f}, z={self.state.z:.2f}", (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Vel: vx={self.state.vx:.1f}, vy={self.state.vy:.1f}, vz={self.state.vz:.2f}", (16, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, f"FPS: {fps:.1f}  Latency: {latency_ms:.1f} ms  Device: {device}", (16, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 120), 1, cv2.LINE_AA)

        action_preview = action_text if len(action_text) < 80 else action_text[:77] + "..."
        cv2.putText(panel, action_preview, (16, self.height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

        return panel
