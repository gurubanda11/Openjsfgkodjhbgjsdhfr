"""
Webcam inspection and real-time live inference loop for OpenVLA Auto.
Provides camera auto-detection, HUD overlay rendering, virtual sim panel, and interactive controls.
"""

import platform
import time
from typing import List, Optional

import cv2
import numpy as np

from config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    HUD_HEIGHT,
    INFERENCE_EVERY_N_FRAMES,
    SIM_PANEL_WIDTH,
)
from src.inference import OpenVLAInferenceEngine, InferenceResult
from src.sim import VirtualSim
from src.utils import webcam_logger, errors_logger


class WebcamRunner:
    """
    Manages camera device detection, streaming, and real-time inference HUD overlay.
    """

    def __init__(self, inference_engine: Optional[OpenVLAInferenceEngine] = None) -> None:
        self.engine = inference_engine or OpenVLAInferenceEngine()
        self.sim = VirtualSim(width=SIM_PANEL_WIDTH, height=CAMERA_HEIGHT)

    @staticmethod
    def _capture_backend() -> int:
        if platform.system().lower().startswith("win"):
            return cv2.CAP_DSHOW
        return cv2.CAP_ANY

    @staticmethod
    def detect_webcams(max_tested: int = 6) -> List[int]:
        available_cameras: List[int] = []
        webcam_logger.info("Scanning for available webcam devices...")
        backend = WebcamRunner._capture_backend()
        for index in range(max_tested):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(index)
            cap.release()
        webcam_logger.info(f"Discovered webcams at indices: {available_cameras}")
        return available_cameras

    def select_camera(self) -> int:
        cams = self.detect_webcams()
        if not cams:
            webcam_logger.warning("No physical webcam detected. Defaulting to camera index 0.")
            return 0
        if len(cams) == 1:
            return cams[0]

        print("\nAvailable Webcams Detected:")
        for idx in cams:
            print(f"  [{idx}] Camera Index {idx}")

        try:
            choice = input("Select camera index (default first detected): ").strip()
            if choice.isdigit() and int(choice) in cams:
                return int(choice)
        except Exception:
            pass
        return cams[0]

    def render_hud_overlay(self, frame: np.ndarray, result: InferenceResult, fps: float, mode: str, paused: bool) -> np.ndarray:
        annotated = frame.copy()
        height, width, _ = annotated.shape

        hud_height = min(HUD_HEIGHT, max(130, height // 3))
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, hud_height), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)
        cv2.line(annotated, (0, hud_height), (width, hud_height), (0, 215, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "OPENVLA LIVE + VIRTUAL SIM"
        status = "PAUSED" if paused else "RUNNING"

        action_line = result.action_text
        if len(action_line) > 95:
            action_line = action_line[:92] + "..."

        cv2.putText(annotated, title, (12, 24), font, 0.62, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"State: {status}  Mode: {mode.upper()}  (1 robot / 2 drone)", (12, 50), font, 0.48, (225, 225, 225), 1, cv2.LINE_AA)
        cv2.putText(annotated, action_line, (12, 76), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Latency: {result.latency_ms:.1f} ms  FPS: {fps:.1f}  Device: {result.device}", (12, 102), font, 0.48, (120, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(annotated, "Q/Esc quit  P pause  1 robot  2 drone", (12, 126), font, 0.45, (190, 190, 190), 1, cv2.LINE_AA)

        return annotated

    def run_live_loop(self, camera_index: Optional[int] = None) -> None:
        if camera_index is None:
            camera_index = self.select_camera()

        webcam_logger.info(f"Opening webcam at index {camera_index}...")
        cap = cv2.VideoCapture(camera_index, self._capture_backend())
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():
            err_msg = f"Unable to open webcam at index {camera_index}."
            webcam_logger.error(err_msg)
            errors_logger.error(err_msg)
            print(f"\nError: {err_msg}")
            print("Running in synthetic mode...")
            self.run_synthetic_loop()
            return

        window_name = "OpenVLA Auto - Webcam + Virtual Sim"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, CAMERA_WIDTH + SIM_PANEL_WIDTH, CAMERA_HEIGHT)

        prev_time = time.time()
        prev_frame_time = time.perf_counter()
        fps = 0.0
        frame_idx = 0
        paused = False

        last_result = InferenceResult(
            action_vector=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            action_text="Action: [0,0,0,0,0,0,1]",
            latency_ms=0.0,
            device=self.engine.device.upper(),
            raw_output_text="init",
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    webcam_logger.warning("Failed to grab frame from webcam. Exiting loop.")
                    break

                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

                now = time.time()
                elapsed = now - prev_time
                if elapsed > 0:
                    inst = 1.0 / elapsed
                    fps = 0.9 * fps + 0.1 * inst if fps > 0 else inst
                prev_time = now

                dt = max(0.001, time.perf_counter() - prev_frame_time)
                prev_frame_time = time.perf_counter()

                if not paused and (frame_idx % max(1, INFERENCE_EVERY_N_FRAMES) == 0):
                    last_result = self.engine.run_inference(frame)

                if not paused:
                    self.sim.step(last_result.action_vector, dt)

                cam_view = self.render_hud_overlay(frame, last_result, fps, self.sim.state.mode, paused)
                sim_view = self.sim.render(fps=fps, latency_ms=last_result.latency_ms, device=last_result.device, action_text=last_result.action_text)

                combined = np.hstack([cam_view, sim_view])
                cv2.imshow(window_name, combined)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    webcam_logger.info("User requested quit.")
                    break
                if key in (ord("p"), ord("P")):
                    paused = not paused
                if key == ord("1"):
                    self.sim.set_mode("robot")
                if key == ord("2"):
                    self.sim.set_mode("drone")

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                frame_idx += 1

        except Exception as e:
            webcam_logger.exception(f"Error during webcam streaming: {e}")
            errors_logger.exception(f"Webcam stream error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            webcam_logger.info("Webcam video capture released.")

    def run_synthetic_loop(self, max_frames: int = 600) -> None:
        webcam_logger.info("Starting synthetic webcam + sim loop...")
        blank = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

        window_name = "OpenVLA Auto - Synthetic Webcam + Virtual Sim"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, CAMERA_WIDTH + SIM_PANEL_WIDTH, CAMERA_HEIGHT)

        prev_time = time.time()
        prev_frame_time = time.perf_counter()
        fps = 30.0
        paused = False

        last_result = InferenceResult(
            action_vector=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            action_text="Action: [0,0,0,0,0,0,1]",
            latency_ms=0.0,
            device=self.engine.device.upper(),
            raw_output_text="init",
        )

        for frame_idx in range(max_frames):
            frame = blank.copy()
            x = int(CAMERA_WIDTH * 0.5 + 160 * np.cos(frame_idx * 0.06))
            y = int(CAMERA_HEIGHT * 0.5 + 100 * np.sin(frame_idx * 0.05))
            cv2.circle(frame, (x, y), 34, (0, 215, 255), -1)
            cv2.putText(frame, "Synthetic input", (18, CAMERA_HEIGHT - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

            now = time.time()
            elapsed = now - prev_time
            if elapsed > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / elapsed)
            prev_time = now

            dt = max(0.001, time.perf_counter() - prev_frame_time)
            prev_frame_time = time.perf_counter()

            if not paused and (frame_idx % max(1, INFERENCE_EVERY_N_FRAMES) == 0):
                last_result = self.engine.run_inference(frame)

            if not paused:
                self.sim.step(last_result.action_vector, dt)

            cam_view = self.render_hud_overlay(frame, last_result, fps, self.sim.state.mode, paused)
            sim_view = self.sim.render(fps=fps, latency_ms=last_result.latency_ms, device=last_result.device, action_text=last_result.action_text)
            combined = np.hstack([cam_view, sim_view])
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("p"), ord("P")):
                paused = not paused
            if key == ord("1"):
                self.sim.set_mode("robot")
            if key == ord("2"):
                self.sim.set_mode("drone")

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    runner = WebcamRunner()
    runner.run_live_loop()
