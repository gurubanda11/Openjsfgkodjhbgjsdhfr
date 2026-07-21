"""
Webcam inspection and real-time live inference loop for OpenVLA Auto.
Provides camera auto-detection, HUD overlay rendering, and interactive controls.
"""

import time
from typing import List, Optional
import cv2
import numpy as np

from src.inference import OpenVLAInferenceEngine, InferenceResult
from src.utils import webcam_logger, errors_logger


class WebcamRunner:
    """
    Manages camera device detection, streaming, and real-time inference HUD overlay.
    """

    def __init__(self, inference_engine: Optional[OpenVLAInferenceEngine] = None) -> None:
        self.engine = inference_engine or OpenVLAInferenceEngine()

    @staticmethod
    def detect_webcams(max_tested: int = 5) -> List[int]:
        """
        Scans camera index range to discover available webcams.
        """
        available_cameras = []
        webcam_logger.info("Scanning for available webcam devices...")
        for index in range(max_tested):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if cv2.__version__.startswith("4") and cv2.os.name == "nt" else cv2.CAP_ANY)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(index)
                cap.release()

        webcam_logger.info(f"Discovered webcams at indices: {available_cameras}")
        return available_cameras

    def select_camera(self) -> int:
        """
        Allows interactive webcam selection or picks default index 0.
        """
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
            choice = input("Select camera index (default 0): ").strip()
            if choice.isdigit() and int(choice) in cams:
                return int(choice)
        except Exception:
            pass

        return cams[0]

    def render_hud_overlay(
        self,
        frame: np.ndarray,
        result: InferenceResult,
        fps: float,
    ) -> np.ndarray:
        """
        Overlays action vector, inference latency, FPS, device, and hotkeys onto frame.
        """
        annotated = frame.copy()
        height, width, _ = annotated.shape

        # Semi-transparent top HUD banner bar
        hud_height = 110
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, hud_height), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Draw border line under HUD
        cv2.line(annotated, (0, hud_height), (width, hud_height), (0, 215, 255), 2)

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1

        # Text Lines
        title_str = "OPENVLA LIVE INFERENCE HUD"
        action_str = f"Action: [{', '.join(f'{x:.2f}' for x in result.action_vector)}]"
        latency_str = f"Latency: {result.latency_ms:.1f} ms"
        fps_str = f"FPS: {fps:.1f}"
        device_str = f"Device: {result.device}"
        quit_str = "Press 'Q' to Quit"

        # Render Left Column
        cv2.putText(annotated, title_str, (15, 22), font, 0.6, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, action_str, (15, 52), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(annotated, quit_str, (15, 82), font, 0.5, (180, 180, 180), thickness, cv2.LINE_AA)

        # Render Right Column
        cv2.putText(annotated, device_str, (width - 220, 22), font, font_scale, (0, 255, 150), 2, cv2.LINE_AA)
        cv2.putText(annotated, latency_str, (width - 220, 52), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(annotated, fps_str, (width - 220, 82), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return annotated

    def run_live_loop(self, camera_index: Optional[int] = None) -> None:
        """
        Opens specified webcam and runs live inference streaming loop until 'q' key is pressed.
        """
        if camera_index is None:
            camera_index = self.select_camera()

        webcam_logger.info(f"Opening webcam at index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            err_msg = f"Unable to open webcam at index {camera_index}."
            webcam_logger.error(err_msg)
            errors_logger.error(err_msg)
            print(f"\nError: {err_msg}")
            print("Running in headless synthetic mode...")
            self.run_synthetic_loop()
            return

        window_name = "OpenVLA Auto - Live Inference"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

        prev_time = time.time()
        fps = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    webcam_logger.warning("Failed to grab frame from webcam. Exiting loop.")
                    break

                # Measure FPS
                curr_time = time.time()
                elapsed = curr_time - prev_time
                if elapsed > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps > 0 else (1.0 / elapsed)
                prev_time = curr_time

                # Run OpenVLA Inference on current frame
                res = self.engine.run_inference(frame)

                # Render HUD text overlays
                hud_frame = self.render_hud_overlay(frame, res, fps)

                cv2.imshow(window_name, hud_frame)

                # Key check 'q' or 'Q' to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q") or key == 27:
                    webcam_logger.info("User pressed Quit key. Exiting live streaming.")
                    break

                # Check if window was closed
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

        except Exception as e:
            webcam_logger.error(f"Error during webcam streaming: {e}")
            errors_logger.error(f"Webcam stream error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            webcam_logger.info("Webcam video capture released.")

    def run_synthetic_loop(self, max_frames: int = 100) -> None:
        """
        Runs synthetic frame loop when no hardware webcam is connected.
        """
        webcam_logger.info("Starting synthetic test stream loop...")
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        window_name = "OpenVLA Auto - Live Inference (Synthetic Mode)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        prev_time = time.time()
        fps = 30.0

        for frame_idx in range(max_frames):
            # Generate test image with moving circle
            frame = blank_frame.copy()
            x = int(320 + 150 * np.cos(frame_idx * 0.1))
            y = int(240 + 100 * np.sin(frame_idx * 0.1))
            cv2.circle(frame, (x, y), 30, (0, 215, 255), -1)

            curr_time = time.time()
            elapsed = curr_time - prev_time
            if elapsed > 0:
                fps = 1.0 / elapsed
            prev_time = curr_time

            res = self.engine.run_inference(frame)
            hud_frame = self.render_hud_overlay(frame, res, fps)

            cv2.imshow(window_name, hud_frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == ord("Q") or key == 27:
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    runner = WebcamRunner()
    runner.run_live_loop()
