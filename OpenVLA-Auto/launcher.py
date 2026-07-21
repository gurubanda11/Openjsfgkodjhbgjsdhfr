"""
Master Launcher for OpenVLA Auto.
Orchestrates virtual environment check, hardware inspection, dependency installation,
model download, model loading, built-in self-testing, and live webcam inference.
"""

import sys
import time

from installer import VenvManager, DependencyInstaller
from src.device import detect_system_info
from src.utils import install_logger, errors_logger


def main() -> None:
    """
    Executes full OpenVLA Auto pipeline.
    """
    print("\n" + "=" * 50)
    print("           WELCOME TO OPENVLA AUTO          ")
    print("=" * 50 + "\n")

    # Step 1: Pre-flight Virtual Environment Auto-Activation / Re-execution
    if not VenvManager.is_in_venv():
        print("Initializing Virtual Environment...")
        VenvManager.activate_or_reexec()

    # Step 2: System & Hardware Inspection
    sys_info = detect_system_info()
    print(sys_info.format_terminal_ui())
    print("\n")

    # Step 3: Dependency Check & Installation
    print("Checking system dependencies...")
    installer = DependencyInstaller(system_info=sys_info)
    if not installer.run_installation():
        print("\nFatal: Dependency installation failed. Check logs for details.")
        sys.exit(1)

    # Lazy-import ML and Vision modules AFTER dependencies are verified/installed
    from src.downloader import ModelDownloader
    from src.model import OpenVLAModelLoader
    from src.inference import OpenVLAInferenceEngine
    from src.webcam import WebcamRunner

    # Step 4: Model Check & Download from Hugging Face
    print("\nChecking OpenVLA Model files...")
    downloader = ModelDownloader()
    try:
        model_path = downloader.download()
        downloader.verify_integrity(model_path)
    except Exception as e:
        print(f"\nFatal: Model download failed: {e}")
        errors_logger.error(f"Launcher model download failed: {e}")
        sys.exit(1)

    # Step 5: Model Loader & Precision Assignment
    print("\nLoading OpenVLA Model onto compute backend...")
    loader = OpenVLAModelLoader(system_info=sys_info)
    container = None
    try:
        container = loader.load_model()
    except Exception as e:
        print(f"\nWarning: Could not load full 7B model directly ({e}).")
        print("Falling back to Inference Engine synthetic/mock execution for self-test & webcam.")

    # Step 6: Built-in Self Test
    print("\nRunning Self Test...")
    engine = OpenVLAInferenceEngine(model_container=container)
    self_test_passed = engine.run_self_test()
    if not self_test_passed:
        print("\nWarning: Self test failed. Proceeding with caution.")

    # Step 7: Webcam Selection & Live Stream Inference Loop
    print("\nInitializing Live Webcam Inference Engine...")
    runner = WebcamRunner(inference_engine=engine)
    runner.run_live_loop()

    print("\n" + "=" * 50)
    print("     OpenVLA Auto execution session completed.     ")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        errors_logger.error(f"Unhandled exception in launcher main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
