"""
PoseNet benchmark adapter — native-tfjs-bench Model #9.

Architecture
------------
MediaPipe PoseLandmarker (Python SDK, pose_landmarker_lite.task).

The TF.js ``@tensorflow-models/pose-detection`` (BlazePose backend) and the
Python mediapipe.tasks.vision.PoseLandmarker load the same BlazePose .task
bundle from storage.googleapis.com.

Exactness: near_equivalent
---------------------------
If the paper tested the BlazePose / MoveNet backend in TF.js, the mediapipe
mapping is near-exact (33 landmarks, BlazePose backbone).  If the paper used
the legacy PoseNet (MobileNetV1, 17 COCO keypoints) backend, architectural
mismatch exists.  Benchmark proceeds under "near_equivalent" pending
clarification.

Runtime note
------------
The mediapipe Python SDK on Windows runs inference on CPU (the GPU delegate
is not available for the Desktop Python API).  wall_clock_ms captures true
CPU latency.  kernel_ms ≈ 0 (no CUDA kernels dispatched).

Input contract
--------------
  Shape  : (256, 256, 3) — HWC uint8
  Values : RGB [0, 255]
  Device : CPU numpy array
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.models.base import BaseModel, InputSpec

logger = logging.getLogger(__name__)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_CACHE_DIR = Path(__file__).parent / "_mediapipe_assets"
_TASK_FILE = _CACHE_DIR / "pose_landmarker_lite.task"


class PoseNetModel(BaseModel):
    """
    Concrete adapter: MediaPipe PoseLandmarker-Lite on CPU.
    Near-equivalent to TF.js @tensorflow-models/pose-detection (BlazePose backend).
    """

    paper_model_id: int = 9
    paper_model_name: str = "posenet"
    task_type: str = "pose_estimation"
    paper_arch: str = (
        "Google PoseNet — MobileNet V1, 17 COCO keypoints, 2D heatmap "
        "(legacy; TF.js has since migrated to BlazePose / MoveNet)"
    )
    native_framework: str = "mediapipe"
    native_model_name: str = "pose_landmarker_lite.task"
    exactness_status: str = "near_equivalent"

    input_spec: InputSpec = InputSpec(
        shape=(1, 256, 256, 3),
        dtype="uint8",
        layout="NHWC",
        value_range=(0.0, 255.0),
        description=(
            "RGB uint8 [0, 255] 256×256. "
            "Passed as (256, 256, 3) HWC numpy array to mediapipe.Image."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._detector: Any = None

    def load_model(self, device: str = "cuda") -> None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._device = device
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not _TASK_FILE.exists():
            logger.info("Downloading pose_landmarker_lite.task from %s …", _MODEL_URL)
            urllib.request.urlretrieve(_MODEL_URL, _TASK_FILE)
            logger.info("pose_landmarker_lite.task saved to %s", _TASK_FILE)
        else:
            logger.info("pose_landmarker_lite.task already cached at %s", _TASK_FILE)

        base_opts = BaseOptions(model_asset_path=str(_TASK_FILE))
        options = vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)
        self._loaded = True
        logger.info("PoseLandmarker-Lite loaded (CPU inference)")

    def make_dummy_input(self, seed: int = 12345) -> np.ndarray:
        """Return a deterministic (256, 256, 3) uint8 numpy array."""
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

    def run_inference(self, inputs: np.ndarray) -> Any:
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=inputs)
        return self._detector.detect(mp_image)

    def cleanup(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
        self._loaded = False

    def postprocess_optional(self, output: Any) -> dict:
        n_poses = len(output.pose_landmarks) if output.pose_landmarks else 0
        return {"poses_detected": n_poses}
