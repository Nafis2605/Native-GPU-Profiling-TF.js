"""
Hand Pose 3D benchmark adapter — native-tfjs-bench Model #2.

Architecture
------------
MediaPipe HandLandmarker (Python SDK, hand_landmarker.task).

The TF.js ``@tensorflow-models/hand-pose-detection`` MediaPipe backend
and the Python mediapipe.tasks.vision.HandLandmarker load the SAME .task
bundle from storage.googleapis.com — weights are byte-for-byte identical.

Exactness: exact_official_equivalent
--------------------------------------
Both TF.js and Python mediapipe use the same underlying model bundle.

Runtime note
------------
The mediapipe Python SDK on Windows runs inference on CPU (the GPU delegate
is not available for the Desktop Python API).  Timing values therefore
reflect CPU latency, not CUDA kernel latency.  kernel_ms ≈ 0 for this
model (no CUDA kernels dispatched).  wall_clock_ms is the true end-to-end
CPU inference time.

Input contract
--------------
  Shape  : (224, 224, 3) — HWC uint8
  Values : RGB [0, 255]
  Device : CPU numpy array (mediapipe does not use CUDA tensors)
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
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_CACHE_DIR = Path(__file__).parent / "_mediapipe_assets"
_TASK_FILE = _CACHE_DIR / "hand_landmarker.task"


class HandPose3DModel(BaseModel):
    """
    Concrete adapter: MediaPipe HandLandmarker on CPU.
    Exact equivalent to TF.js @tensorflow-models/hand-pose-detection (MediaPipe backend).
    """

    paper_model_id: int = 2
    paper_model_name: str = "hand_pose_3d"
    task_type: str = "hand_landmark_detection"
    paper_arch: str = "MediaPipe BlazePalm + Hand Landmark (21 3D keypoints)"
    native_framework: str = "mediapipe"
    native_model_name: str = "hand_landmarker.task"
    exactness_status: str = "exact"

    input_spec: InputSpec = InputSpec(
        shape=(1, 224, 224, 3),
        dtype="uint8",
        layout="NHWC",
        value_range=(0.0, 255.0),
        description=(
            "RGB uint8 [0, 255] 224×224. "
            "Passed as (224, 224, 3) HWC numpy array to mediapipe.Image."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._detector: Any = None

    def load_model(self, device: str = "cuda") -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._device = device
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not _TASK_FILE.exists():
            logger.info("Downloading hand_landmarker.task from %s …", _MODEL_URL)
            urllib.request.urlretrieve(_MODEL_URL, _TASK_FILE)
            logger.info("hand_landmarker.task saved to %s", _TASK_FILE)
        else:
            logger.info("hand_landmarker.task already cached at %s", _TASK_FILE)

        base_opts = BaseOptions(model_asset_path=str(_TASK_FILE))
        options = vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=1,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._loaded = True
        logger.info("HandLandmarker loaded (CPU inference)")

    def make_dummy_input(self, seed: int = 12345) -> np.ndarray:
        """Return a deterministic (224, 224, 3) uint8 numpy array."""
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

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
        n_hands = len(output.hand_landmarks) if output.hand_landmarks else 0
        return {"hands_detected": n_hands}
