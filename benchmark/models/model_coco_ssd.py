"""
COCO-SSD benchmark adapter — native-tfjs-bench Model #4.

Architecture
------------
torchvision.models.detection.ssdlite320_mobilenet_v3_large with
SSDLite320_MobileNet_V3_Large_Weights.COCO_V1 pretrained weights.

The TF.js ``@tensorflow-models/coco-ssd`` defaults to SSD MobileNet V1
(300×300 input).  The closest available torchvision equivalent is SSDLite
with MobileNetV3-Large backbone (320×320 input).

Exactness: near_equivalent
---------------------------
Both are SSD-family object detectors on COCO (80/90 classes), but backbone
differs (MobileNetV1 vs MobileNetV3-Large) and input size differs (300 vs 320).
Inference-latency comparison is directionally valid.

Input contract
--------------
  Shape  : [1] list of (3, 320, 320) tensors — torchvision detection API format
  Dtype  : float32
  Values : [0.0, 1.0] (torchvision detection normalises internally)
  Device : CUDA tensor

Note: torchvision detection models expect List[Tensor] in forward().
The measured forward pass includes GPU backbone + SSD head + CPU NMS.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from benchmark.models.base import BaseModel, InputSpec

logger = logging.getLogger(__name__)


class CocoSsdModel(BaseModel):
    """
    Concrete adapter: SSDLite320 MobileNetV3-Large (torchvision) on CUDA.
    Near-equivalent to TF.js @tensorflow-models/coco-ssd.
    """

    paper_model_id: int = 4
    paper_model_name: str = "coco_ssd"
    task_type: str = "object_detection"
    paper_arch: str = "SSD MobileNet V1 / TensorFlow Object Detection API (90 COCO classes)"
    native_framework: str = "pytorch"
    native_model_name: str = "ssdlite320_mobilenet_v3_large (torchvision COCO_V1)"
    exactness_status: str = "near_equivalent"

    input_spec: InputSpec = InputSpec(
        shape=(1, 3, 320, 320),
        dtype="float32",
        layout="NCHW",
        value_range=(0.0, 1.0),
        description=(
            "torchvision detection format: List[Tensor(3, 320, 320)] float32 [0.0, 1.0]. "
            "SSDLite320 uses 320×320 input (vs original SSD MobileNetV1 300×300)."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    def load_model(self, device: str = "cuda") -> None:
        import torchvision.models.detection as tvd

        self._device = device
        logger.info("Loading SSDLite320 MobileNetV3-Large on %s …", device)
        weights = tvd.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self._model = tvd.ssdlite320_mobilenet_v3_large(weights=weights)
        self._model.to(device)
        self._model.eval()
        self._loaded = True
        logger.info("SSDLite320 MobileNetV3-Large loaded on %s", device)

    def make_dummy_input(self, seed: int = 12345) -> list:
        """Return a list containing one (3, 320, 320) CUDA float32 tensor in [0,1]."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        # Clamp to [0, 1] — detection models assert valid pixel range
        x = torch.rand(3, 320, 320, dtype=torch.float32, generator=gen)
        return [x.to(self._device, non_blocking=True)]

    def run_inference(self, inputs: list) -> list:
        with torch.no_grad():
            return self._model(inputs)

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        self._loaded = False

    def postprocess_optional(self, output: list) -> dict:
        n_boxes = len(output[0]["boxes"]) if output else 0
        return {"detections": n_boxes}
