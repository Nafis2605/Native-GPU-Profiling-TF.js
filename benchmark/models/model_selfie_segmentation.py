"""
Selfie Segmentation benchmark adapter — native-tfjs-bench Model #1.

Architecture
------------
torchvision.models.segmentation.lraspp_mobilenet_v3_large with
COCO_WITH_VOC_LABELS_V1 pretrained weights.

LRASPP (Lite R-ASPP) with a MobileNetV3-Large backbone is the closest
publicly-available torchvision model to the TF.js ``selfie_segmentation``
model (which uses a custom Google MobileNetV3 segmentation backbone).

Exactness: near_equivalent
---------------------------
The TF.js model outputs a binary foreground/background mask (1-channel).
LRASPP outputs 21 Pascal VOC class logits.  Both use a MobileNetV3 backbone
for semantic pixel-level classification.  Inference-latency comparison is
methodologically valid; output semantics are different.

Input contract
--------------
  Shape  : (1, 3, 256, 256) — NCHW, batch = 1
  Dtype  : float32
  Values : ImageNet-normalised N(0, 1) per channel
  Device : CUDA tensor
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from benchmark.models.base import BaseModel, InputSpec

logger = logging.getLogger(__name__)


class SelfieSegmentationModel(BaseModel):
    """
    Concrete adapter: LRASPP MobileNetV3-Large (torchvision) on CUDA.
    Near-equivalent to the TF.js selfie_segmentation (MobileNetV3 backbone).
    """

    paper_model_id: int = 1
    paper_model_name: str = "selfie_segmentation"
    task_type: str = "semantic_segmentation"
    paper_arch: str = "Google ML Kit / custom MobileNetV3 segmentation backbone"
    native_framework: str = "pytorch"
    native_model_name: str = "lraspp_mobilenet_v3_large (torchvision COCO_WITH_VOC_LABELS_V1)"
    exactness_status: str = "near_equivalent"

    input_spec: InputSpec = InputSpec(
        shape=(1, 3, 256, 256),
        dtype="float32",
        layout="NCHW",
        value_range=(-3.0, 3.0),
        description=(
            "NCHW float32. Synthetic N(0,1) inputs matching ImageNet "
            "post-normalisation distribution. "
            "torchvision COCO_WITH_VOC_LABELS_V1 checkpoint."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    def load_model(self, device: str = "cuda") -> None:
        import torchvision.models.segmentation as tvseg

        self._device = device
        logger.info("Loading LRASPP MobileNetV3-Large on %s …", device)
        weights = tvseg.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        self._model = tvseg.lraspp_mobilenet_v3_large(weights=weights)
        self._model.to(device)
        self._model.eval()
        self._loaded = True
        logger.info("LRASPP MobileNetV3-Large loaded on %s", device)

    def make_dummy_input(self, seed: int = 12345) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        x = torch.randn(1, 3, 256, 256, dtype=torch.float32, generator=gen)
        return x.to(self._device, non_blocking=True)

    def run_inference(self, inputs: torch.Tensor) -> Any:
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

    def postprocess_optional(self, output: Any) -> dict:
        out_tensor = output["out"] if isinstance(output, dict) else output
        return {"output_shape": list(out_tensor.shape)}
