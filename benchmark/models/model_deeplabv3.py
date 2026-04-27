"""
DeepLabV3 benchmark adapter — native-tfjs-bench Model #10.

Architecture
------------
torchvision.models.segmentation.deeplabv3_mobilenet_v3_large with
DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 pretrained weights.

The TF.js ``@tensorflow-models/deeplab`` uses DeepLabV3+ with MobileNetV2
backbone (Pascal VOC, 21 classes).  The closest publicly-available torchvision
equivalent uses MobileNetV3-Large backbone with the same DeepLabV3 head and
the same 21-class Pascal VOC label set.

Exactness: near_equivalent
---------------------------
Same DeepLabV3 head and Pascal VOC 21-class output; backbone differs
(MobileNetV2 vs MobileNetV3-Large).  inference-latency comparison is
methodologically valid.

Input contract
--------------
  Shape  : (1, 3, 513, 513) — NCHW, batch = 1
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


class DeepLabV3Model(BaseModel):
    """
    Concrete adapter: DeepLabV3 MobileNetV3-Large (torchvision) on CUDA.
    Near-equivalent to TF.js @tensorflow-models/deeplab (DeepLabV3+ MobileNetV2).
    """

    paper_model_id: int = 10
    paper_model_name: str = "deeplabv3"
    task_type: str = "semantic_segmentation"
    paper_arch: str = (
        "DeepLabV3+ / MobileNetV2 backbone / Pascal VOC 21 classes "
        "(dilated convolutions + ASPP)"
    )
    native_framework: str = "pytorch"
    native_model_name: str = "deeplabv3_mobilenet_v3_large (torchvision COCO_WITH_VOC_LABELS_V1)"
    exactness_status: str = "near_equivalent"

    input_spec: InputSpec = InputSpec(
        shape=(1, 3, 513, 513),
        dtype="float32",
        layout="NCHW",
        value_range=(-3.0, 3.0),
        description=(
            "NCHW float32. Synthetic N(0,1) inputs matching ImageNet "
            "post-normalisation distribution. "
            "torchvision COCO_WITH_VOC_LABELS_V1 checkpoint (21 VOC classes)."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    def load_model(self, device: str = "cuda") -> None:
        import torchvision.models.segmentation as tvseg

        self._device = device
        logger.info("Loading DeepLabV3 MobileNetV3-Large on %s …", device)
        weights = tvseg.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        self._model = tvseg.deeplabv3_mobilenet_v3_large(weights=weights)
        self._model.to(device)
        self._model.eval()
        self._loaded = True
        logger.info("DeepLabV3 MobileNetV3-Large loaded on %s", device)

    def make_dummy_input(self, seed: int = 12345) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        x = torch.randn(1, 3, 513, 513, dtype=torch.float32, generator=gen)
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
