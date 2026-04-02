"""
MobileNetV3-Small benchmark adapter for native-tfjs-bench.

Architecture
------------
torchvision.models.mobilenet_v3_small with IMAGENET1K_V1 pretrained weights.
The model runs as a pure PyTorch CUDA inference path — no ONNX export or
TensorRT conversion at this stage.  This isolates GPU kernel execution time
from any runtime-compilation overhead and requires no file artefacts.

Exactness status: near_equivalent
----------------------------------
The TF.js ``@tensorflow-models/mobilenet`` package loads a TF Hub checkpoint
trained by Google.  torchvision ships a separately-trained PyTorch checkpoint
(IMAGENET1K_V1, ~67.7 % top-1 on ImageNet-1K validation set).

Inference-latency comparison is methodologically valid because both models
execute the same MobileNetV3-Small graph topology.  Accuracy comparison is
NOT valid (different training runs / data pipelines / checkpoints).

Input contract
--------------
  Shape  : (1, 3, 224, 224) — NCHW, batch = 1
  Dtype  : float32
  Values : ImageNet-normalised, approximately N(0, 1) per channel
               mean = [0.485, 0.456, 0.406]
               std  = [0.229, 0.224, 0.225]
  Device : CUDA tensor (pre-allocated once in make_dummy_input(); stays on GPU
           for the entire benchmark run — no host-to-device transfer inside
           the measured loop)

Timing discipline
-----------------
- ``eval()`` is called at load time: disables BatchNorm running-stat updates
  and Dropout, giving stable, reproducible throughput numbers.
- ``torch.no_grad()`` context is entered inside ``run_inference()`` to
  prevent autograd graph construction on the measured hot path.
- Do **NOT** call ``torch.cuda.synchronize()`` here; CudaEventTimer in
  ``runner.py`` owns that responsibility and must see the async submission
  / completion window accurately.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from benchmark.models.base import BaseModel, InputSpec

logger = logging.getLogger(__name__)

# ImageNet channel statistics — used for synthetic input generation comments;
# the actual dummy tensor is torch.randn which already matches this distribution.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class MobileNetV3Model(BaseModel):
    """
    Concrete adapter: MobileNetV3-Small (torchvision) on CUDA.

    Implements the BaseModel interface consumed by runner.py and the
    trial_manager subprocess orchestration.  All heavy state (the nn.Module)
    is stored in ``self._model``; it is None until ``load_model()`` is called
    and is released by ``cleanup()``.
    """

    # ── Class-level identity attributes (consumed by runner.py) ─────────────
    paper_model_id: int = 6
    paper_model_name: str = "mobilenetv3"
    task_type: str = "image_classification"
    paper_arch: str = "MobileNetV3-Small / ImageNet-1K (1000 classes, 224×224 input)"
    native_framework: str = "pytorch"
    native_model_name: str = "mobilenet_v3_small (torchvision IMAGENET1K_V1)"
    exactness_status: str = "near_equivalent"

    # Input contract: NCHW float32, ImageNet-normalised distribution
    input_spec: InputSpec = InputSpec(
        shape=(1, 3, 224, 224),
        dtype="float32",
        layout="NCHW",
        # Approximately ±3σ after ImageNet normalisation
        value_range=(-3.0, 3.0),
        description=(
            "NCHW float32. Synthetic inputs drawn from N(0, 1) per channel, "
            "which matches the post-normalisation distribution when using "
            "ImageNet statistics (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]). "
            "Torchvision IMAGENET1K_V1 checkpoint — not the TF Hub checkpoint."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None  # torch.nn.Module, set in load_model()

    # ──────────────────────────────────────────────────────────────────────────
    # Primary interface
    # ──────────────────────────────────────────────────────────────────────────

    def load_model(self, device: str = "cuda") -> None:
        """
        Download (on first call) and load MobileNetV3-Small onto *device*.

        Uses the torchvision IMAGENET1K_V1 pretrained weights enum (API
        available since torchvision 0.13; requirements.txt enforces ≥ 0.15).
        Sets the model to eval() before returning.

        Raises
        ------
        RuntimeError
            If CUDA is requested but unavailable, or if torchvision fails to
            download the weights (network access required on first run).
        """
        import torchvision.models as tvm  # import deferred to subprocess scope

        self._device = device
        logger.info(
            "Loading MobileNetV3-Small (torchvision IMAGENET1K_V1) on %s …", device
        )

        try:
            weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self._model = tvm.mobilenet_v3_small(weights=weights)
        except AttributeError:
            # Defensive fallback — should not reach here given torchvision ≥ 0.15
            logger.warning(
                "MobileNet_V3_Small_Weights not found in this torchvision version; "
                "falling back to pretrained=True (deprecated API)"
            )
            self._model = tvm.mobilenet_v3_small(pretrained=True)  # type: ignore[call-arg]

        self._model.to(device)
        self._model.eval()
        self._loaded = True
        logger.info("MobileNetV3-Small loaded and ready on %s", device)

    def make_dummy_input(self, seed: int = 12345) -> torch.Tensor:
        """
        Generate a deterministic ImageNet-normalised dummy tensor on the GPU.

        Shape  : (1, 3, 224, 224)
        Dtype  : float32
        Device : self._device (CUDA)

        The tensor is allocated once by runner.py before the benchmark loop
        and reused for all warm-up and measured iterations — benchmarks
        inference, not data-transfer latency.

        A CPU-side Generator is seeded on the host and the result is
        transferred to GPU once, keeping generation deterministic regardless
        of GPU state.
        """
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        x = torch.randn(1, 3, 224, 224, dtype=torch.float32, generator=gen)
        return x.to(self._device, non_blocking=True)

    def run_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Execute one MobileNetV3-Small forward pass.

        The call is wrapped in ``torch.no_grad()`` to suppress autograd
        overhead on the measured hot path.  Do NOT call synchronize() here;
        CudaEventTimer in runner.py handles that.

        Returns the raw (1, 1000) logit tensor (still on GPU).
        """
        with torch.no_grad():
            return self._model(inputs)

    def cleanup(self) -> None:
        """
        Release all GPU and host memory held by this model instance.

        Called once by runner.py after the measured phase finishes (or on any
        error exit path).  Safe to call even when load_model() was never
        invoked.
        """
        if self._model is not None:
            del self._model
            self._model = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        self._loaded = False
        logger.debug("MobileNetV3 cleanup complete")

    # ──────────────────────────────────────────────────────────────────────────
    # Optional override
    # ──────────────────────────────────────────────────────────────────────────

    def postprocess_optional(self, output: torch.Tensor) -> dict:
        """
        Validate output shape and extract the top-1 predicted class index.

        Called once during environment validation — never inside the timed loop.
        Asserts the expected (1, 1000) output shape so misconfigurations are
        caught early.
        """
        if output.shape != (1, 1000):
            raise ValueError(
                f"Unexpected MobileNetV3 output shape: {list(output.shape)} "
                f"(expected [1, 1000])"
            )
        top1_idx = int(output.argmax(dim=1).item())
        return {"top1_class_idx": top1_idx, "output_shape": list(output.shape)}
