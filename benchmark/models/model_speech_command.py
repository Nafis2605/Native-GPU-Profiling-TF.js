"""
Speech Command Recognizer benchmark adapter — native-tfjs-bench Model #3.

Architecture
------------
Depthwise-separable CNN operating on log-mel spectrograms.
Mirrors the topology of Google's TF.js speech-commands model (18-class
audio classification on STFT spectrogram input).

Since no pre-converted PyTorch checkpoint exists for the exact TF.js bundle,
the model is built from scratch with randomly-initialized weights using the
canonical depthwise-separable block structure.  For throughput benchmarking,
random weights produce identical runtime to pretrained weights — only the
arithmetic is relevant, not the parameter values.

Exactness: near_equivalent
---------------------------
Same depthwise-separable CNN topology on spectrogram input; different
checkpoint (randomly initialized vs. TF.js model trained on Speech Commands
v2 dataset).  Latency comparison is valid; accuracy comparison is NOT.

Input contract
--------------
  Shape  : (1, 1, 124, 129) — NCHW, 1 frequency channel, 124 time frames,
           129 frequency bins (log-mel spectrogram)
  Dtype  : float32
  Values : N(0, 1) — synthetic log-mel spectrogram approximation
  Device : CUDA tensor
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from benchmark.models.base import BaseModel, InputSpec

logger = logging.getLogger(__name__)

_NUM_CLASSES = 18


def _dw_sep_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Depthwise-separable convolution block (Conv DW → BN → ReLU → Conv PW → BN → ReLU)."""
    return nn.Sequential(
        # Depthwise
        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                  padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        # Pointwise
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _SpeechCNN(nn.Module):
    """
    Tiny depthwise-separable CNN matching the topology of Google's
    speech-commands model on 2-D log-mel spectrogram input.

    Input : (N, 1, 124, 129)
    Output: (N, 18)
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Stem: standard conv to lift channel count
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Depthwise-sep blocks
            _dw_sep_block(32, 64),
            _dw_sep_block(64, 128, stride=2),
            _dw_sep_block(128, 128),
            _dw_sep_block(128, 256, stride=2),
            _dw_sep_block(256, 256),
            _dw_sep_block(256, 512, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, _NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


class SpeechCommandModel(BaseModel):
    """
    Concrete adapter: depthwise-sep spectrogram CNN on CUDA.
    Near-equivalent to TF.js @tensorflow-models/speech-commands.
    """

    paper_model_id: int = 3
    paper_model_name: str = "speech_command_recognizer"
    task_type: str = "audio_classification"
    paper_arch: str = "Google / depthwise-separable CNN on STFT spectrogram (18 classes)"
    native_framework: str = "pytorch"
    native_model_name: str = "SpeechCNN (custom depthwise-sep, random init)"
    exactness_status: str = "near_equivalent"

    input_spec: InputSpec = InputSpec(
        shape=(1, 1, 124, 129),
        dtype="float32",
        layout="NCHW",
        value_range=(-3.0, 3.0),
        description=(
            "NCHW float32. [batch=1, channels=1, time_frames=124, freq_bins=129]. "
            "Synthetic N(0,1) inputs approximating log-mel spectrogram distribution."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    def load_model(self, device: str = "cuda") -> None:
        self._device = device
        logger.info("Building SpeechCNN (depthwise-sep, random init) on %s …", device)
        self._model = _SpeechCNN()
        self._model.to(device)
        self._model.eval()
        self._loaded = True
        logger.info("SpeechCNN ready on %s", device)

    def make_dummy_input(self, seed: int = 12345) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        x = torch.randn(1, 1, 124, 129, dtype=torch.float32, generator=gen)
        return x.to(self._device, non_blocking=True)

    def run_inference(self, inputs: torch.Tensor) -> torch.Tensor:
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

    def postprocess_optional(self, output: torch.Tensor) -> dict:
        if output.shape != (1, _NUM_CLASSES):
            raise ValueError(
                f"Unexpected SpeechCNN output shape: {list(output.shape)} "
                f"(expected [1, {_NUM_CLASSES}])"
            )
        top1_idx = int(output.argmax(dim=1).item())
        return {"top1_class_idx": top1_idx, "output_shape": list(output.shape)}
