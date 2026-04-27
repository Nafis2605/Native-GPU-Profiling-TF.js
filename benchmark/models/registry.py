"""
Model registry for native-tfjs-bench.

MODEL_REGISTRY maps paper_model_id (int 1–10) to a no-argument factory
callable that returns a BaseModel subclass instance.

Phase 1B state: all entries are _StubModel instances that raise
NotImplementedError on load_model(). runner.py intercepts this and records
the trial as STATUS_UNSUPPORTED, then continues without crashing.

Phase 1C task: for each model, create benchmark/models/model_<name>.py,
subclass BaseModel with a real implementation, then replace the corresponding
_stub_entry(...) call below with a direct reference to the concrete class.

Exactness vocabulary (see base.py for full definitions)
-------------------------------------------------------
  "exact"           – Identical weights + architecture via official SDK.
  "near_equivalent" – Same task; architecture and weights differ to some
                      degree but inference latency comparison is valid.
  "unresolved"      – No verified faithful native equivalent yet.
                      Benchmarking MUST NOT be used to compare against the
                      paper until this status is upgraded.

Audit findings (updated 2026-03-31)
------------------------------------
  UNRESOLVED models (do not compare to paper without further work):
    #7  AR PortraitDepth — MediaPipe has no depth-estimation Python API.
    #8  BodyPix          — TF.js BodyPix deprecated; proposed MediaPipe
                           fallback changes task semantics (foreground
                           segmentation ≠ 24-part body segmentation).

  See docs/model_mapping_audit.md for per-model rationale and mismatch log.

Replacement pattern (Phase 1C)
-------------------------------
  # Before (stub):
  6: _stub_entry(model_id=6, model_name="mobilenetv3", ...),

  # After (real implementation):
  from benchmark.models.model_mobilenetv3 import MobileNetV3Model
  6: MobileNetV3Model,
"""

from __future__ import annotations

from typing import Callable

from benchmark.models.base import BaseModel, InputSpec
from benchmark.models.model_mobilenetv3 import MobileNetV3Model
from benchmark.models.model_selfie_segmentation import SelfieSegmentationModel
from benchmark.models.model_hand_pose_3d import HandPose3DModel
from benchmark.models.model_speech_command import SpeechCommandModel
from benchmark.models.model_coco_ssd import CocoSsdModel
from benchmark.models.model_mobilebert import MobileBertModel
from benchmark.models.model_posenet import PoseNetModel
from benchmark.models.model_deeplabv3 import DeepLabV3Model


# ── Stub implementation ───────────────────────────────────────────────────────

class _StubModel(BaseModel):
    """
    Placeholder used for any model not yet implemented in Phase 1B.

    load_model() raises NotImplementedError so runner.py records the trial
    as STATUS_UNSUPPORTED and continues to the next model without crashing.
    All identity and input-spec metadata is fully populated so that list_models()
    and the CLI can display accurate information even before real adapters exist.
    """

    def __init__(
        self,
        paper_model_id: int,
        paper_model_name: str,
        task_type: str,
        paper_arch: str,
        native_framework: str,
        native_model_name: str,
        exactness_status: str,
        input_spec: InputSpec,
        can_benchmark: bool,
        benchmark_blocker: str,
    ) -> None:
        super().__init__()
        self.paper_model_id = paper_model_id
        self.paper_model_name = paper_model_name
        self.task_type = task_type
        self.paper_arch = paper_arch
        self.native_framework = native_framework
        self.native_model_name = native_model_name
        self.exactness_status = exactness_status
        self.input_spec = input_spec
        self._can_benchmark = can_benchmark
        self._benchmark_blocker = benchmark_blocker

    def load_model(self, device: str = "cuda") -> None:
        raise NotImplementedError(
            f"Model '{self.paper_model_name}' (id={self.paper_model_id}) "
            "is not yet implemented (Phase 1B stub). "
            f"Blocker: {self._benchmark_blocker}. "
            "See docs/model_mapping_audit.md for implementation guidance."
        )

    def make_dummy_input(self, seed: int = 12345):
        raise NotImplementedError(
            f"Model '{self.paper_model_name}' is not yet implemented."
        )

    def run_inference(self, inputs):
        raise NotImplementedError(
            f"Model '{self.paper_model_name}' is not yet implemented."
        )

    def cleanup(self) -> None:
        self._loaded = False

    def get_metadata(self) -> dict:
        meta = super().get_metadata()
        meta["can_benchmark"] = self._can_benchmark
        meta["benchmark_blocker"] = self._benchmark_blocker
        return meta


# ── Factory helper ────────────────────────────────────────────────────────────

def _stub_entry(
    *,
    model_id: int,
    model_name: str,
    task_type: str,
    paper_arch: str,
    native_framework: str,
    native_model_name: str,
    exactness_status: str,
    input_spec: InputSpec,
    can_benchmark: bool,
    benchmark_blocker: str,
) -> Callable[[], BaseModel]:
    """
    Return a no-arg factory callable that creates the described _StubModel.

    Using a factory (rather than storing the instance directly) lets runner.py
    instantiate a fresh object for each trial without holding a reference to a
    previously-loaded model across subprocess invocations.
    """
    def factory() -> BaseModel:
        return _StubModel(
            paper_model_id=model_id,
            paper_model_name=model_name,
            task_type=task_type,
            paper_arch=paper_arch,
            native_framework=native_framework,
            native_model_name=native_model_name,
            exactness_status=exactness_status,
            input_spec=input_spec,
            can_benchmark=can_benchmark,
            benchmark_blocker=benchmark_blocker,
        )
    factory.__name__ = model_name
    return factory


# ── Registry ──────────────────────────────────────────────────────────────────
# Keys are paper model IDs 1–10.  Values are no-arg callables → BaseModel.
# NOTE: Models 7 (ar_portrait_depth) and 8 (bodypix) are DISABLED via
#       experiment_manifest.yaml (enabled: false) due to unresolved equivalents.
#       Benchmark runs on 8 models: 1-6, 9-10.
#
# can_benchmark   : True when the model COULD be benchmarked after obtaining
#                   the ONNX/task artefact.
# benchmark_blocker: One-line description of what prevents benchmarking now.

MODEL_REGISTRY: dict[int, Callable[[], BaseModel]] = {

    # ── 1: Selfie Segmentation ───────────────────────────────────────────────
    # Phase 1C: LRASPP MobileNetV3-Large (torchvision) — near-equivalent.
    # MobileNetV3 segmentation backbone; 21 VOC classes vs binary selfie mask.
    1: SelfieSegmentationModel,

    # ── 2: Hand Pose 3D ──────────────────────────────────────────────────────
    # Phase 1C: MediaPipe HandLandmarker — exact (same .task bundle as TF.js).
    # Runs on CPU on Windows; wall_clock_ms reflects CPU inference latency.
    2: HandPose3DModel,

    # ── 3: Speech Command Recognizer ─────────────────────────────────────────
    # Phase 1C: Custom depthwise-sep CNN (random init) on (1,1,124,129) spectrogram.
    # Architecture is near-equivalent to Google's TF.js speech-commands model.
    3: SpeechCommandModel,

    # ── 4: COCO-SSD ──────────────────────────────────────────────────────────
    # Phase 1C: SSDLite320 MobileNetV3-Large (torchvision COCO_V1) — near-equiv.
    # MobileNetV3-Large backbone vs original MobileNetV1; 320 vs 300 input.
    4: CocoSsdModel,

    # ── 5: MobileBERT ────────────────────────────────────────────────────────
    # Phase 1C: google/mobilebert-uncased (HuggingFace transformers) — near-equiv.
    # Same MobileBERT encoder; no SQuAD QA head; synthetic token sequences.
    5: MobileBertModel,

    # ── 6: MobileNetV3 ───────────────────────────────────────────────────────
    # TF.js model: @tensorflow-models/mobilenet (MobileNetV3-Small default).
    # Native impl: torchvision.models.mobilenet_v3_small (IMAGENET1K_V1 weights).
    # Exactness: near_equivalent — same topology, independently trained checkpoint.
    # See benchmark/models/model_mobilenetv3.py for full implementation notes.
    6: MobileNetV3Model,

    # ── 7: AR PortraitDepth ──────────────────────────────────────────────────
    # TF.js model: @tensorflow-models/depth-estimation (ARPortraitDepth backend).
    # UNRESOLVED: No exact PyTorch equivalent available in standard libraries.
    #
    # Why:
    #   • torchvision: has NO depth estimation models
    #   • MediaPipe: has NO depth-estimation Python task
    #   • MiDaS v3: Exact same task (monocular depth estimation) but requires
    #     'timm' package; dependency installation fails in this environment
    #   • Monodepth2, PackNet: Research models, not in standard PyTorch libraries
    #
    # Path forward: Fix MiDaS timm dependency OR accept custom CNN.
    7: _stub_entry(
        model_id=7,
        model_name="ar_portrait_depth",
        task_type="monocular_depth_estimation",
        paper_arch="Google AR PortraitDepth — custom monocular depth CNN (DepthLab team)",
        native_framework="pytorch",
        native_model_name="(no standard equivalent)",
        exactness_status="unresolved",
        input_spec=InputSpec(
            shape=(1, 3, 256, 256),
            dtype="float32",
            layout="NCHW",
            value_range=(-3.0, 3.0),
            description=(
                "256×256 RGB portrait image, ImageNet-normalized. "
                "Output: per-pixel depth map (float32)."
            ),
        ),
        can_benchmark=False,
        benchmark_blocker=(
            "UNRESOLVED — No exact PyTorch equivalent in standard libraries. "
            "MiDaS v3 (exact task) requires 'timm' — dependency fails. "
            "Alternative: use custom CNN or fix MiDaS installation."
        ),
    ),

    # ── 8: BodyPix ───────────────────────────────────────────────────────────
    # TF.js model: @tensorflow-models/body-segmentation (BodyPix backend).
    # UNRESOLVED: No exact PyTorch equivalent available.
    #
    # Why:
    #   • torchvision: has NO 24-part body segmentation models
    #   • MediaPipe Selfie Segmentation: binary foreground/background (1 class)
    #     vs BodyPix 24-part labels — DIFFERENT TASK, not a valid substitute
    #   • No other public PyTorch library provides 24-part body segmentation
    #
    # Path forward: Extract & convert BodyPix ONNX from npm package,
    # OR accept that no standard equivalent exists.
    8: _stub_entry(
        model_id=8,
        model_name="bodypix",
        task_type="body_part_segmentation",
        paper_arch=(
            "Google BodyPix — MobileNet v1 backbone, 24-part body segmentation "
            "(deprecated 2022; used in benchmarking paper)"
        ),
        native_framework="pytorch",
        native_model_name="(no standard equivalent)",
        exactness_status="unresolved",
        input_spec=InputSpec(
            shape=(1, 3, 513, 513),
            dtype="float32",
            layout="NCHW",
            value_range=(0.0, 1.0),
            description=(
                "513×513 RGB image [0, 1]. "
                "Output: [1, H, W] int32 part labels (0=background, 1–24=body parts)."
            ),
        ),
        can_benchmark=False,
        benchmark_blocker=(
            "UNRESOLVED — No 24-part body segmentation in standard PyTorch libraries. "
            "BodyPix is deprecated (2022). MediaPipe Selfie Segmentation is binary "
            "foreground/background (different task). "
            "Alternative: extract & convert BodyPix from npm ONNX."
        ),
    ),

    # ── 9: PoseNet ───────────────────────────────────────────────────────────
    # Phase 1C: MediaPipe PoseLandmarker-Lite — near-equivalent (BlazePose).
    # Runs on CPU on Windows; wall_clock_ms reflects CPU inference latency.
    9: PoseNetModel,

    # ── 10: DeepLabV3 ────────────────────────────────────────────────────────
    # Phase 1C: DeepLabV3 MobileNetV3-Large (torchvision) — near-equivalent.
    # MobileNetV3-Large backbone vs original MobileNetV2; same 21 VOC classes.
    10: DeepLabV3Model,
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_model(model_id: int) -> BaseModel:
    """
    Instantiate a registered model by its paper model ID.

    Returns an unloaded instance; call .load_model() (or legacy .load())
    before starting the benchmark.

    Raises:
        KeyError: model_id not in MODEL_REGISTRY.
    """
    if model_id not in MODEL_REGISTRY:
        raise KeyError(
            f"Model ID {model_id} is not registered. "
            f"Valid IDs: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_id]()


def list_models() -> list[dict]:
    """Return metadata dicts for all registered models, sorted by paper_model_id."""
    return [get_model(mid).get_metadata() for mid in sorted(MODEL_REGISTRY)]


def list_unresolved() -> list[dict]:
    """Return metadata dicts for models whose exactness_status is 'unresolved'."""
    return [m for m in list_models() if m["exactness_status"] == "unresolved"]


def list_benchmarkable() -> list[dict]:
    """Return metadata dicts for models that can proceed to benchmarking."""
    return [m for m in list_models() if m.get("can_benchmark", False)]


def get_enabled_model_ids(manifest_models: list[dict]) -> list[int]:
    """
    Filter a manifest model list to IDs that are both enabled and registered.

    Args:
        manifest_models: List of model dicts from experiment_manifest.yaml.

    Returns:
        Sorted list of model IDs.
    """
    registered = set(MODEL_REGISTRY.keys())
    return sorted(
        m["model_id"]
        for m in manifest_models
        if m.get("enabled", True) and m["model_id"] in registered
    )


# Map of lowercase model name → model_id for name-based CLI lookup.
# Populated lazily from MODEL_REGISTRY so it stays in sync with the registry.
_NAME_TO_ID: dict[str, int] = {}


def _build_name_map() -> None:
    global _NAME_TO_ID
    if _NAME_TO_ID:
        return
    for mid in MODEL_REGISTRY:
        try:
            meta = get_model(mid).get_metadata()
            _NAME_TO_ID[meta["model_name"].lower()] = mid
        except Exception:
            pass


def get_model_id_by_name(name: str) -> int:
    """
    Resolve a model name string to its registry model ID.

    Name matching is case-insensitive.  Raises ``KeyError`` if the name is
    not found.

    Args:
        name: Model name, e.g. ``"mobilenetv3"``.

    Returns:
        Integer model ID.

    Raises:
        KeyError: The name does not match any registered model.
    """
    _build_name_map()
    key = name.lower()
    if key not in _NAME_TO_ID:
        known = sorted(_NAME_TO_ID.keys())
        raise KeyError(
            f"Unknown model name {name!r}. "
            f"Registered names: {known}"
        )
    return _NAME_TO_ID[key]
