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
from benchmark.models.model_mobilenetv3 import MobileNetV3Model  # Phase 1C: first real impl


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
#
# can_benchmark   : True when the model COULD be benchmarked after obtaining
#                   the ONNX/task artefact.
# benchmark_blocker: One-line description of what prevents benchmarking now.

MODEL_REGISTRY: dict[int, Callable[[], BaseModel]] = {

    # ── 1: Selfie Segmentation ───────────────────────────────────────────────
    # TF.js model: @tensorflow-models/body-segmentation / selfie_segmentation
    # Backing model: Google MediaPipe Selfie Segmentation (TFLite-based).
    # Conversion path: tensorflowjs_converter → TFLite → tf2onnx → ONNX.
    # Mismatches: TF.js bilateral filtering post-process not in exported graph.
    1: _stub_entry(
        model_id=1,
        model_name="selfie_segmentation",
        task_type="semantic_segmentation",
        paper_arch="Google ML Kit / custom MobileNetV3 segmentation backbone",
        native_framework="onnxruntime-gpu",
        native_model_name="selfie_segmentation.onnx",
        exactness_status="near_equivalent",
        input_spec=InputSpec(
            shape=(1, 256, 256, 3),
            dtype="float32",
            layout="NHWC",
            value_range=(0.0, 1.0),
            description="RGB image [0.0, 1.0], 256×256",
        ),
        can_benchmark=True,
        benchmark_blocker=(
            "Requires tf2onnx conversion of TF.js selfie_segmentation TFLite model. "
            "python -m tf2onnx.convert --tflite selfie_segmentation.tflite "
            "--output selfie_segmentation.onnx"
        ),
    ),

    # ── 2: Hand Pose 3D ──────────────────────────────────────────────────────
    # TF.js model: @tensorflow-models/hand-pose-detection (MediaPipe backend).
    # Native equivalent: mediapipe.tasks.vision.HandLandmarker.
    # Exactness: TF.js MediaPipe backend and Python SDK load the SAME .task
    #   bundle from storage.googleapis.com — weights are byte-for-byte identical.
    2: _stub_entry(
        model_id=2,
        model_name="hand_pose_3d",
        task_type="hand_landmark_detection",
        paper_arch="MediaPipe BlazePalm + Hand Landmark (21 3D keypoints)",
        native_framework="mediapipe",
        native_model_name="hand_landmarker.task",
        exactness_status="exact",
        input_spec=InputSpec(
            shape=(1, 224, 224, 3),
            dtype="float32",
            layout="NHWC",
            value_range=(0.0, 1.0),
            description=(
                "RGB [0.0, 1.0]; resized to 224×224. "
                "MediaPipe SDK handles normalisation internally."
            ),
        ),
        can_benchmark=True,
        benchmark_blocker=(
            "Download hand_landmarker.task from "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        ),
    ),

    # ── 3: Speech Command Recognizer ─────────────────────────────────────────
    # TF.js model: @tensorflow-models/speech-commands
    # Backing model: Google 18-class CNN on STFT log-mel spectrograms.
    # Conversion path: TF.js package → TFLite → tf2onnx → ONNX.
    # Mismatches: TF.js preprocessing (FFT, mel filterbank) is external;
    #   benchmark feeds synthetic spectrogram tensors directly.
    3: _stub_entry(
        model_id=3,
        model_name="speech_command_recognizer",
        task_type="audio_classification",
        paper_arch="Google / depthwise-separable CNN on STFT spectrogram (18 classes)",
        native_framework="onnxruntime-gpu",
        native_model_name="speech_commands.onnx",
        exactness_status="near_equivalent",
        input_spec=InputSpec(
            shape=(1, 1, 124, 129),
            dtype="float32",
            layout="NCHW",
            value_range=(-3.0, 3.0),
            description=(
                "[batch, 1, time_frames=124, freq_bins=129] log-mel spectrogram. "
                "Synthetic inputs sample from N(0,1). Real audio requires "
                "external STFT + log-mel filterbank preprocessing."
            ),
        ),
        can_benchmark=True,
        benchmark_blocker=(
            "Extract speech_commands TFLite model from TF.js npm package, "
            "then: python -m tf2onnx.convert --saved-model <dir> "
            "--output speech_commands.onnx"
        ),
    ),

    # ── 4: COCO-SSD ──────────────────────────────────────────────────────────
    # TF.js model: @tensorflow-models/coco-ssd
    # Backing model: SSD MobileNet V1 from TF Object Detection API.
    # TF.js defaults to V1 (faster); match this in the native benchmark.
    # Mismatches: (a) NMS post-processing must survive tf2onnx (use --opset 13).
    #             (b) Input dtype may change uint8→float32 during conversion.
    4: _stub_entry(
        model_id=4,
        model_name="coco_ssd",
        task_type="object_detection",
        paper_arch="SSD MobileNet V1 / TensorFlow Object Detection API (90 COCO classes)",
        native_framework="onnxruntime-gpu",
        native_model_name="ssd_mobilenet_v1_coco.onnx",
        exactness_status="near_equivalent",
        input_spec=InputSpec(
            shape=(1, 300, 300, 3),
            dtype="uint8",
            layout="NHWC",
            value_range=(0.0, 255.0),
            description=(
                "RGB uint8 [0, 255] 300×300. "
                "Verify input dtype after tf2onnx — some graph transforms "
                "change uint8 to float32 at the input node."
            ),
        ),
        can_benchmark=True,
        benchmark_blocker=(
            "Download SSD MobileNet V1 COCO from TF Model Zoo, then: "
            "python -m tf2onnx.convert --saved-model <dir> "
            "--opset 13 --output ssd_mobilenet_v1_coco.onnx"
        ),
    ),

    # ── 5: MobileBERT ────────────────────────────────────────────────────────
    # TF.js model: @tensorflow-models/bert-qa (MobileBERT extractive QA).
    # ONNX export: official HuggingFace ONNX export of google/mobilebert-uncased.
    # Mismatches: (a) Benchmark feeds synthetic int64 token IDs, not real text.
    #             (b) TF.js wrapper includes tokenizer; we benchmark model only.
    5: _stub_entry(
        model_id=5,
        model_name="mobilebert",
        task_type="text_embedding",
        paper_arch="Google MobileBERT-uncased (24 blocks × 128-dim, SQuAD 1.1 fine-tuned)",
        native_framework="onnxruntime-gpu",
        native_model_name="mobilebert.onnx",
        exactness_status="near_equivalent",
        input_spec=InputSpec(
            shape=(1, 128),
            dtype="int64",
            layout="NL",
            value_range=(0.0, 30522.0),
            description=(
                "[batch=1, seq_len=128] int64 token IDs in [0, vocab_size=30522). "
                "Expects 3 inputs: input_ids, attention_mask, token_type_ids. "
                "Synthetic: input_ids=randint(0,30522); "
                "attention_mask=ones; token_type_ids=zeros."
            ),
        ),
        can_benchmark=True,
        benchmark_blocker=(
            "Download from HuggingFace Optimum: "
            "from optimum.onnxruntime import ORTModelForQuestionAnswering; "
            "model = ORTModelForQuestionAnswering.from_pretrained("
            "'google/mobilebert-uncased', export=True)"
        ),
    ),

    # ── 6: MobileNetV3 ───────────────────────────────────────────────────────
    # TF.js model: @tensorflow-models/mobilenet (MobileNetV3-Small default).
    # Native impl: torchvision.models.mobilenet_v3_small (IMAGENET1K_V1 weights).
    # Exactness: near_equivalent — same topology, independently trained checkpoint.
    # See benchmark/models/model_mobilenetv3.py for full implementation notes.
    6: MobileNetV3Model,

    # ── 7: AR PortraitDepth ──────────────────────────────────────────────────
    # TF.js model: @tensorflow-models/depth-estimation (ARPortraitDepth backend).
    # UNRESOLVED: MediaPipe Python API (as of 2026-03) has NO depth-estimation
    #   task.  Available MediaPipe vision tasks: ObjectDetector, ImageClassifier,
    #   ImageSegmenter, FaceDetector, FaceLandmarker, HandLandmarker,
    #   PoseLandmarker, GestureRecognizer, ImageEmbedder, InteractiveSegmenter.
    #   Depth estimation is NOT in this list.
    # The portrait_depth.task bundle claimed in earlier scaffold does not exist
    #   as a documented MediaPipe Python API task.
    # Conversion path exists but is unverified: extract GraphModel JSON from
    #   npm package → tf2onnx → ONNX.
    # ⚠ DO NOT substitute MediaPipe ImageSegmenter — different task entirely.
    7: _stub_entry(
        model_id=7,
        model_name="ar_portrait_depth",
        task_type="monocular_depth_estimation",
        paper_arch="Google AR PortraitDepth — custom monocular depth CNN (DepthLab team)",
        native_framework="onnxruntime-gpu",
        native_model_name="ar_portrait_depth.onnx",
        exactness_status="unresolved",
        input_spec=InputSpec(
            shape=(1, 256, 256, 3),
            dtype="float32",
            layout="NHWC",
            value_range=(0.0, 1.0),
            description=(
                "Tentative: 256×256 RGB portrait image [0.0, 1.0]. "
                "Input shape must be verified from the TF.js GraphModel JSON. "
                "Output: per-pixel disparity map (float32)."
            ),
        ),
        can_benchmark=False,
        benchmark_blocker=(
            "UNRESOLVED — No public native equivalent verified. "
            "MediaPipe has no depth-estimation API. "
            "Required: extract GraphModel from @tensorflow-models/depth-estimation, "
            "convert via tf2onnx, verify I/O shapes, run parity check. "
            "DO NOT substitute MediaPipe segmentation models."
        ),
    ),

    # ── 8: BodyPix ───────────────────────────────────────────────────────────
    # TF.js model: @tensorflow-models/body-segmentation (BodyPix backend).
    # UNRESOLVED: BodyPix was deprecated by Google in 2022. No official ONNX
    #   or maintained TFLite release exists.
    # Rejected mapping: the prior scaffold proposed MediaPipe Selfie Segmentation
    #   as "near_equivalent".  This is INVALID:
    #     Selfie Segmentation = binary foreground/background (1-channel mask)
    #     BodyPix             = 24-part body-part segmentation (24-class labels)
    #   These are different tasks with different outputs.  Substitution would
    #   invalidate any comparison to the paper.
    # Conversion path: TFLite model extractable from npm package via
    #   tf2onnx, but 24-part output correctness has not been verified.
    # ⚠ DO NOT substitute MediaPipe Selfie Segmentation.
    8: _stub_entry(
        model_id=8,
        model_name="bodypix",
        task_type="body_part_segmentation",
        paper_arch=(
            "Google BodyPix — MobileNet v1 backbone, 24-part body segmentation "
            "(deprecated 2022; used in benchmarking paper)"
        ),
        native_framework="onnxruntime-gpu",
        native_model_name="bodypix_mobilenet.onnx",
        exactness_status="unresolved",
        input_spec=InputSpec(
            shape=(1, 3, 513, 513),
            dtype="float32",
            layout="NCHW",
            value_range=(0.0, 1.0),
            description=(
                "Tentative: 513×513 RGB [0.0, 1.0]. "
                "Original TF.js model used NHWC; verify layout after tf2onnx. "
                "Output: [1, H, W] int32 part labels (0=background, 1–24=body parts)."
            ),
        ),
        can_benchmark=False,
        benchmark_blocker=(
            "UNRESOLVED — No verified ONNX export of BodyPix exists. "
            "Required: extract TFLite from @tensorflow-models/body-segmentation npm, "
            "convert via tf2onnx, validate all 24 output part labels. "
            "DO NOT substitute MediaPipe Selfie Segmentation (different task)."
        ),
    ),

    # ── 9: PoseNet ───────────────────────────────────────────────────────────
    # TF.js model: @tensorflow-models/pose-detection (PoseNet backend, legacy).
    # Mapping: MediaPipe PoseLandmarker (Python) = official BlazePose successor.
    # Mismatches:
    #   (a) Keypoints: PoseNet = 17 COCO keypoints (2D heatmap output);
    #       MediaPipe Pose = 33 landmarks (3D + visibility).
    #   (b) Backbone: PoseNet = MobileNet V1; MediaPipe = BlazePose.
    #   (c) Output format: completely different.
    # ⚠ Clarify from paper which TF.js backend was tested: if the paper used
    #   the legacy PoseNet (MobileNet) backend, this mapping has architectural
    #   mismatches; if the paper used BlazePose backend, MediaPipe is exact.
    9: _stub_entry(
        model_id=9,
        model_name="posenet",
        task_type="pose_estimation",
        paper_arch=(
            "Google PoseNet — MobileNet V1, 17 COCO keypoints, 2D heatmap "
            "(legacy; TF.js has since migrated to BlazePose / MoveNet)"
        ),
        native_framework="mediapipe",
        native_model_name="pose_landmarker_lite.task",
        exactness_status="near_equivalent",
        input_spec=InputSpec(
            shape=(1, 256, 256, 3),
            dtype="float32",
            layout="NHWC",
            value_range=(0.0, 1.0),
            description=(
                "256×256 RGB [0.0, 1.0] for MediaPipe PoseLandmarker Lite. "
                "Note: original PoseNet used 257×257 or 353×353; "
                "MediaPipe Pose accepts arbitrary resolution."
            ),
        ),
        can_benchmark=True,
        benchmark_blocker=(
            "ARCHITECTURE MISMATCH: PoseNet (17 kp, MobileNet) vs MediaPipe "
            "PoseLandmarker (33 landmarks, BlazePose). Proceed only if paper "
            "tested BlazePose backend. "
            "Download: https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        ),
    ),

    # ── 10: DeepLabV3 ────────────────────────────────────────────────────────
    # TF.js model: @tensorflow-models/deeplab (DeepLabV3+ MobileNetV2 default).
    # TF Model Zoo: DeepLabV3+ MobileNetV2 SavedModel publicly available.
    # Conversion: tf2onnx conversion of this SavedModel is well-tested.
    # Mismatches: (a) BN folding may differ; run parity check.
    #             (b) Dilated convolution implementation may affect throughput.
    #             (c) Verify Pascal VOC 21-class label order is preserved.
    10: _stub_entry(
        model_id=10,
        model_name="deeplabv3",
        task_type="semantic_segmentation",
        paper_arch=(
            "DeepLabV3+ / MobileNetV2 backbone / Pascal VOC 21 classes "
            "(dilated convolutions + ASPP)"
        ),
        native_framework="onnxruntime-gpu",
        native_model_name="deeplabv3_mobilenet_v2.onnx",
        exactness_status="near_equivalent",
        input_spec=InputSpec(
            shape=(1, 3, 513, 513),
            dtype="float32",
            layout="NCHW",
            value_range=(-1.0, 1.0),
            description=(
                "513×513 RGB normalised to [−1.0, 1.0] (TF DeepLab style). "
                "Output: [1, 21, 513, 513] or [1, 513, 513] depending on "
                "whether the argmax op is included in the exported graph."
            ),
        ),
        can_benchmark=True,
        benchmark_blocker=(
            "Download DeepLabV3+ MobileNetV2 from TF Model Zoo, then: "
            "python -m tf2onnx.convert --saved-model deeplabv3_mnv2_saved_model "
            "--opset 13 --output deeplabv3_mobilenet_v2.onnx"
        ),
    ),
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
