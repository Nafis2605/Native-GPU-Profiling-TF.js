"""
Abstract base class for all native benchmark models.

Interface summary
-----------------
Concrete model adapters must implement four abstract methods:

  load_model(device)     – Load weights; build inference session
  make_dummy_input(seed) – Generate a reproducible synthetic input tensor
  run_inference(inputs)  – Execute one forward pass; return raw output
  cleanup()              – Release all GPU / CPU memory

Optional override:

  postprocess_optional(output) – Validate/reshape output; default is identity

Class-level attributes (set on the subclass, not in __init__) declare the
identity and input contract for the model. They are consumed by runner.py
to populate TrialResult schema columns.

Exactness vocabulary
--------------------
  "exact"          – Identical weights and architecture to the TF.js paper
                     model, deployed via an official native SDK.
  "near_equivalent"– Same task and broadly the same architecture; pre-trained
                     weights differ (different training run, conversion
                     artefact, or official port) but results are directly
                     comparable for inference-latency benchmarking.
  "unresolved"     – No clear faithful native equivalent has been identified
                     or verified. Benchmarking proceeds but the comparison
                     to the paper model is not methodologically sound until
                     this status is resolved.

Backward compatibility
----------------------
The three primary abstract methods are also accessible under their legacy
names used by runner.py:

  load()           → load_model()
  generate_input() → make_dummy_input()
  infer()          → run_inference()

This allows runner.py to remain unchanged during the Phase 1B→1C transition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


# ── InputSpec ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InputSpec:
    """
    Full specification of a model's expected input tensor.

    Use this instead of the bare (input_shape, input_dtype) pair when you
    need layout or value-range information for correct synthetic-input
    generation.

    Fields
    ------
    shape        : Tensor shape, e.g. (1, 3, 224, 224).
    dtype        : NumPy dtype string: "float32" | "uint8" | "int64".
    layout       : Axis labelling: "NCHW" | "NHWC" | "NL" | "NLHW" etc.
    value_range  : (min, max) of valid input values used for synthetic data
                   generation.  Correct normalisation is model-specific:
                   float32 models often need [0.0, 1.0] or ImageNet-normalised
                   values; uint8 models expect [0, 255].
    description  : Human-readable note on preprocessing, e.g. "ImageNet mean/std".
    """
    shape: tuple[int, ...]
    dtype: str
    layout: str
    value_range: tuple[float, float]
    description: str = ""

    # Convenience accessors matching the flat attributes runner.py expects
    @property
    def shape_list(self) -> list[int]:
        return list(self.shape)


# ── BaseModel ─────────────────────────────────────────────────────────────────

class BaseModel(ABC):
    """
    Abstract interface for a single benchmarked inference model.

    Class-level identity attributes
    --------------------------------
    paper_model_id   : int    Paper model number (1–10).
    paper_model_name : str    Human-readable name from the paper.
    task_type        : str    Inference task, e.g. "image_classification".
    paper_arch       : str    Architecture / source in the paper.
    native_framework : str    Runtime for native inference.
    native_model_name: str    Artefact filename or SDK bundle name.
    exactness_status : str    "exact" | "near_equivalent" | "unresolved".
    input_spec       : InputSpec  Full input contract for this model.

    Backward-compat flat attributes (derived from input_spec)
    ----------------------------------------------------------
    input_shape : list[int]   Shortcut for input_spec.shape_list.
    input_dtype : str         Shortcut for input_spec.dtype.
    """

    # ── Identity (override in every concrete subclass) ────────────────────
    paper_model_id: int = 0
    paper_model_name: str = "unnamed"
    task_type: str = ""
    paper_arch: str = ""
    native_framework: str = ""
    native_model_name: str = ""
    # "exact" | "near_equivalent" | "unresolved"
    exactness_status: str = "unresolved"

    # Input contract — set to an InputSpec instance in concrete subclasses
    input_spec: InputSpec = InputSpec(
        shape=(1,), dtype="float32", layout="N",
        value_range=(0.0, 1.0), description="unset",
    )

    # ── Backward-compat flat attributes (aliases of input_spec fields) ────
    # These are kept so runner.py and result_schema code can still do
    #   meta["input_shape"], meta["input_dtype"]
    # without changes during the Phase 1B→1C transition.
    @property
    def input_shape(self) -> list[int]:
        return self.input_spec.shape_list

    @property
    def input_dtype(self) -> str:
        return self.input_spec.dtype

    # Alias kept for registry _StubModel which sets these as instance attrs
    @property
    def model_id(self) -> int:
        return self.paper_model_id

    @property
    def model_name(self) -> str:
        return self.paper_model_name

    def __init__(self) -> None:
        self._loaded: bool = False
        self._device: str = "cuda"

    # ------------------------------------------------------------------
    # Primary abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self, device: str = "cuda") -> None:
        """
        Load model weights and initialise the inference session.

        Contract
        --------
        - Set ``self._loaded = True`` on success.
        - Raise ``RuntimeError`` if the model artefact is missing or the
          framework fails to compile the graph.  runner.py will record the
          trial as STATUS_FAILED and continue.
        - Raise ``NotImplementedError`` if this class is a stub (Phase 1B).
          runner.py will record the trial as STATUS_UNSUPPORTED and continue
          without crashing.
        - Do NOT start CUDA streams, telemetry, or timers here; runner.py
          owns those resources.

        Args:
            device: CUDA device string, e.g. "cuda" or "cuda:0".
        """
        ...

    @abstractmethod
    def make_dummy_input(self, seed: int = 12345) -> Any:
        """
        Generate a deterministic synthetic input tensor for benchmarking.

        Contract
        --------
        - Shape and dtype must match ``self.input_spec``.
        - Must NOT read from disk (avoids I/O latency confounding GPU timing).
        - Must be deterministic: the same ``seed`` must produce the same
          tensor across calls and across fresh subprocess invocations.
        - Return format must be directly passable to ``run_inference()``.

        Typical pattern for ONNX Runtime models:
            rng = np.random.default_rng(seed)
            return rng.uniform(*self.input_spec.value_range,
                                size=self.input_spec.shape).astype(self.input_spec.dtype)

        Args:
            seed: Integer seed for reproducible generation (default 12345).
        """
        ...

    @abstractmethod
    def run_inference(self, inputs: Any) -> Any:
        """
        Execute a single inference forward pass.

        Contract
        --------
        - Must be side-effect–free (no external writes, no state mutation).
        - Must accept the return value of ``make_dummy_input()`` directly.
        - Should NOT call ``torch.cuda.synchronize()`` internally; the caller
          (runner.py + CudaEventTimer) manages synchronisation so that
          kernel time is measured correctly.
        - Return the raw model output; postprocess_optional() is called
          separately for output validation.

        Args:
            inputs: Return value of make_dummy_input().

        Returns:
            Raw model output (shape/dtype used for post-hoc validation only).
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """
        Release all GPU and CPU memory held by this model instance.

        Contract
        --------
        - Must set ``self._loaded = False``.
        - Called once after the measured phase completes (or on error exit).
        - Must be idempotent: safe to call even if load_model() was never
          invoked (e.g. in the stub path).
        """
        ...

    # ------------------------------------------------------------------
    # Optional override
    # ------------------------------------------------------------------

    def postprocess_optional(self, output: Any) -> Any:
        """
        Optional output post-processing for shape and semantic validation.

        Default implementation is identity (returns output unchanged).
        Override in concrete subclasses to:
          - Reshape raw output tensors into labelled structures.
          - Assert expected output shape and dtype.
          - Apply framework-specific decode steps (e.g. NMS decode, softmax).

        This method is never called inside the timed measurement loop.
        It is invoked once during environment validation to assert that
        the model produces a sensible output before benchmarking begins.

        Args:
            output: Return value of run_inference().

        Returns:
            Processed output (or the original output when no processing is needed).
        """
        return output

    # ------------------------------------------------------------------
    # Backward-compat aliases (runner.py uses these names)
    # ------------------------------------------------------------------

    def load(self, device: str = "cuda") -> None:
        """Alias for load_model(). Kept for runner.py compatibility."""
        return self.load_model(device=device)

    def generate_input(self, seed: int = 12345) -> Any:
        """Alias for make_dummy_input(). Kept for runner.py compatibility."""
        return self.make_dummy_input(seed=seed)

    def infer(self, inputs: Any) -> Any:
        """Alias for run_inference(). Kept for runner.py compatibility."""
        return self.run_inference(inputs)

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """True after a successful load_model() call."""
        return self._loaded

    def get_metadata(self) -> dict:
        """
        Return all model identity fields as a plain dict.

        Consumed by runner.py to populate TrialResult identity columns.
        Includes both the structured InputSpec and flat compat fields.
        """
        return {
            # New canonical names
            "paper_model_id":    self.paper_model_id,
            "paper_model_name":  self.paper_model_name,
            "task_type":         self.task_type,
            # Legacy names (same values) expected by TrialResult schema
            "model_id":          self.paper_model_id,
            "model_name":        self.paper_model_name,
            "paper_arch":        self.paper_arch,
            "native_framework":  self.native_framework,
            "native_model_name": self.native_model_name,
            "exactness_status":  self.exactness_status,
            # Flat input attrs (TrialResult schema columns)
            "input_shape":       self.input_spec.shape_list,
            "input_dtype":       self.input_spec.dtype,
            # Structured input contract
            "input_spec": {
                "shape":       self.input_spec.shape,
                "dtype":       self.input_spec.dtype,
                "layout":      self.input_spec.layout,
                "value_range": self.input_spec.value_range,
                "description": self.input_spec.description,
            },
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.paper_model_id}, "
            f"name={self.paper_model_name!r}, "
            f"status={self.exactness_status!r}, "
            f"framework={self.native_framework!r}, "
            f"loaded={self._loaded})"
        )

    # ------------------------------------------------------------------
    # Profiling hint (optional override)
    # ------------------------------------------------------------------

    def get_profiling_hint(self) -> Optional["ProfilingHint"]:
        """
        Return a ProfilingHint that guides Nsight Compute kernel-level profiling.

        The default implementation returns None (no hint), meaning the profiler
        script falls back to its own defaults (launch_skip=0, launch_count=10).

        Concrete subclasses MAY override this to provide:
          - A regex that matches representative kernel names for this model,
            allowing ``ncu --kernel-regex`` to filter profiling to those kernels.
          - A launch_skip count to skip driver/framework initialization kernels
            and land the profiling window on steady-state inference kernels.
          - A launch_count appropriate for the model's kernel launch pattern.

        This method is called in the *parent* process (the profiler script),
        NOT inside the benchmarked subprocess, so it must not perform any
        GPU operations or require the model to be loaded.
        """
        return None


# ── ProfilingHint ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProfilingHint:
    """
    Guidance for kernel-level profiling with Nsight Compute.

    Returned by BaseModel.get_profiling_hint().  The profiler script
    (profile_with_ncu.py) uses these hints to configure the ncu launch window,
    keeping overhead tractable by restricting profiling to representative
    steady-state kernels rather than the entire run.

    Fields
    ------
    representative_kernel_regex : str | None
        Python-compatible regex passed to ncu --kernel-regex.
        Restricts profiling to kernel names matching this pattern.
        None = profile all observed kernels (may be slow for large models).
    launch_skip : int
        Number of kernel launches to skip before profiling begins.
        Use this to skip CUDA driver / framework initialisation kernels
        and warmup kernels that are unrepresentative of steady-state latency.
    launch_count : int
        Number of kernel launches to profile after the skip window.
        A value of 5–20 is usually sufficient to characterise a model.
    notes : str
        Human-readable explanation of the hint rationale.
    """
    representative_kernel_regex: Optional[str] = None
    launch_skip: int = 0
    launch_count: int = 10
    notes: str = ""


# ── DummyCudaModel ────────────────────────────────────────────────────────────

class DummyCudaModel(BaseModel):
    """
    Concrete placeholder model for profiling pipeline validation.

    Runs repeated SGEMM (1024×1024 matrix multiply) on CUDA without loading
    any pre-trained weights.  Useful for:
      - Verifying the Nsight Compute integration end-to-end.
      - Confirming that ncu captures cuBLAS GEMM kernels correctly.
      - Smoke-testing ``profile_with_ncu.py --use-dummy`` without any model artefacts.

    **NOT a benchmark model** — results have no scientific meaning and
    must not appear in benchmark reports.  ``paper_model_id = 0`` marks
    this model as outside the registered benchmark suite.

    Requirements
    ------------
    PyTorch must be importable.  If not, load_model raises NotImplementedError
    so runner.py records STATUS_UNSUPPORTED and continues gracefully.
    """

    paper_model_id: int = 0
    paper_model_name: str = "dummy_cuda_matmul"
    task_type: str = "synthetic"
    paper_arch: str = "N/A"
    native_framework: str = "pytorch"
    native_model_name: str = "N/A"
    exactness_status: str = "N/A"

    input_spec: InputSpec = InputSpec(
        shape=(1024, 1024),
        dtype="float32",
        layout="NC",
        value_range=(-1.0, 1.0),
        description="Random 1024×1024 float32 matrix for synthetic SGEMM.",
    )

    _MATRIX_DIM: int = 1024

    def load_model(self, device: str = "cuda") -> None:
        try:
            import torch  # type: ignore[import]
        except ImportError as exc:
            raise NotImplementedError(
                "DummyCudaModel requires PyTorch (pip install torch)"
            ) from exc
        self._device = device
        self._weight = torch.randn(
            self._MATRIX_DIM, self._MATRIX_DIM,
            device=device, dtype=torch.float32,
        )
        self._loaded = True

    def make_dummy_input(self, seed: int = 12345) -> Any:
        try:
            import torch  # type: ignore[import]
            gen = torch.Generator()
            gen.manual_seed(seed)
            return torch.randn(
                self._MATRIX_DIM, self._MATRIX_DIM,
                generator=gen,
                dtype=torch.float32,
                device=getattr(self, "_device", "cpu"),
            )
        except ImportError:
            import numpy as np  # type: ignore[import]
            rng = np.random.default_rng(seed)
            return rng.standard_normal(
                (self._MATRIX_DIM, self._MATRIX_DIM)
            ).astype("float32")

    def run_inference(self, inputs: Any) -> Any:
        import torch  # type: ignore[import]
        return torch.mm(inputs, self._weight)

    def cleanup(self) -> None:
        if hasattr(self, "_weight"):
            del self._weight
        try:
            import torch  # type: ignore[import]
            torch.cuda.empty_cache()
        except Exception:
            pass
        self._loaded = False

    def get_profiling_hint(self) -> ProfilingHint:
        return ProfilingHint(
            representative_kernel_regex=(
                r"gemm|sgemm|ampere_s|volta_s|turing_s|cutlass"
            ),
            launch_skip=5,
            launch_count=20,
            notes=(
                "1024×1024 SGEMM repeated matrix multiplication. "
                "Primary kernel: cuBLAS SGEMM (e.g. ampere_sgemm_128x64_nt). "
                "Skip first 5 launches to skip CUDA/cuBLAS initialisation."
            ),
        )
