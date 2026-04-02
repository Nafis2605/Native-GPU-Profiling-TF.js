"""
Profiler backends for native-tfjs-bench.

Layers
------
  base        – RunMode constants and ProfilerBase ABC
  nsys_runner – Nsight Systems CLI runner (subprocess wrapper)
  nsys_parser – Nsight Systems report parser (nsys stats + SQLite)
  ncu_runner  – Nsight Compute CLI runner (subprocess wrapper + metric categories)
  ncu_parser  – Nsight Compute CSV parser

Import convenience re-exports:
"""

from benchmark.profilers.base import ProfilerBase, RunMode
from benchmark.profilers.nsys_runner import NsysRunConfig, NsysRunResult, NsysRunner
from benchmark.profilers.nsys_parser import NsysParser, NsysSummary
from benchmark.profilers.ncu_runner import (
    MetricCategory,
    NcuRunConfig,
    NcuRunResult,
    NcuRunner,
    categorize_metric,
    DEFAULT_METRICS,
    KERNEL_DURATION_METRICS,
    OCCUPANCY_METRICS,
    MEMORY_THROUGHPUT_METRICS,
    CACHE_METRICS,
    SM_EFFICIENCY_METRICS,
    TENSOR_CORE_METRICS,
    WARP_SCHEDULER_METRICS,
)
from benchmark.profilers.ncu_parser import (
    NcuMetricValue,
    NcuKernelMetrics,
    NcuProfilingResult,
    NcuParser,
)

__all__ = [
    # Base
    "RunMode",
    "ProfilerBase",
    # nsys
    "NsysRunner",
    "NsysRunConfig",
    "NsysRunResult",
    "NsysParser",
    "NsysSummary",
    # ncu
    "MetricCategory",
    "NcuRunner",
    "NcuRunConfig",
    "NcuRunResult",
    "NcuParser",
    "NcuMetricValue",
    "NcuKernelMetrics",
    "NcuProfilingResult",
    "categorize_metric",
    "DEFAULT_METRICS",
    "KERNEL_DURATION_METRICS",
    "OCCUPANCY_METRICS",
    "MEMORY_THROUGHPUT_METRICS",
    "CACHE_METRICS",
    "SM_EFFICIENCY_METRICS",
    "TENSOR_CORE_METRICS",
    "WARP_SCHEDULER_METRICS",
]

