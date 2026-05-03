"""FPGA backend optimization package.

This package contains the implementation behind ``dcp_optimizer.py``:

- :mod:`optimizer.base` — shared MCP infrastructure and parsing helpers
- :mod:`optimizer.llm_optimizer` — contest's original LLM-guided optimizer
- :mod:`optimizer.strategies` — deterministic strategy implementations
  (``run_deterministic_baseline``, ``try_pblock``, ``try_cell_replacement``,
  ``analyze_design_characteristics``)
- :mod:`optimizer.dispatcher` — the heuristic dispatcher + LLM-augmented
  pblock-ranges retry hook
- :mod:`optimizer.cli` — argparse + ``main()`` entry point

The top-level ``dcp_optimizer.py`` script is intentionally kept thin so
that the alpha-submission flow ``python3 dcp_optimizer.py <DCP>`` keeps
working unchanged.
"""

from .base import DCPOptimizerBase, DEFAULT_MODEL, parse_timing_summary_static

__all__ = [
    "DCPOptimizerBase",
    "DEFAULT_MODEL",
    "parse_timing_summary_static",
]
