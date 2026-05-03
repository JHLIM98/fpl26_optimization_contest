"""FPGA backend optimization package.

This package contains the implementation behind ``dcp_optimizer.py``:

- :mod:`optimizer.base` — shared MCP infrastructure and parsing helpers
- :mod:`optimizer.strategies` — deterministic strategy implementations
  (``run_deterministic_baseline``, ``try_pblock``, ``try_cell_replacement``,
  ``analyze_design_characteristics``)
- :mod:`optimizer.dispatcher` — heuristic dispatcher (static-analysis
  strategy picker + best-of selection); the alpha-submission default path
- :mod:`optimizer.llm_augment` — single-call OpenRouter helpers used by
  the dispatcher to propose alternative pblock_ranges when the auto
  attempt is weak
- :mod:`optimizer.cli` — argparse + ``main()`` entry point

The contest's original LLM-guided full optimizer has been relocated to
:mod:`reference.contest_llm_optimizer` because it is not part of the
alpha submission runtime. It remains reachable via ``--llm`` for study
or future experimentation.

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
