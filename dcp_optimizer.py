#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

"""
FPGA Design Optimization Agent — entry script.

This is a thin entry point so the alpha-submission flow
``python3 dcp_optimizer.py <DCP>`` and the corresponding
``make run_optimizer DCP=...`` target keep working unchanged. The actual
implementation lives in the :mod:`optimizer` package:

- :mod:`optimizer.base` — MCP infrastructure and timing/Fmax helpers
- :mod:`optimizer.llm_optimizer` — contest's original LLM-guided optimizer
  (only reachable via ``--llm``)
- :mod:`optimizer.strategies` — deterministic strategy implementations
  (``run_deterministic_baseline``, ``try_pblock``, ``try_cell_replacement``,
  ``analyze_design_characteristics``)
- :mod:`optimizer.dispatcher` — heuristic dispatcher + LLM-augmented
  pblock-ranges retry hook
- :mod:`optimizer.cli` — argparse + ``main()``
"""

import asyncio

from optimizer.cli import main


if __name__ == "__main__":
    asyncio.run(main())
