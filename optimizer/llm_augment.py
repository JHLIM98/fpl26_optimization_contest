#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

"""
LLM augment for the deterministic dispatcher (single targeted call).

This module is the *only* place in the alpha-submission runtime that
talks to OpenRouter. The dispatcher in :mod:`optimizer.dispatcher` calls
``propose_pblock_ranges`` at most once per benchmark when its
auto-derived pblock attempt comes back weak. The response is a
``SLICE_X..Y..:SLICE_X..Y..`` string that can be fed straight back into
``FPGAOptimizerTest.try_pblock(pblock_ranges=...)`` for a retry.

Cost: ~$0.001 / call on the default ``x-ai/grok-4.1-fast`` model — well
under the contest's $1/benchmark cap.

Public API:
    parse_pblock_ranges_from_llm(text)   -> Optional[str]
    propose_pblock_ranges(api_key, model, context) -> Optional[str]
"""

import logging
import re
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


# Compiled regex for extracting pblock ranges from LLM output. Module-private.
_PBLOCK_RANGE_RE = re.compile(
    r"SLICE_X(\d+)Y(\d+)\s*:\s*SLICE_X(\d+)Y(\d+)",
    re.IGNORECASE,
)


def parse_pblock_ranges_from_llm(text: str) -> Optional[str]:
    """Pull a normalized SLICE_X..Y..:SLICE_X..Y.. (optionally comma-joined)
    string out of an LLM response. Returns None if no valid range found.
    Tolerates code fences, prose, multiple regions, and lower/upper case.
    """
    if not text:
        return None
    matches = _PBLOCK_RANGE_RE.findall(text)
    if not matches:
        return None
    parts = []
    for x1, y1, x2, y2 in matches:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Normalize so the first corner is lower-left.
        x_lo, x_hi = sorted((x1, x2))
        y_lo, y_hi = sorted((y1, y2))
        if x_lo == x_hi or y_lo == y_hi:
            # Degenerate (zero-area) region — reject.
            continue
        parts.append(f"SLICE_X{x_lo}Y{y_lo}:SLICE_X{x_hi}Y{y_hi}")
    if not parts:
        return None
    return ", ".join(parts)


# Module-private: the system prompt embeds the experimental hint
# (logicnets hardcoded ranges +110 MHz vs auto +20 MHz) that motivated
# this hook in the first place.
_PBLOCK_LLM_SYSTEM_PROMPT = """\
You are an FPGA backend placement specialist. Your only job is to propose a
single pblock ranges string that can be passed to Vivado's
create_pblock + add_cells_to_pblock + resize_pblock flow to constrain a
design to a smaller region for better timing.

OUTPUT REQUIREMENTS — VERY IMPORTANT:
- Reply with ONLY a pblock ranges string. No prose, no markdown.
- Format: SLICE_X<col_min>Y<row_min>:SLICE_X<col_max>Y<row_max>
- Multiple regions allowed, comma-separated. Use only SLICE columns
  (no DSP48/RAMB columns — Vivado will auto-include those inside the
  bounding box).

PLACEMENT PRINCIPLES:
- Smaller pblock => more compact placement => shorter wire delay => better Fmax.
- Too small => routing failure or timing regression. Aim for ~1.3-1.8x of
  the area implied by current LUT/FF utilization.
- Aspect ratio matters. Tall narrow regions help register-heavy designs;
  wider regions help DSP-cascading designs.

KNOWN GOOD PRECEDENT (logicnets_jscl_2025.1, ~60K LUT, ~30K FF):
- RapidWright auto-derived ranges via center-of-mass yielded only +20 MHz.
- A hand-tuned region 'SLICE_X55Y60:SLICE_X111Y254' (~57 cols x 195 rows,
  shifted away from the corner the auto heuristic picked) yielded +110 MHz.
- This shows the auto heuristic often picks a region that is the right SIZE
  but the wrong POSITION/SHAPE. Adjust position/aspect ratio first.

Respond with the ranges string only.
"""


def propose_pblock_ranges(
    api_key: str,
    model: str,
    context: dict,
    timeout: float = 60.0,
    max_output_tokens: int = 200,
) -> Optional[str]:
    """One-shot OpenRouter call to propose alternative pblock_ranges.

    Returns a normalized ranges string (caller-validated by the regex parser),
    or None on any failure. The caller MUST be prepared to fall back to the
    auto-derived ranges. Cost cap is enforced by max_output_tokens + a tight
    user prompt; on default Grok this works out to ~$0.001 per call.
    """
    if OpenAI is None:
        logger.warning("openai package missing; cannot propose pblock_ranges via LLM")
        return None
    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    except Exception as e:
        logger.warning(f"OpenAI client init failed: {e}")
        return None

    user_prompt = (
        "Design characteristics:\n"
        f"  Used LUT:  {context.get('used_lut')}\n"
        f"  Used FF:   {context.get('used_ff')}\n"
        f"  Used DSP:  {context.get('used_dsp')}\n"
        f"  Used BRAM: {context.get('used_bram')}\n"
        f"  Critical-path spread (max/avg cell-tile distance): "
        f"{context.get('max_spread_distance')}/{context.get('avg_spread_distance')}\n"
        f"  Initial WNS:    {context.get('initial_wns')} ns\n"
        f"  Clock period:   {context.get('clock_period')} ns "
        f"(target Fmax {context.get('target_fmax')} MHz)\n"
        "\nAuto-derived attempt (RapidWright analyze_fabric_for_pblock):\n"
        f"  Ranges:          {context.get('auto_ranges')}\n"
        f"  Result Fmax:     {context.get('auto_final_fmax')} MHz\n"
        f"  Improvement:     {context.get('auto_delta_fmax')} MHz\n"
        f"  Routing errors:  {context.get('auto_routing_errors')}\n"
        "\nThe auto attempt was weak. Propose alternative pblock_ranges that "
        "may give better Fmax. Reply with the ranges string only."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _PBLOCK_LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_output_tokens,
            timeout=timeout,
            temperature=0.2,
        )
    except Exception as e:
        logger.warning(f"LLM pblock proposal call failed: {e}")
        return None

    raw = ""
    try:
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"LLM response parse failed: {e}")
        return None

    parsed = parse_pblock_ranges_from_llm(raw)
    logger.info(
        f"LLM pblock proposal: raw={raw!r}\n  parsed={parsed!r}"
    )
    return parsed
