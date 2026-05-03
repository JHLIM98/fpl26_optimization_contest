#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

"""
Heuristic strategy dispatcher (alpha-submission default path).

This module decides *which* strategy to run based on static analysis of
the input DCP, then runs it through :mod:`optimizer.strategies` and
keeps the best result. The flow is:

  1. ``analyze_design_characteristics`` — sample utilization, spread,
     fanout candidates.
  2. ``_dispatcher_pick_primary`` — heuristic chooses ``fanout`` or
     ``pblock``.
  3. Run primary; for ``pblock`` keep auto-derived ranges and stats so
     the optional LLM augment can decide whether to retry.
  4. Optional LLM retry (delegated to :mod:`optimizer.llm_augment`) if
     the auto pblock attempt looks weak — single OpenRouter call.
  5. Best-of {auto, llm} selection. If everything regresses vs initial,
     fall back to ``RuntimeOptimized`` phys_opt.
  6. Always emit a final DCP plus ``.edf`` for the validator.

LLM-related code lives entirely in :mod:`optimizer.llm_augment`; this
module only invokes its public helpers.
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from .base import DEFAULT_MODEL
from .llm_augment import parse_pblock_ranges_from_llm, propose_pblock_ranges
from .strategies import FPGAOptimizerTest

logger = logging.getLogger(__name__)


def _dispatcher_pick_primary(
    info: dict,
    small_lut_threshold: int,
    dsp_heavy_threshold: int = 20,
    bram_heavy_threshold: int = 10,
) -> str:
    """Choose the primary deterministic strategy from analyze() output.

    Matrix sweep showed pblock wins on 10/12 benchmarks; the two exceptions
    (vexriscv ~2K LUT, 3d-rendering ~14K LUT) respond better to
    AggressiveFanoutOpt-flavored phys_opt than to area constraint. Within the
    small-LUT band the dispatcher first checks for DSP- or BRAM-heavy designs
    (e.g. amd_mini-isp: 3181 LUT but 40 DSPs / 12 BRAMs) where pblock wins
    despite the small LUT count.
    """
    used_lut  = int(info.get("used_lut")  or 0)
    used_dsp  = int(info.get("used_dsp")  or 0)
    used_bram = int(info.get("used_bram") or 0)
    fanout_cands = int(info.get("fanout_candidate_count") or 0)

    dsp_or_bram_heavy = (
        used_dsp >= dsp_heavy_threshold or used_bram >= bram_heavy_threshold
    )

    if 0 < used_lut < small_lut_threshold and not dsp_or_bram_heavy:
        return "fanout"
    # Catch fanout-dominated designs that happen to be a bit larger than
    # the small-LUT threshold but still respond best to phys_opt fanout
    # rebalancing (heuristic; pblock will run as the fallback regardless).
    if fanout_cands >= 5 and used_lut < 2 * small_lut_threshold and not dsp_or_bram_heavy:
        return "fanout"
    return "pblock"


async def _dispatcher_run_strategy(
    tester: "FPGAOptimizerTest",
    strategy: str,
    input_dcp: Path,
    output_dcp: Path,
    util_factor: float,
    phys_opt_directive: str,
) -> dict:
    """Run one dispatcher candidate strategy and normalize the result dict."""
    if strategy == "pblock":
        stats = await tester.try_pblock(
            input_dcp, output_dcp, util_factor=util_factor
        )
        return {
            "strategy": "pblock",
            "success": bool(stats.get("success")) and output_dcp.exists(),
            "final_wns": stats.get("final_wns"),
            "notes": list(stats.get("notes") or []),
        }
    if strategy == "fanout":
        ok = await tester.run_deterministic_baseline(
            input_dcp,
            output_dcp,
            max_nets_to_optimize=2,
            phys_opt_directive=phys_opt_directive,
        )
        return {
            "strategy": f"fanout({phys_opt_directive})",
            "success": bool(ok) and output_dcp.exists(),
            "final_wns": tester.final_wns,
            "notes": [],
        }
    if strategy == "bl_runtime":
        ok = await tester.run_deterministic_baseline(
            input_dcp,
            output_dcp,
            max_nets_to_optimize=2,
            phys_opt_directive="RuntimeOptimized",
        )
        return {
            "strategy": "bl_runtime",
            "success": bool(ok) and output_dcp.exists(),
            "final_wns": tester.final_wns,
            "notes": [],
        }
    return {"strategy": strategy, "success": False, "final_wns": None, "notes": [f"Unknown strategy: {strategy}"]}


async def _dispatcher_emit_passthrough(
    tester: "FPGAOptimizerTest",
    input_dcp: Path,
    output_dcp: Path,
) -> None:
    """Copy the original DCP to output_dcp and emit a matching .edf so the
    validator's RapidWright phase 1 read works."""
    shutil.copyfile(input_dcp, output_dcp)
    try:
        await tester.call_vivado_tool("open_checkpoint", {
            "dcp_path": str(output_dcp.resolve())
        }, timeout=600.0)
        await tester.call_vivado_tool("write_edif", {
            "edif_path": str(output_dcp.with_suffix(".edf").resolve()),
            "force": True,
        }, timeout=600.0)
    except Exception as e:
        logger.warning(f"Passthrough EDIF write failed: {e}")
        print(f"[DISPATCH] EDIF write skipped: {e}")


async def run_dispatcher_mode(
    input_dcp: Path,
    output_dcp: Path,
    debug: bool = False,
    run_dir: Optional[Path] = None,
    util_factor: float = 1.5,
    small_lut_threshold: int = 5000,
    fanout_phys_opt_directive: str = "AggressiveFanoutOpt",
    llm_api_key: Optional[str] = None,
    llm_model: str = DEFAULT_MODEL,
    llm_budget: int = 0,
    llm_weak_threshold_mhz: float = 25.0,
    llm_time_cap_seconds: int = 1800,
    llm_primary_runtime_cap_seconds: int = 1080,
    llm_mock_ranges: Optional[str] = None,
):
    """Deterministic dispatcher: analyze the design, pick a primary strategy,
    fall back to RuntimeOptimized phys_opt if the primary regresses, and
    always emit a final DCP (+ .edf for the validator).

    Optional LLM augment: when ``llm_budget > 0`` and ``llm_api_key`` is set,
    a single targeted LLM call is made to propose alternative pblock_ranges
    *only* if the auto-derived pblock attempt was weak (delta_fmax under
    ``llm_weak_threshold_mhz``, or routing errors, or regression). Time
    budget guards prevent the retry from pushing the run past the contest's
    1-hour wall-clock cap.
    """
    tester = FPGAOptimizerTest(debug=debug, run_dir=run_dir)
    overall_start = time.time()
    llm_calls_used = 0
    # Mock mode bypasses the real API but still consumes budget so the same
    # weak-result trigger logic is exercised end-to-end.
    llm_calls_budget = (
        max(1, int(llm_budget) or 1) if llm_mock_ranges
        else (max(0, int(llm_budget)) if llm_api_key else 0)
    )

    def _emit_summary(selected_label: str, final_wns: Optional[float]) -> None:
        elapsed = time.time() - overall_start
        tester.print_test_summary(
            title="TEST SUMMARY - DISPATCHER",
            elapsed_seconds=elapsed,
            initial_wns=tester.initial_wns,
            final_wns=final_wns,
            clock_period=tester.clock_period,
            extra_info=(
                f"Strategy: {selected_label}\n"
                f"LLM calls: {llm_calls_used}/{llm_calls_budget}"
            ),
        )

    try:
        await tester.start_servers()

        print("\n" + "=" * 70)
        print("DISPATCHER MODE - design analysis")
        print("=" * 70)
        info = await tester.analyze_design_characteristics(input_dcp)
        for note in info["notes"]:
            print(f"[DISPATCH] {note}")

        initial_wns = info.get("initial_wns")
        if initial_wns is None:
            print("[DISPATCH] Could not measure initial WNS; emitting original DCP unchanged")
            await _dispatcher_emit_passthrough(tester, input_dcp, output_dcp)
            _emit_summary("passthrough (no initial WNS)", None)
            return 1

        if initial_wns >= 0:
            print(f"[DISPATCH] Initial WNS={initial_wns:.3f} >= 0 (timing met); emitting original DCP unchanged")
            await _dispatcher_emit_passthrough(tester, input_dcp, output_dcp)
            _emit_summary("passthrough (timing met)", initial_wns)
            return 0

        primary = _dispatcher_pick_primary(info, small_lut_threshold=small_lut_threshold)
        print(
            f"[DISPATCH] Primary strategy: {primary} "
            f"(LUT={info.get('used_lut')}, fanout_cands={info.get('fanout_candidate_count')}, "
            f"avg_spread={info.get('avg_spread_distance')})"
        )

        candidates: list[dict] = []

        # ---- Run primary strategy ----
        # Special-case pblock so we keep the underlying try_pblock stats
        # (auto-derived ranges, routing_errors). This lets the LLM retry hook
        # decide whether to attempt a single proposal call.
        primary_step_start = time.time()
        if primary == "pblock":
            print("\n" + "-" * 60)
            print("DISPATCH STEP A: primary strategy = pblock (auto-derived ranges)")
            print("-" * 60)
            primary_dcp = Path(tester.temp_dir) / "dispatcher_primary_pblock_auto.dcp"
            auto_stats = await tester.try_pblock(
                input_dcp, primary_dcp, util_factor=util_factor
            )
            for note in auto_stats.get("notes") or []:
                print(f"[DISPATCH] pblock(auto): {note}")
            primary_result = {
                "strategy": "pblock(auto)",
                "success": bool(auto_stats.get("success")) and primary_dcp.exists(),
                "final_wns": auto_stats.get("final_wns"),
                "notes": list(auto_stats.get("notes") or []),
                "dcp": primary_dcp if (auto_stats.get("success") and primary_dcp.exists()) else None,
                "ranges": auto_stats.get("pblock_ranges"),
                "routing_errors": auto_stats.get("routing_errors"),
            }
            candidates.append(primary_result)
            print(
                f"[DISPATCH] pblock(auto): success={primary_result['success']} "
                f"final_wns={primary_result['final_wns']} "
                f"routing_errors={primary_result['routing_errors']}"
            )
        else:
            print("\n" + "-" * 60)
            print(f"DISPATCH STEP A: primary strategy = {primary}")
            print("-" * 60)
            primary_dcp = Path(tester.temp_dir) / "dispatcher_primary.dcp"
            primary_result = await _dispatcher_run_strategy(
                tester, primary, input_dcp, primary_dcp,
                util_factor=util_factor,
                phys_opt_directive=fanout_phys_opt_directive,
            )
            primary_result["dcp"] = primary_dcp if primary_result.get("success") else None
            candidates.append(primary_result)
            for note in primary_result["notes"]:
                print(f"[DISPATCH] {primary_result['strategy']}: {note}")
            print(
                f"[DISPATCH] {primary_result['strategy']}: "
                f"success={primary_result['success']} final_wns={primary_result['final_wns']}"
            )

        primary_runtime = time.time() - primary_step_start

        # ---- Optional LLM-proposed pblock retry ----
        # Single targeted call when the auto-pblock attempt was weak (or
        # outright failed) and we still have time + budget. Note: when ranges
        # are supplied directly, try_pblock skips RW analyze and goes straight
        # to Vivado place+route, so this path can also rescue failures of the
        # RW auto-derive step itself.
        if primary == "pblock" and llm_calls_budget > llm_calls_used:
            auto_final_wns = primary_result.get("final_wns")
            clock_period = info.get("clock_period")
            initial_fmax = tester.calculate_fmax(initial_wns, clock_period)
            auto_final_fmax = tester.calculate_fmax(auto_final_wns, clock_period) if auto_final_wns is not None else None
            auto_delta_fmax = (
                auto_final_fmax - initial_fmax
                if (auto_final_fmax is not None and initial_fmax is not None)
                else None
            )
            auto_routing_errors = primary_result.get("routing_errors") or 0
            auto_failed = not primary_result.get("success")
            auto_regressed = (
                auto_final_wns is None
                or initial_wns is None
                or auto_final_wns <= initial_wns
            )

            weak = (
                auto_failed
                or auto_regressed
                or auto_routing_errors > 0
                or (auto_delta_fmax is not None and auto_delta_fmax < llm_weak_threshold_mhz)
            )

            elapsed_so_far = time.time() - overall_start
            time_ok = (
                elapsed_so_far < llm_time_cap_seconds
                and primary_runtime < llm_primary_runtime_cap_seconds
            )

            if weak and time_ok:
                print("\n" + "-" * 60)
                print(
                    f"DISPATCH STEP A.1: LLM-proposed pblock retry "
                    f"(weak auto delta={auto_delta_fmax} MHz, "
                    f"routing_errors={auto_routing_errors}, elapsed={elapsed_so_far:.0f}s)"
                )
                print("-" * 60)
                target_fmax = (1000.0 / clock_period) if clock_period else None
                if llm_mock_ranges is not None:
                    # Test path: pretend the LLM returned `llm_mock_ranges`.
                    # Still goes through the parser to confirm validity.
                    print(f"[DISPATCH] LLM mock active; using ranges {llm_mock_ranges!r}")
                    proposed = parse_pblock_ranges_from_llm(llm_mock_ranges)
                else:
                    proposed = propose_pblock_ranges(
                        api_key=llm_api_key,
                        model=llm_model,
                        context={
                            "used_lut":  info.get("used_lut"),
                            "used_ff":   info.get("used_ff"),
                            "used_dsp":  info.get("used_dsp"),
                            "used_bram": info.get("used_bram"),
                            "max_spread_distance": info.get("max_spread_distance"),
                            "avg_spread_distance": info.get("avg_spread_distance"),
                            "initial_wns":  initial_wns,
                            "clock_period": clock_period,
                            "target_fmax":  f"{target_fmax:.2f}" if target_fmax else None,
                            "auto_ranges":  primary_result.get("ranges"),
                            "auto_final_fmax": (f"{auto_final_fmax:.2f}" if auto_final_fmax else None),
                            "auto_delta_fmax": (f"{auto_delta_fmax:+.2f}" if auto_delta_fmax is not None else None),
                            "auto_routing_errors": auto_routing_errors,
                        },
                    )
                llm_calls_used += 1
                if proposed and proposed != primary_result.get("ranges"):
                    print(f"[DISPATCH] LLM proposed pblock_ranges: {proposed}")
                    llm_dcp = Path(tester.temp_dir) / "dispatcher_primary_pblock_llm.dcp"
                    llm_stats = await tester.try_pblock(
                        input_dcp, llm_dcp,
                        pblock_ranges=proposed,
                        util_factor=util_factor,
                    )
                    for note in llm_stats.get("notes") or []:
                        print(f"[DISPATCH] pblock(llm): {note}")
                    llm_result = {
                        "strategy": "pblock(llm)",
                        "success": bool(llm_stats.get("success")) and llm_dcp.exists(),
                        "final_wns": llm_stats.get("final_wns"),
                        "notes": list(llm_stats.get("notes") or []),
                        "dcp": llm_dcp if (llm_stats.get("success") and llm_dcp.exists()) else None,
                        "ranges": llm_stats.get("pblock_ranges"),
                        "routing_errors": llm_stats.get("routing_errors"),
                    }
                    candidates.append(llm_result)
                    print(
                        f"[DISPATCH] pblock(llm): success={llm_result['success']} "
                        f"final_wns={llm_result['final_wns']} "
                        f"routing_errors={llm_result['routing_errors']}"
                    )
                elif proposed:
                    print("[DISPATCH] LLM proposed identical ranges; skipping retry")
                else:
                    print("[DISPATCH] LLM did not return parseable pblock_ranges; skipping retry")
            elif weak and not time_ok:
                print(
                    f"[DISPATCH] Skipping LLM retry: time budget tight "
                    f"(elapsed={elapsed_so_far:.0f}s, primary={primary_runtime:.0f}s)"
                )

        # `primary_improved` is now defined as the best-so-far across all
        # primary candidates (auto + optional LLM retry). Falls back to
        # bl_runtime only if everything we've tried still regresses.
        best_so_far = None
        for c in candidates:
            if c.get("success") and c.get("final_wns") is not None:
                if best_so_far is None or c["final_wns"] > best_so_far["final_wns"]:
                    best_so_far = c
        primary_improved = (
            best_so_far is not None
            and best_so_far["final_wns"] > initial_wns
        )

        # ---- Fallback: bl_runtime ----
        if not primary_improved:
            print("\n" + "-" * 60)
            print("DISPATCH STEP B: fallback strategy = bl_runtime (RuntimeOptimized)")
            print("-" * 60)
            fallback_dcp = Path(tester.temp_dir) / "dispatcher_fallback.dcp"
            fallback_result = await _dispatcher_run_strategy(
                tester, "bl_runtime", input_dcp, fallback_dcp,
                util_factor=util_factor,
                phys_opt_directive="RuntimeOptimized",
            )
            fallback_result["dcp"] = fallback_dcp if fallback_result.get("success") else None
            candidates.append(fallback_result)
            print(
                f"[DISPATCH] {fallback_result['strategy']}: "
                f"success={fallback_result['success']} final_wns={fallback_result['final_wns']}"
            )

        # ---- Pick best ----
        scored = [
            c for c in candidates
            if c.get("success") and c.get("final_wns") is not None and c.get("dcp") is not None
        ]
        scored.sort(key=lambda c: c["final_wns"], reverse=True)

        if not scored or scored[0]["final_wns"] <= initial_wns:
            print(
                f"[DISPATCH] No strategy improved on initial WNS={initial_wns:.3f}; "
                f"emitting original DCP unchanged"
            )
            await _dispatcher_emit_passthrough(tester, input_dcp, output_dcp)
            print("\n" + "=" * 70)
            print("DISPATCH SUMMARY")
            print("=" * 70)
            for c in candidates:
                print(f"  {c['strategy']}: success={c['success']} final_wns={c['final_wns']}")
            print(f"  selected: passthrough (initial_wns={initial_wns})")
            _emit_summary("passthrough (no improvement)", initial_wns)
            return 0

        best = scored[0]
        print(
            f"\n[DISPATCH] Best strategy: {best['strategy']} "
            f"(final WNS={best['final_wns']:.3f}, initial={initial_wns:.3f}, "
            f"delta_wns={best['final_wns'] - initial_wns:+.3f} ns)"
        )

        shutil.copyfile(best["dcp"], output_dcp)
        best_edif = best["dcp"].with_suffix(".edf")
        out_edif = output_dcp.with_suffix(".edf")
        if best_edif.exists():
            shutil.copyfile(best_edif, out_edif)
        else:
            try:
                await tester.call_vivado_tool("open_checkpoint", {
                    "dcp_path": str(output_dcp.resolve())
                }, timeout=600.0)
                await tester.call_vivado_tool("write_edif", {
                    "edif_path": str(out_edif.resolve()),
                    "force": True,
                }, timeout=600.0)
            except Exception as e:
                logger.warning(f"Best-DCP EDIF write failed: {e}")
                print(f"[DISPATCH] EDIF regen skipped: {e}")

        print("\n" + "=" * 70)
        print("DISPATCH SUMMARY")
        print("=" * 70)
        for c in candidates:
            marker = "  *" if c is best else "   "
            print(f"{marker} {c['strategy']}: success={c['success']} final_wns={c['final_wns']}")
        print(f"  selected: {best['strategy']} -> {output_dcp}")
        print(f"  run_dir:  {tester.run_dir}")
        _emit_summary(best["strategy"], best["final_wns"])
        return 0

    except KeyboardInterrupt:
        print("\n[DISPATCH] Interrupted by user")
        print(f"[DISPATCH] Run directory: {tester.run_dir}")
        return 130
    except Exception as e:
        logger.exception(f"Dispatcher fatal error: {e}")
        print(f"\n[DISPATCH] Fatal error: {e}")
        print(f"[DISPATCH] Run directory: {tester.run_dir}")
        return 1
    finally:
        await tester.cleanup()


