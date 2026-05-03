#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

"""
Deterministic optimization strategies (no LLM required).

This module hosts :class:`FPGAOptimizerTest`, which exposes:

- ``run_deterministic_baseline`` — limited high-fanout optimization +
  configurable phys_opt directive (``RuntimeOptimized``,
  ``AggressiveFanoutOpt``, etc.). Used by the dispatcher's "fanout" path.
- ``try_pblock`` — Vivado pblock area-constraint flow with auto-derived
  ranges (RW analyze_fabric_for_pblock) or user-supplied ranges. Used by
  the dispatcher's "pblock" path AND by the optional LLM-proposed retry.
- ``try_cell_replacement`` — RapidWright detour analysis + centroid
  re-placement. Kept for future strategy-set expansion (matrix sweep
  showed it underperforms the other two on most benchmarks).
- ``analyze_design_characteristics`` — single-shot design fingerprint
  (utilization, spread, fanout candidates) feeding the dispatcher.
- ``run_test``, ``run_test_logicnets``, ``run_test_vexriscv`` — contest
  examples reachable via ``dcp_optimizer.py --test``.
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from .base import DCPOptimizerBase, parse_timing_summary_static

logger = logging.getLogger(__name__)


class FPGAOptimizerTest(DCPOptimizerBase):
    """
    Test mode for FPGA Design Optimization - hardcodes all tool calls to diagnose issues.
    
    This class runs a deterministic optimization flow without using any LLM, 
    making it easier to identify where MCP servers or Vivado might hang.
    """
    
    def __init__(self, debug: bool = False, run_dir: Optional[Path] = None):
        super().__init__(debug=debug, run_dir=run_dir)
        self.final_wns = None
    
    async def start_servers(self):
        """Start and connect to both MCP servers."""
        await super().start_servers(log_prefix="[TEST]")
    
    async def call_vivado_tool(self, tool_name: str, arguments: dict, timeout: float = 300.0) -> str:
        """Execute a Vivado tool call with timing and logging."""
        logger.info(f"[VIVADO] Calling {tool_name} with args: {json.dumps(arguments)[:200]}...")
        print(f"[TEST] Calling vivado_{tool_name}...")
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self.vivado_session.call_tool(tool_name, arguments),
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[VIVADO] {tool_name} completed in {elapsed:.2f}s")
            print(f"[TEST] vivado_{tool_name} completed in {elapsed:.2f}s")
            
            # Extract text content from result
            if result.content:
                text_parts = [c.text for c in result.content if hasattr(c, 'text')]
                return "\n".join(text_parts)
            return "(no output)"
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[VIVADO] {tool_name} TIMED OUT after {elapsed:.2f}s")
            print(f"[TEST] ERROR: vivado_{tool_name} TIMED OUT after {elapsed:.2f}s")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VIVADO] {tool_name} FAILED after {elapsed:.2f}s: {e}")
            print(f"[TEST] ERROR: vivado_{tool_name} failed after {elapsed:.2f}s: {e}")
            raise
    
    async def call_rapidwright_tool(self, tool_name: str, arguments: dict, timeout: float = 300.0) -> str:
        """Execute a RapidWright tool call with timing and logging."""
        logger.info(f"[RAPIDWRIGHT] Calling {tool_name} with args: {json.dumps(arguments)[:200]}...")
        print(f"[TEST] Calling rapidwright_{tool_name}...")
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self.rapidwright_session.call_tool(tool_name, arguments),
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[RAPIDWRIGHT] {tool_name} completed in {elapsed:.2f}s")
            print(f"[TEST] rapidwright_{tool_name} completed in {elapsed:.2f}s")
            
            # Extract text content from result
            if result.content:
                text_parts = [c.text for c in result.content if hasattr(c, 'text')]
                return "\n".join(text_parts)
            return "(no output)"
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[RAPIDWRIGHT] {tool_name} TIMED OUT after {elapsed:.2f}s")
            print(f"[TEST] ERROR: rapidwright_{tool_name} TIMED OUT after {elapsed:.2f}s")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[RAPIDWRIGHT] {tool_name} FAILED after {elapsed:.2f}s: {e}")
            print(f"[TEST] ERROR: rapidwright_{tool_name} failed after {elapsed:.2f}s: {e}")
            raise
    
    def parse_wns_from_timing_report(self, timing_report: str) -> Optional[float]:
        """Extract WNS from timing report using shared parsing logic."""
        return parse_timing_summary_static(timing_report)["wns"]
    
    async def _call_vivado_for_clock(self, tool_name: str, arguments: dict) -> str:
        """Helper to call Vivado tools for clock period query."""
        return await self.call_vivado_tool(tool_name, arguments, timeout=60.0)
    
    async def fetch_clock_period(self) -> Optional[float]:
        """Query clock period with test-mode logging."""
        period = await super().get_clock_period(self._call_vivado_for_clock)
        if period is not None:
            clock_info = f" (target clock: {self.target_clock})" if self.target_clock else ""
            print(f"[TEST] Clock period: {period:.3f} ns{clock_info}")
        else:
            print("[TEST] WARNING: Could not parse clock period from Vivado")
        return period

    def _parse_json_text(self, text: str) -> dict:
        """Best-effort JSON parsing for MCP tool outputs."""
        try:
            return json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return {}

    def _select_general_fanout_candidates(
        self,
        nets_report: str,
        max_nets: int,
        min_fanout: int = 100
    ) -> list[tuple[str, int, int]]:
        """
        Pick a small number of high-value fanout candidates for the general baseline.

        Stage 1 keeps this conservative: we only touch the worst 1-2 nets that
        appear on multiple critical paths and already have high fanout.
        """
        nets = self.parse_high_fanout_nets(nets_report)
        filtered = [
            (net_name, fanout, path_count)
            for net_name, fanout, path_count in nets
            if fanout >= min_fanout and path_count >= 2
        ]
        filtered.sort(key=lambda item: (-item[2], -item[1], item[0]))
        return filtered[:max(0, min(max_nets, 2))]

    async def _measure_timing(
        self,
        label: str,
        timeout: float = 300.0,
        preview_chars: int = 2000
    ) -> tuple[str, Optional[float]]:
        """Report timing and return the WNS for the target clock if available."""
        result = await self.call_vivado_tool("report_timing_summary", {}, timeout=timeout)
        print(f"{label} timing summary (first {preview_chars} chars):\n{result[:preview_chars]}...")
        logger.info(f"{label} timing summary: {result}")

        target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
        if target_wns is not None:
            return result, target_wns
        return result, self.parse_wns_from_timing_report(result)

    async def _route_and_report_status(
        self,
        directive: str = "Default",
        timeout: float = 3600.0
    ) -> str:
        """Route the current design and print a short route-status preview."""
        route_result = await self.call_vivado_tool("route_design", {
            "directive": directive
        }, timeout=timeout)
        print(f"Route design result:\n{route_result}")
        logger.info(f"Route design: {route_result}")

        status = await self.call_vivado_tool("report_route_status", {
            "show_unrouted": True,
            "show_errors": True,
            "max_nets": 20
        }, timeout=300.0)
        print(f"Route status after routing:\n{status[:1500]}...")
        logger.info(f"Route status after routing: {status}")
        return status

    async def run_deterministic_baseline(
        self,
        input_dcp: Path,
        output_dcp: Path,
        max_nets_to_optimize: int = 2,
        phys_opt_directive: str = "RuntimeOptimized",
    ) -> bool:
        """
        Run a benchmark-agnostic deterministic fallback flow without any LLM.

        Strategy order is intentionally conservative for alpha submission:
        1. Try limited high-fanout optimization first because it is localized and easy to skip.
        2. Fall back to a safe Vivado phys_opt directive for general routed designs.
        3. Always write a final DCP, even if no improvement is found.
        """
        print("\n" + "="*70)
        print("FPGA OPTIMIZER DETERMINISTIC BASELINE MODE")
        print("="*70)
        print(f"Input DCP:  {input_dcp}")
        print(f"Output DCP: {output_dcp}")
        print(f"Temp dir:   {self.temp_dir}")
        print(f"Fanout candidate limit: {max(0, min(max_nets_to_optimize, 2))}")
        print("="*70 + "\n")

        overall_start = time.time()
        best_checkpoint = Path(self.temp_dir) / "baseline_best.dcp"
        fanout_checkpoint = Path(self.temp_dir) / "baseline_fanout_candidate.dcp"
        best_source_dcp = input_dcp.resolve()
        best_strategy = "no-op"
        strategy_notes: list[str] = []

        try:
            # RapidWright is optional for the Stage 1 baseline. If initialization
            # fails, we still continue with the Vivado-only phys_opt fallback.
            rapidwright_available = False
            print("\n" + "-"*60)
            print("STEP 0: Initialize RapidWright (optional)")
            print("-"*60)
            try:
                rw_init = await self.call_rapidwright_tool("initialize_rapidwright", {
                    "jvm_max_memory": "8G"
                }, timeout=120.0)
                rw_init_json = self._parse_json_text(rw_init)
                rapidwright_available = rw_init_json.get("status") in {"success", "already_initialized"}
                if rapidwright_available:
                    print("RapidWright initialized successfully for fanout optimization.")
                else:
                    print("RapidWright unavailable; baseline will skip fanout optimization.")
                    strategy_notes.append("RapidWright initialization failed, skipping fanout optimization")
            except Exception as e:
                print(f"RapidWright initialization failed: {e}")
                logger.warning(f"Baseline continuing without RapidWright: {e}")
                strategy_notes.append("RapidWright initialization failed, skipping fanout optimization")

            print("\n" + "-"*60)
            print("STEP 1: Open input DCP in Vivado")
            print("-"*60)
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"Open checkpoint result:\n{result}")
            logger.info(f"Open checkpoint result: {result}")

            print("\n" + "-"*60)
            print("STEP 2: Measure baseline timing")
            print("-"*60)
            baseline_report, baseline_wns = await self._measure_timing("Baseline")
            self.clock_period = await self.fetch_clock_period()
            self.initial_wns = baseline_wns
            self.final_wns = baseline_wns
            self.print_fmax_status("Initial", self.initial_wns)
            logger.info(f"Baseline initial WNS: {self.initial_wns} ns")

            if self.initial_wns is not None and self.initial_wns >= 0:
                print("\nTiming already met; writing checkpoint without additional optimization.")
                strategy_notes.append("Timing already met; emitted original routed checkpoint")
                result = await self.call_vivado_tool("write_checkpoint", {
                    "dcp_path": str(output_dcp.resolve()),
                    "force": True
                }, timeout=600.0)
                print(f"Write final DCP result:\n{result}")
                elapsed = time.time() - overall_start
                self.print_test_summary(
                    title="TEST SUMMARY - DETERMINISTIC BASELINE",
                    elapsed_seconds=elapsed,
                    initial_wns=self.initial_wns,
                    final_wns=self.final_wns,
                    clock_period=self.clock_period,
                    extra_info="Strategy: no-op (timing already met)"
                )
                return True

            print("\n" + "-"*60)
            print("STEP 3: Select deterministic strategy")
            print("-"*60)
            nets_report = await self.call_vivado_tool("get_critical_high_fanout_nets", {
                "num_paths": 30,
                "min_fanout": 80,
                "exclude_clocks": True
            }, timeout=600.0)
            logger.info(f"High fanout nets report: {nets_report}")
            fanout_candidates = self._select_general_fanout_candidates(
                nets_report,
                max_nets=max_nets_to_optimize,
                min_fanout=100
            )

            if fanout_candidates:
                print("Selected limited fanout optimization because strong shared high-fanout nets were found:")
                for net_name, fanout, path_count in fanout_candidates:
                    print(f"  - {net_name} (fanout={fanout}, critical_paths={path_count})")
                strategy_notes.append(
                    f"Selected fanout-first path due to {len(fanout_candidates)} strong high-fanout candidates"
                )
            else:
                print("No strong shared high-fanout candidates found; using phys_opt as the primary baseline strategy.")
                strategy_notes.append("No strong high-fanout candidates found; using phys_opt fallback")

            current_best_wns = self.initial_wns

            # Fanout comes first because it is localized, deterministic, and easy
            # to abandon if the design does not respond well.
            if rapidwright_available and fanout_candidates:
                print("\n" + "-"*60)
                print("STEP 4: Try limited fanout optimization")
                print("-"*60)
                try:
                    rw_read = await self.call_rapidwright_tool("read_checkpoint", {
                        "dcp_path": str(input_dcp.resolve())
                    }, timeout=600.0)
                    rw_read_json = self._parse_json_text(rw_read)
                    if rw_read_json.get("status") != "success":
                        raise RuntimeError(rw_read)

                    successful_optimizations = 0
                    for i, (net_name, fanout, path_count) in enumerate(fanout_candidates, start=1):
                        split_factor = max(3, min(4, fanout // 100))
                        print(f"[{i}/{len(fanout_candidates)}] Fanout optimization: {net_name}")
                        print(f"    Fanout: {fanout}, Critical paths: {path_count}, Split factor: {split_factor}")
                        opt_result = await self.call_rapidwright_tool("optimize_fanout", {
                            "net_name": net_name,
                            "split_factor": split_factor
                        }, timeout=300.0)
                        opt_result_json = self._parse_json_text(opt_result)
                        if opt_result_json.get("status") == "success":
                            successful_optimizations += 1
                            print(f"    Success: {opt_result_json.get('message', 'fanout split complete')}")
                        else:
                            print(f"    Skipped: {opt_result_json.get('error', opt_result)}")

                    if successful_optimizations > 0:
                        result = await self.call_rapidwright_tool("write_checkpoint", {
                            "dcp_path": str(fanout_checkpoint.resolve()),
                            "overwrite": True
                        }, timeout=600.0)
                        print(f"RapidWright write checkpoint result:\n{result}")

                        await self.call_vivado_tool("open_checkpoint", {
                            "dcp_path": str(fanout_checkpoint.resolve())
                        }, timeout=600.0)
                        await self._route_and_report_status()
                        _, fanout_wns = await self._measure_timing("Post-fanout")

                        if fanout_wns is not None and (current_best_wns is None or fanout_wns > current_best_wns):
                            current_best_wns = fanout_wns
                            best_strategy = "fanout"
                            print("Fanout optimization improved timing; capturing best checkpoint.")
                            strategy_notes.append(
                                f"Fanout optimization improved WNS to {fanout_wns:.3f} ns"
                            )
                            await self.call_vivado_tool("write_checkpoint", {
                                "dcp_path": str(best_checkpoint.resolve()),
                                "force": True
                            }, timeout=600.0)
                            best_source_dcp = best_checkpoint.resolve()
                        else:
                            print("Fanout optimization did not improve timing; reverting to original checkpoint.")
                            strategy_notes.append("Fanout attempt completed but did not improve timing")
                            await self.call_vivado_tool("open_checkpoint", {
                                "dcp_path": str(best_source_dcp)
                            }, timeout=600.0)
                    else:
                        print("No fanout optimizations succeeded; proceeding to phys_opt fallback.")
                        strategy_notes.append("Fanout candidates selected, but no RapidWright fanout edits succeeded")
                except Exception as e:
                    print(f"Fanout optimization path failed safely: {e}")
                    logger.warning(f"Fanout path failed; continuing with phys_opt fallback: {e}")
                    strategy_notes.append(f"Fanout path failed safely: {e}")
                    await self.call_vivado_tool("open_checkpoint", {
                        "dcp_path": str(best_source_dcp)
                    }, timeout=600.0)

            print("\n" + "-"*60)
            print("STEP 5: Run conservative phys_opt fallback")
            print("-"*60)
            await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(best_source_dcp)
            }, timeout=600.0)
            print(f"Using phys_opt_design -directive {phys_opt_directive} as the general fallback.")
            try:
                result = await self.call_vivado_tool("phys_opt_design", {
                    "directive": phys_opt_directive
                }, timeout=3600.0)
                print(f"Phys opt result:\n{result}")
                logger.info(f"Phys opt result: {result}")
                await self._route_and_report_status()
                _, physopt_wns = await self._measure_timing("Post-phys_opt")

                if physopt_wns is not None and (current_best_wns is None or physopt_wns > current_best_wns):
                    current_best_wns = physopt_wns
                    best_strategy = "phys_opt"
                    print("Phys opt improved timing; capturing best checkpoint.")
                    strategy_notes.append(f"phys_opt improved WNS to {physopt_wns:.3f} ns")
                    await self.call_vivado_tool("write_checkpoint", {
                        "dcp_path": str(best_checkpoint.resolve()),
                        "force": True
                    }, timeout=600.0)
                    best_source_dcp = best_checkpoint.resolve()
                else:
                    print("Phys opt did not improve timing; keeping previously best checkpoint.")
                    strategy_notes.append("phys_opt completed but did not improve timing")
            except Exception as e:
                print(f"phys_opt fallback failed safely: {e}")
                logger.warning(f"phys_opt fallback failed: {e}")
                strategy_notes.append(f"phys_opt fallback failed safely: {e}")

            print("\n" + "-"*60)
            print("STEP 6: Write final deterministic baseline DCP")
            print("-"*60)
            await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(best_source_dcp)
            }, timeout=600.0)
            _, self.final_wns = await self._measure_timing("Final")
            self.print_fmax_status("Final", self.final_wns)
            self.print_wns_change(self.initial_wns, self.final_wns, self.clock_period)

            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            print(f"Write final DCP result:\n{result}")

            # Validation Phase 1 reopens the final DCP in RapidWright, so emit a
            # readable EDIF alongside the Vivado checkpoint.
            final_edif = output_dcp.with_suffix(".edf")
            result = await self.call_vivado_tool("write_edif", {
                "edif_path": str(final_edif.resolve()),
                "force": True
            }, timeout=600.0)
            print(f"Write final EDIF result:\n{result}")

            elapsed = time.time() - overall_start
            extra_info = (
                f"Strategy: {best_strategy}\n"
                f"Notes: {'; '.join(strategy_notes)}"
            )
            self.print_test_summary(
                title="TEST SUMMARY - DETERMINISTIC BASELINE",
                elapsed_seconds=elapsed,
                initial_wns=self.initial_wns,
                final_wns=self.final_wns,
                clock_period=self.clock_period,
                extra_info=extra_info
            )
            return True

        except Exception as e:
            logger.exception(f"Deterministic baseline failed with exception: {e}")
            print(f"\n*** DETERMINISTIC BASELINE FAILED ***")
            print(f"Exception: {type(e).__name__}: {e}")
            return False

    async def analyze_design_characteristics(self, input_dcp: Path) -> dict:
        """Sample initial timing, utilization, fanout structure, and critical-path
        spread for the given DCP. Returns a dict the dispatcher uses to pick a
        strategy. Best-effort: each subsection swallows its own exceptions and
        records a note.

        Assumes start_servers() has been called by the caller.
        """
        notes: list[str] = []
        info = {
            "initial_wns": None,
            "clock_period": None,
            "used_lut": 0,
            "used_ff": 0,
            "used_dsp": 0,
            "used_bram": 0,
            "max_spread_distance": None,
            "avg_spread_distance": None,
            "paths_analyzed": 0,
            "fanout_candidate_count": 0,
            "top_fanout_max": 0,
            "notes": notes,
        }

        # ---- (1) Open + initial timing ----
        try:
            await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            self.clock_period = await self.fetch_clock_period()
            info["clock_period"] = self.clock_period
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                ts = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
                self.initial_wns = self.parse_wns_from_timing_report(ts)
            info["initial_wns"] = self.initial_wns
            notes.append(f"Initial WNS={self.initial_wns} clock_period={self.clock_period}")
        except Exception as e:
            logger.warning(f"analyze: initial timing failed: {e}")
            notes.append(f"Initial timing failed: {e}")
            return info

        # ---- (2) Utilization ----
        try:
            util_raw = await self.call_vivado_tool("run_tcl", {
                "command": "report_utilization -return_string"
            }, timeout=300.0)
            if not isinstance(util_raw, str):
                util_raw = str(util_raw)

            def _grab(label_pattern: str) -> int:
                m = re.search(r"\|\s*" + label_pattern + r"\s*\|\s*([\d,]+)", util_raw)
                if not m:
                    return 0
                try:
                    return int(m.group(1).replace(",", ""))
                except (TypeError, ValueError):
                    return 0

            info["used_lut"]  = _grab(r"(?:Slice LUTs|CLB LUTs|LUT as Logic)")
            info["used_ff"]   = _grab(r"(?:Register as Flip Flop|Slice Registers|CLB Registers)")
            info["used_dsp"]  = _grab(r"DSPs")
            info["used_bram"] = _grab(r"Block RAM Tile")
            notes.append(
                f"Util LUT={info['used_lut']} FF={info['used_ff']} "
                f"DSP={info['used_dsp']} BRAM={info['used_bram']}"
            )
        except Exception as e:
            logger.warning(f"analyze: utilization parse failed: {e}")
            notes.append(f"Utilization parse failed: {e}")

        # ---- (3) High-fanout candidate inventory ----
        try:
            nets_report = await self.call_vivado_tool("get_critical_high_fanout_nets", {
                "num_paths": 30,
                "min_fanout": 80,
                "exclude_clocks": True,
            }, timeout=600.0)
            high_nets = self.parse_high_fanout_nets(nets_report) or []
            strong = [
                (name, fanout, paths)
                for (name, fanout, paths) in high_nets
                if fanout >= 100 and paths >= 2
            ]
            info["fanout_candidate_count"] = len(strong)
            info["top_fanout_max"] = max((f for _, f, _ in high_nets), default=0)
            notes.append(
                f"Fanout candidates (>=100 fanout, >=2 paths)={len(strong)} "
                f"top_fanout={info['top_fanout_max']}"
            )
        except Exception as e:
            logger.warning(f"analyze: fanout query failed: {e}")
            notes.append(f"Fanout query failed: {e}")

        # ---- (4) RapidWright critical-path spread ----
        try:
            rw_init = await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            rw_init_json = self._parse_json_text(rw_init)
            if rw_init_json.get("status") in {"success", "already_initialized"}:
                await self.call_rapidwright_tool("read_checkpoint", {
                    "dcp_path": str(input_dcp.resolve())
                }, timeout=600.0)
                cells_file = Path(self.temp_dir) / "dispatcher_critical_path_cells.json"
                await self.call_vivado_tool("extract_critical_path_cells", {
                    "num_paths": 50,
                    "output_file": str(cells_file),
                }, timeout=600.0)
                spread_raw = await self.call_rapidwright_tool("analyze_critical_path_spread", {
                    "input_file": str(cells_file)
                }, timeout=300.0)
                spread = self._parse_json_text(spread_raw) if isinstance(spread_raw, str) else spread_raw
                if isinstance(spread, dict) and "error" not in spread:
                    info["max_spread_distance"] = spread.get("max_distance_found")
                    info["avg_spread_distance"] = spread.get("avg_max_distance")
                    info["paths_analyzed"] = spread.get("paths_analyzed", 0) or 0
                    notes.append(
                        f"Spread max={info['max_spread_distance']} "
                        f"avg={info['avg_spread_distance']} paths={info['paths_analyzed']}"
                    )
                else:
                    notes.append(f"Spread analysis returned no data: {str(spread)[:200]}")
            else:
                notes.append("RapidWright unavailable; skipped spread analysis")
        except Exception as e:
            logger.warning(f"analyze: spread analysis failed: {e}")
            notes.append(f"Spread analysis failed: {e}")

        return info

    async def try_cell_replacement(
        self,
        input_dcp: Path,
        output_dcp: Path,
        detour_threshold: float = 2.0,
        target_paths_max: int = 2,
        extract_num_paths: int = 10,
    ) -> dict:
        """Benchmark-agnostic cell re-placement using RapidWright detour analysis.

        Mirrors run_test_vexriscv but returns structured stats so callers
        (matrix sweep, baseline dispatcher) can decide what to do with the
        result. Assumes start_servers() has already been called.

        Flow:
          1. Vivado: open input + measure baseline + extract critical path pins
          2. RapidWright: read DCP + analyze_net_detour
          3. RapidWright: optimize_cell_placement on detour candidates + write
          4. Vivado: open optimized DCP + route_design + measure final timing
        """
        notes: list[str] = []
        stats = {
            "success": False,
            "initial_wns": None,
            "final_wns": None,
            "clock_period": None,
            "candidates_found": 0,
            "cells_replaced": [],
            "routing_errors": -1,
            "notes": notes,
        }

        try:
            # Step 1: Vivado baseline + extract critical path pins
            await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)

            self.clock_period = await self.fetch_clock_period()
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                ts = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
                self.initial_wns = self.parse_wns_from_timing_report(ts)
            stats["initial_wns"] = self.initial_wns
            stats["clock_period"] = self.clock_period

            pins_file = Path(self.temp_dir) / "critical_path_pins.json"
            extract_result = await self.call_vivado_tool("extract_critical_path_pins", {
                "num_paths": extract_num_paths,
                "output_file": str(pins_file)
            }, timeout=600.0)
            critical_paths = (
                json.loads(Path(pins_file).read_text())
                if pins_file.exists()
                else json.loads(extract_result)
            )
            notes.append(f"Extracted {len(critical_paths)} critical path pin lists")

            # Step 2: RapidWright detour analysis
            await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            await self.call_rapidwright_tool("read_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)

            analyze_result = await self.call_rapidwright_tool("analyze_net_detour", {
                "input_file": str(pins_file),
                "detour_threshold": detour_threshold
            }, timeout=300.0)
            analysis = (
                json.loads(analyze_result)
                if isinstance(analyze_result, str)
                else analyze_result
            )
            if "error" in analysis:
                notes.append(f"analyze_net_detour error: {analysis['error']}")
                return stats

            candidates = analysis.get("candidates", [])
            stats["candidates_found"] = len(candidates)
            notes.append(
                f"Detour analysis: {analysis.get('cells_analyzed', '?')} cells inspected, "
                f"{len(candidates)} above threshold {detour_threshold}"
            )

            if not candidates:
                # Nothing to re-place; treat as a success but produce no output
                stats["final_wns"] = self.initial_wns
                stats["success"] = True
                notes.append("No detour candidates above threshold - re-placement skipped")
                return stats

            worst_path_cells = list(set(
                str(c["cell"]) for c in candidates
                if c.get("path", 0) <= target_paths_max
            ))
            if not worst_path_cells:
                worst_path_cells = [str(candidates[0]["cell"])]
            stats["cells_replaced"] = worst_path_cells
            notes.append(f"Targeting {len(worst_path_cells)} cells on paths 1-{target_paths_max}")

            # Step 3: RapidWright optimize_cell_placement + write intermediate DCP
            await self.call_rapidwright_tool("optimize_cell_placement", {
                "cell_names": worst_path_cells
            }, timeout=300.0)

            rw_intermediate = Path(self.temp_dir) / "cell_replaced_intermediate.dcp"
            await self.call_rapidwright_tool("write_checkpoint", {
                "dcp_path": str(rw_intermediate)
            }, timeout=600.0)

            # Step 4: Vivado route + measure final + write output DCP
            await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(rw_intermediate)
            }, timeout=600.0)
            await self.call_vivado_tool("route_design", {
                "directive": "Default"
            }, timeout=3600.0)

            route_status = await self.call_vivado_tool("report_route_status", {}, timeout=300.0)
            error_match = re.search(r"# of nets with routing errors.*?:\s+(\d+)", route_status)
            stats["routing_errors"] = int(error_match.group(1)) if error_match else -1

            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                ts = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
                self.final_wns = self.parse_wns_from_timing_report(ts)
            stats["final_wns"] = self.final_wns

            await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            # Validation Phase 1 reopens the DCP in RapidWright, which needs a
            # readable EDIF alongside the checkpoint.
            edif_path = output_dcp.with_suffix(".edf")
            await self.call_vivado_tool("write_edif", {
                "edif_path": str(edif_path.resolve()),
                "force": True
            }, timeout=600.0)

            stats["success"] = True
            return stats

        except Exception as e:
            logger.exception(f"try_cell_replacement failed: {e}")
            notes.append(f"Exception: {type(e).__name__}: {e}")
            return stats

    async def try_pblock(
        self,
        input_dcp: Path,
        output_dcp: Path,
        pblock_ranges: Optional[str] = None,
        place_directive: str = "Default",
        route_directive: str = "Default",
        util_factor: float = 1.5,
    ) -> dict:
        """Benchmark-agnostic pblock area-constraint flow.

        Walks the contest-recommended 8-step pipeline:
            (1) Open DCP in Vivado, measure initial WNS
            (2) Pull utilization (1.5x targets) for pblock sizing
            (3) Initialize RapidWright, read DCP
            (4) analyze_fabric_for_pblock -> col/row bounds
            (5) convert_fabric_region_to_pblock -> Vivado pblock_ranges string
            (6) Vivado: place_design -unplace + create_and_apply_pblock
            (7) Vivado: place_design + route_design under the constraint
            (8) Final timing, write checkpoint

        If `pblock_ranges` is provided, steps (2)/(4)/(5) are skipped and the
        provided string is applied directly (useful for known-optimal ranges
        like LogicNets's SLICE_X55Y60:SLICE_X111Y254).

        Assumes start_servers() has been called by the caller. Returns
        structured stats so a matrix sweep / dispatcher can branch on result.
        """
        notes = []
        stats = {
            "success": False,
            "initial_wns": None,
            "final_wns": None,
            "clock_period": None,
            "pblock_ranges": pblock_ranges,
            "routing_errors": -1,
            "notes": notes,
        }

        try:
            # ---- STEP 1: open + initial WNS ----
            await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)

            self.clock_period = await self.fetch_clock_period()
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                ts = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
                self.initial_wns = self.parse_wns_from_timing_report(ts)
            stats["initial_wns"] = self.initial_wns
            stats["clock_period"] = self.clock_period
            notes.append(f"Initial WNS={self.initial_wns} clock_period={self.clock_period}")

            # ---- STEPs 2-5: auto-derive pblock_ranges if not supplied ----
            if pblock_ranges is None:
                # (2) Utilization. Bypass report_utilization_for_pblock (its
                # FF parser miscounts on at least some designs - silently
                # zeroes out the count) by calling raw report_utilization and
                # parsing with our own regex. We then apply the 1.5x multiplier
                # ourselves.
                util_raw = await self.call_vivado_tool("run_tcl", {
                    "command": "report_utilization -return_string"
                }, timeout=300.0)
                if not isinstance(util_raw, str):
                    util_raw = str(util_raw)

                def _grab_used(label_pattern: str) -> int:
                    m = re.search(
                        r"\|\s*" + label_pattern + r"\s*\|\s*([\d,]+)",
                        util_raw,
                    )
                    if not m:
                        return 0
                    try:
                        return int(m.group(1).replace(",", ""))
                    except (TypeError, ValueError):
                        return 0

                used_lut  = _grab_used(r"(?:Slice LUTs|CLB LUTs|LUT as Logic)")
                used_ff   = _grab_used(r"(?:Register as Flip Flop|Slice Registers|CLB Registers)")
                used_dsp  = _grab_used(r"DSPs")
                used_bram = _grab_used(r"Block RAM Tile")
                if used_lut == 0:
                    notes.append(
                        f"Could not parse utilization (LUT used = 0); "
                        f"snippet={util_raw[:300]!r}"
                    )
                    return stats
                target_lut  = int(used_lut  * util_factor)
                target_ff   = int(used_ff   * util_factor)
                target_dsp  = int(used_dsp  * util_factor)
                target_bram = int(used_bram * util_factor)
                notes.append(f"Used: LUT={used_lut} FF={used_ff} DSP={used_dsp} BRAM={used_bram}")
                notes.append(f"Targets ({util_factor}x): LUT={target_lut} FF={target_ff} DSP={target_dsp} BRAM={target_bram}")

                # (3) RapidWright init + read
                await self.call_rapidwright_tool("initialize_rapidwright", {
                    "jvm_max_memory": "8G"
                }, timeout=120.0)
                await self.call_rapidwright_tool("read_checkpoint", {
                    "dcp_path": str(input_dcp.resolve())
                }, timeout=600.0)

                # (4) analyze_fabric_for_pblock -> col/row bounds
                fabric_args = {
                    "target_lut_count": target_lut,
                    "target_ff_count": target_ff,
                }
                if target_dsp > 0:
                    fabric_args["target_dsp_count"] = target_dsp
                if target_bram > 0:
                    fabric_args["target_bram_count"] = target_bram
                fabric_result = await self.call_rapidwright_tool(
                    "analyze_fabric_for_pblock", fabric_args, timeout=300.0
                )
                fabric = self._parse_json_text(fabric_result) if isinstance(fabric_result, str) else fabric_result
                if not isinstance(fabric, dict) or "error" in fabric:
                    notes.append(f"analyze_fabric_for_pblock failed: {str(fabric)[:300]}")
                    return stats
                # The bounds live in fabric["recommended_region"] per the RW tool spec.
                region = fabric.get("recommended_region") or {}
                try:
                    col_min = int(region.get("col_min"))
                    col_max = int(region.get("col_max"))
                    row_min = int(region.get("row_min"))
                    row_max = int(region.get("row_max"))
                except (TypeError, ValueError):
                    notes.append(f"recommended_region missing col/row bounds: {str(region)[:200]}")
                    return stats
                notes.append(f"Fabric region: col[{col_min},{col_max}] row[{row_min},{row_max}]")

                # (5) convert_fabric_region_to_pblock -> Vivado ranges string
                conv_result = await self.call_rapidwright_tool(
                    "convert_fabric_region_to_pblock", {
                        "col_min": col_min, "col_max": col_max,
                        "row_min": row_min, "row_max": row_max,
                    }, timeout=120.0
                )
                conv = self._parse_json_text(conv_result) if isinstance(conv_result, str) else conv_result
                pblock_ranges = (conv.get("pblock_ranges") if isinstance(conv, dict) else None) or \
                                (conv_result.strip() if isinstance(conv_result, str) else None)
                if not pblock_ranges:
                    notes.append(f"convert_fabric_region_to_pblock returned no string: {str(conv)[:200]}")
                    return stats
                stats["pblock_ranges"] = pblock_ranges
                notes.append(f"Auto pblock_ranges: {pblock_ranges[:120]}{'...' if len(pblock_ranges) > 120 else ''}")
            else:
                notes.append(f"Using supplied pblock_ranges: {pblock_ranges[:120]}{'...' if len(pblock_ranges) > 120 else ''}")

            # ---- STEP 6: unplace + apply pblock ----
            await self.call_vivado_tool("run_tcl", {
                "command": "place_design -unplace"
            }, timeout=300.0)
            await self.call_vivado_tool("create_and_apply_pblock", {
                "pblock_name": "pblock_opt_auto",
                "ranges": pblock_ranges,
                "apply_to": "current_design",
                "is_soft": False,
            }, timeout=300.0)

            # ---- STEP 7: place + route under constraint ----
            await self.call_vivado_tool("place_design", {
                "directive": place_directive,
            }, timeout=3600.0)
            await self.call_vivado_tool("route_design", {
                "directive": route_directive,
            }, timeout=3600.0)

            route_status = await self.call_vivado_tool("report_route_status", {}, timeout=300.0)
            error_match = re.search(r"# of nets with routing errors.*?:\s+(\d+)", route_status)
            stats["routing_errors"] = int(error_match.group(1)) if error_match else -1

            # ---- STEP 8: final timing + write ----
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                ts = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
                self.final_wns = self.parse_wns_from_timing_report(ts)
            stats["final_wns"] = self.final_wns
            notes.append(f"Final WNS={self.final_wns}")

            await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True,
            }, timeout=600.0)
            # Validation Phase 1 reopens the DCP in RapidWright, which needs a
            # readable EDIF alongside the checkpoint.
            edif_path = output_dcp.with_suffix(".edf")
            await self.call_vivado_tool("write_edif", {
                "edif_path": str(edif_path.resolve()),
                "force": True,
            }, timeout=600.0)

            stats["success"] = True
            return stats

        except Exception as e:
            logger.exception(f"try_pblock failed: {e}")
            notes.append(f"Exception: {type(e).__name__}: {e}")
            return stats

    async def run_test(self, input_dcp: Path, output_dcp: Path, max_nets_to_optimize: int = 5) -> bool:
        """
        Run the deterministic test optimization flow.
        
        Steps:
        1. Open the input DCP in Vivado
        2. Report timing in Vivado
        3. Get the critical high fan out nets from Vivado
        4. Open the DCP in RapidWright
        5. Apply the fanout optimization for each high fanout net
        6. Write a DCP out from RapidWright
        7. Read the RapidWright generated DCP into Vivado
        8. Route the design in Vivado
        9. Report timing and compare WNS
        """
        print("\n" + "="*70)
        print("FPGA OPTIMIZER TEST MODE")
        print("="*70)
        print(f"Input DCP:  {input_dcp}")
        print(f"Output DCP: {output_dcp}")
        print(f"Temp dir:   {self.temp_dir}")
        print(f"Max nets to optimize: {max_nets_to_optimize}")
        print("="*70 + "\n")
        
        overall_start = time.time()
        
        try:
            # ================================================================
            # Step 0: Initialize RapidWright (Vivado starts automatically)
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 0: Initialize RapidWright")
            print("-"*60)
            
            # Initialize RapidWright (Vivado will auto-start when first used)
            result = await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            print(f"RapidWright init result:\n{result[:500]}...")
            logger.info(f"RapidWright init result: {result}")
            
            # ================================================================
            # Step 1: Open the input DCP in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 1: Open input DCP in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"Open checkpoint result:\n{result}")
            logger.info(f"Open checkpoint result: {result}")
            
            # ================================================================
            # Step 2: Report timing in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 2: Report timing in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Initial timing summary: {result}")
            
            # Get clock period for fmax calculation (also detects target clock)
            self.clock_period = await self.fetch_clock_period()
            
            # Get WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                self.initial_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Initial", self.initial_wns)
            logger.info(f"Initial WNS: {self.initial_wns} ns")
            print()
            
            # ================================================================
            # Step 3: Get critical high fanout nets
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 3: Get critical high fanout nets")
            print("-"*60)
            
            result = await self.call_vivado_tool("get_critical_high_fanout_nets", {
                "num_paths": 50,
                "min_fanout": 100,
                "exclude_clocks": True
            }, timeout=600.0)
            print(f"High fanout nets report:\n{result}")
            logger.info(f"High fanout nets: {result}")
            
            # Parse the nets
            self.high_fanout_nets = self.parse_high_fanout_nets(result)
            print(f"\nParsed {len(self.high_fanout_nets)} high fanout nets")
            
            if not self.high_fanout_nets:
                print("WARNING: No high fanout nets found to optimize!")
                logger.warning("No high fanout nets found to optimize")
            
            # Select top nets to optimize
            nets_to_optimize = self.high_fanout_nets[:max_nets_to_optimize]
            print(f"Will optimize {len(nets_to_optimize)} nets:")
            for net_name, fanout, path_count in nets_to_optimize:
                print(f"  - {net_name} (fanout={fanout}, paths={path_count})")
            
            # ================================================================
            # Step 4: Open the DCP in RapidWright
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 4: Open DCP in RapidWright")
            print("-"*60)
            
            result = await self.call_rapidwright_tool("read_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"RapidWright read checkpoint result:\n{result}")
            logger.info(f"RapidWright read checkpoint: {result}")
            
            # ================================================================
            # Step 5: Apply fanout optimization for each high fanout net
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 5: Apply fanout optimizations in RapidWright")
            print("-"*60)
            
            successful_optimizations = 0
            for i, (net_name, fanout, path_count) in enumerate(nets_to_optimize):
                print(f"\n[{i+1}/{len(nets_to_optimize)}] Optimizing net: {net_name}")
                print(f"    Fanout: {fanout}, Critical paths: {path_count}")
                
                # Calculate split factor: fanout/100, min 2, max 8
                split_factor = max(2, min(8, fanout // 100))
                print(f"    Split factor: {split_factor}")
                
                try:
                    result = await self.call_rapidwright_tool("optimize_fanout", {
                        "net_name": net_name,
                        "split_factor": split_factor
                    }, timeout=300.0)
                    print(f"    Result: {result[:500]}...")
                    logger.info(f"Optimize fanout {net_name}: {result}")
                    
                    # Check if successful
                    if "error" not in result.lower() or "success" in result.lower():
                        successful_optimizations += 1
                except Exception as e:
                    print(f"    FAILED: {e}")
                    logger.error(f"Failed to optimize {net_name}: {e}")
            
            print(f"\nSuccessfully optimized {successful_optimizations}/{len(nets_to_optimize)} nets")
            
            # ================================================================
            # Step 6: Write DCP from RapidWright
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 6: Write DCP from RapidWright")
            print("-"*60)
            
            rapidwright_dcp = Path(self.temp_dir) / "rapidwright_optimized.dcp"
            result = await self.call_rapidwright_tool("write_checkpoint", {
                "dcp_path": str(rapidwright_dcp),
                "overwrite": True
            }, timeout=600.0)
            print(f"Write checkpoint result:\n{result}")
            logger.info(f"RapidWright write checkpoint: {result}")
            
            # Check if the file was created
            if rapidwright_dcp.exists():
                print(f"DCP file created: {rapidwright_dcp} ({rapidwright_dcp.stat().st_size} bytes)")
            else:
                print("WARNING: DCP file was not created!")
                logger.warning("RapidWright DCP file not created")
            
            # ================================================================
            # Step 7: Read RapidWright DCP into Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 7: Read RapidWright DCP into Vivado")
            print("-"*60)
            
            # Note: Opening a RapidWright-generated DCP takes MUCH longer than
            # opening the original DCP because:
            # 1. Vivado must reload encrypted IP blocks from disk
            # 2. Vivado must reconstruct internal data structures
            # For large designs, this can take 10-30 minutes
            RAPIDWRIGHT_DCP_TIMEOUT = 300.0  # 5 minutes
            
            # Check if there's a Tcl script we need to source first (for encrypted IP)
            tcl_script = rapidwright_dcp.with_suffix('.tcl')
            if tcl_script.exists():
                print(f"Found Tcl script for encrypted IP: {tcl_script}")
                print(f"Note: This may take 10-30 minutes for large designs...")
                # Source the Tcl script instead of directly opening the DCP
                result = await self.call_vivado_tool("run_tcl", {
                    "command": f"source {{{tcl_script}}}"
                }, timeout=RAPIDWRIGHT_DCP_TIMEOUT)
                print(f"Source Tcl script result:\n{result}")
            else:
                # Opening a RapidWright-generated DCP can take longer than original
                # because Vivado needs to reconstruct some internal data structures
                result = await self.call_vivado_tool("open_checkpoint", {
                    "dcp_path": str(rapidwright_dcp)
                }, timeout=RAPIDWRIGHT_DCP_TIMEOUT)
                print(f"Open RapidWright DCP result:\n{result}")
            logger.info(f"Open RapidWright DCP: {result}")
            
            # ================================================================
            # Step 8: Route the design in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 8: Route design in Vivado")
            print("-"*60)
            
            # First check route status
            result = await self.call_vivado_tool("report_route_status", {
                "show_unrouted": True,
                "show_errors": True,
                "max_nets": 20
            }, timeout=300.0)
            print(f"Route status before routing:\n{result[:1500]}...")
            logger.info(f"Route status before routing: {result}")
            
            # Route the design
            result = await self.call_vivado_tool("route_design", {
                "directive": "Default",
            }, timeout=600.0)  # 2 hour timeout for routing
            print(f"Route design result:\n{result}")
            logger.info(f"Route design: {result}")
            
            # Check route status again
            result = await self.call_vivado_tool("report_route_status", {
                "show_unrouted": True,
                "show_errors": True,
                "max_nets": 20
            }, timeout=300.0)
            print(f"Route status after routing:\n{result[:1500]}...")
            logger.info(f"Route status after routing: {result}")
            
            # ================================================================
            # Step 9: Report final timing
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 9: Report final timing")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Final timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Final timing summary: {result}")
            
            # Get final WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                self.final_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Final", self.final_wns)
            logger.info(f"Final WNS: {self.final_wns} ns")
            print()
            
            # ================================================================
            # Write final DCP and report results
            # ================================================================
            self.print_wns_change(self.initial_wns, self.final_wns, self.clock_period)
            
            # Always write the final checkpoint (regardless of improvement)
            print(f"\nWriting final DCP to: {output_dcp}")
            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            print(f"Write final DCP result:\n{result}")
            
            # ================================================================
            # Summary
            # ================================================================
            elapsed = time.time() - overall_start
            self.print_test_summary(
                title="TEST SUMMARY",
                elapsed_seconds=elapsed,
                initial_wns=self.initial_wns,
                final_wns=self.final_wns,
                clock_period=self.clock_period,
                extra_info=f"Nets optimized: {successful_optimizations}/{len(nets_to_optimize)}"
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"Test failed with exception: {e}")
            print(f"\n*** TEST FAILED ***")
            print(f"Exception: {type(e).__name__}: {e}")
            return False
    
    async def run_test_logicnets(self, input_dcp: Path, output_dcp: Path) -> bool:
        """
        Run the pblock-based optimization flow for LogicNets designs.
        
        Steps:
        1. Open the input DCP in Vivado
        2. Report timing in Vivado (Initialize WNS)
        3. Run the Vivado tool extract_critical_path_cells
        4. Run the RapidWright tool analyze_critical_path_spread
        5. Use known-optimal pblock range for LogicNets (SLICE_X55Y60:SLICE_X111Y254)
        6. Unplace the design in Vivado
        7. Create and apply pblock to entire design
        8. Place the design in Vivado
        9. Route the design in Vivado
        10. Report timing in Vivado (compare against initial WNS)
        """
        pblock_ranges = "SLICE_X55Y60:SLICE_X111Y254"
        
        print("\n" + "="*70)
        print("FPGA OPTIMIZER TEST MODE - LOGICNETS PBLOCK FLOW")
        print("="*70)
        print(f"Input DCP:  {input_dcp}")
        print(f"Output DCP: {output_dcp}")
        print(f"Temp dir:   {self.temp_dir}")
        print("="*70 + "\n")
        
        overall_start = time.time()
        
        try:
            # ================================================================
            # Step 0: Initialize RapidWright (Vivado starts automatically)
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 0: Initialize RapidWright")
            print("-"*60)
            
            result = await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            print(f"RapidWright init result:\n{result[:500]}...")
            logger.info(f"RapidWright init result: {result}")
            
            # ================================================================
            # Step 1: Open the input DCP in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 1: Open input DCP in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"Open checkpoint result:\n{result}")
            logger.info(f"Open checkpoint result: {result}")
            
            # ================================================================
            # Step 2: Report timing in Vivado (Initialize WNS)
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 2: Report initial timing in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Initial timing summary: {result}")
            
            # Get clock period for fmax calculation (also detects target clock)
            self.clock_period = await self.fetch_clock_period()
            
            # Get WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                self.initial_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Initial", self.initial_wns)
            logger.info(f"Initial WNS: {self.initial_wns} ns")
            print()
            
            # ================================================================
            # Step 3: Extract critical path cells from Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 3: Extract critical path cells")
            print("-"*60)
            
            # Write to a file for efficient data transfer
            critical_paths_file = Path(self.temp_dir) / "critical_paths.json"
            result = await self.call_vivado_tool("extract_critical_path_cells", {
                "num_paths": 50,
                "output_file": str(critical_paths_file)
            }, timeout=600.0)
            print(f"Extract critical paths result:\n{result[:2000]}...")
            logger.info(f"Extract critical paths: {result}")
            
            # ================================================================
            # Step 4: Open DCP in RapidWright and analyze critical path spread
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 4: Analyze critical path spread in RapidWright")
            print("-"*60)
            
            # First, open the DCP in RapidWright
            result = await self.call_rapidwright_tool("read_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"RapidWright read checkpoint result:\n{result}")
            logger.info(f"RapidWright read checkpoint: {result}")
            
            # Analyze critical path spread
            result = await self.call_rapidwright_tool("analyze_critical_path_spread", {
                "input_file": str(critical_paths_file)
            }, timeout=300.0)
            print(f"Critical path spread analysis:\n{result[:3000] if isinstance(result, str) else str(result)[:3000]}...")
            logger.info(f"Critical path spread: {result}")
            
            # Parse the spread analysis result to check if pblock is recommended
            spread_result = result if isinstance(result, str) else str(result)
            pblock_recommended = "spread-out" in spread_result.lower() or "pblock" in spread_result.lower()
            print(f"\n*** Pblock optimization {'RECOMMENDED' if pblock_recommended else 'may not be needed'} ***")
            
            # ================================================================
            # Step 5: Apply pblock constraint for LogicNets
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 5: Apply pblock for LogicNets")
            print("-"*60)
            
            print(f"Using pblock range: {pblock_ranges}")
            
            # ================================================================
            # Step 6: Unplace the design in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 6: Unplace the design in Vivado")
            print("-"*60)
            
            # Use place_design -unplace to remove all placement
            result = await self.call_vivado_tool("run_tcl", {
                "command": "place_design -unplace"
            }, timeout=300.0)
            print(f"Unplace result:\n{result}")
            logger.info(f"Unplace result: {result}")
            
            # ================================================================
            # Step 7: Create and apply pblock to entire design
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 7: Create and apply pblock to entire design")
            print("-"*60)
            
            result = await self.call_vivado_tool("create_and_apply_pblock", {
                "pblock_name": "pblock_opt",
                "ranges": pblock_ranges,
                "apply_to": "current_design",  # Apply to entire design
                "is_soft": False  # Hard constraint
            }, timeout=300.0)
            print(f"Create and apply pblock result:\n{result}")
            logger.info(f"Create pblock result: {result}")
            
            # ================================================================
            # Step 8: Place the design in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 8: Place the design in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("place_design", {
                "directive": "Default"
            }, timeout=3600.0)  # 1 hour timeout for placement
            print(f"Place design result:\n{result}")
            logger.info(f"Place design: {result}")
            
            # ================================================================
            # Step 9: Route the design in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 9: Route the design in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("route_design", {
                "directive": "Default"
            }, timeout=3600.0)  # 1 hour timeout for routing
            print(f"Route design result:\n{result}")
            logger.info(f"Route design: {result}")
            
            # Check route status
            result = await self.call_vivado_tool("report_route_status", {}, timeout=300.0)
            print(f"Route status after routing:\n{result[:1500]}...")
            logger.info(f"Route status after routing: {result}")
            
            # ================================================================
            # Step 10: Report timing and compare WNS
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 10: Report final timing")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Final timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Final timing summary: {result}")
            
            # Get final WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                self.final_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Final", self.final_wns)
            logger.info(f"Final WNS: {self.final_wns} ns")
            print()
            
            # ================================================================
            # Write final DCP and report results
            # ================================================================
            self.print_wns_change(self.initial_wns, self.final_wns, self.clock_period)
            
            # Always write the final checkpoint
            print(f"\nWriting final DCP to: {output_dcp}")
            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            print(f"Write final DCP result:\n{result}")
            
            # ================================================================
            # Summary
            # ================================================================
            elapsed = time.time() - overall_start
            self.print_test_summary(
                title="TEST SUMMARY - LOGICNETS PBLOCK OPTIMIZATION",
                elapsed_seconds=elapsed,
                initial_wns=self.initial_wns,
                final_wns=self.final_wns,
                clock_period=self.clock_period,
                extra_info=f"Pblock applied: {pblock_ranges}"
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"LogicNets test failed with exception: {e}")
            print(f"\n*** TEST FAILED ***")
            print(f"Exception: {type(e).__name__}: {e}")
            return False

    async def run_test_vexriscv(self, input_dcp: Path, output_dcp: Path) -> bool:
        """
        Cell re-placement optimization flow for VexRiscv.
        
        Mirrors the script in docs/optimization_example.md:
          Step 1 — Vivado baseline (open, get Fmax, extract critical path pins)
          Step 2 — RapidWright analysis (analyze_net_detour, filter candidates)
          Step 3 — RapidWright optimization (optimize_cell_placement, write DCP)
          Step 4 — Vivado verification (open optimized DCP, route, measure Fmax)
        """
        overall_start = time.time()
        
        try:
            # ==============================================================
            # Step 1: Vivado baseline
            # ==============================================================
            print("=" * 60)
            print("Step 1  Vivado baseline")
            print("=" * 60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            logger.info(f"Open checkpoint result: {result}")
            
            self.clock_period = await self.fetch_clock_period()
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                ts = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
                self.initial_wns = self.parse_wns_from_timing_report(ts)
            
            baseline_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            print(f"  Clock period:   {self.clock_period} ns")
            print(f"  Baseline WNS:   {self.initial_wns} ns")
            if baseline_fmax is not None:
                print(f"  Baseline Fmax:  {baseline_fmax:.2f} MHz")
            
            pins_file = Path(self.temp_dir) / "critical_path_pins.json"
            result = await self.call_vivado_tool("extract_critical_path_pins", {
                "num_paths": 10,
                "output_file": str(pins_file)
            }, timeout=600.0)
            
            critical_paths = json.loads(Path(pins_file).read_text()) if pins_file.exists() else json.loads(result)
            print(f"  Extracted {len(critical_paths)} critical path pin lists")
            
            # ==============================================================
            # Step 2: RapidWright analysis
            # ==============================================================
            print("\n" + "=" * 60)
            print("Step 2  RapidWright analysis")
            print("=" * 60)
            
            result = await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            logger.info(f"RapidWright init: {result}")
            
            result = await self.call_rapidwright_tool("read_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            logger.info(f"RapidWright read checkpoint: {result}")
            
            result = await self.call_rapidwright_tool("analyze_net_detour", {
                "input_file": str(pins_file),
                "detour_threshold": 2.0
            }, timeout=300.0)
            logger.info(f"analyze_net_detour: {result}")
            
            analysis = json.loads(result) if isinstance(result, str) else result
            if "error" in analysis:
                raise RuntimeError(f"analyze_net_detour failed: {analysis['error']}")
            candidates = analysis.get("candidates", [])
            print(f"  Cells analyzed: {analysis.get('cells_analyzed', '?')}")
            print(f"  Candidates (detour > 2.0): {len(candidates)}")
            for c in candidates[:5]:
                print(f"    {str(c['cell']):55s}  ratio={c['max_detour_ratio']}")
            
            if not candidates:
                print("\n  No candidates found — nothing to optimize")
                self.final_wns = self.initial_wns
                return True
            
            worst_path_cells = list(set(
                str(c["cell"]) for c in candidates if c.get("path", 0) <= 2
            ))
            if not worst_path_cells:
                worst_path_cells = [str(candidates[0]["cell"])]
            
            print(f"\n  Targeting {len(worst_path_cells)} cells on paths 1-2:")
            for name in worst_path_cells:
                print(f"    {name}")
            
            # ==============================================================
            # Step 3: RapidWright optimization
            # ==============================================================
            print("\n" + "=" * 60)
            print("Step 3  RapidWright optimization")
            print("=" * 60)
            
            result = await self.call_rapidwright_tool("optimize_cell_placement", {
                "cell_names": worst_path_cells
            }, timeout=300.0)
            logger.info(f"optimize_cell_placement: {result}")
            
            opt_result = json.loads(result) if isinstance(result, str) else result
            for r in opt_result.get("results", []):
                print(f"  {r['cell']}: {r['status']} — {r['message']}")
            
            rw_output = Path(self.temp_dir) / "vexriscv_rw_optimized.dcp"
            result = await self.call_rapidwright_tool("write_checkpoint", {
                "dcp_path": str(rw_output)
            }, timeout=600.0)
            print(f"  Wrote {rw_output.name}")
            
            # ==============================================================
            # Step 4: Vivado verification
            # ==============================================================
            print("\n" + "=" * 60)
            print("Step 4  Vivado verification")
            print("=" * 60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(rw_output)
            }, timeout=600.0)
            logger.info(f"Open optimized checkpoint: {result}")
            
            result = await self.call_vivado_tool("route_design", {
                "directive": "Default"
            }, timeout=3600.0)
            logger.info(f"Route design: {result}")
            
            route_result = await self.call_vivado_tool("report_route_status", {}, timeout=300.0)
            error_match = re.search(r"# of nets with routing errors.*?:\s+(\d+)", route_result)
            error_count = int(error_match.group(1)) if error_match else -1
            
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                ts = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
                self.final_wns = self.parse_wns_from_timing_report(ts)
            
            new_fmax = self.calculate_fmax(self.final_wns, self.clock_period)
            
            print(f"  Routing errors:  {error_count}")
            if baseline_fmax is not None and new_fmax is not None:
                print(f"  Baseline WNS:    {self.initial_wns} ns  →  Fmax {baseline_fmax:.2f} MHz")
                print(f"  Optimized WNS:   {self.final_wns} ns  →  Fmax {new_fmax:.2f} MHz")
                delta = new_fmax - baseline_fmax
                print(f"  Fmax improvement: {delta:+.2f} MHz")
            else:
                print(f"  Baseline WNS:  {self.initial_wns} ns")
                print(f"  Optimized WNS: {self.final_wns} ns")
            
            # Write final DCP
            print(f"\nWriting final DCP to: {output_dcp}")
            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            
            # Summary
            elapsed = time.time() - overall_start
            cells_info = ", ".join(worst_path_cells)
            self.print_test_summary(
                title="TEST SUMMARY - VEXRISCV CELL RE-PLACEMENT",
                elapsed_seconds=elapsed,
                initial_wns=self.initial_wns,
                final_wns=self.final_wns,
                clock_period=self.clock_period,
                extra_info=f"Cells re-placed: {cells_info}"
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"VexRiscv test failed with exception: {e}")
            print(f"\n*** TEST FAILED ***")
            print(f"Exception: {type(e).__name__}: {e}")
            return False

    async def cleanup(self):
        """Clean up resources."""
        print("\n[TEST] Cleaning up...")
        await super().cleanup()
        print(f"[TEST] Run directory preserved at: {self.run_dir}")


