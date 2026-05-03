#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

"""
Command-line entry point + thin per-mode wrappers.

The thin module-level wrappers (``run_test_mode``, ``run_baseline_mode``,
``run_pblock_mode``, ``run_replace_mode``) exist to keep the corresponding
CLI flags (``--test``, ``--baseline``, ``--strategy {pblock,replace}``)
working as before. They delegate to the strategy methods on
:class:`FPGAOptimizerTest`. The dispatcher is the alpha-submission default
path; LLM-guided full optimization stays opt-in via ``--llm``.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .base import DEFAULT_MODEL
from .dispatcher import run_dispatcher_mode
from .llm_optimizer import DCPOptimizer
from .strategies import FPGAOptimizerTest

logger = logging.getLogger(__name__)


async def run_test_mode(input_dcp: Path, output_dcp: Path, debug: bool = False, max_nets: int = 5, run_dir: Optional[Path] = None):
    """Run the test mode optimization.
    
    Detects which example DCP is being used and applies the appropriate optimization flow:
    - logicnets_jscl: Pblock-based placement optimization flow
    - vexriscv_re-place: Cell re-placement flow (same recipe as docs/optimization_example.md)
    """
    # Detect which DCP is being used based on filename
    dcp_name = input_dcp.name.lower()
    
    if "logicnets" in dcp_name:
        design_type = "logicnets"
        print(f"[TEST] Detected LogicNets design - using pblock optimization flow")
    elif "vexriscv" in dcp_name:
        design_type = "vexriscv"
        print(f"[TEST] Detected VexRiscv design - using cell re-placement flow")
    else:
        print(f"\n[TEST] ERROR: Unsupported DCP file: {input_dcp.name}")
        print(f"[TEST] Test mode supports these benchmark DCPs:")
        print(f"[TEST]   - fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp")
        print(f"[TEST]   - fpl26_contest_benchmarks/vexriscv_re-place_2025.1.dcp")
        print(f"[TEST]")
        print(f"[TEST] For custom DCPs, run without --test to use the LLM-guided optimizer.")
        return 1
    
    tester = FPGAOptimizerTest(debug=debug, run_dir=run_dir)
    
    try:
        await tester.start_servers()
        
        if design_type == "logicnets":
            success = await tester.run_test_logicnets(input_dcp, output_dcp)
        else:
            success = await tester.run_test_vexriscv(input_dcp, output_dcp)
        
        if success:
            print("\n[TEST] Test completed successfully")
            print(f"\n[TEST] Output files:")
            print(f"[TEST]   Optimized DCP: {output_dcp}")
            print(f"[TEST]   Run directory: {tester.run_dir}")
            return 0
        else:
            print("\n[TEST] Test failed")
            print(f"[TEST] Run directory: {tester.run_dir}")
            return 1
            
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
        print(f"[TEST] Run directory: {tester.run_dir}")
        return 130
    except Exception as e:
        logger.exception(f"Test mode fatal error: {e}")
        print(f"\n[TEST] Fatal error: {e}")
        print(f"[TEST] Run directory: {tester.run_dir}")
        return 1
    finally:
        await tester.cleanup()


async def run_baseline_mode(
    input_dcp: Path,
    output_dcp: Path,
    debug: bool = False,
    max_nets: int = 2,
    run_dir: Optional[Path] = None,
    phys_opt_directive: str = "RuntimeOptimized",
):
    """Run the benchmark-agnostic deterministic fallback baseline."""
    tester = FPGAOptimizerTest(debug=debug, run_dir=run_dir)

    try:
        await tester.start_servers()
        success = await tester.run_deterministic_baseline(
            input_dcp,
            output_dcp,
            max_nets_to_optimize=max_nets,
            phys_opt_directive=phys_opt_directive,
        )

        if success:
            print("\n[BASELINE] Deterministic baseline completed successfully")
            print(f"\n[BASELINE] Output files:")
            print(f"[BASELINE]   Optimized DCP: {output_dcp}")
            print(f"[BASELINE]   Run directory: {tester.run_dir}")
            return 0

        print("\n[BASELINE] Deterministic baseline failed")
        print(f"[BASELINE] Run directory: {tester.run_dir}")
        return 1

    except KeyboardInterrupt:
        print("\n[BASELINE] Interrupted by user")
        print(f"[BASELINE] Run directory: {tester.run_dir}")
        return 130
    except Exception as e:
        logger.exception(f"Deterministic baseline fatal error: {e}")
        print(f"\n[BASELINE] Fatal error: {e}")
        print(f"[BASELINE] Run directory: {tester.run_dir}")
        return 1
    finally:
        await tester.cleanup()


async def run_pblock_mode(
    input_dcp: Path,
    output_dcp: Path,
    debug: bool = False,
    run_dir: Optional[Path] = None,
    pblock_ranges: Optional[str] = None,
    util_factor: float = 1.5,
):
    """Run the pblock-based area-constraint strategy in isolation."""
    tester = FPGAOptimizerTest(debug=debug, run_dir=run_dir)

    try:
        await tester.start_servers()
        stats = await tester.try_pblock(input_dcp, output_dcp, pblock_ranges=pblock_ranges, util_factor=util_factor)

        for note in stats["notes"]:
            print(f"[PBLOCK] {note}")

        if stats["success"]:
            print("\n[PBLOCK] Pblock optimization completed successfully")
            print(f"[PBLOCK]   Initial WNS:    {stats['initial_wns']}")
            print(f"[PBLOCK]   Final WNS:      {stats['final_wns']}")
            print(f"[PBLOCK]   Pblock ranges:  {stats['pblock_ranges']}")
            print(f"[PBLOCK]   Routing errors: {stats['routing_errors']}")
            print(f"[PBLOCK]   Output DCP:     {output_dcp}")
            print(f"[PBLOCK]   Run directory:  {tester.run_dir}")
            return 0

        print("\n[PBLOCK] Pblock optimization failed")
        print(f"[PBLOCK] Run directory: {tester.run_dir}")
        return 1

    except KeyboardInterrupt:
        print("\n[PBLOCK] Interrupted by user")
        print(f"[PBLOCK] Run directory: {tester.run_dir}")
        return 130
    except Exception as e:
        logger.exception(f"Pblock mode fatal error: {e}")
        print(f"\n[PBLOCK] Fatal error: {e}")
        print(f"[PBLOCK] Run directory: {tester.run_dir}")
        return 1
    finally:
        await tester.cleanup()


async def run_replace_mode(
    input_dcp: Path,
    output_dcp: Path,
    debug: bool = False,
    run_dir: Optional[Path] = None,
):
    """Run the cell re-placement strategy in isolation (RapidWright detour + centroid + reroute)."""
    tester = FPGAOptimizerTest(debug=debug, run_dir=run_dir)

    try:
        await tester.start_servers()
        stats = await tester.try_cell_replacement(input_dcp, output_dcp)

        for note in stats["notes"]:
            print(f"[REPLACE] {note}")

        if stats["success"]:
            initial = stats["initial_wns"]
            final = stats["final_wns"]
            cells = stats["cells_replaced"]
            errors = stats["routing_errors"]
            print("\n[REPLACE] Cell re-placement completed successfully")
            print(f"[REPLACE]   Initial WNS:    {initial}")
            print(f"[REPLACE]   Final WNS:      {final}")
            print(f"[REPLACE]   Cells replaced: {len(cells)}")
            print(f"[REPLACE]   Routing errors: {errors}")
            print(f"[REPLACE]   Output DCP:     {output_dcp}")
            print(f"[REPLACE]   Run directory:  {tester.run_dir}")
            return 0

        print("\n[REPLACE] Cell re-placement failed")
        print(f"[REPLACE] Run directory: {tester.run_dir}")
        return 1

    except KeyboardInterrupt:
        print("\n[REPLACE] Interrupted by user")
        print(f"[REPLACE] Run directory: {tester.run_dir}")
        return 130
    except Exception as e:
        logger.exception(f"Replace mode fatal error: {e}")
        print(f"\n[REPLACE] Fatal error: {e}")
        print(f"[REPLACE] Run directory: {tester.run_dir}")
        return 1
    finally:
        await tester.cleanup()


async def main():
    parser = argparse.ArgumentParser(
        description="FPGA Design Optimization Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dcp_optimizer.py input.dcp
  python dcp_optimizer.py input.dcp --output output.dcp
  python dcp_optimizer.py input.dcp --baseline
  python dcp_optimizer.py input.dcp --model anthropic/claude-sonnet-4
  python dcp_optimizer.py input.dcp --debug
  python dcp_optimizer.py fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp --test
  python dcp_optimizer.py fpl26_contest_benchmarks/vexriscv_re-place_2025.1.dcp --test
        """
    )
    parser.add_argument("input_dcp", type=Path, help="Input design checkpoint (.dcp)")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        dest="output_dcp",
        help="Output optimized checkpoint (.dcp). Default: <input_name>_optimized-<timestamp>.dcp in same directory as input"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (default: OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging, save intermediate checkpoints)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: run without LLM. Pblock for LogicNets, cell re-placement for VexRiscv (see docs/optimization_example.md)."
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Deterministic fallback mode: run a benchmark-agnostic non-LLM baseline using limited fanout optimization and phys_opt."
    )
    parser.add_argument(
        "--strategy",
        choices=["replace", "pblock"],
        default=None,
        help="Run a single benchmark-agnostic strategy in isolation (no LLM). "
             "'replace' = RapidWright cell re-placement (detour analysis + centroid placement + reroute). "
             "'pblock' = Vivado pblock area constraint (auto-derived ranges via RW analyze_fabric_for_pblock; "
             "override with --pblock-ranges)."
    )
    parser.add_argument(
        "--pblock-ranges",
        default=None,
        help="Optional pblock range string (e.g. 'SLICE_X55Y60:SLICE_X111Y254') for --strategy pblock. "
             "If omitted, ranges are auto-derived from utilization + RW fabric analysis."
    )
    parser.add_argument(
        "--pblock-util-factor",
        type=float,
        default=1.5,
        help="Multiplier applied to current utilization to size the pblock (default: 1.5). "
             "Larger values give the placer more room (often helps LUT-dominant designs); "
             "ignored when --pblock-ranges is supplied."
    )
    parser.add_argument(
        "--phys-opt-directive",
        default="RuntimeOptimized",
        help="phys_opt_design -directive used by the baseline mode's STEP 5 fallback "
             "(e.g. RuntimeOptimized, Default, Explore, AggressiveExplore, AggressiveFanoutOpt). "
             "Default: RuntimeOptimized."
    )
    parser.add_argument(
        "--max-nets",
        type=int,
        default=5,
        help="Maximum number of high fanout nets to optimize in non-LLM modes (default: 5)"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Directory for logs, journals, and intermediate files. Default: dcp_optimizer_run-<timestamp> in current directory"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Run the LLM-guided optimizer (requires OPENROUTER_API_KEY). "
             "Without this flag the default mode is the deterministic dispatcher "
             "(analyze design -> pick pblock or fanout phys_opt -> fallback to RuntimeOptimized phys_opt)."
    )
    parser.add_argument(
        "--dispatcher",
        action="store_true",
        help="Force dispatcher mode (this is also the default when no other mode flag is given)."
    )
    parser.add_argument(
        "--dispatcher-fanout-directive",
        default="AggressiveFanoutOpt",
        help="phys_opt -directive the dispatcher uses for the fanout primary path "
             "(default: AggressiveFanoutOpt; matrix sweep winner on small / fanout-heavy designs)."
    )
    parser.add_argument(
        "--dispatcher-small-lut-threshold",
        type=int,
        default=5000,
        help="Used-LUT cutoff below which the dispatcher prefers the fanout phys_opt strategy "
             "over pblock (default: 5000). Designs above the cutoff default to pblock."
    )
    parser.add_argument(
        "--llm-budget",
        type=int,
        default=None,
        help="Max LLM calls per dispatcher run for the pblock-ranges retry hook. "
             "Default: 1 if OPENROUTER_API_KEY is set, else 0. Set to 0 to disable LLM "
             "augment even when an API key is available."
    )
    parser.add_argument(
        "--llm-weak-threshold-mhz",
        type=float,
        default=25.0,
        help="When pblock(auto) yields delta_fmax below this threshold (MHz), the "
             "dispatcher spends its single LLM call to propose alternative ranges. "
             "Default: 25.0. Lowering it makes the LLM trigger more cautious; "
             "raising it makes the LLM trigger more aggressive."
    )
    parser.add_argument(
        "--llm-mock-ranges",
        default=None,
        help="Test-only: skip the real OpenRouter call and treat this string as the "
             "LLM-proposed pblock_ranges. Used to verify the retry-pblock integration "
             "path end-to-end without burning API credit."
    )

    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_dcp.exists():
        print(f"Error: Input file not found: {args.input_dcp}", file=sys.stderr)
        sys.exit(1)
    
    # Generate default output DCP name if not provided
    if args.output_dcp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        input_stem = args.input_dcp.stem  # Filename without extension
        input_dir = args.input_dcp.parent  # Directory of input file
        args.output_dcp = input_dir / f"{input_stem}_optimized-{timestamp}.dcp"
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory if needed
    args.output_dcp.parent.mkdir(parents=True, exist_ok=True)

    # Normalize run-dir to absolute so that MCP server child processes
    # (whose cwd may be set to run_dir) can still resolve any paths we pass.
    if args.run_dir is not None:
        args.run_dir = args.run_dir.resolve()
    
    # Test mode - run without LLM
    if args.test:
        if args.run_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = Path.cwd() / f"dcp_optimizer_run-{timestamp}"
        else:
            run_dir = args.run_dir
        
        print(f"FPGA Design Optimization - TEST MODE")
        print(f"=====================================")
        print(f"Input:       {args.input_dcp.resolve()}")
        print(f"Output:      {args.output_dcp.resolve()}")
        print(f"Run dir:     {run_dir.resolve()}")
        print(f"Max nets to optimize: {args.max_nets}")
        print()
        
        exit_code = await run_test_mode(
            args.input_dcp, 
            args.output_dcp, 
            debug=args.debug,
            max_nets=args.max_nets,
            run_dir=run_dir
        )
        sys.exit(exit_code)

    # Deterministic baseline mode - benchmark agnostic, no LLM required
    if args.baseline:
        if args.run_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = Path.cwd() / f"dcp_optimizer_run-{timestamp}"
        else:
            run_dir = args.run_dir

        print(f"FPGA Design Optimization - DETERMINISTIC BASELINE MODE")
        print(f"========================================================")
        print(f"Input:       {args.input_dcp.resolve()}")
        print(f"Output:      {args.output_dcp.resolve()}")
        print(f"Run dir:     {run_dir}")
        print(f"Max nets to optimize: {max(0, min(args.max_nets, 2))}")
        print()

        exit_code = await run_baseline_mode(
            args.input_dcp,
            args.output_dcp,
            debug=args.debug,
            max_nets=args.max_nets,
            run_dir=run_dir,
            phys_opt_directive=args.phys_opt_directive,
        )
        sys.exit(exit_code)

    # Single-strategy isolation mode (no LLM): run one named strategy end-to-end
    if args.strategy:
        if args.run_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = Path.cwd() / f"dcp_optimizer_run-{timestamp}"
        else:
            run_dir = args.run_dir

        print(f"FPGA Design Optimization - STRATEGY MODE ({args.strategy})")
        print(f"========================================================")
        print(f"Input:       {args.input_dcp.resolve()}")
        print(f"Output:      {args.output_dcp.resolve()}")
        print(f"Run dir:     {run_dir}")
        print()

        if args.strategy == "replace":
            exit_code = await run_replace_mode(
                args.input_dcp,
                args.output_dcp,
                debug=args.debug,
                run_dir=run_dir,
            )
        elif args.strategy == "pblock":
            exit_code = await run_pblock_mode(
                args.input_dcp,
                args.output_dcp,
                debug=args.debug,
                run_dir=run_dir,
                pblock_ranges=args.pblock_ranges,
                util_factor=args.pblock_util_factor,
            )
        else:
            print(f"Error: unknown strategy '{args.strategy}'", file=sys.stderr)
            exit_code = 2
        sys.exit(exit_code)

    # Default mode = deterministic dispatcher. The LLM-guided path is opt-in
    # via --llm to keep alpha submissions deterministic and free.
    if args.dispatcher or not args.llm:
        if args.run_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = Path.cwd() / f"dcp_optimizer_run-{timestamp}"
        else:
            run_dir = args.run_dir

        # LLM augment auto-enables when OPENROUTER_API_KEY is set (alpha
        # submission environment will set this). Defaults to 1 call/run; set
        # --llm-budget 0 to opt out, or pass --llm-budget N to allow more.
        # --llm-mock-ranges short-circuits the real API call (test-only).
        if args.llm_mock_ranges is not None:
            llm_budget_effective = max(args.llm_budget or 1, 1)
            llm_api_key_effective = None
        else:
            llm_budget_effective = (
                args.llm_budget
                if args.llm_budget is not None
                else (1 if args.api_key else 0)
            )
            llm_api_key_effective = args.api_key if llm_budget_effective > 0 else None

        print(f"FPGA Design Optimization - DISPATCHER MODE")
        print(f"========================================================")
        print(f"Input:       {args.input_dcp.resolve()}")
        print(f"Output:      {args.output_dcp.resolve()}")
        print(f"Run dir:     {run_dir}")
        print(f"Util factor: {args.pblock_util_factor}")
        print(f"Small-LUT threshold: {args.dispatcher_small_lut_threshold}")
        print(f"Fanout phys_opt directive: {args.dispatcher_fanout_directive}")
        if llm_budget_effective > 0:
            print(
                f"LLM augment: ENABLED (budget={llm_budget_effective}, "
                f"weak<{args.llm_weak_threshold_mhz} MHz, model={args.model})"
            )
        else:
            print(
                "LLM augment: disabled "
                "(set OPENROUTER_API_KEY or --api-key to enable, --llm-budget 0 to force off)"
            )
        print()

        exit_code = await run_dispatcher_mode(
            args.input_dcp,
            args.output_dcp,
            debug=args.debug,
            run_dir=run_dir,
            util_factor=args.pblock_util_factor,
            small_lut_threshold=args.dispatcher_small_lut_threshold,
            fanout_phys_opt_directive=args.dispatcher_fanout_directive,
            llm_api_key=llm_api_key_effective,
            llm_model=args.model,
            llm_budget=llm_budget_effective,
            llm_weak_threshold_mhz=args.llm_weak_threshold_mhz,
            llm_mock_ranges=args.llm_mock_ranges,
        )
        sys.exit(exit_code)

    # LLM-guided mode (opt-in via --llm)
    if not args.api_key:
        print("Error: OpenRouter API key required. Set OPENROUTER_API_KEY or use --api-key", file=sys.stderr)
        print("       Use the default dispatcher mode (drop --llm) or --baseline / --test to run without LLM", file=sys.stderr)
        sys.exit(1)

    if OpenAI is None:
        print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)
    
    if args.run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path.cwd() / f"dcp_optimizer_run-{timestamp}"
    else:
        run_dir = args.run_dir
    
    print(f"FPGA Design Optimization Agent")
    print(f"================================")
    print(f"Input:       {args.input_dcp.resolve()}")
    print(f"Output:      {args.output_dcp.resolve()}")
    print(f"Run dir:     {run_dir.resolve()}")
    print(f"Model:       {args.model}")
    print()
    
    optimizer = DCPOptimizer(
        api_key=args.api_key,
        model=args.model,
        debug=args.debug,
        run_dir=run_dir
    )
    
    try:
        await optimizer.start_servers()
        success = await optimizer.optimize(args.input_dcp, args.output_dcp)
        
        if success:
            print("\n✓ Optimization completed successfully")
            print(f"\nOutput files:")
            print(f"  Optimized DCP: {args.output_dcp}")
            print(f"  Run directory: {run_dir}")
            sys.exit(0)
        else:
            print("\n✗ Optimization did not complete successfully")
            print(f"\nRun directory: {run_dir}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        print(f"Run directory: {run_dir}")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"Run directory: {run_dir}")
        sys.exit(1)
    finally:
        await optimizer.cleanup()

