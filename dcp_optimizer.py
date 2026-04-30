#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

"""
FPGA Design Optimization Agent

An autonomous AI agent that analyzes FPGA designs and applies optimizations
using RapidWright and Vivado via MCP servers.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Default model
DEFAULT_MODEL = "x-ai/grok-4.1-fast"


def parse_timing_summary_static(timing_report: str) -> dict:
    """
    Parse timing summary report to extract WNS, TNS, and failing endpoints.
    Returns dict with keys: wns, tns, failing_endpoints
    
    Parses the Design Timing Summary table:
        WNS(ns)      TNS(ns)  TNS Failing Endpoints  ...
        -------      -------  ---------------------  ...
         -0.099       -1.449                     42  ...
    
    This is a shared utility function used by both FPGAOptimizer and FPGAOptimizerTest.
    """
    result = {
        "wns": None,
        "tns": None,
        "failing_endpoints": None
    }
    
    lines = timing_report.split('\n')
    
    # Find the line with "WNS(ns)" header
    header_idx = -1
    for i, line in enumerate(lines):
        if 'WNS(ns)' in line and 'TNS(ns)' in line:
            header_idx = i
            break
    
    if header_idx == -1:
        return result
    
    # The data line should be 2 lines after the header (skipping the dashes line)
    # Format: whitespace + values separated by whitespace
    data_idx = header_idx + 2
    if data_idx >= len(lines):
        return result
    
    data_line = lines[data_idx].strip()
    if not data_line:
        return result
    
    # Split by whitespace and extract first 3 values: WNS, TNS, TNS Failing Endpoints
    parts = data_line.split()
    if len(parts) >= 3:
        try:
            result["wns"] = float(parts[0])
            result["tns"] = float(parts[1])
            result["failing_endpoints"] = int(parts[2])
        except (ValueError, IndexError):
            # If parsing fails, leave as None
            pass
    
    return result


def load_system_prompt() -> str:
    """Load system prompt from SYSTEM_PROMPT.TXT file."""
    script_dir = Path(__file__).parent.resolve()
    prompt_file = script_dir / "SYSTEM_PROMPT.TXT"
    
    try:
        with open(prompt_file, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"System prompt file not found: {prompt_file}")
        raise
    except Exception as e:
        logger.error(f"Failed to load system prompt: {e}")
        raise


def convert_mcp_tool_to_openai(tool, server_prefix: str) -> dict:
    """Convert MCP tool definition to OpenAI-compatible format with server prefix."""
    schema = tool.inputSchema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": f"{server_prefix}_{tool.name}",
            "description": tool.description or "",
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }
    }


class DCPOptimizerBase:
    """Base class with shared functionality for FPGA optimization."""
    
    def __init__(self, debug: bool = False, run_dir: Optional[Path] = None):
        self.debug = debug
        
        # Create run directory if not provided
        if run_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path.cwd() / f"dcp_optimizer_run-{timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created run directory: {self.run_dir}")
        else:
            self.run_dir = run_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.exit_stack = AsyncExitStack()
        self.rapidwright_session: Optional[ClientSession] = None
        self.vivado_session: Optional[ClientSession] = None
        
        # Use run directory for all temporary files
        self.temp_dir = self.run_dir
        logger.info(f"Working directory: {self.temp_dir}")
        
        # Timing tracking
        self.initial_wns = None
        self.initial_tns = None
        self.initial_failing_endpoints = None
        self.high_fanout_nets = []
        self.clock_period = None
        self.target_clock = None  # Set to clock name (e.g. "clk_fpl26contest") for clock-specific Fmax
        
        # Log file handles
        self._rw_log_file = None
        self._v_log_file = None
    
    async def start_servers(self, log_prefix: str = ""):
        """Start and connect to both MCP servers."""
        script_dir = Path(__file__).parent.resolve()
        
        # Create log files in run directory
        rapidwright_log = self.run_dir / "rapidwright.log"
        rapidwright_mcp_log = self.run_dir / "rapidwright-mcp.log"
        vivado_log = self.run_dir / "vivado.log"
        vivado_journal = self.run_dir / "vivado.jou"
        vivado_mcp_log = self.run_dir / "vivado-mcp.log"
        
        # Open log files (if not in debug mode, redirect stderr to log)
        if self.debug:
            self._rw_log_file = None
            self._v_log_file = None
            logger.info("Debug mode: MCP server output will be shown in console")
            if log_prefix:
                print(f"{log_prefix} Debug mode: MCP server output will be shown in console")
        else:
            self._rw_log_file = open(rapidwright_mcp_log, 'w')
            self._v_log_file = open(vivado_mcp_log, 'w')
            logger.info(f"RapidWright Java output: {rapidwright_log}")
            logger.info(f"RapidWright MCP output: {rapidwright_mcp_log}")
            logger.info(f"Vivado output: {vivado_log}")
            logger.info(f"Vivado journal: {vivado_journal}")
            logger.info(f"Vivado MCP output: {vivado_mcp_log}")
            print(f"Log files in {self.run_dir.name}/: {rapidwright_log.name}, {rapidwright_mcp_log.name}, {vivado_log.name}, {vivado_journal.name}, {vivado_mcp_log.name}")
        
        # RapidWright MCP server config
        rapidwright_args = [str(script_dir / "RapidWrightMCP" / "server.py")]
        if not self.debug:
            rapidwright_args.extend([
                "--java-log", str(rapidwright_log),
                "--mcp-log", str(rapidwright_mcp_log)
            ])
        
        env = {**os.environ}
        rapidwright_submodule = script_dir / "RapidWright"
        if rapidwright_submodule.is_dir() and "RAPIDWRIGHT_PATH" not in env:
            env["RAPIDWRIGHT_PATH"] = str(rapidwright_submodule)
            env["CLASSPATH"] = f"{rapidwright_submodule}/bin:{rapidwright_submodule}/jars/*"
        
        rapidwright_config = {
            "command": sys.executable,
            "args": rapidwright_args,
            "cwd": str(self.run_dir),
            "env": env
        }
        
        # Vivado MCP server config
        vivado_args = [str(script_dir / "VivadoMCP" / "vivado_mcp_server.py")]
        if not self.debug:
            vivado_args.extend([
                "--vivado-log", str(vivado_log),
                "--vivado-journal", str(vivado_journal)
            ])
        
        vivado_config = {
            "command": sys.executable,
            "args": vivado_args,
            "cwd": str(self.run_dir),
            "env": {**os.environ}
        }
        
        # Start RapidWright MCP
        logger.info("Starting RapidWright MCP server...")
        if log_prefix:
            print(f"{log_prefix} Starting RapidWright MCP server...")
        start_time = time.time()
        
        rw_params = StdioServerParameters(**rapidwright_config)
        rw_transport = await self.exit_stack.enter_async_context(
            stdio_client(rw_params, errlog=self._rw_log_file)
        )
        rw_read, rw_write = rw_transport
        self.rapidwright_session = await self.exit_stack.enter_async_context(
            ClientSession(rw_read, rw_write)
        )
        await self.rapidwright_session.initialize()
        
        elapsed = time.time() - start_time
        logger.info(f"RapidWright MCP server started in {elapsed:.2f}s")
        if log_prefix:
            print(f"{log_prefix} RapidWright MCP server started in {elapsed:.2f}s")
        
        # Start Vivado MCP
        logger.info("Starting Vivado MCP server...")
        if log_prefix:
            print(f"{log_prefix} Starting Vivado MCP server...")
        start_time = time.time()
        
        vivado_params = StdioServerParameters(**vivado_config)
        vivado_transport = await self.exit_stack.enter_async_context(
            stdio_client(vivado_params, errlog=self._v_log_file)
        )
        v_read, v_write = vivado_transport
        self.vivado_session = await self.exit_stack.enter_async_context(
            ClientSession(v_read, v_write)
        )
        await self.vivado_session.initialize()
        
        elapsed = time.time() - start_time
        logger.info(f"Vivado MCP server started in {elapsed:.2f}s")
        if log_prefix:
            print(f"{log_prefix} Vivado MCP server started in {elapsed:.2f}s")
        
        logger.info("Both MCP servers connected")
        if log_prefix:
            print(f"{log_prefix} Both MCP servers connected successfully")
    
    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()
        
        if self._rw_log_file:
            self._rw_log_file.close()
        if self._v_log_file:
            self._v_log_file.close()
        
        logger.info(f"Run directory preserved at: {self.run_dir}")
    
    def calculate_fmax(self, wns: Optional[float], clock_period: Optional[float]) -> Optional[float]:
        """
        Calculate achievable fmax in MHz based on WNS and clock period.
        
        fmax = 1 / (clock_period - WNS) when WNS < 0 (timing violation)
        fmax = 1 / clock_period when WNS >= 0 (timing met)
        
        Returns fmax in MHz, or None if cannot be calculated.
        """
        if clock_period is None or clock_period <= 0:
            return None
        if wns is None:
            return None
        
        achievable_period_ns = clock_period - wns
        if achievable_period_ns <= 0:
            return None
        
        return 1000.0 / achievable_period_ns
    
    async def get_clock_period(self, call_tool_fn) -> Optional[float]:
        """
        Query the clock period of the target clock from Vivado in nanoseconds.
        
        First checks for the contest clock 'clk_fpl26contest'. If found, uses its
        period and sets self.target_clock. Otherwise falls back to the endpoint clock
        of the worst setup timing path.
        
        Args:
            call_tool_fn: Function to call Vivado tools, should accept (tool_name, arguments)
        
        Returns the period of the target clock, or None if no clocks found.
        """
        tcl_cmd = (
            "set contest_clk [get_clocks -quiet clk_fpl26contest]; "
            "if {$contest_clk ne {}} { "
            "  puts \"CLOCK:clk_fpl26contest\"; "
            "  puts [get_property PERIOD $contest_clk]; "
            "} else { "
            "  set tp [get_timing_paths -max_paths 1 -setup]; "
            "  if {$tp ne {}} { "
            "    set clk [get_property ENDPOINT_CLOCK $tp]; "
            "    if {$clk ne {}} { "
            "      puts \"CLOCK:$clk\"; "
            "      puts [get_property PERIOD [get_clocks $clk]]; "
            "    } "
            "  } "
            "}"
        )
        try:
            result = await call_tool_fn("run_tcl", {"command": tcl_cmd})
            
            clock_name = None
            for token in result.strip().split():
                if token.startswith('CLOCK:'):
                    clock_name = token[len('CLOCK:'):]
                    continue
                if token.startswith('ERROR') or token.startswith('WARNING'):
                    continue
                try:
                    period = float(token)
                    if period > 0:
                        if clock_name:
                            self.target_clock = clock_name
                            logger.info(f"Target clock: {clock_name}, period: {period:.3f} ns")
                        else:
                            logger.info(f"Critical clock period: {period:.3f} ns")
                        return period
                except ValueError:
                    continue
        except Exception as e:
            logger.warning(f"Failed to get clock period: {e}")
        
        logger.warning("Could not determine clock period from Vivado")
        return None
    
    async def get_wns_for_target_clock(self, call_tool_fn) -> Optional[float]:
        """
        Get WNS specifically for the target clock domain.
        
        When target_clock is set (e.g. 'clk_fpl26contest'), queries WNS filtered
        to that clock's timing paths. Falls back to overall WNS if no target clock.
        
        Args:
            call_tool_fn: Function to call Vivado tools, should accept (tool_name, arguments)
        
        Returns WNS in nanoseconds, or None if query fails.
        """
        if self.target_clock:
            tcl_cmd = (
                f"set clk_obj [get_clocks -quiet {{{self.target_clock}}}]; "
                f"if {{$clk_obj ne {{}}}} {{ "
                f"  set tp [get_timing_paths -max_paths 1 -setup -to $clk_obj]; "
                f"  if {{[llength $tp] > 0}} {{get_property SLACK $tp}} else {{puts 0.0}} "
                f"}} else {{ "
                f"  set tp [get_timing_paths -max_paths 1 -slack_lesser_than 999]; "
                f"  if {{[llength $tp] > 0}} {{get_property SLACK $tp}} else {{puts 0.0}} "
                f"}}"
            )
        else:
            tcl_cmd = (
                "set tp [get_timing_paths -max_paths 1 -slack_lesser_than 999]; "
                "if {[llength $tp] > 0} {get_property SLACK $tp} else {puts 0.0}"
            )
        
        try:
            result = await call_tool_fn("run_tcl", {"command": tcl_cmd})
            for token in result.strip().split('\n'):
                token = token.strip()
                if not token or token.startswith('ERROR') or token.startswith('WARNING'):
                    continue
                try:
                    wns = float(token)
                    clock_info = f" (clock: {self.target_clock})" if self.target_clock else ""
                    logger.info(f"WNS{clock_info}: {wns:.3f} ns")
                    return wns
                except ValueError:
                    continue
        except Exception as e:
            logger.warning(f"Failed to get WNS for target clock: {e}")
        
        return None
    
    def parse_high_fanout_nets(self, report: str) -> list[tuple[str, int, int]]:
        """
        Parse high fanout nets report and return list of (net_name, fanout, path_count).
        """
        nets = []
        lines = report.split('\n')
        in_net_section = False
        
        for line in lines:
            if 'Paths' in line and 'Fanout' in line and 'Parent Net Name' in line:
                in_net_section = True
                continue
            
            if in_net_section:
                if line.startswith('---') or not line.strip():
                    continue
                if line.startswith('==='):
                    break
                
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        path_count = int(parts[0])
                        fanout = int(parts[1])
                        net_name = parts[2]
                        
                        if (net_name and 
                            '/' in net_name and
                            not net_name.startswith('get_') and
                            not net_name.startswith('ERROR') and
                            not net_name.startswith('WARNING')):
                            nets.append((net_name, fanout, path_count))
                    except ValueError:
                        continue
        
        return nets

    def _format_fmax_results(
        self,
        clock_period: Optional[float],
        initial_wns: Optional[float],
        result_wns: Optional[float],
        result_label: str = "Final",
    ) -> list[str]:
        """Format Fmax/WNS results block as a list of lines.
        
        """
        initial_fmax = self.calculate_fmax(initial_wns, clock_period)
        result_fmax = self.calculate_fmax(result_wns, clock_period)
        result_fmax_label = f"{result_label} Fmax:"
        result_wns_label = f"{result_label} WNS:"
        
        lines: list[str] = []
        if initial_fmax is not None and result_fmax is not None:
            target_fmax = 1000.0 / clock_period
            fmax_change = result_fmax - initial_fmax
            lines.append(f"  {'Target Fmax:':<21s}{target_fmax:8.2f} MHz  (clock period: {clock_period:.3f} ns)")
            lines.append(f"  {'Initial Fmax:':<21s}{initial_fmax:8.2f} MHz  (WNS: {initial_wns:.3f} ns)")
            lines.append(f"  {result_fmax_label:<21s}{result_fmax:8.2f} MHz  (WNS: {result_wns:.3f} ns)")
            lines.append(f"  {'Fmax Improvement:':<21s}{fmax_change:+8.2f} MHz  (WNS: {result_wns - initial_wns:+.3f} ns)")
        else:
            if clock_period is not None:
                target_fmax = 1000.0 / clock_period
                lines.append(f"  {'Clock period:':<21s}{clock_period:8.3f} ns (target: {target_fmax:.2f} MHz)")
            if initial_wns is not None:
                fmax_str = f"  (fmax: {initial_fmax:.2f} MHz)" if initial_fmax else ""
                lines.append(f"  {'Initial WNS:':<21s}{initial_wns:8.3f} ns{fmax_str}")
            if result_wns is not None:
                fmax_str = f"  (fmax: {result_fmax:.2f} MHz)" if result_fmax else ""
                lines.append(f"  {result_wns_label:<21s}{result_wns:8.3f} ns{fmax_str}")
            if initial_wns is not None and result_wns is not None:
                lines.append(f"  {'WNS Improvement:':<21s}{result_wns - initial_wns:+8.3f} ns")
        
        return lines
    
    
    def print_wns_change(
        self,
        initial_wns: Optional[float],
        final_wns: Optional[float],
        clock_period: Optional[float]
    ):
        """Print Fmax/WNS change comparison with improvement/regression status."""
        if final_wns is None or initial_wns is None:
            return
        
        initial_fmax = self.calculate_fmax(initial_wns, clock_period)
        final_fmax = self.calculate_fmax(final_wns, clock_period)
        
        if initial_fmax is not None and final_fmax is not None:
            fmax_improvement = final_fmax - initial_fmax
            pct = (fmax_improvement / initial_fmax) * 100 if initial_fmax else 0
            print(f"\n*** Fmax: {initial_fmax:.2f} -> {final_fmax:.2f} MHz ({fmax_improvement:+.2f} MHz, {pct:+.1f}%) ***")
            print(f"*** WNS:  {initial_wns:.3f} -> {final_wns:.3f} ns ***")
            if fmax_improvement > 0:
                print(f"IMPROVEMENT: Fmax improved by {fmax_improvement:.2f} MHz")
            elif fmax_improvement < 0:
                print(f"REGRESSION: Fmax got worse by {-fmax_improvement:.2f} MHz")
            else:
                print("NO CHANGE: Fmax is the same")
        else:
            wns_improvement = final_wns - initial_wns
            print(f"\n*** WNS: {initial_wns:.3f} -> {final_wns:.3f} ns ({wns_improvement:+.3f} ns) ***")
            if wns_improvement > 0:
                print(f"IMPROVEMENT: WNS improved by {wns_improvement:.3f} ns")
            elif wns_improvement < 0:
                print(f"REGRESSION: WNS got worse by {-wns_improvement:.3f} ns")
            else:
                print("NO CHANGE")
    
    def print_fmax_status(self, label: str, wns: Optional[float]):
        """Print Fmax (primary) and WNS (secondary) for a given measurement point."""
        if wns is None:
            print(f"*** {label}: WNS unknown ***")
            return
        fmax = self.calculate_fmax(wns, self.clock_period)
        clock_info = f" (clock: {self.target_clock})" if self.target_clock else ""
        if fmax is not None:
            print(f"*** {label} Fmax{clock_info}: {fmax:.2f} MHz (WNS: {wns:.3f} ns) ***")
        else:
            print(f"*** {label} WNS{clock_info}: {wns:.3f} ns ***")
    
    def print_test_summary(
        self,
        title: str,
        elapsed_seconds: float,
        initial_wns: Optional[float],
        final_wns: Optional[float],
        clock_period: Optional[float],
        extra_info: str = ""
    ):
        """Print formatted test summary."""
        print("\n" + "="*70)
        print(title)
        print("="*70)
        print(f"Total runtime: {elapsed_seconds:.2f} seconds ({elapsed_seconds/60:.2f} minutes)")
        
        result_lines = self._format_fmax_results(clock_period, initial_wns, final_wns)
        if result_lines:
            print(f"\nFmax Results:")
            print("\n".join(result_lines))
        
        if extra_info:
            print(f"\n{extra_info}")
        print("="*70)


class DCPOptimizer(DCPOptimizerBase):
    """FPGA Design Optimization Agent using RapidWright and Vivado MCPs."""
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        debug: bool = False,
        run_dir: Optional[Path] = None
    ):
        super().__init__(debug=debug, run_dir=run_dir)
        
        self.api_key = api_key
        self.model = model
        self.tools: list[dict] = []
        self.messages: list[dict] = []
        
        self.openai = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Track optimization progress
        self.iteration = 0
        self.best_wns = float('-inf')
        self.no_improvement_count = 0
        self.llm_call_count = 0
        
        # Track token usage and costs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.api_call_details = []
        
        # Track all tool calls with timing and WNS
        self.tool_call_details = []
        
        # Track total runtime
        self.start_time = None
        self.end_time = None
    
    async def start_servers(self):
        """Start and connect to both MCP servers."""
        await super().start_servers()
        await self._collect_tools()
        logger.info(f"Connected to servers with {len(self.tools)} tools available")
    
    async def _collect_tools(self):
        """Collect and convert tools from both MCP servers."""
        self.tools = []
        
        rw_response = await self.rapidwright_session.list_tools()
        for tool in rw_response.tools:
            self.tools.append(convert_mcp_tool_to_openai(tool, "rapidwright"))
        
        v_response = await self.vivado_session.list_tools()
        for tool in v_response.tools:
            self.tools.append(convert_mcp_tool_to_openai(tool, "vivado"))
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call on the appropriate MCP server."""
        # Parse server prefix from tool name
        if tool_name.startswith("rapidwright_"):
            session = self.rapidwright_session
            actual_name = tool_name[len("rapidwright_"):]
        elif tool_name.startswith("vivado_"):
            session = self.vivado_session
            actual_name = tool_name[len("vivado_"):]
        else:
            return json.dumps({"error": f"Unknown tool prefix in: {tool_name}"})
        
        # Track timing for this tool call
        start_time = time.time()
        wns_measured = None
        error_occurred = False
        
        try:
            logger.info(f"Calling {tool_name} with args: {json.dumps(arguments)[:200]}...")
            result = await session.call_tool(actual_name, arguments)
            
            # Extract text content from result
            if result.content:
                text_parts = [c.text for c in result.content if hasattr(c, 'text')]
                result_text = "\n".join(text_parts)
            else:
                result_text = "(no output)"
            
            # Track WNS from timing reports and get_wns calls
            if tool_name == "vivado_report_timing_summary":
                # If target clock is set, get clock-specific WNS instead of overall
                if self.target_clock:
                    try:
                        clock_wns = await super().get_wns_for_target_clock(self._call_vivado_tool)
                        if clock_wns is not None:
                            current_wns = clock_wns
                            wns_measured = current_wns
                            current_fmax = self.calculate_fmax(current_wns, self.clock_period)
                            fmax_str = f", fmax: {current_fmax:.2f} MHz" if current_fmax is not None else ""
                            if current_wns > self.best_wns:
                                logger.info(f"New best WNS (clock: {self.target_clock}): {current_wns:.3f} ns{fmax_str} (improved from {self.best_wns:.3f} ns)")
                                self.best_wns = current_wns
                            else:
                                logger.info(f"Current WNS (clock: {self.target_clock}): {current_wns:.3f} ns{fmax_str} (best is still {self.best_wns:.3f} ns)")
                    except Exception as e:
                        logger.warning(f"Failed to get clock-specific WNS, falling back to overall: {e}")
                        self.target_clock = None  # Fall through to overall WNS parsing
                
                if not self.target_clock or wns_measured is None:
                    timing_info = parse_timing_summary_static(result_text)
                    if timing_info["wns"] is not None:
                        current_wns = timing_info["wns"]
                        wns_measured = current_wns
                        current_fmax = self.calculate_fmax(current_wns, self.clock_period)
                        fmax_str = f", fmax: {current_fmax:.2f} MHz" if current_fmax is not None else ""
                        if current_wns > self.best_wns:
                            logger.info(f"New best WNS: {current_wns:.3f} ns{fmax_str} (improved from {self.best_wns:.3f} ns)")
                            self.best_wns = current_wns
                        else:
                            logger.info(f"Current WNS: {current_wns:.3f} ns{fmax_str} (best is still {self.best_wns:.3f} ns)")
            
            # Also track WNS from get_wns tool (returns just the numeric WNS value)
            elif tool_name == "vivado_get_wns":
                try:
                    current_wns = float(result_text.strip())
                    wns_measured = current_wns
                    current_fmax = self.calculate_fmax(current_wns, self.clock_period)
                    fmax_str = f", fmax: {current_fmax:.2f} MHz" if current_fmax is not None else ""
                    if current_wns > self.best_wns:
                        logger.info(f"New best WNS (from get_wns): {current_wns:.3f} ns{fmax_str} (improved from {self.best_wns:.3f} ns)")
                        self.best_wns = current_wns
                    else:
                        logger.info(f"Current WNS (from get_wns): {current_wns:.3f} ns{fmax_str} (best is still {self.best_wns:.3f} ns)")
                except (ValueError, AttributeError):
                    logger.warning(f"Could not parse WNS from get_wns output: {result_text[:100]}")
            
            elapsed_time = time.time() - start_time
            
            # Record tool call details
            self.tool_call_details.append({
                "tool_name": tool_name,
                "iteration": self.iteration,
                "elapsed_time": elapsed_time,
                "wns": wns_measured,
                "error": False
            })
            
            return result_text
            
        except Exception as e:
            error_occurred = True
            elapsed_time = time.time() - start_time
            
            # Record failed tool call
            self.tool_call_details.append({
                "tool_name": tool_name,
                "iteration": self.iteration,
                "elapsed_time": elapsed_time,
                "wns": None,
                "error": True,
                "error_message": str(e)
            })
            
            logger.error(f"Tool call failed: {e}")
            return json.dumps({"error": str(e)})
    
    async def _call_vivado_tool(self, tool_name: str, arguments: dict) -> str:
        """Helper to call Vivado tools (for use with base class methods)."""
        return await self.call_tool(f"vivado_{tool_name}", arguments)
    
    async def process_response(self, response) -> tuple[str, bool]:
        """Process LLM response, execute tool calls, return final text and done flag."""
        # Validate response structure with detailed logging
        try:
            if not response:
                raise ValueError("Response is None")
            if not hasattr(response, 'choices'):
                raise ValueError(f"Response has no 'choices' attribute. Response type: {type(response)}, Response: {response}")
            if response.choices is None:
                raise ValueError("Response.choices is None")
            if len(response.choices) == 0:
                raise ValueError("Response choices list is empty")
            
            message = response.choices[0].message
            if not message:
                raise ValueError("Message is None")
        except Exception as e:
            logger.error(f"Failed to parse response structure: {e}")
            logger.error(f"Response object: {response}")
            raise
        
        # Convert message to dict, excluding None values which can cause issues
        message_dict = message.model_dump(exclude_none=True)
        self.messages.append(message_dict)
        
        if self.debug:
            logger.debug(f"Added message to conversation: {json.dumps(message_dict, indent=2)[:500]}...")
        
        # Check for tool calls
        if message.tool_calls:
            tool_results = []
            
            for tool_call in message.tool_calls:
                # Validate tool_call structure
                if not tool_call or not hasattr(tool_call, 'function') or not tool_call.function:
                    logger.warning(f"Invalid tool_call structure: {tool_call}")
                    continue
                
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError:
                    tool_args = {}
                
                result = await self.call_tool(tool_name, tool_args)
                
                # Truncate very long results to avoid API issues
                MAX_RESULT_LENGTH = 50000  # characters
                if len(result) > MAX_RESULT_LENGTH:
                    logger.warning(f"Tool result from {tool_name} is {len(result)} chars, truncating to {MAX_RESULT_LENGTH}")
                    result = result[:MAX_RESULT_LENGTH] + f"\n...[truncated {len(result) - MAX_RESULT_LENGTH} characters]"
                
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result
                })
                
                # Debug logging
                if self.debug:
                    logger.debug(f"Tool {tool_name} result: {result[:500]}...")
            
            # Add tool results to messages
            self.messages.extend(tool_results)
            
            # Continue conversation
            return await self.get_completion()
        
        # No tool calls - check if we're done
        content = message.content or ""
        
        # Check for completion indicators
        is_done = any(phrase in content.lower() for phrase in [
            "optimization complete",
            "timing is met",
            "wns >= 0",
            "no more optimizations",
            "design meets timing",
            "successfully saved",
            "final design saved"
        ])
        
        return content, is_done
    
    async def perform_initial_analysis(self, input_dcp: Path) -> str:
        """
        Perform initial analysis without LLM:
        1. Initialize RapidWright
        2. Open checkpoint in Vivado
        3. Report timing summary
        4. Get critical high fanout nets
        
        Returns a formatted summary of the analysis.
        """
        logger.info("Performing initial design analysis...")
        print("\n=== Initial Design Analysis ===\n")
        
        # Step 1: Initialize RapidWright
        logger.info("Initializing RapidWright...")
        print("Initializing RapidWright...")
        result = await self.call_tool("rapidwright_initialize_rapidwright", {})
        if "error" in result.lower() and "success" not in result.lower():
            raise RuntimeError(f"Failed to initialize RapidWright: {result}")
        print("✓ RapidWright initialized\n")
        
        # Step 2: Open checkpoint in Vivado
        logger.info(f"Opening checkpoint: {input_dcp}")
        print(f"Opening checkpoint: {input_dcp.name}")
        result = await self.call_tool("vivado_open_checkpoint", {
            "dcp_path": str(input_dcp.resolve())
        })
        if "error" in result.lower() and "opened successfully" not in result.lower():
            raise RuntimeError(f"Failed to open checkpoint: {result}")
        print("✓ Checkpoint opened in Vivado\n")
        
        # Step 3: Report timing summary
        logger.info("Analyzing timing...")
        print("Analyzing timing...")
        timing_report = await self.call_tool("vivado_report_timing_summary", {})
        
        # Parse timing
        timing_info = parse_timing_summary_static(timing_report)
        self.initial_tns = timing_info["tns"]
        self.initial_failing_endpoints = timing_info["failing_endpoints"]
        
        # Get clock period for fmax calculation (also detects target clock)
        self.clock_period = await super().get_clock_period(self._call_vivado_tool)
        
        # Get WNS for the target clock domain
        target_wns = await super().get_wns_for_target_clock(self._call_vivado_tool)
        if target_wns is not None:
            self.initial_wns = target_wns
        else:
            self.initial_wns = timing_info["wns"]
        self.best_wns = self.initial_wns if self.initial_wns is not None else float('-inf')
        
        clock_info = f" (clock: {self.target_clock})" if self.target_clock else ""
        print(f"✓ Timing analyzed:")
        if self.clock_period is not None:
            target_fmax = 1000.0 / self.clock_period
            print(f"  - Clock period: {self.clock_period:.3f} ns (target fmax: {target_fmax:.2f} MHz)")
        if self.target_clock:
            print(f"  - Target clock: {self.target_clock}")
        if self.initial_wns is not None:
            print(f"  - WNS{clock_info}: {self.initial_wns:.3f} ns")
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            if initial_fmax is not None:
                print(f"  - Achievable fmax: {initial_fmax:.2f} MHz")
        if self.initial_tns is not None:
            print(f"  - TNS: {self.initial_tns:.3f} ns")
        if self.initial_failing_endpoints is not None:
            print(f"  - Failing endpoints: {self.initial_failing_endpoints}")
        print()
        
        # Step 4: Get critical high fanout nets
        logger.info("Identifying critical high fanout nets...")
        print("Identifying critical high fanout nets...")
        nets_report = await self.call_tool("vivado_get_critical_high_fanout_nets", {
            "num_paths": 50,
            "min_fanout": 100
        })
        
        # Parse high fanout nets
        self.high_fanout_nets = self.parse_high_fanout_nets(nets_report)
        print(f"✓ Found {len(self.high_fanout_nets)} high fanout nets (>100 fanout)\n")
        
        # Step 5: Load design in RapidWright for spread analysis
        critical_path_spread_info = None  # Initialize
        
        logger.info("Loading design in RapidWright...")
        print("Loading design in RapidWright for spread analysis...")
        result = await self.call_tool("rapidwright_read_checkpoint", {
            "dcp_path": str(input_dcp.resolve())
        })
        if "error" in result.lower() and "success" not in result.lower():
            print(f"⚠ Warning: Could not load design in RapidWright: {result}")
        else:
            print("✓ Design loaded in RapidWright\n")
            
            # Step 6: Extract critical path cells and analyze spread
            logger.info("Extracting and analyzing critical path spread...")
            print("Analyzing critical path spread...")
            
            # Extract critical path cells from Vivado
            temp_path = Path(self.temp_dir) / "initial_critical_paths.json"
            cells_json = await self.call_tool("vivado_extract_critical_path_cells", {
                "num_paths": 50,
                "output_file": str(temp_path)
            })
            
            # Analyze spread in RapidWright
            spread_result = await self.call_tool("rapidwright_analyze_critical_path_spread", {
                "input_file": str(temp_path)
            })
            
            # Parse spread results
            import json
            try:
                spread_data = json.loads(spread_result)
                critical_path_spread_info = {
                    "max_distance": spread_data.get("max_distance_found", 0),
                    "avg_distance": spread_data.get("avg_max_distance", 0),
                    "paths_analyzed": spread_data.get("paths_analyzed", 0)
                }
                print(f"✓ Critical path spread analyzed:")
                print(f"  - Max distance: {critical_path_spread_info['max_distance']} tiles")
                print(f"  - Avg distance: {critical_path_spread_info['avg_distance']:.1f} tiles")
                print(f"  - Paths analyzed: {critical_path_spread_info['paths_analyzed']}")
                print()
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠ Warning: Could not parse spread results: {e}")
                critical_path_spread_info = None
        
        # Create concise summary for LLM
        summary = []
        summary.append("=== Initial Design Analysis ===\n")
        
        # Timing status
        summary.append("TIMING STATUS:")
        if self.clock_period is not None:
            target_fmax = 1000.0 / self.clock_period
            summary.append(f"  Clock period: {self.clock_period:.3f} ns (target fmax: {target_fmax:.2f} MHz)")
        if self.initial_wns is not None:
            if self.initial_wns >= 0:
                summary.append(f"  WNS: {self.initial_wns:.3f} ns - TIMING MET ✓")
            else:
                summary.append(f"  WNS: {self.initial_wns:.3f} ns - TIMING VIOLATED")
            # Add fmax information
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            if initial_fmax is not None:
                summary.append(f"  Achievable fmax: {initial_fmax:.2f} MHz")
        if self.initial_tns is not None:
            summary.append(f"  TNS: {self.initial_tns:.3f} ns")
        if self.initial_failing_endpoints is not None:
            summary.append(f"  Failing endpoints: {self.initial_failing_endpoints}")
        summary.append("")
        
        # Critical path spread analysis
        if critical_path_spread_info:
            summary.append("CRITICAL PATH SPREAD ANALYSIS:")
            summary.append(f"  Max cell distance: {critical_path_spread_info['max_distance']} tiles")
            summary.append(f"  Avg cell distance: {critical_path_spread_info['avg_distance']:.1f} tiles")
            summary.append(f"  Paths analyzed: {critical_path_spread_info['paths_analyzed']}")
            
            # Recommendation based on spread
            if critical_path_spread_info['avg_distance'] > 70 and critical_path_spread_info['paths_analyzed'] >= 5:
                summary.append(f"  ⚠ RECOMMENDATION: Use PBLOCK strategy (high spread detected)")
            summary.append("")
        
        # High fanout nets (show top 10)
        if self.high_fanout_nets:
            summary.append("CRITICAL HIGH FANOUT NETS (top 10):")
            for i, (net_name, fanout, path_count) in enumerate(self.high_fanout_nets[:10]):
                summary.append(f"  {i+1}. {net_name}")
                summary.append(f"     Fanout: {fanout}, Critical paths: {path_count}")
            if len(self.high_fanout_nets) > 10:
                summary.append(f"  ... and {len(self.high_fanout_nets) - 10} more nets")
        else:
            summary.append("CRITICAL HIGH FANOUT NETS: None found")
        
        summary.append("")
        summary.append(f"Total nets available for optimization: {len(self.high_fanout_nets)}")
        
        summary_text = "\n".join(summary)
        print(summary_text)
        print()
        
        return summary_text
    
    async def get_completion(self) -> tuple[str, bool]:
        """Get LLM completion and process it."""
        try:
            self.llm_call_count += 1
            logger.info(f"LLM API call #{self.llm_call_count}")
            
            # Request usage accounting from OpenRouter
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=4096,
                extra_body={
                    "usage": {
                        "include": True
                    }
                }
            )
            
            # Validate response immediately
            if response is None:
                raise ValueError("API returned None response")
            
            # Extract token usage information from OpenRouter
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                # Update cumulative totals
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                
                # Get actual cost from OpenRouter (in credits/dollars)
                call_cost = 0.0
                if hasattr(response.usage, 'cost') and response.usage.cost is not None:
                    call_cost = float(response.usage.cost)
                    self.total_cost += call_cost
                else:
                    logger.warning("OpenRouter did not provide cost information")
                
                # Extract additional usage details if available
                cached_tokens = 0
                reasoning_tokens = 0
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    if hasattr(response.usage.prompt_tokens_details, 'cached_tokens'):
                        cached_tokens = response.usage.prompt_tokens_details.cached_tokens or 0
                if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                    if hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                        reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens or 0
                
                # Store details for this call
                call_detail = {
                    "call_number": self.llm_call_count,
                    "iteration": self.iteration,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": call_cost,
                    "cached_tokens": cached_tokens,
                    "reasoning_tokens": reasoning_tokens
                }
                self.api_call_details.append(call_detail)
                
                # Log token usage
                cache_info = f", Cached: {cached_tokens:,}" if cached_tokens > 0 else ""
                reasoning_info = f", Reasoning: {reasoning_tokens:,}" if reasoning_tokens > 0 else ""
                cost_info = f" | Cost: ${call_cost:.4f}" if call_cost > 0 else ""
                
                logger.info(f"API call #{self.llm_call_count} - Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total{cost_info}{cache_info}{reasoning_info}")
                print(f"[API Call #{self.llm_call_count}] Tokens: {total_tokens:,} (Prompt: {prompt_tokens:,}, Completion: {completion_tokens:,}{cache_info}{reasoning_info}){cost_info}")
            else:
                logger.warning("No usage information in API response")
            
            # Debug logging
            if self.debug:
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response: {response}")
            
            # Check if response has error
            if hasattr(response, 'error') and response.error:
                raise ValueError(f"API returned error: {response.error}")
            
            return await self.process_response(response)
            
        except Exception as e:
            logger.error(f"Error in get_completion: {e}")
            logger.error(f"Number of messages in conversation: {len(self.messages)}")
            if self.messages:
                logger.error(f"Last message: {self.messages[-1]}")
            raise
    
    async def optimize(self, input_dcp: Path, output_dcp: Path) -> bool:
        """Run the optimization workflow."""
        # Start timing the optimization process
        self.start_time = time.time()
        
        # Perform initial analysis without LLM
        try:
            initial_analysis = await self.perform_initial_analysis(input_dcp)
        except Exception as e:
            logger.exception(f"Initial analysis failed: {e}")
            print(f"\n✗ Initial analysis failed: {e}\n")
            self.end_time = time.time()
            return False
        
        # Check if timing is already met
        if self.initial_wns is not None and self.initial_wns >= 0:
            print("✓ Design already meets timing! No optimization needed.\n")
            logger.info("Design already meets timing")
            # Save the design as-is
            result = await self.call_tool("vivado_write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            })
            print(f"Saved design to: {output_dcp}\n")
            
            # End timing
            self.end_time = time.time()
            total_runtime = self.end_time - self.start_time
            
            # Print summary even for early exit
            print("\n=== No Optimization Required ===")
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            if initial_fmax is not None:
                print(f"Design already meets timing - Fmax: {initial_fmax:.2f} MHz (WNS: {self.initial_wns:.3f} ns)")
            else:
                print(f"Design already meets timing (WNS: {self.initial_wns:.3f} ns)")
            print(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
            print(f"LLM API calls: 0 (analysis performed without LLM)")
            print(f"Estimated cost: $0.00")
            print("="*70 + "\n")
            return True
        
        # Load and fill in system prompt with temp directory and input DCP path
        system_prompt_template = load_system_prompt()
        system_prompt = system_prompt_template.format(
            temp_dir=self.temp_dir,
            input_dcp=input_dcp.resolve()
        )
        
        # Initialize conversation with analysis results
        self.messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Optimize this FPGA design for timing.

PATHS:
- Input DCP: {input_dcp.resolve()}
- Output DCP (save final result here): {output_dcp.resolve()}
- Run directory (for intermediate files): {self.temp_dir}

CURRENT STATE:
- Vivado has the input design ALREADY OPEN and analyzed
- RapidWright has the input design ALREADY LOADED (from initial analysis)

INITIAL ANALYSIS RESULTS:
{initial_analysis}

Proceed with optimization strategy based on the analysis above. Do NOT reload the design in either Vivado or RapidWright - both already have it loaded."""
            }
        ]
        
        max_iterations = 50  # Safety limit
        
        print("=== Starting LLM-Driven Optimization ===\n")
        
        while self.iteration < max_iterations:
            self.iteration += 1
            logger.info(f"=== Iteration {self.iteration} ===")
            
            try:
                response_text, is_done = await self.get_completion()
                print(f"\n{response_text}\n")
                
                if is_done:
                    logger.info("Optimization workflow completed")
                    self.end_time = time.time()
                    self._print_optimization_summary()
                    return True
                    
            except Exception as e:
                logger.exception(f"Error during optimization: {e}")
                # Add error context to conversation
                self.messages.append({
                    "role": "user",
                    "content": f"An error occurred: {e}. Please verify your approach and continue or report if unrecoverable."
                })
        
        logger.warning("Reached maximum iterations")
        self.end_time = time.time()
        self._print_optimization_summary(max_iterations_reached=True)
        return False
    
    def save_token_usage_report(self, output_path: Path):
        """Save detailed token usage report to JSON file."""
        # Calculate total cached and reasoning tokens
        total_cached = sum(detail.get('cached_tokens', 0) for detail in self.api_call_details)
        total_reasoning = sum(detail.get('reasoning_tokens', 0) for detail in self.api_call_details)
        
        # Calculate tool call statistics
        total_tool_time = sum(detail['elapsed_time'] for detail in self.tool_call_details)
        tool_counts = {}
        for detail in self.tool_call_details:
            tool_name = detail['tool_name']
            if tool_name not in tool_counts:
                tool_counts[tool_name] = 0
            tool_counts[tool_name] += 1
        
        # Calculate total runtime
        total_runtime = None
        if self.start_time is not None:
            total_runtime = (self.end_time or time.time()) - self.start_time
        
        # Calculate fmax values
        initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
        best_fmax = self.calculate_fmax(self.best_wns, self.clock_period) if self.best_wns > float('-inf') else None
        fmax_improvement = (best_fmax - initial_fmax) if (initial_fmax is not None and best_fmax is not None) else None
        
        report = {
            "model": self.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_runtime_seconds": total_runtime,
                "total_llm_calls": self.llm_call_count,
                "total_iterations": self.iteration,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
                "total_cached_tokens": total_cached,
                "total_reasoning_tokens": total_reasoning,
                "total_cost": self.total_cost,
                "clock_period_ns": self.clock_period,
                "initial_wns": self.initial_wns,
                "best_wns": self.best_wns,
                "wns_improvement": self.best_wns - self.initial_wns if self.initial_wns is not None else None,
                "initial_fmax_mhz": initial_fmax,
                "best_fmax_mhz": best_fmax,
                "fmax_improvement_mhz": fmax_improvement,
                "total_tool_calls": len(self.tool_call_details),
                "total_tool_time_seconds": total_tool_time,
                "tool_call_counts": tool_counts
            },
            "per_llm_call_details": self.api_call_details,
            "per_tool_call_details": self.tool_call_details
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Token usage report saved to {output_path}")
    
    def _print_optimization_summary(self, max_iterations_reached: bool = False):
        """Print detailed optimization summary including token usage and costs."""
        title = "Optimization Summary (Max Iterations Reached)" if max_iterations_reached else "Optimization Summary"
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")
        
        # Calculate total runtime
        if self.start_time is not None:
            total_runtime = (self.end_time or time.time()) - self.start_time
            print(f"\nTOTAL RUNTIME: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
        
        best_wns = self.best_wns if self.best_wns > float('-inf') else None
        result_lines = self._format_fmax_results(
            self.clock_period, self.initial_wns, best_wns, result_label="Best"
        )
        if result_lines:
            print(f"\nFMAX RESULTS:")
            print("\n".join(result_lines))
        
        # Iteration stats
        print(f"\nITERATION STATS:")
        print(f"  Total iterations:    {self.iteration}")
        print(f"  LLM API calls:       {self.llm_call_count}")
        
        # Token usage
        print(f"\nTOKEN USAGE:")
        print(f"  Prompt tokens:       {self.total_prompt_tokens:,}")
        print(f"  Completion tokens:   {self.total_completion_tokens:,}")
        print(f"  Total tokens:        {self.total_tokens:,}")
        
        # Calculate total cached and reasoning tokens
        total_cached = sum(detail.get('cached_tokens', 0) for detail in self.api_call_details)
        total_reasoning = sum(detail.get('reasoning_tokens', 0) for detail in self.api_call_details)
        
        if total_cached > 0:
            print(f"  Cached tokens:       {total_cached:,} (saved cost)")
        if total_reasoning > 0:
            print(f"  Reasoning tokens:    {total_reasoning:,}")
        
        # Cost
        print(f"\nCOST:")
        print(f"  Model:               {self.model}")
        if self.total_cost > 0:
            print(f"  Total cost:          ${self.total_cost:.4f}")
        else:
            print(f"  Total cost:          Not available")
        
        # Tool call summary
        if self.tool_call_details:
            print(f"\nTOOL CALLS SUMMARY:")
            print(f"  Total tool calls:    {len(self.tool_call_details)}")
            
            # Calculate total time spent in tool calls
            total_tool_time = sum(detail['elapsed_time'] for detail in self.tool_call_details)
            print(f"  Total tool time:     {total_tool_time:.2f}s")
            
            # Count by tool type
            tool_counts = {}
            for detail in self.tool_call_details:
                tool_name = detail['tool_name']
                if tool_name not in tool_counts:
                    tool_counts[tool_name] = 0
                tool_counts[tool_name] += 1
            
            print(f"\n  Tool call breakdown:")
            for tool_name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
                print(f"    {tool_name}: {count}")
            
            # Detailed tool call list
            print(f"\n  Detailed tool call log:")
            print(f"  {'#':<5} {'Iter':<6} {'Tool':<40} {'Time (s)':<12} {'WNS (ns)':<12} {'Status':<10}")
            print(f"  {'-'*5} {'-'*6} {'-'*40} {'-'*12} {'-'*12} {'-'*10}")
            
            for i, detail in enumerate(self.tool_call_details, 1):
                tool_name = detail['tool_name']
                iteration = detail.get('iteration', 0)
                elapsed = detail['elapsed_time']
                wns = detail.get('wns')
                error = detail.get('error', False)
                
                # Format WNS column
                wns_str = f"{wns:.3f}" if wns is not None else "-"
                
                # Format status
                status_str = "ERROR" if error else "OK"
                
                print(f"  {i:<5} {iteration:<6} {tool_name:<40} {elapsed:<12.2f} {wns_str:<12} {status_str:<10}")
                
                # If error, show error message on next line
                if error and 'error_message' in detail:
                    print(f"        Error: {detail['error_message'][:80]}")
        
        # Per-call breakdown if debug mode
        if self.debug and self.api_call_details:
            print(f"\nPER-CALL BREAKDOWN:")
            
            # Check if we have cached or reasoning tokens to display
            has_cached = any(detail.get('cached_tokens', 0) > 0 for detail in self.api_call_details)
            has_reasoning = any(detail.get('reasoning_tokens', 0) > 0 for detail in self.api_call_details)
            has_cost = any(detail.get('cost', 0) > 0 for detail in self.api_call_details)
            
            # Build header
            header = f"  {'Call':<6} {'Iter':<6} {'Prompt':<10} {'Completion':<12}"
            if has_cached:
                header += f" {'Cached':<10}"
            if has_reasoning:
                header += f" {'Reasoning':<10}"
            header += f" {'Total':<10}"
            if has_cost:
                header += f" {'Cost':<12}"
            print(header)
            
            # Build separator
            separator = f"  {'-'*6} {'-'*6} {'-'*10} {'-'*12}"
            if has_cached:
                separator += f" {'-'*10}"
            if has_reasoning:
                separator += f" {'-'*10}"
            separator += f" {'-'*10}"
            if has_cost:
                separator += f" {'-'*12}"
            print(separator)
            
            # Print details
            for detail in self.api_call_details:
                line = (f"  {detail['call_number']:<6} {detail['iteration']:<6} "
                       f"{detail['prompt_tokens']:<10,} {detail['completion_tokens']:<12,}")
                if has_cached:
                    line += f" {detail.get('cached_tokens', 0):<10,}"
                if has_reasoning:
                    line += f" {detail.get('reasoning_tokens', 0):<10,}"
                line += f" {detail['total_tokens']:<10,}"
                if has_cost:
                    cost = detail.get('cost', 0)
                    line += f" ${cost:<11.4f}" if cost > 0 else f" {'N/A':<12}"
                print(line)
        
        print(f"\n{'='*70}\n")
        
        # Save detailed report to JSON in run directory
        try:
            report_path = self.run_dir / "token_usage.json"
            self.save_token_usage_report(report_path)
            print(f"Detailed token usage report saved to: {report_path}\n")
        except Exception as e:
            logger.warning(f"Failed to save token usage report: {e}")
    


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

    # Normal mode - requires API key and LLM
    if not args.api_key:
        print("Error: OpenRouter API key required. Set OPENROUTER_API_KEY or use --api-key", file=sys.stderr)
        print("       Use --test or --baseline to run without LLM", file=sys.stderr)
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


if __name__ == "__main__":
    asyncio.run(main())
