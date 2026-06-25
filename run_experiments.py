import os
import sys
from contextlib import redirect_stdout, redirect_stderr
import io
import re

from config import Config
from evaluate_agents import main as evaluate_agents_main

scenarios = [
    {"name": "Baseline (Load=6.0, Fading=10, BW=20.0)", "replacements": {}},
    {"name": "Low Traffic Load (λ = 2.0)", "replacements": {"arrival_rate_packets_per_ts": 2.0}},
    {"name": "High Traffic Load (λ = 10.0)", "replacements": {"arrival_rate_packets_per_ts": 10.0}},
    {"name": "High Mobility / Fast Fading (correlation = 1)", "replacements": {"fading_correlation_time": 1}},
    {"name": "Low Mobility / Slow Fading (correlation = 50)", "replacements": {"fading_correlation_time": 50}},
    {"name": "High Bandwidth / Abundance (BW = 50.0 MHz)", "replacements": {"total_bandwidth_mhz": 50.0}},
    {"name": "Low Bandwidth / Scarcity (BW = 10.0 MHz)", "replacements": {"total_bandwidth_mhz": 10.0}},
]

md_content = "# 6G Spectrum Allocation - Sensitivity Analysis Results\n\n"
md_content += "This document contains the evaluation of all algorithms (including zero-shot generalization of DQN and QI-DQN) under varying network conditions.\n\n"

for s in scenarios:
    print(f"Running scenario: {s['name']}", flush=True)
    
    # Create fresh default config
    config = Config()
    
    # Apply replacements
    for key, val in s["replacements"].items():
        if hasattr(config.env, key):
            setattr(config.env, key, val)
        elif hasattr(config.channel, key):
            setattr(config.channel, key, val)
        elif hasattr(config.traffic, key):
            setattr(config.traffic, key, val)
    
    # Recalculate derived config properties if necessary
    config.env.bandwidth_per_rb_mhz = config.env.total_bandwidth_mhz / config.env.num_resource_blocks
            
    # Capture output
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        evaluate_agents_main(config)
    output = f.getvalue()
    
    match = re.search(r"ALGORITHM COMPARISON RESULTS.*?(Algorithm\s+\|.*?)\n(?:INFO|$)", output, re.DOTALL)
    
    md_content += f"## Scenario: {s['name']}\n\n"
    if match:
        table = match.group(1).strip()
        md_content += "```text\n" + table + "\n```\n\n"
    else:
        # Fallback regex if INFO tags are missing
        match2 = re.search(r"(Algorithm\s+\|.*?)\Z", output, re.DOTALL)
        if match2:
            table = match2.group(1).strip()
            md_content += "```text\n" + table + "\n```\n\n"
        else:
            md_content += "```text\nFailed to extract table.\n```\n\n"
            print("Failed output snippet:", output[-500:])

output_path = "sensitivity_analysis.md"
with open(output_path, "w") as f:
    f.write(md_content)
    
print(f"All scenarios completed. Results written to {output_path}", flush=True)
