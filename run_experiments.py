import os
import re
import subprocess

scenarios = [
    {"name": "Baseline (Load=6.0, Fading=10, BW=20.0)", "replacements": {}},
    {"name": "Low Traffic Load (λ = 2.0)", "replacements": {"arrival_rate_packets_per_ts": "2.0"}},
    {"name": "High Traffic Load (λ = 10.0)", "replacements": {"arrival_rate_packets_per_ts": "10.0"}},
    {"name": "High Mobility / Fast Fading (correlation = 1)", "replacements": {"arrival_rate_packets_per_ts": "6.0", "fading_correlation_time": "1"}},
    {"name": "Low Mobility / Slow Fading (correlation = 50)", "replacements": {"fading_correlation_time": "50"}},
    {"name": "High Bandwidth / Abundance (BW = 50.0 MHz)", "replacements": {"fading_correlation_time": "10", "total_bandwidth_mhz": "50.0"}},
    {"name": "Low Bandwidth / Scarcity (BW = 10.0 MHz)", "replacements": {"total_bandwidth_mhz": "10.0"}},
]

def update_config(replacements):
    with open("config.py", "r") as f:
        content = f.read()
    for key, val in replacements.items():
        content = re.sub(rf"({key}.*?=\s*)[\d\.]+", rf"\g<1>{val}", content)
    with open("config.py", "w") as f:
        f.write(content)

md_content = "# 6G Spectrum Allocation - Sensitivity Analysis Results\n\n"
md_content += "This document contains the evaluation of all algorithms (including zero-shot generalization of DQN and QI-DQN) under varying network conditions.\n\n"

for s in scenarios:
    print(f"Running scenario: {s['name']}", flush=True)
    update_config(s["replacements"])
    result = subprocess.run(["venv/bin/python", "evaluate_agents.py"], capture_output=True, text=True)
    
    output = result.stderr + result.stdout
    match = re.search(r"ALGORITHM COMPARISON RESULTS.*?(Algorithm\s+\|.*?)\n(?:INFO|$)", output, re.DOTALL)
    
    md_content += f"## Scenario: {s['name']}\n\n"
    if match:
        table = match.group(1).strip()
        md_content += "```text\n" + table + "\n```\n\n"
    else:
        md_content += "```text\nFailed to extract table.\n```\n\n"

# Restore config to defaults
update_config({"arrival_rate_packets_per_ts": "6.0", "fading_correlation_time": "10", "total_bandwidth_mhz": "20.0"})

with open("/home/vishwas/.gemini/antigravity-cli/brain/0fbd9be0-ee38-41ce-bf58-f1e4e5516153/sensitivity_analysis.md", "w") as f:
    f.write(md_content)
    
print("All scenarios completed. Results written to sensitivity_analysis.md", flush=True)
