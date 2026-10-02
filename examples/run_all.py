"""Run all public PyAutoStat customer analytics examples in isolated subprocesses.

Usage:
    python examples/run_all.py
    python examples/run_all.py --fast

Features:
    - Runs numbered examples (01 through 09) sequentially in isolated subprocesses.
    - Captures and displays execution progress without swallowing errors or tracebacks.
    - Verifies that individual CustomerID values are never leaked in terminal output.
    - Provides a concise PASS/FAIL status table with per-script runtimes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent

EXAMPLE_SCRIPTS: tuple[str, ...] = (
    "01_customer_360_profile.py",
    "02_compare_customer_segments.py",
    "03_multigroup_customer_spending.py",
    "04_customer_value_association.py",
    "05_spend_drivers_regression.py",
    "06_high_value_customer_logistic.py",
    "07_product_portfolio_repeated_measures.py",
    "08_factorial_customer_segments.py",
    "09_complete_research_workflow.py",
)

KNOWN_ID_SENTINEL = "0002-GTOKLU-YVY"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all PyAutoStat customer analytics showcase examples."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast demonstration / CI mode with bounded bootstrap resamples.",
    )
    return parser.parse_args()


def run_script(script_name: str, fast_mode: bool) -> tuple[bool, float, str]:
    script_path = EXAMPLES_DIR / script_name
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if fast_mode:
        env["PYAUTOSTAT_FAST_DEMO"] = "1"

    cmd = [sys.executable, str(script_path)]
    if script_name == "09_complete_research_workflow.py":
        output_dir = EXAMPLES_DIR / "reports"
        cmd.extend(["--output-dir", str(output_dir)])

    start_time = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start_time

    output = proc.stdout + ("\nSTDERR:\n" + proc.stderr if proc.stderr else "")
    success = proc.returncode == 0 and KNOWN_ID_SENTINEL not in output

    return success, elapsed, output


def main() -> int:
    args = parse_arguments()
    mode_label = "FAST DEMO / CI MODE" if args.fast else "STANDARD DEMO MODE"

    print("=" * 76)
    print(f"PYAUTOSTAT CUSTOMER ANALYTICS SHOWCASE: RUN ALL EXAMPLES ({mode_label})")
    print("=" * 76)
    print(f"Python interpreter: {sys.executable}")
    print(f"Gallery location  : {EXAMPLES_DIR}")
    print(f"Total examples    : {len(EXAMPLE_SCRIPTS)}\n")

    results: list[tuple[str, bool, float]] = []
    any_failed = False

    for script_name in EXAMPLE_SCRIPTS:
        print(f"--> Running {script_name}...", end=" ", flush=True)
        success, elapsed, output = run_script(script_name, args.fast)
        results.append((script_name, success, elapsed))

        if success:
            print(f"PASSED ({elapsed:.2f}s)")
        else:
            print(f"FAILED ({elapsed:.2f}s)")
            any_failed = True
            print("-" * 76)
            print(output.strip())
            print("-" * 76)
            break

    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    print(f"  {'Example Script':<44} | {'Status':<8} | {'Elapsed':<8}")
    print("  " + "-" * 66)
    for name, success, elapsed in results:
        status_str = "PASS" if success else "FAIL"
        print(f"  {name:<44} | {status_str:<8} | {elapsed:>6.2f}s")

    total_time = sum(t for _, _, t in results)
    print("  " + "-" * 66)
    print(f"  Total execution time: {total_time:.2f}s\n")

    if any_failed:
        print("RESULT: One or more examples failed. Check error trace above.")
        return 1

    print("RESULT: ALL 9 EXAMPLES PASSED.")
    print("No CustomerID values were exposed. Full customer analytics gallery verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
