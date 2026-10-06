#!/usr/bin/env python3
"""Run the complete clean-distribution five-seed experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--out-dir", default="outputs_ptbxl/p8")
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    python = sys.executable
    for seed in config["seeds"]:
        run(
            [
                python,
                "scripts/run_seed.py",
                "--seed",
                str(seed),
                "--config",
                str(config_path),
                "--out-dir",
                args.out_dir,
            ]
        )

    run([python, "scripts/aggregate_seed_results.py", "--out-dir", args.out_dir])
    run([python, "scripts/paired_comparison.py", "--out-dir", args.out_dir])
    run(
        [
            python,
            "scripts/reproduce_results.py",
            "--root",
            str(ROOT),
            "--artifacts-dir",
            args.out_dir,
            "--metadata",
            "outputs_ptbxl/data/ptbxl_metadata_processed.csv",
            "--output-dir",
            f"{args.out_dir}/revision",
            "--bootstrap",
            str(args.bootstrap),
        ]
    )


if __name__ == "__main__":
    main()
