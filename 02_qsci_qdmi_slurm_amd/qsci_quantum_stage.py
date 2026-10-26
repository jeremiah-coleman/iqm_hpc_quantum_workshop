#!/usr/bin/env python3
"""Slurm stage 1 (quantum partition): VQE + bitstring sampling through QDMI.

Usage: python qsci_quantum_stage.py --run-dir /jobs/qsci-run-001
The run directory must contain config.json (written by the notebook / QSCIConfig.save).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qsci_lib  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run-dir", default=os.environ.get("QSCI_RUN_DIR", "."))
args = p.parse_args()
qsci_lib.quantum_stage(args.run_dir, log=lambda m: print(m, flush=True))
