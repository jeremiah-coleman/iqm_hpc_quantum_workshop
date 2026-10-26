#!/usr/bin/env python3
"""Slurm stage 2 (GPU / compute partition): subspace Hamiltonian + JAX diagonalisation.

Usage: python qsci_classical_stage.py --run-dir /jobs/qsci-run-001
Reads quantum.json from the run directory, writes result.json.
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
qsci_lib.classical_stage(args.run_dir, log=lambda m: print(m, flush=True))
