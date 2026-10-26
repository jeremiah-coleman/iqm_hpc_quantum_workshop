#!/bin/bash
# Lab 1c: the same qdmi_demo.py, now scheduled by Slurm on a quantum access node (QAN).
# Submit from the login node:
#   sbatch --licenses=mqt.ddsim.default bell.sbatch      # local simulator through QDMI
#   sbatch --licenses=iqm.garnet.mock   bell.sbatch      # IQM Resonance (use an ID from `qdmi_demo.py list`)
# The device is chosen by the licence; the script reads it from SLURM_JOB_LICENSES.
#SBATCH --job-name=qdmi-bell
#SBATCH --partition=quantum
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
set -euo pipefail
echo "QAN $(hostname) got licence(s): ${SLURM_JOB_LICENSES:-none}"
srun python /jobs/qdmi_demo.py bell --device slurm --via raw --shots 1000 --out "/jobs/bell-${SLURM_JOB_ID}.json"
