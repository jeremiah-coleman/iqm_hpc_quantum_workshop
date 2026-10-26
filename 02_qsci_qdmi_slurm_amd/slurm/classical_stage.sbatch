#!/bin/bash
# Stage 2 -- runs on a classical compute node, ideally with an AMD GPU (ROCm JAX).
# The notebook adds --dependency=afterok:<quantum job>, --partition and (optionally) --gpus.
#SBATCH --job-name=qsci-classical
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --output=%x-%j.out
set -euo pipefail

echo "Classical node: $(hostname)  GPUs visible: ${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-none}}"

# --- AMD GPU environment (uncomment what your site / droplet needs) ---------------------
# module load rocm                                  # e.g. LUMI: module load LUMI partition/G rocm
# export JAX_PLATFORMS=rocm                         # force the ROCm backend (fails loudly if absent)
# export XLA_PYTHON_CLIENT_PREALLOCATE=false        # don't grab all HBM for a tiny matrix
# source /path/to/venv-rocm/bin/activate
command -v rocm-smi >/dev/null && rocm-smi --showproductname || true

srun python "${QSCI_CODE_DIR:-$SLURM_SUBMIT_DIR}/qsci_classical_stage.py" --run-dir "${QSCI_RUN_DIR:-$PWD}"
