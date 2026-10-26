#!/bin/bash
# Stage 1 -- runs on a Quantum Access Node (QAN).
# Partition / licence / chdir are normally passed by the notebook (qsci_lib.submit_pipeline),
# so the same file works on the MQT Docker playground, a single-node droplet, or a real cluster.
#SBATCH --job-name=qsci-quantum
#SBATCH --partition=quantum
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --output=%x-%j.out
set -euo pipefail

# Credentials are exported from the submission environment (sbatch --export=ALL is the default):
#   RESONANCE_API_KEY / IQM_TOKEN or IQM_TOKENS_FILE, optional IQM_BASE_URL, IQM_QC_ALIAS
# On production sites a SPANK plugin can inject these instead (see QDMI-on-IQM/spank).
echo "QAN: $(hostname)  licences: ${SLURM_JOB_LICENSES:-none}  run dir: ${QSCI_RUN_DIR:-$PWD}"

# module load python/3.12            # <- site specific
# source /path/to/venv/bin/activate  # <- not needed on the MQT playground (/opt/venv on PATH)

srun python "${QSCI_CODE_DIR:-$SLURM_SUBMIT_DIR}/qsci_quantum_stage.py" --run-dir "${QSCI_RUN_DIR:-$PWD}"
