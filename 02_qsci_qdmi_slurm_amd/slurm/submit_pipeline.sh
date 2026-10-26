#!/bin/bash
# Submit the two-stage QSCI pipeline from a login node without the notebook.
#   ./slurm/submit_pipeline.sh <run_dir> [device_licence] [classical_partition] [gpus]
# Example (MQT playground, DDSIM):   ./slurm/submit_pipeline.sh /jobs/qsci-001 mqt.ddsim.default compute 0
# Example (droplet, IQM + AMD GPU):  ./slurm/submit_pipeline.sh $HOME/runs/qsci-002 iqm.garnet gpu 1
set -euo pipefail
RUN_DIR=$(realpath "$1"); LIC=${2:-mqt.ddsim.default}; CPART=${3:-compute}; GPUS=${4:-0}
CODE_DIR=$(cd "$(dirname "$0")/.." && pwd)
test -f "$RUN_DIR/config.json" || { echo "missing $RUN_DIR/config.json (create it with QSCIConfig(...).save(run_dir))"; exit 1; }
export QSCI_RUN_DIR="$RUN_DIR" QSCI_CODE_DIR="$CODE_DIR"

QID=$(sbatch --parsable --partition=quantum --licenses="$LIC" --chdir="$RUN_DIR" "$CODE_DIR/slurm/quantum_stage.sbatch")
GPU_ARG=(); [ "$GPUS" -gt 0 ] && GPU_ARG=(--gpus="$GPUS")
CID=$(sbatch --parsable --partition="$CPART" "${GPU_ARG[@]}" --chdir="$RUN_DIR" \
      --dependency=afterok:"$QID" "$CODE_DIR/slurm/classical_stage.sbatch")
echo "quantum job $QID  ->  classical job $CID (afterok)"
echo "watch:   squeue --me    |   result: $RUN_DIR/result.json"
