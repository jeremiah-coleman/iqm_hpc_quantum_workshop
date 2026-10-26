#!/bin/bash
# Classical follow-up stage: runs on the compute partition, holds NO quantum licence.
#   QID=$(sbatch --parsable --licenses=mqt.ddsim.default bell.sbatch)
#   sbatch --dependency=afterok:$QID --export=ALL,BELL_JOB=$QID analyze.sbatch
#SBATCH --job-name=qdmi-analyze
#SBATCH --partition=compute
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out
set -euo pipefail
srun python /jobs/qdmi_demo.py analyze --in "/jobs/bell-${BELL_JOB}.json"
