#!/bin/bash
# Lab 1c: a complete mini HPC centre on your laptop, from the MQT Core Slurm playground:
#   login | controller | 2x classical "node" (compute partition) | 2x "qan" (quantum partition) | slurmdbd+MariaDB
# Requirements: Docker Engine (Linux) or Docker Desktop (macOS) with >= 4 CPUs and 12 GiB RAM,
# cgroup v2, Enhanced Container Isolation disabled. First build takes a while (it compiles MQT Core).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HOME/qdmi-workshop}
mkdir -p "$WORK" && cd "$WORK"
[ -d core ] || git clone --depth 1 https://github.com/munich-quantum-toolkit/core.git
cd core

sh examples/slurm/prepare.sh                       # creates build/slurm/{slurm.conf,jobs,...}

# 1) Install the IQM QDMI device (and anything your jobs import) into the shared /opt/venv
export MQT_CORE_SLURM_PACKAGES="$(grep -v '^#' "$HERE/devices.txt" | xargs)"

# 2) Add Slurm licences named after the IQM device IDs you want to schedule.
#    Check the exact IDs first with:  python ../qdmi_demo.py list   (on your laptop venv)
IQM_LICENSES=${IQM_LICENSES:-"iqm.garnet.mock:2,iqm.garnet:1"}
if grep -q '^Licenses=' build/slurm/slurm.conf; then
  sed -i.bak -E "s|^(Licenses=.*)$|\1,${IQM_LICENSES}|" build/slurm/slurm.conf
else
  echo "Licenses=${IQM_LICENSES}" >> build/slurm/slurm.conf
fi
ACC=$(echo "$IQM_LICENSES" | tr ',' '\n' | cut -d: -f1 | sed 's|^|license/|' | paste -sd, -)
sed -i.bak -E "s|^(AccountingStorageTRES=.*)$|\1,${ACC}|" build/slurm/slurm.conf
grep -E '^(Licenses|AccountingStorageTRES)=' build/slurm/slurm.conf

# 3) Make the workshop scripts visible as /jobs inside every container
mkdir -p build/slurm/jobs
cp "$HERE/../qdmi_demo.py" "$HERE/bell.sbatch" "$HERE/analyze.sbatch" build/slurm/jobs/
[ -d "$HERE/../../02_qsci_qdmi_slurm_amd" ] && cp -r "$HERE/../../02_qsci_qdmi_slurm_amd" build/slurm/jobs/qsci

docker compose -f examples/slurm/compose.yml up --build -d --wait
echo
echo "Log in with:"
echo "  docker compose -f $WORK/core/examples/slurm/compose.yml exec --user 10000:10000 login bash --login"
echo "Then:  sinfo; scontrol show licenses; export RESONANCE_API_KEY=...; cd /jobs; sbatch --licenses=mqt.ddsim.default bell.sbatch"
