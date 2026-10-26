#!/bin/bash
# Turn ONE Ubuntu machine (e.g. an AMD Developer Cloud MI300X droplet, or your workstation)
# into a minimal Slurm "cluster" with two partitions on the same node:
#     quantum : CPU slots that talk to QDMI devices (IQM Resonance over HTTPS)
#     gpu     : the AMD GPU(s) for the classical QSCI stage
# plus Slurm licences named after QDMI device IDs, so jobs can request --licenses=<device id>.
#
# This is for workshops / experimentation only -- no accounting DB, no cgroup isolation.
# Tested pattern: Ubuntu 22.04/24.04, slurm-wlm from the distro. Run as a sudo-capable user.
set -euo pipefail

DEVICE_LICENSES=${DEVICE_LICENSES:-"mqt.ddsim.default:4,iqm.garnet:1,iqm.garnet.mock:2"}  # edit to your device IDs
GPU_COUNT=${GPU_COUNT:-$(ls /dev/dri/renderD* 2>/dev/null | wc -l)}

sudo apt-get update -y
sudo apt-get install -y slurm-wlm munge
sudo systemctl enable --now munge

NODE=$(hostname -s)
CPUS=$(nproc)
MEM=$(( $(free -m | awk '/Mem:/{print $2}') - 2048 ))
GRES_LINE=""; GRES_NODE=""
if [ "${GPU_COUNT}" -gt 0 ]; then
  GRES_NODE="Gres=gpu:${GPU_COUNT}"
  FIRST=128; LAST=$((128 + GPU_COUNT - 1))
  GRES_LINE="NodeName=${NODE} Name=gpu File=/dev/dri/renderD[${FIRST}-${LAST}]"
fi

sudo mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
sudo chown slurm:slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm

sudo tee /etc/slurm/slurm.conf >/dev/null <<CONF
ClusterName=workshop
SlurmctldHost=${NODE}
AuthType=auth/munge
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
GresTypes=gpu
Licenses=${DEVICE_LICENSES}
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
ReturnToService=2
NodeName=${NODE} CPUs=${CPUS} RealMemory=${MEM} ${GRES_NODE} State=UNKNOWN
PartitionName=quantum Nodes=${NODE} MaxTime=02:00:00 State=UP
PartitionName=gpu     Nodes=${NODE} MaxTime=02:00:00 State=UP Default=YES
PartitionName=compute Nodes=${NODE} MaxTime=02:00:00 State=UP
CONF

if [ -n "${GRES_LINE}" ]; then echo "${GRES_LINE}" | sudo tee /etc/slurm/gres.conf >/dev/null; fi

sudo systemctl enable --now slurmctld slurmd
sleep 3
sudo scontrol update nodename="${NODE}" state=idle || true
sinfo
scontrol show licenses
echo "Done. Try:  srun -p gpu --gpus=1 rocm-smi   and   srun -p quantum --licenses=mqt.ddsim.default hostname"
