#!/bin/bash
# Laptop / workstation setup for Lab 1 (Linux, macOS, WSL2). Python >= 3.11.
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "mqt-core[qiskit]" "iqm-qdmi[qiskit]" jupyterlab matplotlib
python -c "import mqt.core, qiskit; print('mqt-core', mqt.core.__version__, '| qiskit', qiskit.__version__)"
python qdmi_demo.py list
echo
echo "Next:  python qdmi_demo.py bell --device local            # classical node only"
echo "       cp env.example .env && \$EDITOR .env && set -a && source .env && set +a"
echo "       python qdmi_demo.py info --device auto-iqm          # quantum node (IQM Resonance)"
