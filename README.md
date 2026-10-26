# Hybrid Quantum + HPC with IQM: 3-hour hands-on workshop

Materials for an HPC audience that is new to quantum: an IQM-style slide deck, three Jupyter labs and the
scripts to run them on a laptop, a Docker "mini HPC centre", an AMD GPU droplet or a real Slurm cluster.

The thread through the whole workshop is a single idea: **a QPU is another scheduled accelerator.** The
standard device interface **QDMI** lets the same code run on a laptop simulator, on IQM Resonance in the cloud, or on
an on-prem IQM Radiance that appears as a Slurm node. Later blocks show why hybrid algorithms and error
correction push that integration from seconds to microseconds.

## Run of show (180 min)

| Time | Block | Format | Material |
|---|---|---|---|
| 0:00 | Welcome, goals, environment check | talk | deck §0 |
| 0:05 | **1 · Who is IQM, IQM Resonance** (15) | talk + live demo of the Resonance UI | deck §1 |
| 0:20 | **2 · Structure of an HPC + QC supercomputer** (15) | talk | deck §2 |
| 0:35 | **3 · QDMI, and running it on your computer** (30) | 10 min talk, 20 min lab | deck §3 · `01_qdmi_local_to_resonance/` |
| 1:05 | Break (10) | | |
| 1:15 | **4 · Hybrid algorithms on Resonance, then through QDMI** (30) | 8 min talk, 22 min lab | deck §4 · `03_qrisp_intro/` |
| 1:45 | **5 · (Optional) QSCI: IQM QPU + AMD GPU through Slurm** (30) | 8 min talk, 22 min lab/demo | deck §5 · `02_qsci_qdmi_slurm_amd/` |
| 2:15 | Break (5) | | |
| 2:20 | **6 · Future of hybrid algorithms: why tight integration** (15) | talk | deck §6 |
| 2:35 | **7 · Tight integration for QEC** (15) | talk | deck §7 |
| 2:50 | Wrap-up, resources, Q&A (10) | discussion | deck §8 |

If block 5 is skipped, run it as an instructor demo (5 min) and give the time to Q&A or to Lab 1 step 5 (Slurm playground).

## Slide deck
`IQM_Hybrid_Quantum_HPC_Workshop.pptx` is built on the IQM presentation template: 42 slides with speaker notes, following the run of show above.
Fill in the placeholders on the cover (presenter, venue, date) and the repo URL on the closing slide.

## Before the workshop

**Participants (send 1 week ahead)**
1. Create a free IQM Resonance account (https://resonance.meetiqm.com/signup), then generate an API token.
2. Python ≥ 3.11 and `git`. Then in `01_qdmi_local_to_resonance/`: `./setup_local.sh`. It must end by printing a device list containing `mqt.ddsim.default` and at least one `iqm.*` ID.
3. *(Optional, for the Slurm part)* Docker Desktop/Engine with ≥4 CPUs and 12 GiB RAM. Pre-build the playground: `./slurm/start_playground.sh` (the first build compiles MQT Core and takes a while).
4. `pip install "qrisp[iqm]" networkx` for Lab 2.

**Instructor**
- Confirm the IQM device IDs your accounts see (`python qdmi_demo.py list`) and put them in `slurm/start_playground.sh` (`IQM_LICENSES`) and in the slides' lab slides if they differ from `iqm.garnet` / `iqm.garnet.mock`.
- Decide the hardware budget. Everything defaults to `:mock` devices (zero credits). One real-QPU Bell run plus QSCI sampling per participant is small; VQE on hardware is not (see Lab 3, §6).
- For block 5, either provision an AMD Developer Cloud GPU droplet and run `02_qsci_qdmi_slurm_amd/slurm/single_node_slurm_droplet.sh`, or use your site's AMD GPU partition.
- Dry-run all three notebooks end to end on the machine you will project from.

## The labs

### `01_qdmi_local_to_resonance/`: Lab 1, QDMI from your laptop to Resonance
| File | Purpose |
|---|---|
| `QDMI_Tutorial_Local_to_Resonance.ipynb` | guided lab: discover devices → local DDSIM (classical node) → IQM Resonance (quantum node) → property queries → raw IQM-JSON job → Qiskit via `QDMIBackend` → round-trip latency loop → Slurm |
| `qdmi_demo.py` | the same steps as a CLI (`list`, `info`, `bell --via raw/qiskit`, `loop`, `analyze`); runs unchanged on a laptop and inside `sbatch` |
| `setup_local.sh`, `env.example` | venv with `mqt-core[qiskit]` + `iqm-qdmi[qiskit]`; credentials template |
| `slurm/start_playground.sh` | builds the MQT Core Slurm playground (login, controller, compute nodes, quantum access nodes, accounting) with the IQM device installed and IQM licences added |
| `slurm/bell.sbatch`, `slurm/analyze.sbatch` | quantum stage on the `quantum` partition with `--licenses=<device id>`, classical stage chained with `afterok` |

Quick check without Jupyter:
```bash
python qdmi_demo.py list
python qdmi_demo.py bell --device local
set -a; source .env; set +a
python qdmi_demo.py info --device auto-iqm
python qdmi_demo.py bell --device auto-iqm --via qiskit --shots 500
python qdmi_demo.py loop --device auto-iqm --steps 6
```

### `03_qrisp_intro/`: Lab 2, Qrisp on IQM
`Intro_to_Qrisp_on_IQM.ipynb` follows the IQM Academy *Get started with Qrisp on IQM* structure (install → imports →
simulator → token → mock vs. real) and extends it with Grover (`x² = 0.25`), QAOA MaxCut on simulator and
Resonance mock, and a **Qrisp → QDMI bridge** (`VirtualBackend` wrapping `QDMIBackend`), so Qrisp code runs on any QDMI device.

### `02_qsci_qdmi_slurm_amd/`: Lab 3, QSCI through QDMI, Slurm and an AMD GPU
**What it does.** You want to do linear algebra on a very large matrix, in principle too large to work with classically.
You input the matrix. The quantum part of the workflow shrinks the matrix in a way that shouldn't change the output you are
looking for. The classical (GPU) part then performs the linear algebra on the smaller matrix. In the application, the input is
the Hamiltonian of a molecule, which gives the energy of the molecule as a function of the state of its electrons. The output is
the approximate ground-state energy (the lowest possible energy) of the electrons. The lab runs H₂, but the same code runs on
more complicated molecules.

| File | Purpose |
|---|---|
| `QSCI_QDMI_Slurm_AMD.ipynb` | the QSCI workflow with switchable backends (`statevector`, `qdmi:mqt.ddsim.default`, `qdmi:<iqm id>`) and run modes (`inline`, `slurm`) |
| `qsci_lib.py` | all algorithm code: QDMI energy estimator, COBYLA VQE, sampling, Slater–Condon subspace Hamiltonian, JAX post-processing, Slurm helpers. `python qsci_lib.py` runs a self-test |
| `qsci_quantum_stage.py`, `qsci_classical_stage.py` | the two stages as batch entry points, sharing a run directory (`config.json` → `quantum.json` → `result.json`) |
| `slurm/quantum_stage.sbatch`, `slurm/classical_stage.sbatch`, `slurm/submit_pipeline.sh` | quantum partition + device licence → GPU partition with `--dependency=afterok` |
| `slurm/single_node_slurm_droplet.sh` | turns one Ubuntu GPU machine (e.g. AMD MI300X droplet) into a Slurm "cluster" with `quantum` + `gpu` partitions and QDMI device licences |
| `QSCI_on_AMD_original.ipynb`, `qsci_workflow.png` | the original IQM notebook and figure, for reference |

**Changes from the original notebook.**
- Quantum access goes through QDMI instead of `IQMProvider`.
- The helper functions the original referenced (`estimate_iqm`, `sample_iqm`, `_full_mbh_canonical_from_qsci`) are implemented in `qsci_lib.py`. The Hamiltonian builder is verified against brute-force second quantisation.
- VQE uses COBYLA, because L-BFGS-B with finite differences does not cope with shot noise.
- The exact reference is PySCF FCI, which removes the need for the qiskit-algorithms SciPy patch.
- `from __future__ import annotations` moved to the top. It must be the first statement in a cell.

**AMD GPU (ROCm) JAX.** On the GPU node, install a ROCm build of JAX matching the node's ROCm version (AMD publishes
ROCm JAX wheels and containers; on DigitalOcean AMD droplets the ROCm stack is preinstalled). Verify with
`python -c "import jax; print(jax.devices())"` and expect `rocm` devices. In `classical_stage.sbatch` you can set
`JAX_PLATFORMS=rocm` to fail loudly instead of silently falling back to CPU.

## Three ways to run the Slurm parts
| Where | Classical node | Quantum node | Setup |
|---|---|---|---|
| Laptop + Docker | playground `node` containers (CPU) | playground `qan` → DDSIM or IQM Resonance via QDMI | `01_.../slurm/start_playground.sh` |
| AMD GPU droplet | the droplet's GPU (`gpu` partition) | same droplet, `quantum` partition → IQM Resonance | `02_.../slurm/single_node_slurm_droplet.sh` |
| Real cluster (e.g. LUMI-G) | your AMD GPU partition | a partition/QAN with outbound HTTPS, or an on-prem IQM Radiance | load modules, set partitions/account in the notebook's `SLURM` dict |

## Verification status
The Slater–Condon builder, the IQM-native Bell program and all scripts and notebooks were checked offline (unit tests, syntax).
Calls into `mqt-core`, `iqm-qdmi`, Qiskit Nature, Qrisp and Resonance follow the published documentation as of October 2026 but
were **not executed** in the authoring environment, so do the instructor dry-run. Two known risks:
- **IQM device IDs.** Names such as `iqm.garnet.mock` depend on your installed `iqm-qdmi` version and account. Always check with `qdmi_demo.py list`.
- **Qiskit Nature 0.7.2 and Qiskit 2.x.** These may disagree. If imports fail, pin the versions from the `uv.lock` of `iqm-finland/QDMI-on-IQM`, whose examples run QSCI through QDMI.

## References
- QDMI-on-IQM: https://github.com/iqm-finland/QDMI-on-IQM · docs https://iqm-finland.github.io/QDMI-on-IQM/ · PyPI `iqm-qdmi`
- QDMI specification: https://github.com/Munich-Quantum-Software-Stack/QDMI
- Burgholzer et al., *Practical HPCQC Integration with QDMI: A Real-Hardware Case Study with IQM Systems*, arXiv:2604.19869
- MQT Core QDMI Qiskit backend and Slurm playground: https://mqt.readthedocs.io/projects/core/en/latest/qdmi/
- IQM Resonance https://resonance.meetiqm.com · IQM Academy https://www.iqmacademy.com · Qrisp https://qrisp.eu · IQM QAOA https://docs.meetiqm.com/iqm-qaoa/
