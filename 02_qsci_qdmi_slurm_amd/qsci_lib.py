"""qsci_lib -- Quantum-Selected Configuration Interaction split across HPC resources.

    login node        : classical preparation (PySCF + Qiskit Nature), orchestration
    quantum partition : VQE + bitstring sampling through QDMI (DDSIM locally, IQM Resonance remotely)
    gpu partition     : subspace Hamiltonian + diagonalisation with JAX (AMD GPU via ROCm, or CPU)

The same functions are used inline (no Slurm) and by the two Slurm stage scripts
(qsci_quantum_stage.py, qsci_classical_stage.py), so the notebook and the batch jobs
run identical code. Only the *backend spec strings* change between runs:

    "statevector"                  exact local simulation, no QDMI involved
    "qdmi:mqt.ddsim.default"       QDMI -> MQT DDSIM simulator on the local / classical node
    "qdmi:<iqm device id>"         QDMI -> QDMI-on-IQM -> IQM Resonance (cloud) or Radiance (on-prem)

Copyright 2026. Licensed under the Apache License, Version 2.0.
Builds on the "QSCI on AMD" tutorial by IQM Quantum Computers (N. Steinmann, D. Bulmash).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from functools import reduce
from pathlib import Path
from typing import Callable

import numpy as np

# --------------------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------------------


@dataclass
class QSCIConfig:
    """Everything a run needs. Serialised to <run_dir>/config.json and shared by all stages."""

    atom: str = "H 0 0 0; H 0 0 1"
    basis: str = "sto-3g"
    # Backend specs: "statevector" | "qdmi:<device id>"
    vqe_backend: str = "statevector"
    sample_backend: str = "qdmi:mqt.ddsim.default"
    vqe_shots: int = 1000
    vqe_maxiter: int = 60
    sample_shots: int = 2000
    cutoff: int = 10  # max number of selected configurations
    tol: float = 1e-10  # drop |H_ij| below this when assembling the subspace matrix
    seed: int = 7
    # Optional IQM session overrides (otherwise taken from env / installed manifest)
    iqm_base_url: str | None = None
    iqm_qc_alias: str | None = None

    def save(self, run_dir: str | Path) -> Path:
        path = Path(run_dir) / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, run_dir: str | Path) -> "QSCIConfig":
        return cls(**json.loads((Path(run_dir) / "config.json").read_text()))


def _write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=float))


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------------------
# 2. QDMI connection helpers
# --------------------------------------------------------------------------------------


def qdmi_session_kwargs(device_id: str, base_url: str | None = None, qc_alias: str | None = None) -> dict:
    """Session parameters for QDMI devices.

    For IQM devices the credentials normally come from the environment (RESONANCE_API_KEY /
    IQM_TOKEN / IQM_TOKENS_FILE) so the *same* code runs interactively and inside sbatch.
    Local devices such as mqt.ddsim.default need no parameters.
    """
    if not device_id.lower().startswith("iqm"):
        return {}
    kw: dict = {}
    url = base_url or os.environ.get("IQM_BASE_URL")
    token = os.environ.get("RESONANCE_API_KEY") or os.environ.get("IQM_TOKEN")
    tokens_file = os.environ.get("IQM_TOKENS_FILE")
    if url:
        kw["base_url"] = url
    if token:
        kw["token"] = token
    elif tokens_file:
        kw["auth_file"] = tokens_file
    alias = qc_alias or os.environ.get("IQM_QC_ALIAS")
    if alias:
        kw["custom2"] = alias  # QDMI-on-IQM: CUSTOM2 = quantum-computer alias selector
    return kw


def open_qdmi_backend(device_id: str, base_url: str | None = None, qc_alias: str | None = None):
    """Return a Qiskit BackendV2 that talks to `device_id` through QDMI (MQT Core adapter)."""
    from mqt.core.plugins.qiskit import QDMIBackend

    return QDMIBackend.from_device_id(device_id, **qdmi_session_kwargs(device_id, base_url, qc_alias))


def parse_backend_spec(spec: str) -> tuple[str, str | None]:
    if spec == "statevector":
        return "statevector", None
    if spec.startswith("qdmi:"):
        return "qdmi", spec.split(":", 1)[1]
    raise ValueError(f"Unknown backend spec {spec!r}; use 'statevector' or 'qdmi:<device id>'")


# --------------------------------------------------------------------------------------
# 3. Classical preparation (login node)
# --------------------------------------------------------------------------------------


def build_problem(atom: str, basis: str):
    """Electronic-structure problem from PySCF via Qiskit Nature."""
    from qiskit_nature.second_q.drivers import PySCFDriver

    return PySCFDriver(atom=atom, basis=basis).run()


def build_ansatz_and_observable(problem):
    """HF-initialised UCCSD ansatz and the Jordan-Wigner qubit Hamiltonian (electronic part)."""
    from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    mapper = JordanWignerMapper()
    hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    ansatz = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper, initial_state=hf)
    observable = mapper.map(problem.second_q_ops()[0])
    # Flatten the blueprint circuit once so every backend sees plain gates.
    return ansatz.decompose(reps=3), observable


def _bind(circuit, params, theta):
    values = {p: float(v) for p, v in zip(params, theta) if p in circuit.parameters}
    return circuit.assign_parameters(values)


# --------------------------------------------------------------------------------------
# 4. Quantum stage: energy evaluation, VQE and sampling
# --------------------------------------------------------------------------------------


def _counts_parity(counts: dict[str, int], support: list[int]) -> float:
    """<Z...Z> on `support` from counts. Bitstrings are Qiskit-ordered (qubit 0 rightmost)."""
    total, acc = 0, 0.0
    for key, c in counts.items():
        bits = key.replace(" ", "")
        ones = sum(bits[-1 - q] == "1" for q in support)
        acc += c * (-1.0 if ones % 2 else 1.0)
        total += c
    return acc / max(total, 1)


class StatevectorEnergy:
    """Exact <H>(theta) -- fast local reference, no QDMI, no shots."""

    def __init__(self, ansatz, observable):
        from qiskit.quantum_info import Statevector

        self._sv = Statevector
        self.ansatz, self.observable = ansatz, observable
        self.params = list(ansatz.parameters)
        self.evaluations = 0

    def __call__(self, theta) -> float:
        self.evaluations += 1
        state = self._sv(_bind(self.ansatz, self.params, theta))
        return float(np.real(state.expectation_value(self.observable)))


class QDMIEnergy:
    """Shot-based <H>(theta) on any QDMI device.

    Paulis are grouped into qubit-wise commuting sets; one measurement circuit per group is
    transpiled ONCE to the device target (IQM: prx + cz) and re-bound every iteration.
    Each call submits len(groups) circuits through QDMI -- this is the loop a tightly
    coupled HPC+QC system makes cheap.
    """

    def __init__(self, backend, ansatz, observable, shots: int, seed: int = 7):
        from qiskit import QuantumCircuit, transpile

        self.backend, self.shots = backend, shots
        self.params = list(ansatz.parameters)
        n = ansatz.num_qubits
        self.groups, templates = [], []
        for group in observable.group_commuting(qubit_wise=True):
            basis = ["I"] * n
            for label in group.paulis.to_labels():
                for q, ch in enumerate(reversed(label)):
                    if ch != "I":
                        basis[q] = ch
            circ = QuantumCircuit(n, n)
            circ.compose(ansatz, qubits=range(n), inplace=True)
            for q, b in enumerate(basis):
                if b == "X":
                    circ.h(q)
                elif b == "Y":
                    circ.sdg(q)
                    circ.h(q)
            circ.measure(range(n), range(n))
            templates.append(circ)
            self.groups.append(group)
        self.templates = transpile(templates, backend=backend, optimization_level=1, seed_transpiler=seed)
        self.evaluations, self.circuits_submitted = 0, 0

    def __call__(self, theta) -> float:
        self.evaluations += 1
        bound = [_bind(t, self.params, theta) for t in self.templates]
        result = self.backend.run(bound, shots=self.shots).result()
        self.circuits_submitted += len(bound)
        energy = 0.0
        for idx, group in enumerate(self.groups):
            counts = result.get_counts(idx)
            for label, coeff in zip(group.paulis.to_labels(), group.coeffs):
                support = [q for q, ch in enumerate(reversed(label)) if ch != "I"]
                energy += float(np.real(coeff)) * _counts_parity(counts, support)
        return energy


def make_energy_fn(spec: str, ansatz, observable, cfg: QSCIConfig):
    kind, device_id = parse_backend_spec(spec)
    if kind == "statevector":
        return StatevectorEnergy(ansatz, observable)
    backend = open_qdmi_backend(device_id, cfg.iqm_base_url, cfg.iqm_qc_alias)
    return QDMIEnergy(backend, ansatz, observable, cfg.vqe_shots, cfg.seed)


def run_vqe(energy_fn: Callable, n_params: int, maxiter: int, log: Callable | None = print) -> dict:
    """COBYLA from the Hartree-Fock point (theta = 0). Gradient-free, so it tolerates shot noise."""
    from scipy.optimize import minimize

    history: list[float] = []

    def wrapped(theta):
        e = energy_fn(theta)
        history.append(e)
        if log and (len(history) == 1 or len(history) % 10 == 0):
            log(f"  VQE eval {len(history):3d}: E_elec = {e:+.6f} Ha")
        return e

    t0 = time.perf_counter()
    res = minimize(wrapped, np.zeros(n_params), method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
    return {
        "theta": [float(x) for x in res.x],
        "energy": float(res.fun),
        "evaluations": len(history),
        "history": history,
        "seconds": time.perf_counter() - t0,
    }


def sample_counts(spec: str, ansatz, theta, shots: int, cfg: QSCIConfig) -> dict[str, int]:
    """Sample the trained ansatz in the computational basis."""
    from qiskit import QuantumCircuit, transpile

    kind, device_id = parse_backend_spec(spec)
    bound = _bind(ansatz, list(ansatz.parameters), theta)
    if kind == "statevector":
        from qiskit.quantum_info import Statevector

        sv = Statevector(bound)
        sv.seed(cfg.seed)
        return {k: int(v) for k, v in sv.sample_counts(shots).items()}
    n = bound.num_qubits
    circ = QuantumCircuit(n, n)
    circ.compose(bound, qubits=range(n), inplace=True)
    circ.measure(range(n), range(n))
    backend = open_qdmi_backend(device_id, cfg.iqm_base_url, cfg.iqm_qc_alias)
    tqc = transpile(circ, backend=backend, optimization_level=1, seed_transpiler=cfg.seed)
    counts = backend.run(tqc, shots=shots).result().get_counts()
    return {k.replace(" ", ""): int(v) for k, v in counts.items()}


# --------------------------------------------------------------------------------------
# 5. Classical stage: subspace Hamiltonian (Slater-Condon rules) + GPU diagonalisation
# --------------------------------------------------------------------------------------


def _popcount(x: int) -> int:
    return bin(x).count("1")


def _annihilate(det: int, p: int) -> tuple[int, int]:
    if not (det >> p) & 1:
        return 0, det
    return (-1 if _popcount(det & ((1 << p) - 1)) & 1 else 1), det ^ (1 << p)


def _create(det: int, p: int) -> tuple[int, int]:
    if (det >> p) & 1:
        return 0, det
    return (-1 if _popcount(det & ((1 << p) - 1)) & 1 else 1), det | (1 << p)


def _phase(det: int, annihilate: list[int], create: list[int]) -> tuple[int, int]:
    """Apply a_{annihilate[0]}, a_{annihilate[1]}, ..., then a^dag_{create[0]}, ... (right to left)."""
    sign, d = 1, det
    for p in annihilate:
        s, d = _annihilate(d, p)
        sign *= s
    for p in create:
        s, d = _create(d, p)
        sign *= s
    return sign, d


def _full_mbh_canonical_from_qsci(selected_states, h_diag, h_offdiag, h2, num_qubits, tol=1e-10):
    """Many-body Hamiltonian restricted to the QSCI-selected determinants (COO triplets).

    Conventions (match Qiskit Nature's Jordan-Wigner mapping):
      * spin orbital p = qubit p; p < N is alpha orbital p, p >= N is beta orbital p - N
      * h_diag    : length-2N diagonal of the one-body operator (alpha block, beta block)
      * h_offdiag : N x N off-diagonal part of the spatial one-body integrals (MO basis)
      * h2        : N x N x N x N spatial two-electron integrals, chemists' notation (pq|rs)
    Returns (rows, cols, vals) for the *electronic* Hamiltonian (no nuclear repulsion).
    """
    h_diag = np.asarray(h_diag, dtype=float)
    h2 = np.asarray(h2, dtype=float)
    n_spatial = num_qubits // 2
    h1 = np.asarray(h_offdiag, dtype=float) + np.diag(h_diag[:n_spatial])

    def h1s(p, q):
        return h1[p % n_spatial, q % n_spatial] if (p // n_spatial) == (q // n_spatial) else 0.0

    def g(p, q, r, s):  # <pq|rs> = (pr|qs) with spin deltas
        if (p // n_spatial) != (r // n_spatial) or (q // n_spatial) != (s // n_spatial):
            return 0.0
        return h2[p % n_spatial, r % n_spatial, q % n_spatial, s % n_spatial]

    def anti(p, q, r, s):  # <pq||rs>
        return g(p, q, r, s) - g(p, q, s, r)

    dets = [int(d) for d in selected_states]
    rows, cols, vals = [], [], []
    for k, bra in enumerate(dets):
        for l, ket in enumerate(dets):
            diff = bra ^ ket
            degree = _popcount(diff) // 2
            if degree > 2:
                continue
            if degree == 0:
                occ = [p for p in range(num_qubits) if (ket >> p) & 1]
                v = sum(h1s(i, i) for i in occ)
                v += 0.5 * sum(anti(i, j, i, j) for i in occ for j in occ)
            else:
                holes = [p for p in range(num_qubits) if (diff >> p) & 1 and (ket >> p) & 1]
                parts = [p for p in range(num_qubits) if (diff >> p) & 1 and (bra >> p) & 1]
                if degree == 1:
                    (i,), (a,) = holes, parts
                    sign, d = _phase(ket, [i], [a])
                    occ = [p for p in range(num_qubits) if (ket >> p) & 1 and p != i]
                    v = sign * (h1s(a, i) + sum(anti(a, j, i, j) for j in occ))
                else:
                    (i, j), (a, b) = holes, parts
                    sign, d = _phase(ket, [i, j], [b, a])
                    v = sign * anti(a, b, i, j)
                assert d == bra, "fermionic phase bookkeeping failed"
            if abs(v) > tol:
                rows.append(k)
                cols.append(l)
                vals.append(float(v))
    return rows, cols, vals


def select_configurations(counts: dict[str, int], num_qubits: int, num_particles: tuple[int, int], cutoff: int):
    """Keep the most frequent bitstrings with the right (N_alpha, N_beta). Right half = alpha qubits."""
    n_alpha, n_beta = num_particles
    half = num_qubits // 2
    selected = []
    for state, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        bits = state.replace(" ", "").zfill(num_qubits)
        if bits[half:].count("1") == n_alpha and bits[:half].count("1") == n_beta:
            selected.append(int(bits, 2))
            if len(selected) >= cutoff:
                break
    return selected


def jax_device_report() -> str:
    try:
        import jax

        return ", ".join(f"{d.platform}:{d.device_kind}" for d in jax.devices())
    except Exception as exc:  # pragma: no cover
        return f"JAX unavailable ({exc.__class__.__name__}); using NumPy"


def postprocess_gpu(atom: str, basis: str, counts: dict[str, int], cutoff: int = 10, tol: float = 1e-10) -> dict:
    """Filter -> select -> build H_S -> eigvalsh. JAX runs on the AMD GPU when ROCm JAX is installed."""
    from pyscf import ao2mo, gto, scf

    try:
        import jax
        import jax.numpy as xp

        jax.config.update("jax_enable_x64", True)
        backend_name = "jax"
    except ImportError:  # still runs, just not on the GPU
        xp, backend_name = np, "numpy"

    num_qubits = len(next(iter(counts)).replace(" ", ""))
    mol = gto.Mole(verbose=0)
    mol.build(atom=atom, basis=basis)
    mf = scf.RHF(mol)
    mf.kernel()
    nelec = mol.nelec  # (alpha, beta)
    selected = select_configurations(counts, num_qubits, nelec, cutoff)
    if len(selected) < 2:
        raise ValueError("Fewer than 2 physical configurations sampled -- increase shots or check the ansatz.")

    mo = np.asarray(mf.mo_coeff)
    h1 = reduce(np.dot, (mo.T, mf.get_hcore(), mo))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, mo), mol.nao)
    h_diag = np.kron(np.array([1.0, 1.0]), np.diag(h1))
    h_offdiag = h1 - np.diag(np.diag(h1))

    t0 = time.perf_counter()
    r, c, v = _full_mbh_canonical_from_qsci(selected, h_diag, h_offdiag, h2, num_qubits, tol)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    n = len(selected)
    if backend_name == "jax":
        hr = xp.zeros((n, n), dtype=xp.float64).at[xp.array(r), xp.array(c)].set(xp.array(v))
        eigvals = xp.linalg.eigvalsh(hr)
        e0 = float(eigvals[0])
    else:
        hr = np.zeros((n, n))
        hr[r, c] = v
        e0 = float(np.linalg.eigvalsh(hr)[0])
    t_diag = time.perf_counter() - t0

    return {
        "e_electronic": e0,
        "e_nuclear": float(mol.energy_nuc()),
        "e_total": e0 + float(mol.energy_nuc()),
        "e_hf": float(mf.e_tot),
        "n_selected": n,
        "selected_states": [format(s, f"0{num_qubits}b") for s in selected],
        "nnz": len(v),
        "seconds_build": t_build,
        "seconds_diag": t_diag,
        "linear_algebra": backend_name,
        "devices": jax_device_report() if backend_name == "jax" else "cpu (numpy)",
    }


def fci_reference(atom: str, basis: str) -> float:
    """Exact (FCI) total energy from PySCF -- replaces NumPyMinimumEigensolver from the original notebook."""
    from pyscf import fci, gto, scf

    mol = gto.Mole(verbose=0)
    mol.build(atom=atom, basis=basis)
    mf = scf.RHF(mol)
    mf.kernel()
    e, _ = fci.FCI(mf).kernel()
    return float(e)


# --------------------------------------------------------------------------------------
# 6. Stages (called inline by the notebook or by the sbatch scripts)
# --------------------------------------------------------------------------------------


def quantum_stage(run_dir: str | Path, log: Callable = print) -> dict:
    """VQE + sampling. Runs on a quantum access node (QAN) or inline."""
    run_dir = Path(run_dir)
    cfg = QSCIConfig.load(run_dir)
    # Inside Slurm, the device licence chosen at submission time wins (as in the MQT playground).
    lic = os.environ.get("SLURM_JOB_LICENSES", "")
    if lic and cfg.sample_backend.startswith("qdmi:"):
        cfg.sample_backend = "qdmi:" + lic.split(",")[0].split(":")[0]
    log(f"[quantum] host={os.uname().nodename} vqe={cfg.vqe_backend} sample={cfg.sample_backend}")

    problem = build_problem(cfg.atom, cfg.basis)
    ansatz, observable = build_ansatz_and_observable(problem)
    log(f"[quantum] {ansatz.num_qubits} qubits, {ansatz.num_parameters} parameters, {len(observable)} Pauli terms")

    energy_fn = make_energy_fn(cfg.vqe_backend, ansatz, observable, cfg)
    vqe = run_vqe(energy_fn, ansatz.num_parameters, cfg.vqe_maxiter, log)
    log(f"[quantum] VQE done: E_elec = {vqe['energy']:+.6f} Ha after {vqe['evaluations']} evaluations")

    t0 = time.perf_counter()
    counts = sample_counts(cfg.sample_backend, ansatz, vqe["theta"], cfg.sample_shots, cfg)
    t_sample = time.perf_counter() - t0
    out = {
        "counts": counts,
        "vqe": vqe,
        "num_qubits": ansatz.num_qubits,
        "nuclear_repulsion_energy": float(problem.nuclear_repulsion_energy),
        "sample_backend": cfg.sample_backend,
        "seconds_sampling": t_sample,
        "host": os.uname().nodename,
    }
    _write_json(run_dir / "quantum.json", out)
    log(f"[quantum] wrote {run_dir / 'quantum.json'} ({len(counts)} distinct bitstrings)")
    return out


def classical_stage(run_dir: str | Path, log: Callable = print) -> dict:
    """Subspace construction + diagonalisation. Runs on a (GPU) compute node or inline."""
    run_dir = Path(run_dir)
    cfg = QSCIConfig.load(run_dir)
    q = _read_json(run_dir / "quantum.json")
    log(f"[classical] host={os.uname().nodename} devices={jax_device_report()}")
    res = postprocess_gpu(cfg.atom, cfg.basis, q["counts"], cfg.cutoff, cfg.tol)
    res["e_fci"] = fci_reference(cfg.atom, cfg.basis)
    res["error_mHa"] = 1000.0 * (res["e_total"] - res["e_fci"])
    res["host"] = os.uname().nodename
    _write_json(run_dir / "result.json", res)
    log(
        f"[classical] E_QSCI = {res['e_total']:.8f} Ha | E_FCI = {res['e_fci']:.8f} Ha | "
        f"error = {res['error_mHa']:.3f} mHa | |S| = {res['n_selected']}"
    )
    return res


def run_inline(cfg: QSCIConfig, run_dir: str | Path, log: Callable = print) -> dict:
    cfg.save(run_dir)
    quantum_stage(run_dir, log)
    return classical_stage(run_dir, log)


# --------------------------------------------------------------------------------------
# 7. Slurm orchestration (login node)
# --------------------------------------------------------------------------------------


def slurm_available() -> bool:
    return shutil.which("sbatch") is not None


def sbatch(script: str | Path, *args: str, env: dict | None = None) -> str:
    cmd = ["sbatch", "--parsable", *args, str(script)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, env={**os.environ, **(env or {})})
    return out.stdout.strip().split(";")[0]


def submit_pipeline(
    run_dir: str | Path,
    quantum_script: str | Path,
    classical_script: str | Path,
    device_license: str | None = None,
    quantum_partition: str = "quantum",
    classical_partition: str = "compute",
    gpus: int = 0,
    extra_quantum: tuple[str, ...] = (),
    extra_classical: tuple[str, ...] = (),
) -> tuple[str, str]:
    """quantum stage -> (afterok) -> classical stage. Returns both job ids."""
    run_dir = str(Path(run_dir).resolve())
    env = {"QSCI_RUN_DIR": run_dir, "QSCI_CODE_DIR": str(Path(__file__).resolve().parent)}
    q_args = [f"--partition={quantum_partition}", f"--chdir={run_dir}", *extra_quantum]
    if device_license:
        q_args.append(f"--licenses={device_license}")
    qid = sbatch(quantum_script, *q_args, env=env)
    c_args = [f"--partition={classical_partition}", f"--chdir={run_dir}", f"--dependency=afterok:{qid}", *extra_classical]
    if gpus:
        c_args.append(f"--gpus={gpus}")
    cid = sbatch(classical_script, *c_args, env=env)
    return qid, cid


def job_state(job_id: str) -> str:
    out = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T"], capture_output=True, text=True)
    state = out.stdout.strip()
    if state:
        return state
    out = subprocess.run(["sacct", "-n", "-X", "-j", job_id, "-o", "State"], capture_output=True, text=True)
    return (out.stdout.strip().split() or ["UNKNOWN"])[0]


def wait_for(job_ids: list[str], poll: float = 5.0, timeout: float = 3600.0, log: Callable = print) -> dict:
    t0, last = time.time(), {}
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
    while time.time() - t0 < timeout:
        states = {j: job_state(j) for j in job_ids}
        if states != last:
            log("  " + "  ".join(f"{j}:{s}" for j, s in states.items()))
            last = states
        if all(s.split()[0] in terminal for s in states.values()):
            return states
        time.sleep(poll)
    raise TimeoutError(f"jobs {job_ids} not finished after {timeout:.0f}s")


# --------------------------------------------------------------------------------------
# 8. Self-test of the Slater-Condon builder (no quantum or chemistry packages needed)
# --------------------------------------------------------------------------------------


def _selftest_h2() -> float:
    """H2/STO-3G at R = 1.4 bohr, MO integrals from Szabo & Ostlund. Exact E_elec = -1.8516 Ha (E_tot = -1.1373 Ha)."""
    h1 = np.diag([-1.2528, -0.4756])
    h2 = np.zeros((2, 2, 2, 2))
    h2[0, 0, 0, 0], h2[1, 1, 1, 1] = 0.6746, 0.6975
    h2[0, 0, 1, 1] = h2[1, 1, 0, 0] = 0.6636
    for idx in [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]:
        h2[idx] = 0.1813
    dets = [0b0101, 0b0110, 0b1001, 0b1010]  # one alpha (bits 0-1) + one beta (bits 2-3)
    r, c, v = _full_mbh_canonical_from_qsci(dets, np.kron([1, 1], np.diag(h1)), h1 - np.diag(np.diag(h1)), h2, 4)
    H = np.zeros((4, 4))
    H[r, c] = v
    assert np.allclose(H, H.T), "Hamiltonian not symmetric"
    return float(np.linalg.eigvalsh(H)[0])


if __name__ == "__main__":
    e = _selftest_h2()
    print(f"Slater-Condon self-test: E_elec(H2) = {e:.4f} Ha (expected -1.8516)")
