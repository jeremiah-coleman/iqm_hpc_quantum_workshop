#!/usr/bin/env python3
"""qdmi_demo.py -- one script, three places to run it:

    1. your laptop (the "classical node"), talking to the local MQT DDSIM QDMI device
    2. your laptop, talking to IQM Resonance (the "quantum node") through QDMI-on-IQM
    3. inside a Slurm job on a quantum access node (see slurm/bell.sbatch)

Nothing in the code changes between the three -- only the device ID and the environment.

Sub-commands
    list                              registered QDMI device IDs (no network, no libraries loaded)
    info    --device ID               device, site (qubit) and operation properties via QDMI queries
    bell    --device ID [--via raw|qiskit] [--shots N] [--out file.json]
    loop    --device ID [--steps N]   classical node drives quantum node: theta sweep + round-trip timing
    analyze --in file.json            classical post-processing of saved counts (e.g. on a compute node)

Environment (only needed for IQM devices; see env.example)
    RESONANCE_API_KEY or IQM_TOKEN   API token from https://resonance.meetiqm.com  ("Create token")
    IQM_TOKENS_FILE                  alternative: path to an IQM tokens file
    IQM_BASE_URL                     default comes from the installed device manifest
    IQM_QC_ALIAS                     e.g. garnet, emerald, sirius (QDMI CUSTOM2 selector)
    IQM_DEVICE_ID                    default device ID for the IQM sub-commands

Copyright 2026. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

LOCAL_DEVICE = "mqt.ddsim.default"


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def _prop(obj, name, *args):
    """Call a QDMI property getter if it exists; QDMI devices may legally not support a property."""
    attr = getattr(obj, name, None)
    if attr is None:
        return None
    try:
        return attr(*args) if callable(attr) else attr
    except Exception as exc:  # QDMI_ERROR_NOTSUPPORTED and friends
        return f"<{exc.__class__.__name__}>"


def session_kwargs(device_id: str) -> dict:
    """Explicit session parameters for IQM devices; empty for local devices."""
    if not device_id.lower().startswith("iqm"):
        return {}
    kw = {}
    if os.environ.get("IQM_BASE_URL"):
        kw["base_url"] = os.environ["IQM_BASE_URL"]
    token = os.environ.get("RESONANCE_API_KEY") or os.environ.get("IQM_TOKEN")
    if token:
        kw["token"] = token
    elif os.environ.get("IQM_TOKENS_FILE"):
        kw["auth_file"] = os.environ["IQM_TOKENS_FILE"]
    if os.environ.get("IQM_QC_ALIAS"):
        kw["custom2"] = os.environ["IQM_QC_ALIAS"]  # QDMI-on-IQM: CUSTOM2 = QC alias selector
    if not (token or os.environ.get("IQM_TOKENS_FILE")):
        print("! no RESONANCE_API_KEY / IQM_TOKEN / IQM_TOKENS_FILE set -- the IQM session will likely fail")
    return kw


def resolve_device(device: str | None) -> str:
    if device in (None, "", "auto-iqm"):
        device = os.environ.get("IQM_DEVICE_ID") or _first_iqm_device()
    if device == "local":
        device = LOCAL_DEVICE
    if device.startswith("slurm"):  # take the device licence granted to this Slurm job
        device = os.environ["SLURM_JOB_LICENSES"].split(",")[0].split(":")[0]
    return device


def _first_iqm_device() -> str:
    from mqt.core.qdmi import builtin_driver

    ids = [d for d in builtin_driver.registered_device_ids() if d.lower().startswith("iqm")]
    if not ids:
        sys.exit("No IQM QDMI device registered. Did you `pip install iqm-qdmi`? Run `qdmi_demo.py list`.")
    return ids[0]


def open_device(device_id: str):
    """Open ONE device by stable ID through the builtin MQT Core QDMI driver."""
    from mqt.core.qdmi import builtin_driver

    t0 = time.perf_counter()
    kw = session_kwargs(device_id)
    try:
        dev = builtin_driver.open_device(device_id, **kw)
    except TypeError:  # older driver without keyword session parameters -> rely on environment
        dev = builtin_driver.open_device(device_id)
    print(f"opened {device_id!r} in {time.perf_counter() - t0:.2f}s (session init = auth + calibration fetch)")
    return dev


def coupled_pair(dev) -> tuple:
    cmap = _prop(dev, "coupling_map")
    if isinstance(cmap, (list, tuple)) and cmap:
        return tuple(cmap[0])
    sites = _prop(dev, "sites") or []
    return tuple(sites[:2])


def site_label(site, fallback_index: int) -> str:
    name = _prop(site, "name")
    if isinstance(name, str) and not name.startswith("<"):
        return name
    idx = _prop(site, "index")
    return f"QB{(idx if isinstance(idx, int) else fallback_index) + 1}"


# ----------------------------------------------------------------------------------------
# programs
# ----------------------------------------------------------------------------------------
BELL_QASM2 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""


def bell_iqm_json(q0: str = "QB1", q1: str = "QB2") -> str:
    """Bell state in IQM's native gate set: PRX (angles in full turns) + CZ.

    RY(pi/2) on both, CZ, RY(-pi/2) on the target  ==  H . CNOT up to a phase.
    PRX(theta, phi): angle_t = theta / 2pi, phase_t = phi / 2pi  (phi = pi/2 -> RY).
    """
    prog = {
        "name": "bell_state",
        "instructions": [
            {"name": "prx", "locus": [q0], "args": {"angle_t": 0.25, "phase_t": 0.25}},
            {"name": "prx", "locus": [q1], "args": {"angle_t": 0.25, "phase_t": 0.25}},
            {"name": "cz", "locus": [q0, q1], "args": {}},
            {"name": "prx", "locus": [q1], "args": {"angle_t": 0.25, "phase_t": 0.75}},
            {"name": "measure", "locus": [q0, q1], "args": {"key": "m"}},
        ],
    }
    return json.dumps(prog)


def bell_qiskit():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


# ----------------------------------------------------------------------------------------
# sub-commands
# ----------------------------------------------------------------------------------------
def cmd_list(_args) -> None:
    from mqt.core.qdmi import builtin_driver

    ids = builtin_driver.registered_device_ids()
    print(f"{len(ids)} registered QDMI device(s):")
    for d in ids:
        where = "IQM (remote)" if d.lower().startswith("iqm") else "local"
        print(f"  {d:<40} {where}")


def cmd_info(args) -> None:
    dev = open_device(resolve_device(args.device))
    print("\n-- device properties (QDMI_DEVICE_PROPERTY_*) --")
    for p in ["name", "version", "status", "library_version", "qubits_num", "needs_calibration",
              "duration_unit", "duration_scale_factor", "supported_program_formats"]:
        print(f"  {p:<26} {_prop(dev, p)}")
    sites = _prop(dev, "sites") or []
    cmap = _prop(dev, "coupling_map") or []
    print(f"  {'sites':<26} {len(sites) if isinstance(sites, list) else sites}")
    print(f"  {'coupling_map (edges)':<26} {len(cmap) if isinstance(cmap, list) else cmap}")

    print("\n-- first sites (QDMI_SITE_PROPERTY_*) --")
    for k, s in enumerate(sites[: args.max_sites] if isinstance(sites, list) else []):
        print(f"  {site_label(s, k):<6} index={_prop(s, 'index')}  T1={_prop(s, 't1')}  T2={_prop(s, 't2')}")

    print("\n-- operations (QDMI_OPERATION_PROPERTY_*) --")
    for op in (_prop(dev, "operations") or [])[:10]:
        print(f"  {str(_prop(op, 'name')):<10} qubits={_prop(op, 'qubits_num')}  params={_prop(op, 'parameters_num')}"
              f"  fidelity={_prop(op, 'fidelity')}  duration={_prop(op, 'duration')}")


def _save(counts: dict, args, meta: dict) -> None:
    print(f"counts: {dict(sorted(counts.items()))}")
    if args.out:
        Path(args.out).write_text(json.dumps({"counts": counts, **meta}, indent=2))
        print(f"saved -> {args.out}")


def cmd_bell(args) -> None:
    device_id = resolve_device(args.device)
    meta = {"device": device_id, "shots": args.shots, "via": args.via, "host": os.uname().nodename}
    if args.via == "qiskit":
        from mqt.core.plugins.qiskit import QDMIBackend
        from qiskit import transpile

        t0 = time.perf_counter()
        backend = QDMIBackend.from_device_id(device_id, **session_kwargs(device_id))
        print(f"QDMIBackend {backend.name!r}: {backend.target.num_qubits} qubits, "
              f"ops={sorted(backend.target.operation_names)[:8]}... ({time.perf_counter() - t0:.2f}s)")
        tqc = transpile(bell_qiskit(), backend=backend, optimization_level=1, seed_transpiler=1)
        print(f"transpiled to native gates: {dict(tqc.count_ops())}")
        t0 = time.perf_counter()
        counts = backend.run(tqc, shots=args.shots).result().get_counts()
        meta["seconds_execute"] = time.perf_counter() - t0
        print(f"executed in {meta['seconds_execute']:.2f}s (submit -> queue -> run -> results)")
        return _save({k.replace(' ', ''): int(v) for k, v in counts.items()}, args, meta)

    # --- raw QDMI path: the same calls you would make from C/C++/Fortran ---
    from mqt.core.qdmi import ProgramFormat

    dev = open_device(device_id)
    formats = _prop(dev, "supported_program_formats") or []
    if device_id.lower().startswith("iqm"):
        a, b = coupled_pair(dev)
        program, fmt = bell_iqm_json(site_label(a, 0), site_label(b, 1)), ProgramFormat.IQM_JSON
    else:
        program, fmt = BELL_QASM2, ProgramFormat.QASM2
    print(f"device formats: {formats}\nsubmitting {fmt} program, {args.shots} shots")
    t0 = time.perf_counter()
    job = dev.submit_job(program, fmt, args.shots)  # QDMI_device_job_set_parameter + _submit
    print(f"job id: {_prop(job, 'id')}  status: {_prop(job, 'check')}")
    if not job.wait(args.timeout):  # QDMI_device_job_wait
        sys.exit(f"job not finished after {args.timeout}s (status {_prop(job, 'check')})")
    meta["seconds_execute"] = time.perf_counter() - t0
    print(f"done in {meta['seconds_execute']:.2f}s")
    _save({str(k): int(v) for k, v in job.get_counts().items()}, args, meta)  # QDMI_JOB_RESULT_HIST_*


def cmd_loop(args) -> None:
    """A minimal hybrid loop. The classical node picks theta, the quantum node returns <Z>.

    Every iteration is one QDMI job, so the printed round-trip time IS the coupling latency
    that later decides whether VQE/QAOA/QSCI-style loops are practical on your system.
    """
    from mqt.core.plugins.qiskit import QDMIBackend
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter

    device_id = resolve_device(args.device)
    backend = QDMIBackend.from_device_id(device_id, **session_kwargs(device_id))
    theta = Parameter("theta")
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)
    tqc = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=1)  # transpile once
    print(f"{'step':>4} {'theta':>7} {'<Z> meas':>9} {'cos(theta)':>10} {'round trip':>11}")
    rtts = []
    for k in range(args.steps):
        t = math.pi * k / max(args.steps - 1, 1)
        t0 = time.perf_counter()
        counts = backend.run(tqc.assign_parameters({theta: t}), shots=args.shots).result().get_counts()
        rtts.append(time.perf_counter() - t0)
        z = (counts.get("0", 0) - counts.get("1", 0)) / max(sum(counts.values()), 1)
        print(f"{k:>4} {t:7.3f} {z:9.3f} {math.cos(t):10.3f} {rtts[-1]:10.2f}s")
    print(f"\nmean round trip {sum(rtts) / len(rtts):.2f}s over {len(rtts)} jobs on {device_id}"
          f"  -> a 100-iteration VQE with 5 circuits/iteration would spend ~{500 * sum(rtts) / len(rtts) / 60:.0f} min waiting")


def cmd_analyze(args) -> None:
    data = json.loads(Path(args.inp).read_text())
    counts = data["counts"]
    shots = sum(counts.values())
    p_same = (counts.get("00", 0) + counts.get("11", 0)) / shots
    print(f"device={data.get('device')} host={data.get('host')} shots={shots}")
    print(f"P(00)+P(11) = {p_same:.3f}   (ideal Bell state: 1.000; uniform noise: 0.500)")
    print(f"analysed on {os.uname().nodename}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("info")
    p.add_argument("--device", default=LOCAL_DEVICE, help="device ID, 'local', 'auto-iqm' or 'slurm'")
    p.add_argument("--max-sites", type=int, default=6)
    p = sub.add_parser("bell")
    p.add_argument("--device", default=LOCAL_DEVICE)
    p.add_argument("--via", choices=["raw", "qiskit"], default="raw")
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--out")
    p = sub.add_parser("loop")
    p.add_argument("--device", default=LOCAL_DEVICE)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--shots", type=int, default=500)
    p = sub.add_parser("analyze")
    p.add_argument("--in", dest="inp", required=True)
    args = ap.parse_args()
    {"list": cmd_list, "info": cmd_info, "bell": cmd_bell, "loop": cmd_loop, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
