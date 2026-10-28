"""OPTIONAL: a small check on a real IBM quantum computer (free IBM Quantum account, a few minutes of QPU time).
NOT part of the main results; run it only if the team wants a hardware data point.

It estimates the IQP kernel for a few test queries against a subset of the pool with the compute-uncompute
circuit U(x')^dagger U(x) |0> and measures P(all zeros), then compares the hardware picks with the exact picks
on the same subset (top-k overlap and rank agreement).

Setup in Colab (once):
    !pip install -q qiskit qiskit-ibm-runtime
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<your IBM API key>")   # never in a file you share
Run:  python -m src.pipeline hardware --dataset trec          (or add --simulate to test the circuits locally first)
This script was written without access to qiskit or IBM hardware: run with --simulate first."""
import argparse, json, numpy as np, pandas as pd
from src.paths import CFG, DATASET, DATA, FEAT, RES, TUNED
from src import selectors_classical as sc, quantum_kernels as qk, metrics_extra as mx


def iqp_circuit(x, reps):
    """Same gates as PennyLane's IQPEmbedding (and the NumPy reference): H, RZ(x_i), RZZ(x_i x_j); repeated."""
    from qiskit import QuantumCircuit
    d = len(x); qc = QuantumCircuit(d)
    for _ in range(reps):
        qc.h(range(d))
        for i in range(d):
            qc.rz(float(x[i]), i)
        for i in range(d):
            for j in range(i + 1, d):
                qc.rzz(float(x[i] * x[j]), i, j)
    return qc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--dims", type=int, default=4)
    ap.add_argument("--shots", type=int, default=1024); ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--backend-name", default=None, help="IBM backend; default = least busy")
    a = ap.parse_args()
    from qiskit import transpile
    tag = f"seed{a.seed}_lda{a.dims}"; c = json.loads(TUNED.read_text())[tag]["bandwidth"]; reps = CFG["quantum"]["iqp_repeats"]
    nq, npool = CFG["studies"]["hardware_queries"], CFG["studies"]["hardware_pool"]
    rng = np.random.default_rng(a.seed)
    Sp_all, Sq_all = np.load(FEAT / f"{tag}_pool_std.npy") * c, np.load(FEAT / f"{tag}_query_std.npy") * c
    pool_sel = np.sort(rng.choice(len(Sp_all), npool, replace=False)); q_sel = np.sort(rng.choice(len(Sq_all), nq, replace=False))
    Sp, Sq = Sp_all[pool_sel], Sq_all[q_sel]
    circuits = []
    for xq in Sq:
        Uq = iqp_circuit(xq, reps)
        for xp in Sp:
            qc = Uq.compose(iqp_circuit(xp, reps).inverse()); qc.measure_all(); circuits.append(qc)
    if a.simulate:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.backend(a.backend_name) if a.backend_name else service.least_busy(operational=True, simulator=False)
    tc = transpile(circuits, backend=backend, optimization_level=3)
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    job = Sampler(mode=backend).run(tc, shots=a.shots)
    res = job.result()
    zero = "0" * a.dims; K_hw = np.zeros((nq, npool))
    for n, r in enumerate(res):
        counts = r.data.meas.get_counts()
        K_hw[n // npool, n % npool] = counts.get(zero, 0) / a.shots
    K_ex = qk.kernel_from_states(qk.np_states(Sq, "iqp", reps), qk.np_states(Sp, "iqp", reps))
    P_hw, P_ex = sc.top_k_from_scores(K_hw, CFG["experiment"]["k_main"]), sc.top_k_from_scores(K_ex, CFG["experiment"]["k_main"])
    out = {"backend": getattr(backend, "name", str(backend)), "queries": nq, "pool": npool, "qubits": a.dims, "shots": a.shots,
           "simulated": a.simulate, "jaccard_hw_vs_exact": float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(P_hw, P_ex)])),
           "rank_agreement_hw_vs_exact": mx.rank_agreement(K_hw, K_ex), "mean_abs_kernel_error": float(np.abs(K_hw - K_ex).mean())}
    json.dump(out, open(RES / "hardware.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
