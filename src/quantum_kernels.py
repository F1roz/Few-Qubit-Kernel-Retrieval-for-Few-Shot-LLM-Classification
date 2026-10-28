"""Phase 5: quantum kernels. Owner: quantum role.

Feature maps on d qubits (d = number of features):
  Map A  AngleEmbedding (RY only, no entanglement)   -> selector S6
  Map B  IQPEmbedding (entangling, ZZ style, 2 reps)  -> selector S7
Fidelity kernel k(x, y) = |<psi(x)|psi(y)>|^2.
Projected kernel (S8, Huang et al. 2021): compare the one qubit states (Bloch vectors) of Map B.
Finite shots (S9): the fidelity kernel estimated from a limited number of measurements.
"""
import numpy as np

# ---------- PennyLane version (use this in the project) ----------
def pl_states(X, fmap="iqp", n_repeats=2):
    import pennylane as qml
    d = X.shape[1]
    dev = qml.device("default.qubit", wires=d)

    @qml.qnode(dev)
    def circ(x):
        if fmap == "angle":
            qml.AngleEmbedding(x, wires=range(d), rotation="Y")
        else:
            qml.IQPEmbedding(x, wires=range(d), n_repeats=n_repeats)
        return qml.state()

    return np.array([circ(x) for x in X])


# ---------- NumPy reference version (unit tests and cross checks) ----------
def _bits(d):
    """Z eigenvalue (+1 for |0>, -1 for |1>) of every qubit, for all 2^d basis states.
    Qubit 0 is the leftmost (most significant) bit, like PennyLane."""
    z = (np.arange(2 ** d)[:, None] >> np.arange(d - 1, -1, -1)) & 1
    return 1 - 2 * z


def np_angle_state(x):
    s = np.array([1.0])
    for xi in x:
        s = np.kron(s, np.array([np.cos(xi / 2), np.sin(xi / 2)]))
    return s


def _hadamard_all(s, d):
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    s = s.reshape([2] * d)
    for q in range(d):
        s = np.moveaxis(np.tensordot(H, np.moveaxis(s, q, 0), axes=1), 0, q)
    return s.reshape(-1)


def np_iqp_state(x, n_repeats=2):
    """H on all qubits, then RZ(x_i) on each qubit and MultiRZ(x_i * x_j) on every pair; repeat."""
    d = len(x)
    Z = _bits(d)
    phase = -0.5 * (Z @ x)
    for i in range(d):
        for j in range(i + 1, d):
            phase += -0.5 * x[i] * x[j] * Z[:, i] * Z[:, j]
    s = np.zeros(2 ** d, complex); s[0] = 1
    for _ in range(n_repeats):
        s = _hadamard_all(s, d)
        s = np.exp(1j * phase) * s
    return s


def np_states(X, fmap="iqp", n_repeats=2):
    f = np_angle_state if fmap == "angle" else (lambda x: np_iqp_state(x, n_repeats))
    return np.array([f(x) for x in X])


# ---------- kernels ----------
def kernel_from_states(SA, SB):
    """Fidelity kernel: squared overlap between every state in SA and every state in SB."""
    return np.abs(SA.conj() @ SB.T) ** 2


def angle_kernel_closed_form(A, B):
    """Map A has a simple classical formula: product over features of cos^2((a_i - b_i)/2)."""
    return np.prod(np.cos((A[:, None, :] - B[None, :, :]) / 2) ** 2, axis=2)


def bloch_vectors(S):
    """For each state (rows of S, length 2^d), the Bloch vector (<X>, <Y>, <Z>) of every qubit.
    Returns an array of shape (n_states, 3 * d)."""
    n, dim = S.shape; d = int(np.log2(dim)); out = np.zeros((n, 3 * d))
    for i in range(n):
        T = S[i].reshape([2] * d)
        for q in range(d):
            M = np.moveaxis(T, q, 0).reshape(2, -1)
            rho = M @ M.conj().T                       # one qubit reduced density matrix
            out[i, 3 * q:3 * q + 3] = [2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real]
    return out


def projected_kernel(BA, BB, gamma):
    """Projected quantum kernel: exp(-gamma * sum over qubits of ||b_q(x) - b_q(x')||^2)."""
    d2 = (BA ** 2).sum(1)[:, None] + (BB ** 2).sum(1)[None, :] - 2 * BA @ BB.T
    return np.exp(-gamma * np.maximum(d2, 0))


def projected_median_gamma(B):
    d2 = ((B[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return 1.0 / np.median(d2[np.triu_indices_from(d2, 1)])


def shot_kernel(K, shots, seed=0):
    """What a real device reports: each fidelity is the share of 'all zeros' outcomes in `shots` runs."""
    rng = np.random.default_rng(seed)
    return rng.binomial(shots, np.clip(K, 0, 1)) / shots


# ---------- diagnostics ----------
def concentration_stats(K):
    v = K[np.triu_indices_from(K, 1)] if K.shape[0] == K.shape[1] else K.ravel()
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}


def _psd_sqrt(K):
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def geometric_difference(K_quantum, K_classical, ridge=1e-3):
    """Huang et al. 2021: g = sqrt(largest eigenvalue of sqrt(Kq) (Kc + ridge I)^-1 sqrt(Kq)),
    both kernels scaled so trace = N. g near 1: the classical kernel already does what the
    quantum kernel does on this data. Larger g: room for the quantum kernel to differ."""
    n = K_quantum.shape[0]
    Kq = K_quantum * n / np.trace(K_quantum)
    Kc = K_classical * n / np.trace(K_classical)
    Sq = _psd_sqrt(Kq)
    M = Sq @ np.linalg.solve(Kc + ridge * np.eye(n), Sq)
    return float(np.sqrt(np.max(np.linalg.eigvalsh((M + M.T) / 2))))


# ---------------- noise models (version 5) ----------------
def global_depolarize(K, p, d):
    """Global depolarizing noise on the compute-uncompute circuit: K -> (1 - p) K + p / 2^d.
    A positive affine map keeps every row's ORDER, so retrieval picks never change (without shots)."""
    return (1 - p) * K + p / 2 ** d


def _depol_qubit(R, q, d, p):
    """Single-qubit depolarizing channel on qubit q of a density matrix R (2^d x 2^d)."""
    if p <= 0:
        return R
    T = R.reshape([2] * (2 * d))
    red = np.trace(T, axis1=q, axis2=d + q)                         # partial trace over qubit q
    red = np.expand_dims(np.expand_dims(red, q), d + q)            # put the axes back (size 1)
    eye = np.eye(2).reshape([2 if i in (q, d + q) else 1 for i in range(2 * d)])
    mixed = red * eye / 2
    return ((1 - p) * T + p * mixed).reshape(2 ** d, 2 ** d)


def np_iqp_density(x, n_repeats=2, p1=0.0, p2=0.0):
    """Density matrix of the IQP map (Map B) under LOCAL depolarizing noise. After each layer, every qubit
    gets a depolarizing channel whose strength matches the gates that touched it in that layer:
    p1 after the Hadamard layer, p1 + (d - 1) p2 after the phase layer (one RZ and d - 1 ZZ gates per qubit).
    Gate-count-weighted local depolarizing per layer is a standard first-order NISQ noise model."""
    d = len(x); Z = _bits(d)
    phase = -0.5 * (Z @ x)
    for i in range(d):
        for j in range(i + 1, d):
            phase += -0.5 * x[i] * x[j] * Z[:, i] * Z[:, j]
    D = np.exp(1j * phase)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2); Hd = np.array([[1.0]])
    for _ in range(d):
        Hd = np.kron(Hd, H)
    R = np.zeros((2 ** d, 2 ** d), complex); R[0, 0] = 1
    for _ in range(n_repeats):
        R = Hd @ R @ Hd.conj().T
        for q in range(d):
            R = _depol_qubit(R, q, d, p1)
        R = (D[:, None] * R) * D.conj()[None, :]
        for q in range(d):
            R = _depol_qubit(R, q, d, p1 + (d - 1) * p2)
    return R


def density_kernel(RA, RB):
    """Tr(rho rho') for every pair: the overlap a swap test measures (equals |<psi|psi'>|^2 for pure states)."""
    A = np.array([r.reshape(-1) for r in RA]); B = np.array([r.reshape(-1) for r in RB])
    return np.real(A.conj() @ B.T)


def swap_test_shots(K, shots, seed=0):
    """Swap test with finite shots: P(ancilla = 0) = (1 + K) / 2, observed as B / shots; estimate 2 B / shots - 1."""
    rng = np.random.default_rng(seed)
    return 2 * rng.binomial(shots, np.clip((1 + K) / 2, 0, 1)) / shots - 1
