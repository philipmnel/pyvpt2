"""Route-consistency report of the semi-diagonal quartic constants (Hessian route), on a synthetic quartic potential."""
import numpy as np

from pyvpt2.constants import wave_to_hartree
from pyvpt2.quartic import assemble_quartic_from_hessians, report_quartic_routes


def _synthetic_findifrec(n=3, disp=0.05, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    omega = np.array([500.0, 900.0, 1500.0])[:n]
    phi3 = rng.normal(0, 30, (n, n, n)); phi3 = (phi3 + phi3.transpose(1, 0, 2) + phi3.transpose(0, 2, 1) + phi3.transpose(1, 2, 0)
                                                  + phi3.transpose(2, 0, 1) + phi3.transpose(2, 1, 0)) / 6
    phi4 = rng.normal(0, 40, (n, n)); phi4 = 0.5 * (phi4 + phi4.T)

    def hessian(Q):   # second derivatives of V = 1/2 w Q^2 + 1/6 phi3 Q^3 + 1/24 phi_iijj Q_i^2 Q_j^2 (semi-diagonal quartic only)
        H = np.diag(omega).astype(float)
        H = H + np.einsum("ijk,k->ij", phi3, Q)
        for i in range(n):
            for j in range(n):
                if i == j:
                    H[i, i] += 0.5 * phi4[i, i] * Q[i] ** 2
                else:
                    H[i, j] += phi4[i, j] * Q[i] * Q[j]
                    H[i, i] += 0.5 * phi4[i, j] * Q[j] ** 2
        return H * wave_to_hartree          # the assembly divides by wave_to_hartree

    disps = {}
    for i in range(n):
        for s in (1, -1):
            Q = np.zeros(n); Q[i] = s * disp
            H = hessian(Q)
            if noise and (i, s) == (0, 1):
                H = H + noise * wave_to_hartree * rng.normal(0, 1, (n, n))
            disps[f"{i}: {s}"] = {"hessian": H}
    rec = {"harm": {"n_modes": n, "H0": hessian(np.zeros(n)), "modes": np.eye(n), "gamma": np.ones(n), "v_ind": list(range(n))},
           "reference": {}, "displacements": disps, "disp_size": disp}
    return rec, phi3, phi4


def test_routes_agree_on_exact_hessians():
    rec, phi3, phi4 = _synthetic_findifrec()
    got3, got4 = assemble_quartic_from_hessians(rec)
    assert np.allclose(got4, phi4, atol=1e-6)
    assert np.allclose(got3, phi3, atol=1e-6)
    assert rec["quartic_route_report"]["max"] < 1e-6
    assert rec["quartic_route_report"]["noisy"] is False


def test_routes_disagree_when_one_hessian_is_noisy():
    rec, _, _ = _synthetic_findifrec(noise=0.05)      # 0.05 cm-1 noise on one displaced Hessian -> /disp^2 = 20 cm-1 in phi_iijj
    assemble_quartic_from_hessians(rec)
    rep = rec["quartic_route_report"]
    assert rep["max"] > 5.0
    assert rep["noisy"] is True
    assert len(rep["worst"]) >= 1


def test_report_function_handles_single_mode():
    stats = report_quartic_routes(np.zeros((1, 1)), np.zeros((1, 1)), [0])
    assert stats["max"] == 0.0 and stats["noisy"] is False
