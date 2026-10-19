"""The consistency checks of the cubic and semi-diagonal quartic constants assembled from displaced Hessians (FD: HESSIAN).

check_cubic and check_quartic compare permutations of an element and then symmetrize. The Hessian assembly used to average the
finite-difference routes before the check, so the permutations were equal by construction and the check always reported
"No inconsistencies found". Now each element holds one route; the returned constants are unchanged."""
import itertools

import numpy as np

from pyvpt2.constants import wave_to_hartree
from pyvpt2.quartic import assemble_quartic_from_hessians


def _synthetic_findifrec(n=3, disp=0.05, noise=0.0, seed=0):
    """Displaced Hessians of V = 1/2 w Q^2 + 1/6 phi3 Q^3 + 1/24 phi_iijj Q_i^2 Q_j^2 in reduced normal coordinates (cm-1), optionally
    with noise on one displaced Hessian (a finite-difference input Hessian's error)."""
    rng = np.random.default_rng(seed)
    omega = np.array([500.0, 900.0, 1500.0])[:n]
    phi3 = rng.normal(0, 30, (n, n, n))
    phi3 = sum(phi3.transpose(p) for p in itertools.permutations(range(3))) / 6
    phi4 = rng.normal(0, 40, (n, n))
    phi4 = 0.5 * (phi4 + phi4.T)

    def hessian(Q):
        H = np.diag(omega).astype(float) + np.einsum("ijk,k->ij", phi3, Q)
        for i in range(n):
            for j in range(n):
                if i == j:
                    H[i, i] += 0.5 * phi4[i, i] * Q[i] ** 2
                else:
                    H[i, j] += phi4[i, j] * Q[i] * Q[j]
                    H[i, i] += 0.5 * phi4[i, j] * Q[j] ** 2
        return H * wave_to_hartree

    disps = {}
    for i in range(n):
        for s in (1, -1):
            Q = np.zeros(n)
            Q[i] = s * disp
            H = hessian(Q)
            if noise and (i, s) == (0, 1):
                E = rng.normal(0, 1, (n, n))
                H = H + noise * wave_to_hartree * 0.5 * (E + E.T)
            disps[f"{i}: {s}"] = {"hessian": H}
    rec = {"harm": {"n_modes": n, "H0": hessian(np.zeros(n)), "modes": np.eye(n), "gamma": np.ones(n), "v_ind": list(range(n))},
           "reference": {}, "displacements": disps, "disp_size": disp}
    return rec, phi3, phi4


def _averaged_before_the_check(rec):
    """The previous assembly (all routes averaged into every element), for the comparison of the returned constants."""
    n, d = rec["harm"]["n_modes"], rec["disp_size"]
    hp = {i: rec["displacements"][f"{i}: 1"]["hessian"] for i in range(n)}
    hn = {i: rec["displacements"][f"{i}: -1"]["hessian"] for i in range(n)}
    h0 = rec["harm"]["H0"]
    p3, p4 = np.zeros((n, n, n)), np.zeros((n, n))
    for i, j, k in itertools.product(range(n), repeat=3):
        if i == j == k:
            p3[i, i, i] = (hp[i][i, i] - hn[i][i, i]) / (2 * d)
        else:
            p3[i, j, k] = (hp[i][j, k] - hn[i][j, k] + hp[j][k, i] - hn[j][k, i] + hp[k][i, j] - hn[k][i, j]) / (6 * d)
    for i, j in itertools.product(range(n), repeat=2):
        if i == j:
            p4[i, i] = (hp[i][i, i] + hn[i][i, i] - 2 * h0[i, i]) / d ** 2
        else:
            p4[i, j] = (hp[j][i, i] + hn[j][i, i] + hp[i][j, j] + hn[i][j, j] - 2 * h0[i, i] - 2 * h0[j, j]) / (2 * d ** 2)
    return p3 / wave_to_hartree, p4 / wave_to_hartree


def test_exact_hessians_give_the_constants_and_no_inconsistency(capsys):
    rec, phi3, phi4 = _synthetic_findifrec()
    got3, got4 = assemble_quartic_from_hessians(rec)
    assert np.allclose(got3, phi3, atol=1e-6) and np.allclose(got4, phi4, atol=1e-6)
    assert capsys.readouterr().out.count("No inconsistencies found") == 2


def test_a_noisy_hessian_is_reported_and_the_constants_are_unchanged(capsys):
    """0.05 cm-1 of noise on one displaced Hessian becomes ~20 cm-1 in phi_iijj (divided by disp^2) and ~0.5 cm-1 in phi_ijk (divided by
    2 disp, below the check's 1 cm-1): the quartic check now reports the two routes, the cubic one stays quiet. The returned (symmetrized)
    constants are the ones the previous assembly returned."""
    rec, _, _ = _synthetic_findifrec(noise=0.05)
    ref3, ref4 = _averaged_before_the_check(rec)
    got3, got4 = assemble_quartic_from_hessians(rec)
    quartic, cubic = capsys.readouterr().out.split("Checking for numerical inconsistencies in cubic terms...")
    assert "(0, 1) (1, 0)" in quartic and "No inconsistencies found" not in quartic
    assert "No inconsistencies found" in cubic
    assert np.allclose(got3, ref3, rtol=0, atol=1e-9) and np.allclose(got4, ref4, rtol=0, atol=1e-9)   # summation order: ~1e-10


def test_larger_noise_reaches_the_cubic_check_too(capsys):
    rec, _, _ = _synthetic_findifrec(noise=0.5)
    assemble_quartic_from_hessians(rec)
    assert capsys.readouterr().out.count("No inconsistencies found") == 0
