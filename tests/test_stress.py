"""
Stress tests for transformerlens_fdr, one function at a time.

Run:      pytest test_stress.py -v
Quick:    FDR_REPS=60 pytest test_stress.py -v          (fewer Monte Carlo reps)
Report:   pytest test_stress.py -v -s -k diagnostic     (prints skew robustness table)

Test tiers
  * plain tests      -> must pass; a failure is a bug (or an adapter needs updating)
  * xfail(strict=False) -> document a KNOWN limitation. They XPASS when you fix it.
  * diagnostic       -> print numbers, assert nothing.

ADAPTERS: the few places that depend on your exact signatures are marked ADAPTER.
Assumed contracts (edit if yours differ):
  calculate_empirical_p_values(observed[L,C], null_1d)      -> p[L,C]
  calculate_p_values(effects, mu, sigma)                    -> p
  compute_robust_baseline(effects)                          -> (mu, sigma)
  apply_fdr_adjustment(p, alpha=, method=)                  -> (q, mask)
  normalized_patching_effect(patched, clean, corrupted)     -> tensor
  circuit_localization_index(effects[L,C])                  -> float
  signflip_p_values(effects[L,C,B], n_perm=, generator=)    -> p[L,C]   (skipped if absent)
  metric_fn(logits, correct, incorrect) -> scalar mean (or [B] if PER_PROMPT_METRIC)
"""
import inspect
import math
import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

scipy_stats = pytest.importorskip("scipy.stats")

from transformerlens_fdr import metrics as M          # noqa: E402
from transformerlens_fdr import stats as S            # noqa: E402
from transformerlens_fdr.auditor import PatchingAuditor  # noqa: E402

REPS = int(os.environ.get("FDR_REPS", "300"))
PER_PROMPT_METRIC = True    # ADAPTER: your auditor now requires metric_fn -> shape [B]

signflip = getattr(S, "signflip_p_values", None)   # ADAPTER: name of your sign-flip function
needs_signflip = pytest.mark.skipif(signflip is None, reason="signflip_p_values not in stats.py")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def T(x, dtype=torch.float64):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def N(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def scalar(a):
    return float(np.asarray(a).reshape(-1)[0])


def G(seed):
    return torch.Generator().manual_seed(seed)


def mc_tol(p, n, z=4.0):
    """Monte Carlo tolerance for a proportion estimated from n draws (+ plus-one slack)."""
    return z * math.sqrt(max(p * (1 - p), 1e-12) / n) + 1.0 / n


def fdr(p, alpha=0.05, method="fdr_bh"):
    q, m = S.apply_fdr_adjustment(T(p), alpha=alpha, method=method)
    return N(q).astype(float), N(m).astype(bool)


def ref_bh(p, alpha, by=False):
    """Independent reference BH/BY implementation (numpy, no library code)."""
    p = np.asarray(p, float).ravel()
    n = p.size
    order = np.argsort(p, kind="stable")
    c = np.sum(1.0 / np.arange(1, n + 1)) if by else 1.0
    q_sorted = p[order] * n * c / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(q_sorted, 1.0)
    return q, q <= alpha


# ============================================================================
# 1. normalized_patching_effect
# ============================================================================
def npe(patched, clean, corrupted):
    return M.normalized_patching_effect(T(patched), T(clean), T(corrupted))


class TestNormalizedPatchingEffect:
    def test_endpoints(self):
        assert float(npe(3.0, 3.0, -1.0)) == pytest.approx(1.0)   # fully restored
        assert float(npe(-1.0, 3.0, -1.0)) == pytest.approx(0.0)  # nothing restored

    @pytest.mark.parametrize("t", [-1.0, 0.0, 0.25, 0.5, 1.0, 2.0])
    def test_linear_interpolation(self, t):
        clean, corr = 5.0, 1.0
        assert float(npe(corr + t * (clean - corr), clean, corr)) == pytest.approx(t)

    @pytest.mark.parametrize("a,b", [(2.0, 0.0), (-3.0, 7.0), (0.01, -4.0)])
    def test_affine_invariance(self, a, b):
        base = float(npe(2.2, 5.0, 1.0))
        moved = float(npe(a * 2.2 + b, a * 5.0 + b, a * 1.0 + b))
        assert moved == pytest.approx(base, rel=1e-9, abs=1e-9)

    def test_negative_gap_same_formula(self):
        assert float(npe(0.0, -2.0, 2.0)) == pytest.approx(0.5)

    def test_vectorised_over_patched(self):
        out = M.normalized_patching_effect(T([1.0, 3.0, 5.0]), T(5.0), T(1.0))
        np.testing.assert_allclose(N(out), [0.0, 0.5, 1.0])

    def test_zero_gap_never_looks_valid(self):
        try:
            out = float(npe(1.0, 2.0, 2.0))
        except Exception:
            return
        assert not math.isfinite(out), "zero clean-corrupted gap must not return a finite effect"


# ============================================================================
# 2. circuit_localization_index  (property tests; adapt if your CLI differs)
# ============================================================================
def cli(x):
    return float(M.circuit_localization_index(T(x)))


class TestCLI:
    def test_concentrated_beats_uniform(self):
        conc = np.zeros((4, 6)); conc[1, 2] = 1.0
        unif = np.full((4, 6), 1.0 / 24)
        assert cli(conc) > cli(unif)

    def test_permutation_invariant(self):
        rng = np.random.default_rng(0)
        x = rng.random((4, 6))
        shuffled = rng.permutation(x.ravel()).reshape(4, 6)
        assert cli(x) == pytest.approx(cli(shuffled), abs=1e-6)

    def test_finite_on_mixed_signs(self):
        rng = np.random.default_rng(1)
        assert math.isfinite(cli(rng.normal(size=(5, 5))))

    def test_all_zero_is_finite_or_raises(self):
        try:
            v = cli(np.zeros((3, 3)))
        except Exception:
            return
        assert math.isfinite(v)


# ============================================================================
# 3. compute_robust_baseline
# ============================================================================
def rb(x):
    mu, sg = S.compute_robust_baseline(T(x))
    return float(mu), float(sg)


class TestRobustBaseline:
    def test_recovers_location_and_scale(self):
        x = np.random.default_rng(0).normal(5, 2, size=5000)
        mu, sg = rb(x)
        assert abs(mu - 5) < 0.15
        assert 1.0 < sg < 2.6          # unscaled MAD ~1.35, scaled MAD ~2.0

    def test_robust_to_10pct_gross_outliers(self):
        x = np.random.default_rng(1).normal(5, 2, size=5000)
        x[:500] = 1e6
        mu, sg = rb(x)
        assert abs(mu - 5) < 0.5
        assert 1.0 < sg < 5.0

    @pytest.mark.parametrize("a,b", [(3.0, 1.0), (-2.0, 4.0)])
    def test_affine_equivariance(self, a, b):
        x = np.random.default_rng(2).normal(size=1001)
        mu, sg = rb(x)
        mu2, sg2 = rb(a * x + b)
        assert mu2 == pytest.approx(a * mu + b, rel=1e-6, abs=1e-6)
        assert sg2 == pytest.approx(abs(a) * sg, rel=1e-6)

    def test_shape_invariant(self):
        x = np.random.default_rng(3).normal(size=(12, 12))
        assert rb(x) == pytest.approx(rb(x.ravel()))

    def test_constant_input_gives_nonnegative_finite_scale(self):
        try:
            mu, sg = rb(np.ones((4, 4)))
        except Exception:
            return
        assert math.isfinite(sg) and sg >= 0


# ============================================================================
# 4. calculate_p_values  (Gaussian helper)
# ============================================================================
def cp(x, mu, sigma):
    return N(S.calculate_p_values(T(x), T(mu), T(sigma)))


class TestGaussianPValues:
    def test_range_and_no_nan_at_extremes(self):
        p = cp(np.array([[-40.0, -3.0, 0.0, 3.0, 40.0]]), 0.0, 1.0)
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()

    def test_uniform_under_exact_gaussian_null(self):
        x = np.random.default_rng(0).standard_normal((100, 200))
        p = cp(x, 0.0, 1.0).ravel()
        assert scipy_stats.kstest(p, "uniform").statistic < 0.02

    def test_known_quantile_and_sidedness_are_explicit(self):
        p_pos = scalar(cp(np.array([[1.96]]), 0.0, 1.0))
        p_neg = scalar(cp(np.array([[-1.96]]), 0.0, 1.0))
        two_sided = math.isclose(p_pos, p_neg, abs_tol=1e-9)
        upper_one_sided = math.isclose(p_pos + p_neg, 1.0, abs_tol=1e-9)
        assert two_sided or upper_one_sided
        assert abs(p_pos - (0.05 if two_sided else 0.025)) < 1e-3
        print(f"[diagnostic] calculate_p_values is {'two' if two_sided else 'upper one'}-sided")

    def test_monotone_in_distance_from_mu(self):
        x = np.linspace(0, 5, 60)[None, :]
        p = cp(x, 0.0, 1.0).ravel()
        assert (np.diff(p) <= 1e-12).all()

    def test_location_scale_invariance(self):
        x = np.random.default_rng(4).standard_normal((1, 50))
        a, b = 3.7, -2.0
        np.testing.assert_allclose(cp(a * x + b, a * 0.3 + b, a * 1.2), cp(x, 0.3, 1.2), atol=1e-5)

    @pytest.mark.xfail(strict=False, reason="policy: degenerate scale must not yield significance")
    def test_zero_sigma_not_silently_significant(self):
        try:
            p = cp(np.array([[0.0, 1.0, -1.0]]), 0.0, 0.0)
        except (ValueError, ZeroDivisionError, RuntimeError):
            return
        assert not (p < 0.05).any()


# ============================================================================
# 5. calculate_empirical_p_values  (pooled 1-D null)   ADAPTER: ep()
# ============================================================================
def ep(obs, null):
    """obs [L,C]. A 1-D null is broadcast to the same null for every component;
    a 3-D null [n_null, L, C] is passed through (per-component nulls)."""
    obs = np.asarray(obs, float)
    null = np.asarray(null, float)
    if null.ndim == 1:
        null = np.broadcast_to(null[:, None, None], (null.shape[0],) + obs.shape)
    return N(S.calculate_empirical_p_values(T(obs), T(np.ascontiguousarray(null))))


class TestEmpiricalPValues:
    def test_plus_one_floor_and_ceiling(self):
        null = np.random.default_rng(0).standard_normal(199)
        p = ep(np.array([[1e6, 0.0, 1.0, 2.0]]), null)
        assert (p >= 1 / 200 - 1e-12).all() and (p <= 1.0 + 1e-12).all()
        assert p[0, 0] == pytest.approx(1 / 200)

    def test_resolution_limited_by_null_size(self):
        for n in (9, 99, 999):
            null = np.random.default_rng(n).standard_normal(n)
            assert scalar(ep(np.array([[1e6]]), null)) >= 1 / (n + 1) - 1e-12

    def test_ties_are_conservative(self):
        null = np.linspace(0.1, 1.0, 50)               # positive, so sidedness is irrelevant
        p = scalar(ep(np.array([[1.0]]), null))          # equals the max of the null
        assert p >= 2 / 51 - 1e-12                      # counts the tie: (1 + #>=) / (n + 1)

    def test_monotone_in_observed(self):
        null = np.random.default_rng(1).standard_normal(500)
        p = ep(np.linspace(0, 4, 40)[None, :], null).ravel()
        assert (np.diff(p) <= 1e-12).all()

    def test_negative_extreme_convention_is_explicit(self):
        null = np.random.default_rng(2).standard_normal(199)
        p = scalar(ep(np.array([[-1e6]]), null))
        two_sided = math.isclose(p, 1 / 200, abs_tol=1e-12)
        upper_only = math.isclose(p, 1.0, abs_tol=1e-12)
        assert two_sided or upper_only
        print(f"[diagnostic] empirical p-values are {'two-sided' if two_sided else 'upper one-sided'}")

    def test_uniform_when_observed_comes_from_the_null(self):
        rng = np.random.default_rng(3)
        ps = [scalar(ep(rng.standard_normal((1, 1)), rng.standard_normal(199))) for _ in range(3000)]
        assert scipy_stats.kstest(ps, "uniform").statistic < 0.04

    def test_does_not_mutate_inputs(self):
        null = np.random.default_rng(4).standard_normal(100)
        obs = np.random.default_rng(5).standard_normal((3, 3))
        n_t, o_t = T(np.broadcast_to(null[:, None, None], (100, 3, 3)).copy()), T(obs)
        n0, o0 = n_t.clone(), o_t.clone()
        S.calculate_empirical_p_values(o_t, n_t)
        assert torch.equal(n_t, n0) and torch.equal(o_t, o0)

    def test_pooled_1d_null_is_rejected_by_contract(self):
        with pytest.raises(ValueError):
            S.calculate_empirical_p_values(T(np.zeros((2, 2))), T(np.zeros(50)))

    def test_per_component_nulls_are_used(self):
        """Components with very different null scales must each be calibrated to their own null."""
        rng = np.random.default_rng(7)
        scales = np.array([[1.0, 10.0]])
        ps = []
        for _ in range(2000):
            ps.append(ep(rng.standard_normal((1, 2)) * scales,
                         rng.standard_normal((199, 1, 2)) * scales)[0])
        ps = np.array(ps)
        for c in range(2):
            assert scipy_stats.kstest(ps[:, c], "uniform").statistic < 0.045

    def test_nonfinite_null_rejected(self):
        null = np.random.default_rng(6).standard_normal(50)
        null[3] = np.nan
        with pytest.raises((ValueError, RuntimeError)):
            ep(np.array([[1.0]]), null)


# ============================================================================
# 6. apply_fdr_adjustment
# ============================================================================
class TestFDRAdjustment:
    @pytest.mark.parametrize("method", ["fdr_bh", "fdr_by"])
    @pytest.mark.parametrize("n", [1, 2, 7, 144, 1000])
    @pytest.mark.parametrize("seed", range(4))
    def test_matches_independent_reference(self, method, n, seed):
        rng = np.random.default_rng(seed)
        p = np.where(rng.random(n) < 0.3, rng.random(n) ** 6, rng.random(n))
        if seed % 2:
            p = np.round(p, 3)                           # heavy ties
        q, m = fdr(p, 0.05, method)
        q_ref, m_ref = ref_bh(p, 0.05, by=(method == "fdr_by"))
        np.testing.assert_allclose(q, q_ref, atol=1e-6)
        border = np.abs(q_ref - 0.05) < 1e-6
        assert (m == m_ref)[~border].all()

    @pytest.mark.parametrize("method", ["fdr_bh", "fdr_by"])
    def test_q_properties(self, method):
        p = np.random.default_rng(9).random(300) ** 3
        q, _ = fdr(p, 0.05, method)
        assert (q >= p - 1e-9).all() and (q <= 1 + 1e-9).all()
        assert (np.diff(q[np.argsort(p)]) >= -1e-9).all()      # order preserving

    def test_by_never_more_liberal_than_bh(self):
        p = np.random.default_rng(10).random(200) ** 4
        _, m_bh = fdr(p, 0.05, "fdr_bh")
        _, m_by = fdr(p, 0.05, "fdr_by")
        assert (m_by <= m_bh).all()

    def test_shape_and_position_preserved(self):
        p = np.ones((3, 5)); p[2, 4] = 1e-9
        q, m = fdr(p, 0.05, "fdr_bh")
        assert q.shape == (3, 5) and m.shape == (3, 5)
        assert m[2, 4] and m.sum() == 1

    def test_input_not_mutated(self):
        p = T(np.random.default_rng(0).random((4, 4)))
        before = p.clone()
        S.apply_fdr_adjustment(p, alpha=0.05, method="fdr_bh")
        assert torch.equal(p, before)

    def test_edge_cases(self):
        _, m = fdr(np.ones(50)); assert not m.any()
        _, m = fdr(np.full(50, 1e-12)); assert m.all()
        q, m = fdr(np.array([0.03])); assert q[0] == pytest.approx(0.03) and m[0]
        q, m = fdr(np.array([0.2])); assert not m[0]

    def test_resolution_arithmetic_by_144(self):
        """n_perm=9999 -> p_min=1e-4 > BY first-step 6.3e-5; rank 2 can still pass."""
        p = np.ones(144); p[0] = 1e-4
        _, m = fdr(p, 0.05, "fdr_by"); assert m.sum() == 0
        p = np.ones(144); p[:2] = 1e-4
        _, m = fdr(p, 0.05, "fdr_by"); assert m[:2].all() and m.sum() == 2
        p = np.ones(144); p[0] = 1e-4
        _, m = fdr(p, 0.05, "fdr_bh"); assert m[0] and m.sum() == 1

    def test_nan_p_never_discovered(self):
        p = np.array([np.nan, 1e-9, 0.5, 0.9])
        try:
            _, m = fdr(p)
        except (ValueError, RuntimeError, AssertionError):
            return
        assert not m[0]

    def test_invalid_method_raises(self):
        with pytest.raises((ValueError, KeyError, NotImplementedError)):
            fdr(np.random.default_rng(0).random(10), 0.05, "fdr_nonsense")

    def test_matches_statsmodels(self):
        sm = pytest.importorskip("statsmodels.stats.multitest")
        p = np.random.default_rng(5).random(120) ** 3
        for method in ("fdr_bh", "fdr_by"):
            rej, q_sm, *_ = sm.multipletests(p, alpha=0.05, method=method)
            q, m = fdr(p, 0.05, method)
            np.testing.assert_allclose(q, q_sm, atol=1e-6)


def sim_fdr(method, n=100, pi0=0.8, rho=0.0, shift=3.0, reps=None, alpha=0.05, seed=0):
    reps = reps or REPS * 5
    rng = np.random.default_rng(seed)
    n1 = int(round(n * (1 - pi0)))
    fdps, pows = [], []
    for _ in range(reps):
        z = math.sqrt(rho) * rng.standard_normal() + math.sqrt(1 - rho) * rng.standard_normal(n)
        z[:n1] += shift
        _, m = fdr(scipy_stats.norm.sf(z), alpha, method)
        R, V = m.sum(), m[n1:].sum()
        fdps.append(V / max(R, 1))
        pows.append(m[:n1].sum() / max(n1, 1))
    fdps = np.array(fdps)
    return fdps.mean(), fdps.std(ddof=1) / math.sqrt(reps), float(np.mean(pows))


class TestFDRRealizedControl:
    @pytest.mark.parametrize("pi0,rho", [(1.0, 0.0), (0.8, 0.0), (0.5, 0.0), (0.8, 0.5), (0.8, 0.9)])
    @pytest.mark.parametrize("method", ["fdr_bh", "fdr_by"])
    def test_fdr_at_most_alpha(self, method, pi0, rho):
        fdr_hat, se, _ = sim_fdr(method, pi0=pi0, rho=rho)
        assert fdr_hat <= 0.05 * pi0 + 4 * se

    @pytest.mark.parametrize("pi0", [0.8, 0.5])
    def test_bh_not_pathologically_conservative(self, pi0):
        """Independent BH has FDR == pi0*alpha exactly; catches Bonferroni-like bugs."""
        fdr_hat, _, _ = sim_fdr("fdr_bh", pi0=pi0)
        assert fdr_hat >= 0.5 * pi0 * 0.05

    def test_has_power(self):
        _, _, power = sim_fdr("fdr_bh", pi0=0.8, shift=4.0)
        assert power > 0.5


# ============================================================================
# 7. sign-flip p-values
# ============================================================================
def sf(x, n_perm=499, seed=0, **kw):
    return N(signflip(T(x), n_perm=n_perm, generator=G(seed), **kw))


def exact_signflip_p(x):
    """Exact two-sided p over all 2^B sign patterns for the studentised mean."""
    B = len(x)
    signs = ((np.arange(2 ** B)[:, None] >> np.arange(B)) & 1) * 2 - 1

    def t(v):
        return v.mean(-1) / (v.std(-1, ddof=1) / math.sqrt(B))
    return float(np.mean(np.abs(t(signs * x)) >= abs(t(x)) - 1e-12))


DISTS = {
    "normal": lambda r, s: r.standard_normal(s),
    "t3": lambda r, s: r.standard_t(3, s),
    "laplace": lambda r, s: r.laplace(size=s),
    "uniform": lambda r, s: r.uniform(-1, 1, s),
    "scale_mixture": lambda r, s: r.standard_normal(s) * np.where(r.random(s) < 0.2, 5, 1),
}


@needs_signflip
class TestSignFlip:
    def test_shape_bounds_and_plus_one(self):
        x = np.random.default_rng(0).normal(size=(3, 5, 12))
        p = sf(x, n_perm=199)
        assert p.shape == (3, 5)
        assert (p >= 1 / 200 - 1e-12).all() and (p <= 1).all()

    def test_component_position_preserved(self):
        x = np.random.default_rng(1).normal(size=(3, 5, 20))
        x[2, 4] += 5.0
        p = sf(x)
        assert np.unravel_index(p.argmin(), p.shape) == (2, 4)

    def test_all_zero_effects_p_is_one(self):
        assert (sf(np.zeros((1, 2, 10))) == 1.0).all()

    def test_overwhelming_effect_hits_resolution_floor(self):
        x = 5 + 0.1 * np.random.default_rng(2).standard_normal((1, 1, 20))
        assert scalar(sf(x, n_perm=499)) == pytest.approx(1 / 500)

    def test_reproducible_with_seed(self):
        x = np.random.default_rng(3).normal(0.3, 1, size=(2, 4, 15))
        np.testing.assert_array_equal(sf(x, seed=7), sf(x, seed=7))

    def test_two_sided_sign_symmetry(self):
        x = np.random.default_rng(4).normal(0.3, 1, size=(2, 4, 15))
        np.testing.assert_array_equal(sf(x, seed=5), sf(-x, seed=5))

    def test_scale_invariance_of_studentised_statistic(self):
        x = np.random.default_rng(5).normal(0.5, 1, size=(1, 6, 20))
        for c in (1e-3, 1e3):
            np.testing.assert_allclose(sf(c * x, seed=1), sf(x, seed=1), atol=0.01)

    @pytest.mark.parametrize("seed", range(6))
    def test_matches_exact_enumeration(self, seed):
        """ADAPTER: edit exact_signflip_p if your statistic is not the studentised mean."""
        x = np.random.default_rng(seed).normal(0.7, 1.0, size=10)
        n_perm = 20000
        p = scalar(sf(x[None, None, :], n_perm=n_perm, seed=seed)[0, 0])
        ref = exact_signflip_p(x)
        assert abs(p - ref) <= mc_tol(ref, n_perm)

    @pytest.mark.parametrize("dist", list(DISTS))
    def test_uniform_under_symmetric_nulls(self, dist):
        C, B = 1200, 20
        x = DISTS[dist](np.random.default_rng(11), (1, C, B))
        p = sf(x, n_perm=399, seed=1).ravel()
        assert scipy_stats.kstest(p, "uniform").statistic < 0.06
        assert np.mean(p <= 0.05) <= 0.05 + 4 * math.sqrt(0.05 * 0.95 / C)

    def test_shared_flips_preserve_dependence(self):
        x = np.random.default_rng(6).normal(0.4, 1, size=(1, 3, 20))
        x[0, 1] = x[0, 0]
        x[0, 2] = -x[0, 0]
        p = sf(x)
        assert p[0, 0] == p[0, 1] == p[0, 2]    # identical/negated columns need identical flips

    def test_single_prompt_never_spuriously_significant(self):
        try:
            p = scalar(sf(np.array([[[2.0]]]), n_perm=99))
        except (ValueError, RuntimeError):
            return
        assert p == 1.0 or math.isnan(p)        # std is undefined at B=1

    def test_nan_effect_never_spuriously_significant(self):
        x = np.random.default_rng(7).normal(size=(1, 2, 12))
        x[0, 0, 3] = np.nan
        try:
            p = sf(x)
        except (ValueError, RuntimeError):
            return
        assert math.isnan(p[0, 0]) or p[0, 0] == 1.0
        assert 1 / 500 - 1e-12 <= p[0, 1] <= 1.0

    @pytest.mark.xfail(strict=False, reason="loophole #2: no absolute tolerance for numerical-noise effects")
    def test_tiny_constant_effect_not_flagged(self):
        assert scalar(sf(np.full((1, 1, 20), 1e-9))) > 0.05

    @pytest.mark.parametrize("pi0,rho", [(1.0, 0.0), (0.8, 0.0), (0.8, 0.5), (0.9, 0.8)])
    @pytest.mark.parametrize("method", ["fdr_bh", "fdr_by"])
    def test_pipeline_realized_fdr(self, pi0, rho, method):
        C, B, reps = 60, 20, max(50, REPS // 2)
        n1 = int(round(C * (1 - pi0)))
        rng = np.random.default_rng(21)
        fdps = []
        for r in range(reps):
            f = rng.standard_t(5, size=(B, 1))
            eps = rng.standard_t(5, size=(B, C))
            e = math.sqrt(rho) * f + math.sqrt(1 - rho) * eps
            e[:, :n1] += 1.2
            p = sf(e.T.reshape(1, C, B), n_perm=999, seed=r)
            _, m = fdr(p.ravel(), 0.05, method)
            fdps.append(m[n1:].sum() / max(m.sum(), 1))
        fdps = np.array(fdps)
        se = fdps.std(ddof=1) / math.sqrt(reps)
        assert fdps.mean() <= 0.05 + 4 * se

    def test_pipeline_has_power(self):
        C, B, reps = 60, 20, max(30, REPS // 4)
        rng = np.random.default_rng(22)
        hits = []
        for r in range(reps):
            e = rng.standard_t(5, size=(B, C)); e[:, :12] += 1.2
            p = sf(e.T.reshape(1, C, B), n_perm=999, seed=r)
            _, m = fdr(p.ravel(), 0.05, "fdr_bh")
            hits.append(m[:12].mean())
        assert np.mean(hits) > 0.3

    # ---- skew: diagnostics + documented failure --------------------------------
    @pytest.mark.parametrize("k", [8.0, 2.0, 0.5])
    @pytest.mark.parametrize("B", [10, 30, 100])
    def test_skew_diagnostic(self, k, B):
        """Mean-zero but skewed null. Prints the rejection rate; asserts nothing."""
        C = 800
        x = np.random.default_rng(31).gamma(k, size=(1, C, B)) - k
        p = sf(x, n_perm=299, seed=2).ravel()
        print(f"[diagnostic] skew={2 / math.sqrt(k):.2f} B={B}: P(p<=.05)={np.mean(p <= .05):.3f}  "
              f"KS={scipy_stats.kstest(p, 'uniform').statistic:.3f}")
        assert np.isfinite(p).all()

    @pytest.mark.xfail(strict=False, reason="documented limitation: symmetry is required for validity")
    def test_strongly_skewed_null_is_not_calibrated(self):
        x = np.random.default_rng(32).gamma(0.2, size=(1, 1500, 10)) - 0.2
        p = sf(x, n_perm=399, seed=3).ravel()
        assert np.mean(p <= 0.05) <= 0.05 + 0.02

    def test_large_family_runs(self):
        x = np.random.default_rng(8).normal(size=(12, 12, 30))
        p = sf(x, n_perm=1999)
        assert p.shape == (12, 12) and np.isfinite(p).all()


# ============================================================================
# 8. Auditor on a toy model with KNOWN ground truth (no TransformerLens needed)
# ============================================================================
# TransformerLens 4 registers legacy names as aliases of canonical hook points.
# Hooks always see hook.name == canonical; run_with_cache stores both spellings.
_ALIASES = {"attn.hook_z": "attn.o.hook_in", "hook_mlp_out": "mlp.hook_out"}


def canon(name):
    for legacy, real in _ALIASES.items():
        if name.endswith("." + legacy):
            return name[: -len(legacy)] + real
    return name


def legacy_spellings(name):
    return [name[: -len(real)] + legacy for legacy, real in _ALIASES.items() if name.endswith("." + real)]


class ToyModel:
    """Linear toy transformer exposing TransformerLens-style hooks.

    Position t only sees token t; heads/MLPs feed the logits additively through
    weights U / Vm. Heads with weight 0 have exactly zero causal effect.
    """

    def __init__(self, head_w=None, mlp_w=None, L=2, H=3, D=6, Dh=2, V=7, V_in=11,
                 seed=0, _params=None):
        self.cfg = SimpleNamespace(n_layers=L, n_heads=H, device="cpu")
        self.V, self.V_in = V, V_in
        if _params is None:
            g = torch.Generator().manual_seed(seed)
            E = torch.randn(V_in, D, generator=g)
            Wz = torch.randn(L, H, D, Dh, generator=g)
            Wm = torch.randn(L, D, D, generator=g)
            Ur = torch.randn(L, H, Dh, V, generator=g)
            Vr = torch.randn(L, D, V, generator=g)
            U, Vm = torch.zeros_like(Ur), torch.zeros_like(Vr)
            for (l, h), w in (head_w or {}).items():
                U[l, h] = w * Ur[l, h]
            for l, w in (mlp_w or {}).items():
                Vm[l] = w * Vr[l]
            _params = dict(E=E, Wz=Wz, Wm=Wm, U=U, Vm=Vm)
        self.p = _params

    def _forward(self, tokens, hooks=(), cache=None):
        by_name = {}
        for name, fn in hooks:
            by_name.setdefault(canon(name), []).append(fn)

        def hp(name, act):
            for fn in by_name.get(name, []):
                out = fn(act, SimpleNamespace(name=name))
                if out is not None:
                    act = out
            if cache is not None:
                cache[name] = act.detach().clone()
                for alias in legacy_spellings(name):
                    cache[alias] = cache[name]
            return act

        with torch.no_grad():
            emb = self.p["E"][tokens]
            logits = 0
            for l in range(self.cfg.n_layers):
                z = torch.einsum("bsd,hde->bshe", emb, self.p["Wz"][l])
                z = hp(f"blocks.{l}.attn.o.hook_in", z)
                m = hp(f"blocks.{l}.mlp.hook_out", emb @ self.p["Wm"][l])
                logits = (logits + torch.einsum("bshe,hev->bsv", z, self.p["U"][l])
                          + m @ self.p["Vm"][l])
        return logits

    def __call__(self, tokens):
        return self._forward(tokens)

    def run_with_cache(self, tokens):
        cache = {}
        return self._forward(tokens, cache=cache), cache

    def run_with_hooks(self, tokens, return_type="logits", fwd_hooks=()):
        return self._forward(tokens, hooks=fwd_hooks)

    def permuted_heads(self, perm):
        P = dict(self.p)
        P["Wz"], P["U"] = self.p["Wz"][:, perm], self.p["U"][:, perm]
        return ToyModel(L=self.cfg.n_layers, H=self.cfg.n_heads, V=self.V,
                        V_in=self.V_in, _params=P)


def per_prompt_metric(logits, correct, incorrect):
    last = logits[:, -1, :]
    return last.gather(1, correct[:, None]).squeeze(1) - last.gather(1, incorrect[:, None]).squeeze(1)


def metric_fn(logits, correct, incorrect):
    d = per_prompt_metric(logits, correct, incorrect)
    return d if PER_PROMPT_METRIC else d.mean()


def make_case(head_w=None, mlp_w=None, B=6, S_=4, seed0=0):
    """Builds a toy model + token batches whose clean/corrupted gap is clearly non-zero."""
    for seed in range(seed0, seed0 + 200):
        model = ToyModel(head_w=head_w, mlp_w=mlp_w, seed=seed)
        g = torch.Generator().manual_seed(1000 + seed)
        clean = torch.randint(0, model.V_in, (B, S_), generator=g)
        corr = clean.clone()
        corr[:, -1] = (clean[:, -1] + 1 + torch.randint(0, model.V_in - 1, (B,), generator=g)) % model.V_in
        correct = torch.randint(0, model.V, (B,), generator=g)
        incorrect = (correct + 1 + torch.randint(0, model.V - 1, (B,), generator=g)) % model.V
        gap = (per_prompt_metric(model(clean), correct, incorrect).mean()
               - per_prompt_metric(model(corr), correct, incorrect).mean())
        if abs(float(gap)) > 0.5:
            return SimpleNamespace(model=model, clean=clean, corr=corr,
                                   correct=correct, incorrect=incorrect)
    raise AssertionError("could not build a toy case with a clear clean/corrupted gap")


def run_audit(case, hook_type="attn_head", corr=None, correct=None, **kw):
    aud = PatchingAuditor(case.model, metric_fn, hook_type=hook_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = aud.run_patching_audit(case.clean, case.corr if corr is None else corr,
                                     case.correct if correct is None else correct,
                                     case.incorrect, **kw)
    return aud, res


def mean_eff(res):
    e = N(res.raw_patching_effects)
    return e.reshape(e.shape[0], e.shape[1], -1).mean(-1)


def reference_effects(case, hook_type="attn_head"):
    """Brute-force one-forward-pass-per-component patching, written independently."""
    m = case.model
    L, H = m.cfg.n_layers, m.cfg.n_heads
    _, cache = m.run_with_cache(case.clean)
    mc = per_prompt_metric(m(case.clean), case.correct, case.incorrect).mean()
    mk = per_prompt_metric(m(case.corr), case.correct, case.incorrect).mean()
    out = np.zeros((L, H if hook_type == "attn_head" else 1))
    for l in range(L):
        for h in range(out.shape[1]):
            if hook_type == "attn_head":
                name = f"blocks.{l}.attn.hook_z"

                def hk(act, hook, h=h):
                    act[:, :, h, :] = cache[hook.name][:, :, h, :]
                    return act
            else:
                name = f"blocks.{l}.hook_mlp_out"

                def hk(act, hook):
                    act[:] = cache[hook.name]
                    return act
            pl = m.run_with_hooks(case.corr, fwd_hooks=[(name, hk)])
            out[l, h] = float((per_prompt_metric(pl, case.correct, case.incorrect).mean() - mk) / (mc - mk))
    return out


class TestAuditorGroundTruth:
    def test_single_active_head_recovered_exactly(self):
        case = make_case(head_w={(0, 1): 1.0})
        expected = np.zeros((2, 3)); expected[0, 1] = 1.0
        np.testing.assert_allclose(mean_eff(run_audit(case)[1]), expected, atol=1e-4)

    def test_batched_path_matches_bruteforce_reference(self):
        rng = np.random.default_rng(0)
        w = {(l, h): float(rng.normal()) for l in range(2) for h in range(3)}
        case = make_case(head_w=w)
        np.testing.assert_allclose(mean_eff(run_audit(case)[1]), reference_effects(case), atol=1e-4)

    def test_head_permutation_equivariance(self):
        rng = np.random.default_rng(1)
        w = {(l, h): float(rng.normal()) for l in range(2) for h in range(3)}
        case = make_case(head_w=w)
        perm = [2, 0, 1]
        case2 = SimpleNamespace(**{**vars(case), "model": case.model.permuted_heads(perm)})
        e1, e2 = mean_eff(run_audit(case)[1]), mean_eff(run_audit(case2)[1])
        for h in range(3):
            np.testing.assert_allclose(e2[:, h], e1[:, perm[h]], atol=1e-4)

    def test_mlp_single_active_layer(self):
        case = make_case(mlp_w={1: 1.0})
        e = mean_eff(run_audit(case, hook_type="mlp_out")[1])
        assert e.shape == (2, 1)
        np.testing.assert_allclose(e[:, 0], [0.0, 1.0], atol=1e-4)

    def test_mlp_matches_bruteforce_reference(self):
        case = make_case(mlp_w={0: 0.7, 1: -1.3})
        np.testing.assert_allclose(mean_eff(run_audit(case, hook_type="mlp_out")[1]),
                                   reference_effects(case, "mlp_out"), atol=1e-4)

    def test_joint_patching_endpoints_and_additivity(self):
        rng = np.random.default_rng(2)
        w = {(l, h): float(rng.normal()) for l in range(2) for h in range(3)}
        case = make_case(head_w=w)
        aud, res = run_audit(case)
        e = mean_eff(res)

        def joint(mask):
            return aud.run_joint_patching(case.clean, case.corr, case.correct, case.incorrect,
                                          torch.as_tensor(mask))
        assert joint(np.ones((2, 3), bool)) == pytest.approx(1.0, abs=1e-4)
        assert joint(np.zeros((2, 3), bool)) == pytest.approx(0.0, abs=1e-4)
        subset = np.zeros((2, 3), bool); subset[0, 0] = subset[1, 2] = subset[1, 0] = True
        assert joint(subset) == pytest.approx(e[subset].sum(), abs=1e-4)   # linear model => additive

    def test_deterministic_and_inputs_untouched(self):
        case = make_case(head_w={(0, 0): 1.0, (1, 1): 0.5})
        before = [t.clone() for t in (case.clean, case.corr, case.correct, case.incorrect)]
        a = mean_eff(run_audit(case)[1]); b = mean_eff(run_audit(case)[1])
        np.testing.assert_array_equal(a, b)
        for t0, t1 in zip(before, (case.clean, case.corr, case.correct, case.incorrect)):
            assert torch.equal(t0, t1)

    def test_summary_df_consistent_with_tensors(self):
        case = make_case(head_w={(0, 1): 1.0, (1, 2): 0.6})
        _, res = run_audit(case)
        df = res.summary_df
        assert len(df) == 6 and "layer" in df.columns
        if getattr(res, "p_values", None) is None:
            pytest.skip("no p-values on this path")
        np.testing.assert_allclose(df["p_value"].to_numpy(), N(res.p_values).reshape(-1), atol=1e-6)
        q = N(res.q_values)
        for name in ("passed_fdr_mask", "significant_mask"):
            mask = getattr(res, name, None)
            if mask is not None:
                border = np.abs(q - 0.05) < 1e-6
                assert (N(mask) == (q <= 0.05))[~border].all()

    @pytest.mark.parametrize("kind", ["identical_inputs", "identical_labels"])
    def test_degenerate_zero_gap_never_yields_discoveries(self, kind):
        case = make_case(head_w={(0, 1): 1.0})
        try:
            if kind == "identical_inputs":
                _, res = run_audit(case, corr=case.clean)
            else:
                _, res = run_audit(case, correct=case.incorrect)
        except Exception:
            return
        assert not np.isfinite(N(res.raw_patching_effects)).all()
        for name in ("passed_fdr_mask", "significant_mask"):
            mask = getattr(res, name, None)
            if mask is not None:
                assert not N(mask).any()


class TestAuditorValidation:
    def test_invalid_hook_type_raises(self):
        case = make_case(head_w={(0, 1): 1.0})
        with pytest.raises(ValueError):
            PatchingAuditor(case.model, metric_fn, hook_type="nonsense")

    def test_mismatched_batch_sizes_rejected(self):
        case = make_case(head_w={(0, 1): 1.0})
        with pytest.raises((ValueError, AssertionError, RuntimeError)):
            run_audit(case, corr=case.corr[:-1])

    def test_wrong_label_length_rejected(self):
        case = make_case(head_w={(0, 1): 1.0})
        with pytest.raises((ValueError, AssertionError, RuntimeError)):
            run_audit(case, correct=case.correct[:-1])

    def test_head_alias_forwards_null_distribution(self):
        params = inspect.signature(PatchingAuditor.run_head_patching_audit).parameters
        assert "null_distribution" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class TestToyModelMatchesTL4HookSemantics:
    """The auditor relies on these TL4 behaviours; the toy model must reproduce them."""

    def test_legacy_name_hooks_see_canonical_name_and_cache_has_both(self):
        case = make_case(head_w={(0, 1): 1.0})
        seen = []

        def spy(act, hook):
            seen.append(hook.name)
            return act
        case.model.run_with_hooks(case.clean, fwd_hooks=[("blocks.0.attn.hook_z", spy)])
        assert seen == ["blocks.0.attn.o.hook_in"]
        _, cache = case.model.run_with_cache(case.clean)
        assert torch.equal(cache["blocks.0.attn.hook_z"], cache["blocks.0.attn.o.hook_in"])
