"""Synthetic calibration checks; no TransformerLens model is loaded.

Quick tests: ``python -m pytest -q tests/test_calibration.py``
Full 1,000-replication validation: ``python -m tests.test_calibration``

For 144 simultaneous tests and alpha=0.05, use at least about 3,000
sign-flips so the minimum attainable p-value is at or below alpha / 144.
The default family below has 12 tests so the full harness remains practical.
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest
import scipy.stats as scipy_stats
import torch

from transformerlens_fdr.stats import (
    apply_fdr_adjustment,
    calculate_empirical_p_values,
    signflip_p_values,
)


def simulate(
    L=3,
    C=4,
    B=30,
    pi0=0.9,
    shift=1.0,
    dist="normal",
    rho=0.3,
    rng=None,
    skew=0.75,
):
    """Create effects [L,C,B]; null effects have mean zero.

    ``pi0`` is the fraction of true null components. Nonsymmetric cases have
    mean-zero nulls but violate the sign-flip test's symmetry assumption.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_components = L * C
    n_signal = int(round((1.0 - pi0) * n_components))
    if dist in {"normal", "correlated"}:
        independent = rng.normal(size=(n_components, B))
        if dist == "correlated":
            shared = rng.normal(size=(1, B))
            independent = math.sqrt(rho) * shared + math.sqrt(1.0 - rho) * independent
    elif dist == "t3":
        independent = rng.standard_t(df=3, size=(n_components, B)) / math.sqrt(3.0)
    elif dist == "symmetric_mixture":
        signs = rng.choice((-1.0, 1.0), size=(n_components, B))
        magnitudes = rng.lognormal(mean=-0.5, sigma=0.7, size=(n_components, B))
        independent = signs * magnitudes
    elif dist in {"mild_skew", "strong_skew"}:
        strength = skew if dist == "mild_skew" else max(skew, 2.0)
        gaussian = rng.normal(size=(n_components, B))
        centered_exponential = rng.exponential(scale=1.0, size=(n_components, B)) - 1.0
        independent = math.sqrt(1.0 / (1.0 + strength**2)) * (
            gaussian + strength * centered_exponential
        )
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    if n_signal:
        independent[-n_signal:, :] += shift
    effects = torch.as_tensor(independent.reshape(L, C, B), dtype=torch.float32)
    true_null = torch.ones((L, C), dtype=torch.bool)
    if n_signal:
        true_null.reshape(-1)[-n_signal:] = False
    return effects, true_null


@dataclass
class ReplicationSummary:
    fdr: float
    fdr_se: float
    power: float
    global_rejection_rate: float
    global_rejection_se: float
    null_p_values: list


def _run_replicates(
    *,
    repetitions,
    n_perm,
    pi0,
    dist,
    seed,
    B=30,
    L=3,
    C=4,
    rho=0.3,
    skew=0.75,
    fdr_method="fdr_by",
):
    rng = np.random.default_rng(seed)
    fdp = []
    powers = []
    global_rejections = []
    null_p_values = []
    for rep in range(repetitions):
        effects, true_null = simulate(
            L=L, C=C, B=B, pi0=pi0, dist=dist, rho=rho,
            rng=rng, skew=skew,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed + rep + 1)
        p_values = signflip_p_values(effects, n_perm=n_perm, generator=generator)
        _, rejected = apply_fdr_adjustment(p_values, alpha=0.05, method=fdr_method)
        false_positives = rejected & true_null
        true_positives = rejected & ~true_null
        n_rejected = int(rejected.sum())
        fdp.append(int(false_positives.sum()) / max(n_rejected, 1))
        n_signal = int((~true_null).sum())
        powers.append(int(true_positives.sum()) / n_signal if n_signal else 0.0)
        global_rejections.append(float(n_rejected > 0))
        if bool(true_null.reshape(-1)[0]):
            null_p_values.append(float(p_values.reshape(-1)[0]))

    fdp_array = np.asarray(fdp, dtype=float)
    rejection_array = np.asarray(global_rejections, dtype=float)
    return ReplicationSummary(
        fdr=float(fdp_array.mean()),
        fdr_se=float(fdp_array.std(ddof=1) / math.sqrt(repetitions)),
        power=float(np.mean(powers)),
        global_rejection_rate=float(rejection_array.mean()),
        global_rejection_se=float(
            math.sqrt(max(rejection_array.mean() * (1 - rejection_array.mean()), 0)
                      / repetitions)
        ),
        null_p_values=null_p_values,
    )


def run_calibration_suite(repetitions=1000, n_perm=999, seed=20261009):
    """Run the slower repeated-simulation validation and print its results."""
    if repetitions < 1000:
        raise ValueError("The full calibration harness requires at least 1,000 replications.")
    family_size = 12
    if 1.0 / (n_perm + 1) > 0.05 / family_size:
        raise ValueError(
            "Too few sign-flips for this family size: increase n_perm so "
            "1/(n_perm+1) <= alpha / number_of_tests."
        )

    symmetric_distributions = ("normal", "t3", "correlated", "symmetric_mixture")
    summaries = {}
    for dist_index, dist in enumerate(symmetric_distributions):
        summary = _run_replicates(
            repetitions=repetitions, n_perm=n_perm, pi0=0.9,
            dist=dist, seed=seed + 10000 * dist_index,
        )
        summaries[dist] = summary
        ks = scipy_stats.kstest(summary.null_p_values, "uniform")
        assert ks.pvalue > 0.001, (
            f"Null p-values failed the KS uniformity check for {dist}: "
            f"D={ks.statistic:.4g}, p={ks.pvalue:.4g}"
        )

    fdr_rows = []
    for pi_index, pi0 in enumerate((0.5, 0.9, 1.0)):
        summary = _run_replicates(
            repetitions=repetitions, n_perm=n_perm, pi0=pi0,
            dist="correlated", seed=seed + 50000 + 1000 * pi_index,
        )
        # Three standard errors allow for Monte Carlo variation at the boundary.
        assert summary.fdr <= 0.05 + 3 * summary.fdr_se
        if pi0 == 1.0:
            assert summary.global_rejection_rate <= (
                0.05 + 3 * summary.global_rejection_se
            )
        fdr_rows.append((pi0, summary.fdr, summary.fdr_se, summary.power,
                         summary.global_rejection_rate))

    # Nonsymmetric nulls are diagnostics, not gating tests. Sign-flip validity
    # assumes symmetry; report how calibration changes with B and skew.
    skew_rows = []
    for B in (10, 30, 100):
        for skew in (0.5, 1.0, 2.0):
            reps = min(repetitions, 250)
            summary = _run_replicates(
                repetitions=reps, n_perm=n_perm, pi0=1.0,
                dist="mild_skew" if skew <= 1 else "strong_skew",
                seed=seed + 80000 + B * 100 + int(skew * 10),
                B=B, skew=skew,
            )
            ks = scipy_stats.kstest(summary.null_p_values, "uniform")
            skew_rows.append((B, skew, ks.statistic, ks.pvalue,
                              summary.global_rejection_rate))

    print("Symmetric-null KS checks (true null p-values):")
    for dist, summary in summaries.items():
        ks = scipy_stats.kstest(summary.null_p_values, "uniform")
        print(f"  {dist:18s} D={ks.statistic:.4f} p={ks.pvalue:.4g}")
    print("FDR/power by null fraction (BY):")
    print("  pi0     FDR     MC-SE   power   global-null-reject-rate")
    for row in fdr_rows:
        print(f"  {row[0]:.1f}   {row[1]:.4f}  {row[2]:.4f}  {row[3]:.4f}  {row[4]:.4f}")
    print("Nonsymmetric-null diagnostics (not pass/fail):")
    print("  B   skew    KS-D    KS-p    global-null-reject-rate")
    for row in skew_rows:
        print(f"  {row[0]:3d}  {row[1]:.1f}   {row[2]:.4f}  {row[3]:.4g}   {row[4]:.4f}")
    return summaries, fdr_rows, skew_rows


def test_global_null_fdr_is_controlled_for_symmetric_effects():
    """Under a symmetric all-null, observed FDR stays within Monte Carlo error."""
    repetitions = 300
    alpha = 0.05
    summary = _run_replicates(
        repetitions=repetitions,
        n_perm=999,
        pi0=1.0,
        dist="normal",
        seed=20261009,
        fdr_method="fdr_bh",
    )

    # Three binomial standard errors allow ordinary Monte Carlo variation.
    mc_tolerance = 3.0 * math.sqrt(alpha * (1.0 - alpha) / repetitions)
    assert summary.fdr <= alpha + mc_tolerance, (
        f"Global-null FDR {summary.fdr:.3f} exceeds nominal {alpha:.2f} "
        f"plus Monte Carlo tolerance {mc_tolerance:.3f}."
    )

def test_signflip_returns_componentwise_plus_one_p_values():
    effects = torch.randn((2, 3, 20), generator=torch.Generator().manual_seed(7))
    p_values = signflip_p_values(
        effects, n_perm=199, generator=torch.Generator().manual_seed(11)
    )
    assert p_values.shape == (2, 3)
    assert torch.isfinite(p_values).all()
    assert (p_values >= 1 / 200).all()
    assert (p_values <= 1).all()


def test_signflip_requires_per_prompt_effects():
    with pytest.raises(ValueError, match=r"shape \[layers, components, prompts\]"):
        signflip_p_values(torch.randn((3, 4)))


def test_empirical_p_values_are_componentwise_and_plus_one_corrected():
    null = torch.tensor([
        [[-1.0, -2.0]],
        [[0.0, 0.0]],
        [[1.0, 2.0]],
    ])
    observed = torch.tensor([[10.0, 2.0]])
    p_values = calculate_empirical_p_values(observed, null)
    assert p_values.shape == observed.shape
    assert p_values[0, 0].item() == pytest.approx(0.25)
    assert p_values[0, 1].item() == pytest.approx(0.75)


def test_simulator_marks_mean_zero_skew_as_non_symmetric_null():
    effects, true_null = simulate(
        L=4, C=8, B=100, pi0=1.0, dist="strong_skew",
        rng=np.random.default_rng(13), skew=2.0,
    )
    null_values = effects[true_null].numpy().reshape(-1)
    assert abs(float(null_values.mean())) < 0.15
    assert abs(float(scipy_stats.skew(null_values))) > 0.5


def test_strongly_skewed_null_exceeds_nominal_signflip_rejection_rate():
    """Pin a known failure mode: sign-flip is anti-conservative for this null."""
    rng = np.random.default_rng(1701)
    repetitions, B = 10000, 6
    values = rng.exponential(size=(repetitions, B)) - 1.0
    observed_mean = values.mean(axis=1)
    observed_sd = values.std(axis=1, ddof=1)
    observed_t = observed_mean / (observed_sd / math.sqrt(B)).clip(min=1e-12)

    bit_patterns = np.arange(1 << B, dtype=np.uint8)[:, None]
    bit_positions = np.arange(B, dtype=np.uint8)[None, :]
    signs = (((bit_patterns >> bit_positions) & 1).astype(np.float64) * 2.0) - 1.0
    null_t = np.empty((len(signs), repetitions), dtype=np.float64)
    for start in range(0, repetitions, 250):
        stop = min(start + 250, repetitions)
        flipped = signs[:, None, :] * values[None, start:stop, :]
        means = flipped.mean(axis=-1)
        stds = flipped.std(axis=-1, ddof=1)
        null_t[:, start:stop] = means / (stds / math.sqrt(B)).clip(min=1e-12)

    exceedances = (np.abs(null_t) >= np.abs(observed_t)[None, :]).sum(axis=0)
    p_values = (exceedances + 1) / (len(signs) + 1)
    rejection_rate = float(np.mean(p_values <= 0.05))
    mc_se = math.sqrt(rejection_rate * (1.0 - rejection_rate) / repetitions)
    assert rejection_rate > 0.05 + 3.0 * mc_se


if __name__ == "__main__":
    run_calibration_suite()
