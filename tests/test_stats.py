import itertools

import torch
import pytest
from transformerlens_fdr.stats import (
    aggregate_effects_by_group,
    apply_fdr_adjustment,
    calculate_p_values,
    compute_control_baseline,
    compute_robust_baseline,
    _compute_robust_baseline_details,
    encode_group_ids,
    signflip_minimum_p,
    signflip_permutation_info,
    signflip_p_values,
)
from transformerlens_fdr.metrics import circuit_localization_index


def test_fdr_bh_adjustment():
    """Benjamini-Hochberg correctly separates signal from noise."""
    p_values = torch.tensor([
        [0.0001, 0.0002, 0.8, 0.9],
        [0.00001, 0.5, 0.6, 0.7]
    ])

    q_values, passed_mask = apply_fdr_adjustment(p_values, alpha=0.05, method='fdr_bh')

    assert passed_mask[0, 0].item() is True
    assert passed_mask[0, 1].item() is True
    assert passed_mask[1, 0].item() is True

    assert passed_mask[0, 2].item() is False
    assert passed_mask[0, 3].item() is False
    assert passed_mask[1, 1].item() is False
    assert passed_mask[1, 2].item() is False
    assert passed_mask[1, 3].item() is False


def test_bh_q_values_match_hand_calculation():
    """BH adjustment is independent of the model that produced the p-values."""
    p_values = torch.tensor([0.01, 0.03, 0.04, 0.5], dtype=torch.float64)
    q_values, _ = apply_fdr_adjustment(p_values, alpha=0.05, method="fdr_bh")
    expected = torch.tensor(
        [0.04, 0.05333333333333334, 0.05333333333333334, 0.5],
        dtype=torch.float64,
    )
    assert torch.allclose(q_values, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "p_values",
    [
        torch.tensor([], dtype=torch.float32),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.tensor([-0.01]),
        torch.tensor([1.01]),
    ],
)
def test_fdr_rejects_invalid_p_values(p_values):
    with pytest.raises((TypeError, ValueError)):
        apply_fdr_adjustment(p_values)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_fdr_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        apply_fdr_adjustment(torch.tensor([0.01, 0.5]), alpha=alpha)


def test_fdr_rejects_unsupported_method():
    with pytest.raises(ValueError, match="method"):
        apply_fdr_adjustment(torch.tensor([0.01, 0.5]), method="bonferroni")


def test_fdr_boundary_uses_float64_threshold_arithmetic():
    # float32 stores 0.0125 slightly above the exact BH first threshold.
    p = torch.tensor([0.0125, 0.5, 0.8, 1.0], dtype=torch.float32)
    _, rejected = apply_fdr_adjustment(p, alpha=0.05, method="fdr_bh")
    assert not rejected.any()


def test_fdr_exact_bh_boundary_tie_is_rejected():
    # At rank 7, the BH threshold is 7 * .05 / 10 == .035.
    p = torch.tensor([0.035] * 7 + [0.5, 0.8, 1.0], dtype=torch.float64)
    q_values, rejected = apply_fdr_adjustment(p, alpha=0.05, method="fdr_bh")
    assert rejected[:7].all()
    assert not rejected[7:].any()
    assert torch.equal(q_values <= 0.05, rejected)


def test_sparse_nonflat_map_does_not_use_epsilon_as_robust_scale():
    effects = torch.tensor([0.0] * 20 + [1.0, -1.0])
    median, scale = compute_robust_baseline(effects)
    assert median.abs() < 1e-9
    assert torch.isclose(scale, 0.1 * effects.std(unbiased=False))


def test_near_degenerate_mad_uses_relative_scale_floor():
    central_values = torch.linspace(-1e-7, 1e-7, 1000)
    effects = torch.cat([central_values, torch.full((10,), 1e-3), torch.full((10,), -1e-3)])
    median, scale, method = _compute_robust_baseline_details(effects)
    expected_floor = 0.1 * effects.std(unbiased=False)
    assert median.abs() < 1e-9
    assert scale == expected_floor
    assert method == "scaled MAD with 0.1 population-standard-deviation floor"


def test_signflip_exact_enumeration_and_resolution_floor():
    effects = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    p = signflip_p_values(effects, n_perm=8)
    signs = torch.tensor(
        [[1.0, -1.0, -1.0, -1.0], [1.0, 1.0, -1.0, -1.0],
         [1.0, -1.0, 1.0, -1.0], [1.0, 1.0, 1.0, -1.0],
         [1.0, -1.0, -1.0, 1.0], [1.0, 1.0, -1.0, 1.0],
         [1.0, -1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float64
    )
    x = effects.reshape(1, 4).double()
    values = signs * x
    stats = values.mean(-1) / (values.std(-1, unbiased=True) / 2.0)
    observed = x.mean(-1) / (x.std(-1, unbiased=True) / 2.0)
    expected = (stats.abs() >= observed.abs()).double().mean()
    assert p.item() == expected.item()
    assert signflip_minimum_p(4, 8) == 1 / 8
    assert signflip_permutation_info(4, 8) == {
        "mode": "exact", "n_patterns": 8, "minimum_p": 1 / 8
    }
    assert signflip_permutation_info(5, 8)["mode"] == "monte_carlo"


def test_signflip_matrix_statistics_match_brute_force_for_multiple_components():
    effects = torch.tensor(
        [[[0.3, -0.8, 1.2, 0.5], [1.0, 1.1, -0.7, 0.2]]],
        dtype=torch.float64,
    )
    observed = signflip_p_values(effects, n_perm=8)
    signs = torch.tensor(
        [[1, -1, -1, -1], [1, 1, -1, -1], [1, -1, 1, -1], [1, 1, 1, -1],
         [1, -1, -1, 1], [1, 1, -1, 1], [1, -1, 1, 1], [1, 1, 1, 1]],
        dtype=torch.float64,
    )
    expected = []
    for component in effects[0]:
        null = signs * component
        null_t = null.mean(-1) / (null.std(-1, unbiased=True) / 2)
        obs_t = component.mean() / (component.std(unbiased=True) / 2)
        expected.append((null_t.abs() >= obs_t.abs()).double().mean())
    assert torch.allclose(observed[0], torch.stack(expected), atol=1e-12, rtol=0)


def test_exact_signflip_near_constant_effects_keep_positive_resolution_floor():
    generator = torch.Generator().manual_seed(20261014)
    effects = 1.0 + 0.01 * torch.randn((1, 200, 10), generator=generator, dtype=torch.float64)
    p_values = signflip_p_values(effects, n_perm=9999)
    assert (p_values >= 1 / 512).all()


def test_exact_signflip_constant_effects_are_not_all_counted_as_ties():
    effects = torch.full((1, 5, 10), 0.83, dtype=torch.float64)
    p_values = signflip_p_values(effects, n_perm=9999)
    assert torch.equal(p_values, torch.full_like(p_values, 1 / 512))


def test_fdr_by_adjustment():
    """Benjamini-Yekutieli is more conservative but should still find strong signals."""
    p_values = torch.tensor([
        [0.0001, 0.0002, 0.8, 0.9],
        [0.00001, 0.5, 0.6, 0.7]
    ])

    q_values_by, passed_mask_by = apply_fdr_adjustment(p_values, alpha=0.05, method='fdr_by')
    q_values_bh, passed_mask_bh = apply_fdr_adjustment(p_values, alpha=0.05, method='fdr_bh')

    # BY q-values should be >= BH q-values (more conservative)
    assert (q_values_by >= q_values_bh - 1e-10).all()

    # The strongest signal should still pass even under BY
    assert passed_mask_by[1, 0].item() is True


def test_z_score_p_values():
    """Verify Z-score p-value calculation produces expected results."""
    control = torch.tensor([0.0, 0.1, -0.05, 0.02, 0.08, -0.03, 0.01, 0.04])
    mu, sigma = compute_control_baseline(control)

    # A value right at the mean should have a high p-value (not significant)
    effects_at_mean = torch.tensor([[mu.item()]])
    p_at_mean = calculate_p_values(effects_at_mean, mu, sigma)
    assert p_at_mean[0, 0].item() > 0.9

    # A value far from the mean should have a low p-value
    effects_far = torch.tensor([[mu.item() + 5 * sigma.item()]])
    p_far = calculate_p_values(effects_far, mu, sigma)
    assert p_far[0, 0].item() < 0.001


def test_cli_bounds():
    # 1. Single active component — highly localised
    effects_single = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    cli_single = circuit_localization_index(effects_single)
    assert 0.99 < cli_single <= 1.0

    # 2. Uniformly distributed components — not localised
    effects_uniform = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    cli_uniform = circuit_localization_index(effects_uniform)
    assert 0.0 <= cli_uniform < 0.01


# ---- Cluster (prompt-group) sign-flip -------------------------------------


def test_encode_group_ids_orders_labels_and_counts_sizes():
    codes, labels, sizes = encode_group_ids(["b", "a", "b", "c", "a", "b"], 6)
    assert codes.tolist() == [0, 1, 0, 2, 1, 0]
    assert labels == ("b", "a", "c")
    assert sizes == (3, 2, 1)


def test_encode_group_ids_accepts_tensor_labels():
    _, labels, sizes = encode_group_ids(torch.tensor([5, 5, 9]), 3)
    assert labels == (5, 9)
    assert sizes == (2, 1)


@pytest.mark.parametrize(
    "bad, n_prompts, error, match",
    [
        (["a", "b"], 3, ValueError, "entries"),
        (["a", None, "b"], 3, ValueError, "None or NaN"),
        (["a", float("nan"), "b"], 3, ValueError, "None or NaN"),
        ([["x"], ["y"]], 2, TypeError, "hashable"),
        (torch.zeros(2, 2), 4, ValueError, "one-dimensional"),
    ],
)
def test_encode_group_ids_rejects_bad_labels(bad, n_prompts, error, match):
    with pytest.raises(error, match=match):
        encode_group_ids(bad, n_prompts)


def test_aggregate_effects_by_group_uses_unweighted_group_means():
    effects = torch.tensor([[[1.0, 3.0, 10.0, 2.0]]])
    grouped, info = aggregate_effects_by_group(effects, ["a", "a", "b", "c"])
    assert grouped.shape == (1, 1, 3)
    expected = torch.tensor([2.0, 10.0, 2.0], dtype=grouped.dtype)
    assert torch.allclose(grouped[0, 0], expected)
    assert info == {"labels": ("a", "b", "c"), "sizes": (2, 1, 1), "n_groups": 3}


def _brute_force_group_p_values(group_means):
    """All 2**G sign patterns, studentized two-sided; group_means is [components, G]."""
    n_groups = group_means.shape[-1]

    def studentized(values):
        spread = values.std(-1, unbiased=True).clamp_min(1e-12)
        return values.mean(-1) / (spread / n_groups ** 0.5)

    observed = studentized(group_means)
    patterns = torch.tensor(
        list(itertools.product([-1.0, 1.0], repeat=n_groups)), dtype=group_means.dtype
    )
    null = studentized(patterns[:, None, :] * group_means[None])
    return (null.abs() >= observed.abs() - 1e-9).to(group_means.dtype).mean(0)


def test_cluster_signflip_matches_brute_force_over_group_patterns():
    generator = torch.Generator().manual_seed(21)
    effects = torch.randn((2, 3, 12), generator=generator) + 0.4
    group_ids = [g for g in range(4) for _ in range(3)]
    p_values = signflip_p_values(effects, n_perm=9999, group_ids=group_ids)
    group_means = effects.double().reshape(2, 3, 4, 3).mean(-1).reshape(6, 4)
    torch.testing.assert_close(
        p_values.reshape(6).double(), _brute_force_group_p_values(group_means)
    )


def test_singleton_groups_match_prompt_level_signflip():
    generator = torch.Generator().manual_seed(5)
    effects = torch.randn((2, 3, 8), generator=generator) + 0.3
    prompt_level = signflip_p_values(effects, n_perm=9999)
    singleton = signflip_p_values(effects, n_perm=9999, group_ids=list(range(8)))
    torch.testing.assert_close(prompt_level, singleton)


def test_cluster_floor_depends_on_groups_not_prompts():
    n_groups, per_group = 5, 8
    generator = torch.Generator().manual_seed(9)
    effects = 1.0 + 0.01 * torch.randn((1, 1, n_groups * per_group), generator=generator)
    group_ids = [g for g in range(n_groups) for _ in range(per_group)]
    cluster = signflip_p_values(effects, n_perm=9999, group_ids=group_ids)
    prompt_level = signflip_p_values(
        effects, n_perm=9999, generator=torch.Generator().manual_seed(1)
    )
    assert cluster.item() == pytest.approx(1 / 16)
    assert prompt_level.item() < 0.01


def test_cluster_signflip_uses_monte_carlo_floor_when_group_patterns_exceed_budget():
    generator = torch.Generator().manual_seed(14)
    effects = torch.randn((1, 3, 20), generator=generator)
    group_ids = [group for group in range(10) for _ in range(2)]
    p_values = signflip_p_values(
        effects,
        n_perm=99,
        group_ids=group_ids,
        generator=torch.Generator().manual_seed(15),
    )
    assert (p_values >= 1 / 100).all()
    torch.testing.assert_close(p_values * 100, (p_values * 100).round())


def test_cluster_signflip_requires_two_groups():
    with pytest.raises(ValueError, match="at least two prompt groups"):
        signflip_p_values(torch.randn((1, 2, 6)), group_ids=["a"] * 6)
