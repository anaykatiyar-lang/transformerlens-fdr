import torch
import pytest
from transformerlens_fdr.stats import apply_fdr_adjustment, calculate_p_values, compute_control_baseline
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
