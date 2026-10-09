"""Adversarial, model-free checks for TransformerLens-FDR.

Run with: python -m pytest -q tests/test_stress_cases.py
"""

import pytest
import torch

from transformerlens_fdr.metrics import (
    circuit_localization_index,
    normalized_patching_effect,
)
from transformerlens_fdr.stats import (
    calculate_p_values,
    compute_control_baseline,
)


def test_single_control_sample_is_rejected():
    """A single observation cannot estimate control variance."""
    with pytest.raises(ValueError, match="at least 2"):
        compute_control_baseline(torch.tensor([0.25]))


def test_zero_variance_control_is_rejected_for_p_values():
    """A degenerate null cannot support meaningful z-test p-values."""
    with pytest.raises(ValueError, match="standard deviation"):
        calculate_p_values(
            torch.tensor([[1.0]]),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )


def test_undefined_normalized_effect_is_rejected_or_flagged():
    """No clean-vs-corrupt gap means normalized patching effect is undefined."""
    with pytest.raises(ValueError, match="Undefined normalized effect"):
        normalized_patching_effect(
            torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.5)
        )



def test_zero_effects_are_not_reported_as_maximally_localized():
    """An absent signal should not look like a perfectly localized circuit."""
    score = circuit_localization_index(torch.zeros((2, 4)))
    assert score == 0.0


def test_fdr_adjustment_handles_extreme_valid_p_values():
    """Sanity check: a strong signal survives correction, nulls do not."""
    from transformerlens_fdr.stats import apply_fdr_adjustment

    p = torch.tensor([[1e-6, 0.4, 0.8]])
    q, passed = apply_fdr_adjustment(p, alpha=0.05, method="fdr_bh")
    assert torch.isfinite(q).all()
    assert passed.tolist() == [[True, False, False]]


def test_cli_uniform_effects_are_not_localized():
    """Regression sanity check for the non-degenerate CLI path."""
    score = circuit_localization_index(torch.ones((2, 4)))
    assert abs(score) < 1e-6
