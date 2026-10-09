"""Adversarial, model-independent checks for the FDR statistics.

TransformerLens integration is covered separately by test_bridge_native.py
and test_auditor.py. These tests exercise the statistical routines directly.
"""

import math

import pytest
import torch

from transformerlens_fdr.metrics import normalized_patching_effect
from transformerlens_fdr.stats import apply_fdr_adjustment


class TestAdversarialFDR:
    def test_pure_null_hypothesis_rejection(self):
        """Under independent uniform null p-values, BH stays within MC error."""
        repetitions = 300
        family_size = 144
        alpha = 0.05
        generator = torch.Generator().manual_seed(20261009)
        any_rejection = 0

        for _ in range(repetitions):
            p_values = torch.rand(family_size, generator=generator)
            _, rejected = apply_fdr_adjustment(p_values, alpha=alpha, method="fdr_bh")
            any_rejection += int(rejected.any())

        rate = any_rejection / repetitions
        mc_tolerance = 3 * math.sqrt(alpha * (1 - alpha) / repetitions)
        assert rate <= alpha + mc_tolerance, (
            f"Global-null rejection rate {rate:.3f} exceeds {alpha + mc_tolerance:.3f}."
        )

    def test_synthetic_planted_circuit_recovery(self):
        """BH recovers planted strong effects and rejects the simulated nulls."""
        n_layers, n_heads = 12, 12
        planted = {(5, 5), (8, 10), (9, 9)}
        generator = torch.Generator().manual_seed(1701)
        effects = torch.randn((n_layers, n_heads), generator=generator) * 0.01
        for layer, head in planted:
            effects[layer, head] = 0.85

        # Two-sided z-test p-values against the known N(0, 0.01) null.
        p_values = torch.special.erfc(torch.abs(effects / 0.01) / math.sqrt(2))
        _, rejected = apply_fdr_adjustment(
            p_values.flatten(), alpha=0.01, method="fdr_bh"
        )
        recovered = {
            (int(index) // n_heads, int(index) % n_heads)
            for index in torch.where(rejected)[0]
        }
        assert recovered == planted

    def test_extreme_collinearity_and_nan_resistance(self):
        """An identical clean/corrupt metric makes normalization undefined."""
        with pytest.raises(ValueError, match="Undefined normalized effect"):
            normalized_patching_effect(
                torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.5)
            )

    def test_massive_scale_oom_resistance(self):
        """FDR adjustment handles 5,120 tests without allocating a model."""
        p_values = torch.full((80, 64), 0.5)
        planted = {(5, 5), (8, 10), (9, 9)}
        for layer, head in planted:
            p_values[layer, head] = 1e-12

        q_values, rejected = apply_fdr_adjustment(
            p_values, alpha=0.05, method="fdr_bh"
        )
        recovered = {
            (int(layer), int(head))
            for layer, head in torch.nonzero(rejected, as_tuple=False).tolist()
        }
        assert q_values.shape == (80, 64)
        assert torch.isfinite(q_values).all()
        assert recovered == planted
