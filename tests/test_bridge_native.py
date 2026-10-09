"""No-download integration coverage for the TransformerLens 4 bridge API."""

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformerlens_fdr.auditor import PatchingAuditor


@pytest.fixture(scope="module")
def native_bridge():
    config = TransformerBridgeConfig(
        n_layers=2,
        d_model=32,
        d_head=8,
        n_heads=4,
        n_ctx=16,
        d_vocab=64,
        act_fn="gelu",
        seed=17,
    )
    return TransformerBridge.boot_native(config, device="cpu")


def _metric(logits, correct_tokens, incorrect_tokens):
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, -1, correct_tokens] - logits[rows, -1, incorrect_tokens]


@pytest.mark.parametrize(
    ("hook_type", "component_count"),
    [("attn_head", 4), ("mlp_out", 1)],
)
def test_auditor_runs_with_native_bridge_and_calibrated_results(
    native_bridge, hook_type, component_count
):
    generator = torch.Generator().manual_seed(23)
    clean = torch.randint(0, 64, (8, 6), generator=generator)
    corrupted = torch.randint(0, 64, (8, 6), generator=generator)
    correct = torch.randint(0, 64, (8,), generator=generator)
    offsets = torch.randint(1, 64, (8,), generator=generator)
    incorrect = (correct + offsets) % 64

    auditor = PatchingAuditor(
        native_bridge,
        metric_fn=_metric,
        n_perm=199,
        fdr_method="fdr_by",
        hook_type=hook_type,
    )
    result = auditor.run_patching_audit(
        clean_tokens=clean,
        corrupted_tokens=corrupted,
        correct_tokens=correct,
        incorrect_tokens=incorrect,
        method="signflip",
    )

    expected = (native_bridge.cfg.n_layers, component_count)
    assert result.calibrated
    assert result.raw_patching_effects.shape == expected
    assert result.per_prompt_effects.shape == (*expected, clean.shape[0])
    assert result.p_values.shape == expected
    assert result.q_values.shape == expected
    assert torch.isfinite(result.p_values).all()
    assert torch.isfinite(result.q_values).all()
