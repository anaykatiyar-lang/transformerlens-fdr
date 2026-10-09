import torch
import pytest
from transformer_lens import TransformerBridge
from transformerlens_fdr.auditor import PatchingAuditor


def _make_mock_metric():
    """Returns a dummy metric function for testing."""
    def mock_metric(logits, correct, incorrect):
        return logits.mean()
    return mock_metric


@pytest.fixture(scope="module")
def tiny_model():
    """Load a small model once per module."""
    # Using a small HF model that works with boot_transformers
    model = TransformerBridge.boot_transformers("NeelNanda/Attn_Only_1L512W_C4_Code")
    model.enable_compatibility_mode()
    return model


@pytest.fixture
def base_tokens():
    return {
        "clean_tokens": torch.tensor([[1, 2, 3]]),
        "corrupted_tokens": torch.tensor([[1, 4, 3]]),
        "correct_tokens": torch.tensor([5]),
        "incorrect_tokens": torch.tensor([6]),
    }


# -------------------------------------------------------------------
# Test: basic attn_head audit (batched path)
# -------------------------------------------------------------------
def test_attn_head_audit(tiny_model, base_tokens):
    auditor = PatchingAuditor(
        tiny_model,
        metric_fn=_make_mock_metric(),
        fdr_threshold=0.05,
        n_control_samples=2,
        hook_type="attn_head",
    )
    results = auditor.run_patching_audit(**base_tokens)

    assert results.raw_patching_effects.shape == (tiny_model.cfg.n_layers, tiny_model.cfg.n_heads)
    assert results.passed_fdr_mask.shape == (tiny_model.cfg.n_layers, tiny_model.cfg.n_heads)
    assert 0 <= results.cli_score <= 1.0
    assert results.hook_type == "attn_head"


# -------------------------------------------------------------------
# Test: backward-compat alias run_head_patching_audit still works
# -------------------------------------------------------------------
def test_backward_compat_alias(tiny_model, base_tokens):
    auditor = PatchingAuditor(
        tiny_model,
        metric_fn=_make_mock_metric(),
        fdr_threshold=0.05,
        n_control_samples=2,
    )
    results = auditor.run_head_patching_audit(**base_tokens)

    assert results.raw_patching_effects.shape == (tiny_model.cfg.n_layers, tiny_model.cfg.n_heads)
    assert 0 <= results.cli_score <= 1.0


# -------------------------------------------------------------------
# Test: fdr_by (Benjamini-Yekutieli) method
# -------------------------------------------------------------------
def test_fdr_by_method(tiny_model, base_tokens):
    auditor = PatchingAuditor(
        tiny_model,
        metric_fn=_make_mock_metric(),
        fdr_threshold=0.05,
        fdr_method="fdr_by",
        n_control_samples=2,
    )
    results = auditor.run_patching_audit(**base_tokens)

    # fdr_by is more conservative — just verify shapes and bounds
    assert results.q_values.shape == results.p_values.shape
    assert 0 <= results.cli_score <= 1.0


# -------------------------------------------------------------------
# Test: invalid hook_type raises
# -------------------------------------------------------------------
def test_invalid_hook_type_raises(tiny_model):
    with pytest.raises(ValueError, match="Unsupported hook_type"):
        PatchingAuditor(
            tiny_model,
            metric_fn=_make_mock_metric(),
            hook_type="bad_hook",
        )
