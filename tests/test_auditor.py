import torch
import pytest
from transformer_lens import TransformerBridge
from transformerlens_fdr.auditor import PatchingAuditor
from transformerlens_fdr.metrics import logit_difference


def _make_mock_metric():
    """Returns a dummy metric function for testing."""
    def mock_metric(logits, correct, incorrect):
        return logits.reshape(logits.shape[0], -1).mean(dim=1)
    return mock_metric


@pytest.fixture(scope="module")
def tiny_model():
    """Load a small model once per module."""
    # Using a small HF model that works with boot_transformers
    model = TransformerBridge.boot_tl_legacy("NeelNanda/Attn_Only_1L512W_C4_Code")
    model.enable_compatibility_mode()
    return model


@pytest.fixture
def base_tokens():
    return {
        "clean_tokens": torch.tensor([[1, 2, 3], [2, 3, 4]]),
        "corrupted_tokens": torch.tensor([[1, 4, 3], [2, 5, 4]]),
        "correct_tokens": torch.tensor([5, 5]),
        "incorrect_tokens": torch.tensor([6, 6]),
    }


# -------------------------------------------------------------------
# Test: basic attn_head audit (batched path)
# -------------------------------------------------------------------
def test_attn_head_audit(tiny_model, base_tokens):
    auditor = PatchingAuditor(
        tiny_model,
        metric_fn=_make_mock_metric(),
        fdr_threshold=0.05,
        hook_type="attn_head",
    )
    results = auditor.run_patching_audit(**base_tokens)

    assert results.raw_patching_effects.shape == (tiny_model.cfg.n_layers, tiny_model.cfg.n_heads)
    assert results.per_prompt_effects.shape == (tiny_model.cfg.n_layers, tiny_model.cfg.n_heads, 2)
    assert results.p_values is None
    assert results.significant_mask is None
    assert not results.calibrated
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
        n_perm=99,
    )
    results = auditor.run_head_patching_audit(**base_tokens, method="signflip")

    assert results.raw_patching_effects.shape == (tiny_model.cfg.n_layers, tiny_model.cfg.n_heads)
    assert results.calibrated
    assert results.p_values.shape == results.raw_patching_effects.shape
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
        n_perm=99,
    )
    results = auditor.run_patching_audit(**base_tokens, method="signflip")

    # fdr_by is more conservative — just verify shapes and bounds
    assert results.q_values.shape == results.p_values.shape
    assert results.significant_mask.shape == results.p_values.shape
    assert 0 <= results.cli_score <= 1.0


def test_per_component_empirical_null_path(tiny_model, base_tokens):
    n_layers = tiny_model.cfg.n_layers
    n_heads = tiny_model.cfg.n_heads
    null = torch.randn((100, n_layers, n_heads)) * 0.1
    auditor = PatchingAuditor(
        tiny_model,
        metric_fn=_make_mock_metric(),
        fdr_method="fdr_by",
    )
    results = auditor.run_patching_audit(**base_tokens, null_distribution=null)
    assert results.calibrated
    assert results.null_description == "caller-supplied per-component empirical null"
    assert results.p_values.shape == (n_layers, n_heads)
    assert torch.isfinite(results.p_values).all()


def test_resid_mid_is_position_specific(tiny_model, base_tokens):
    auditor = PatchingAuditor(
        tiny_model,
        metric_fn=_make_mock_metric(),
        hook_type="resid_mid",
    )
    results = auditor.run_patching_audit(**base_tokens)
    assert results.raw_patching_effects.shape == (
        tiny_model.cfg.n_layers, base_tokens["clean_tokens"].shape[1]
    )
    assert results.per_prompt_effects.shape == (
        tiny_model.cfg.n_layers, base_tokens["clean_tokens"].shape[1], 2
    )


# -------------------------------------------------------------------
# Test: invalid hook_type raises
# -------------------------------------------------------------------
def test_invalid_hook_type_raises():
    with pytest.raises(ValueError, match="Unsupported hook_type"):
        PatchingAuditor(
            None,
            metric_fn=_make_mock_metric(),
            hook_type="bad_hook",
        )


def test_real_model_shuffled_labels_smoke(tiny_model):
    """Exercise the calibrated model path with randomized label orientation."""
    batch = 30
    clean_row = torch.tensor([[1, 2, 3]])
    corrupted_row = torch.tensor([[1, 4, 3]])
    clean_tokens = clean_row.repeat(batch, 1)
    corrupted_tokens = corrupted_row.repeat(batch, 1)
    swap = torch.randint(0, 2, (batch,), generator=torch.Generator().manual_seed(41)).bool()
    correct_tokens = torch.where(swap, torch.tensor(5), torch.tensor(6))
    incorrect_tokens = torch.where(swap, torch.tensor(6), torch.tensor(5))
    device = tiny_model.cfg.device

    auditor = PatchingAuditor(
        tiny_model,
        metric_fn=logit_difference,
        fdr_method="fdr_by",
        n_perm=999,
    )
    results = auditor.run_patching_audit(
        clean_tokens=clean_tokens.to(device),
        corrupted_tokens=corrupted_tokens.to(device),
        correct_tokens=correct_tokens.to(device),
        incorrect_tokens=incorrect_tokens.to(device),
        method="signflip",
    )
    assert results.calibrated
    assert results.p_values.shape == results.raw_patching_effects.shape
    assert torch.isfinite(results.p_values).all()
    assert torch.isfinite(results.q_values).all()
