import torch
import pytest
from collections import UserDict
from types import SimpleNamespace
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
    assert len(results.resolved_hook_names) == tiny_model.cfg.n_layers
    assert results.resolved_hook_names[0] == "blocks.0.attn.hook_z"
    assert results.outlier_scale_method in {
        "scaled MAD",
        "scaled MAD with 0.1 population-standard-deviation floor",
        "scaled MAD with 0.1 population-standard-deviation floor with epsilon floor",
        "scaled MAD with epsilon floor",
    }
    assert results.outlier_scale_method in results.methods_statement


def test_head_batching_handles_non_divisible_batch_size(tiny_model, base_tokens, monkeypatch):
    n_heads = tiny_model.cfg.n_heads
    batch_size = max(1, n_heads - 1)
    calls = []
    original = tiny_model.run_with_hooks

    def counted_run_with_hooks(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(tiny_model, "run_with_hooks", counted_run_with_hooks)
    auditor = PatchingAuditor(
        tiny_model,
        metric_fn=_make_mock_metric(),
        hook_type="attn_head",
        head_batch_size=batch_size,
    )
    results = auditor.run_patching_audit(**base_tokens)
    assert results.raw_patching_effects.shape == (tiny_model.cfg.n_layers, n_heads)
    assert len(calls) == (n_heads + batch_size - 1) // batch_size


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
    assert results.signflip_mode == "exact"
    assert results.signflip_pattern_count == 2
    assert "exact enumeration" in results.format_summary()


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


def test_mlp_in_registered_but_inactive_hook_fails_and_restores_flag(
    tiny_model, base_tokens
):
    original_mlp_in_flag = getattr(tiny_model.cfg, "use_hook_mlp_in", False)
    auditor = PatchingAuditor(
        tiny_model,
        metric_fn=_make_mock_metric(),
        hook_type="mlp_in",
    )
    # This legacy Attn-Only checkpoint registers the hook name but does not
    # emit the hook during its forward pass. Never return empty or misleading
    # audit results for a registered-but-inactive hook.
    with pytest.raises(RuntimeError, match="did not cache requested hook"):
        auditor.run_patching_audit(**base_tokens)
    assert getattr(tiny_model.cfg, "use_hook_mlp_in", False) == original_mlp_in_flag


def test_resid_mid_prefers_whole_stream_hook_when_available():
    model = SimpleNamespace(
        cfg=SimpleNamespace(n_layers=1),
        hook_dict={"blocks.0.hook_resid_mid": object()},
        set_use_hook_mlp_in=lambda enabled: None,
    )
    auditor = PatchingAuditor(model, metric_fn=_make_mock_metric(), hook_type="resid_mid")
    assert auditor._hook_cfg["hook_suffix"] == "hook_resid_mid"
    assert not auditor._needs_mlp_in_flag


def test_resid_mid_validation_accepts_mapping_hook_dict():
    model = SimpleNamespace(
        cfg=SimpleNamespace(n_layers=1),
        hook_dict=UserDict({"blocks.0.hook_resid_mid": object()}),
        set_use_hook_mlp_in=lambda enabled: None,
    )
    auditor = PatchingAuditor(model, metric_fn=_make_mock_metric(), hook_type="resid_mid")
    assert auditor._hook_cfg["hook_suffix"] == "hook_resid_mid"


def test_resid_mid_does_not_silently_alias_to_mlp_input():
    model = SimpleNamespace(
        cfg=SimpleNamespace(n_layers=1),
        hook_dict={"blocks.0.hook_mlp_in": object()},
        set_use_hook_mlp_in=lambda enabled: None,
    )
    with pytest.raises(ValueError, match="use hook_type='mlp_in'"):
        PatchingAuditor(model, metric_fn=_make_mock_metric(), hook_type="resid_mid")


def test_cache_filter_uses_supported_signature_and_falls_back_cleanly():
    class FilteredModel:
        cfg = SimpleNamespace(n_layers=2)

        def __init__(self):
            self.names = None

        def run_with_cache(self, tokens, names_filter=None):
            self.names = names_filter
            return tokens, {name: None for name in (names_filter or [])}

    filtered = FilteredModel()
    auditor = PatchingAuditor(filtered, metric_fn=_make_mock_metric())
    auditor._run_with_filtered_cache("input")
    assert filtered.names == ["blocks.0.attn.hook_z", "blocks.1.attn.hook_z"]

    class LegacyModel:
        cfg = SimpleNamespace(n_layers=1)

        def run_with_cache(self, tokens):
            return tokens, {"blocks.0.attn.hook_z": None}

    legacy = LegacyModel()
    legacy_auditor = PatchingAuditor(legacy, metric_fn=_make_mock_metric())
    assert legacy_auditor._run_with_filtered_cache("legacy")[0] == "legacy"


def test_cache_validation_catches_registered_but_inactive_hook():
    class InactiveHookModel:
        cfg = SimpleNamespace(n_layers=1)
        hook_dict = {"blocks.0.hook_resid_mid": object()}

        def run_with_cache(self, tokens, names_filter=None):
            return tokens, {}

    auditor = PatchingAuditor(
        InactiveHookModel(), metric_fn=_make_mock_metric(), hook_type="resid_mid"
    )
    with pytest.raises(RuntimeError, match="registered but inactive"):
        auditor._run_with_filtered_cache("input")


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
