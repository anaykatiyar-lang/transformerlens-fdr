"""
Integration tests against a real TransformerLens 4 TransformerBridge.

Opt in with FDR_BRIDGE=1. Select a cached model with FDR_MODEL_ID (default: gpt2).

These check the things the toy model in test_stress.py cannot:
  * the hook names in HOOK_CONFIGS resolve on a live bridge
  * shapes at those hooks match what the auditor assumes
  * cfg.n_layers / cfg.n_heads / cfg.device exist, and bridge(tokens) returns logits
  * the batched path matches a brute-force reference on representative real hooks
  * hooks do not leak between runs
  * resid_mid returns position-specific effects
"""
import math
import os
import warnings

import numpy as np
import pytest
import torch

pytest.importorskip("transformer_lens")
pytestmark = pytest.mark.skipif((os.environ.get("FDR_BRIDGE") or "").strip() != "1",
                                reason="set FDR_BRIDGE=1 to run (downloads gpt2)")

from transformer_lens.model_bridge import TransformerBridge  # noqa: E402
from huggingface_hub import snapshot_download                 # noqa: E402

from transformerlens_fdr.auditor import HOOK_CONFIGS, PatchingAuditor  # noqa: E402

MODEL_ID = os.environ.get("FDR_MODEL_ID", "gpt2")
DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
MODEL_DTYPE_NAME = os.environ.get("FDR_DTYPE", "float32").strip().lower()
if MODEL_DTYPE_NAME not in DTYPE_BY_NAME:
    raise ValueError(f"Unsupported FDR_DTYPE {MODEL_DTYPE_NAME!r}; choose {tuple(DTYPE_BY_NAME)}")


def metric_fn(logits, correct, incorrect):
    last = logits[:, -1, :]          # per-prompt [B], as the auditor requires
    return last.gather(1, correct[:, None]).squeeze(1) - last.gather(1, incorrect[:, None]).squeeze(1)


@pytest.fixture(scope="module")
def bridge():
    model_path = snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    b = TransformerBridge.boot_transformers(
        model_path, device="cpu", dtype=DTYPE_BY_NAME[MODEL_DTYPE_NAME]
    )
    b.eval()
    return b


@pytest.fixture(scope="module")
def data(bridge):
    # Synthetic valid token IDs make this integration test tokenizer-independent
    # and keep it fully offline once model weights are cached.
    vocab_size = int(bridge.cfg.d_vocab)
    generator = torch.Generator().manual_seed(17)
    clean = torch.randint(0, vocab_size, (4, 4), generator=generator)
    corr = clean.clone()
    corr[:, 1] = (corr[:, 1] + 1) % vocab_size
    correct = torch.randint(0, vocab_size, (4,), generator=generator)
    incorrect = (correct + 1) % vocab_size
    return clean, corr, correct, incorrect


def audit(bridge, data, hook_type):
    clean, corr, correct, incorrect = data
    aud = PatchingAuditor(bridge, metric_fn, hook_type=hook_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = aud.run_patching_audit(clean, corr, correct, incorrect)
    e = res.raw_patching_effects.detach().cpu().numpy()
    return e.reshape(e.shape[0], e.shape[1], -1).mean(-1)


def test_cfg_and_forward_contract(bridge, data):
    for field in ("n_layers", "n_heads", "device"):
        assert hasattr(bridge.cfg, field), f"bridge.cfg.{field} missing"
    out = bridge(data[0])
    assert torch.is_tensor(out), "bridge(tokens) must return a logits tensor"
    assert out.shape[:2] == data[0].shape


@pytest.mark.parametrize("hook_type", list(HOOK_CONFIGS))
def test_hook_names_resolve_and_have_expected_shapes(bridge, data, hook_type):
    cfg = HOOK_CONFIGS[hook_type]
    name = f"blocks.0.{cfg['hook_suffix']}"
    assert name in bridge.hook_dict, f"{name} not registered on the bridge"
    canonical = bridge.hook_dict[name].name
    _, cache = bridge.run_with_cache(data[0])
    assert canonical in cache, "hooks see hook.name (canonical); it must be in the cache"
    act = cache[canonical]
    B, S = data[0].shape
    if cfg["has_head_dim"]:
        assert act.shape[:3] == (B, S, bridge.cfg.n_heads)
    else:
        assert act.shape[:2] == (B, S) and act.dim() == 3
    print(f"[diagnostic] {name} -> {canonical} {tuple(act.shape)}")


def test_batched_path_matches_bruteforce_on_real_hooks(bridge, data):
    clean, corr, correct, incorrect = data
    n_layers, n_heads = bridge.cfg.n_layers, bridge.cfg.n_heads
    layers = sorted({0, n_layers // 2, n_layers - 1})
    heads = sorted({0, n_heads // 2, n_heads - 1})
    with torch.no_grad():
        clean_logits, cache = bridge.run_with_cache(clean)
        mc = metric_fn(clean_logits, correct, incorrect).mean()
        mk = metric_fn(bridge(corr), correct, incorrect).mean()
        ref = np.zeros((len(layers), len(heads)))
        for layer_index, l in enumerate(layers):
            for head_index, h in enumerate(heads):
                def hk(act, hook, h=h):
                    act[:, :, h, :] = cache[hook.name][:, :, h, :]
                    return act
                pl = bridge.run_with_hooks(corr, return_type="logits",
                                           fwd_hooks=[(f"blocks.{l}.attn.hook_z", hk)])
                ref[layer_index, head_index] = float(
                    (metric_fn(pl, correct, incorrect).mean() - mk) / (mc - mk)
                )
    got = audit(bridge, data, "attn_head")
    np.testing.assert_allclose(got[np.ix_(layers, heads)], ref, atol=2e-3)


def test_no_hook_leakage_after_audit(bridge, data):
    before = bridge(data[1]).detach().clone()
    audit(bridge, data, "mlp_out")
    audit(bridge, data, "attn_head")
    after = bridge(data[1])
    assert torch.allclose(before, after, atol=1e-5), "hooks leaked into later forward passes"


def test_resid_mid_reports_position_specific_effects(bridge, data):
    """The auditor reports one residual-stream effect per token position."""
    e = audit(bridge, data, "resid_mid")
    assert e.shape == (bridge.cfg.n_layers, data[0].shape[1])
    assert np.isfinite(e).all()
    print(f"[diagnostic] {MODEL_ID} ({MODEL_DTYPE_NAME}): resid_mid effects have shape {e.shape}")
