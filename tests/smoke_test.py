import os

import torch
from huggingface_hub import snapshot_download
from transformer_lens.model_bridge import TransformerBridge
from transformerlens_fdr.auditor import PatchingAuditor

# Select a model per run with FDR_MODEL_ID. Defaults to the prior Pythia smoke test.
MODEL_ID = os.environ.get("FDR_MODEL_ID", "EleutherAI/pythia-1b")
DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
MODEL_DTYPE_NAME = os.environ.get("FDR_DTYPE", "float32").strip().lower()
if MODEL_DTYPE_NAME not in DTYPE_BY_NAME:
    raise ValueError(f"Unsupported FDR_DTYPE {MODEL_DTYPE_NAME!r}; choose {tuple(DTYPE_BY_NAME)}")


def metric(logits, correct_tokens, incorrect_tokens):
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, -1, correct_tokens] - logits[rows, -1, incorrect_tokens]


def test_downloaded_model_audit_smoke():
    model_path = snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerBridge.boot_transformers(
        model_path, device=device, dtype=DTYPE_BY_NAME[MODEL_DTYPE_NAME]
    )

    vocab_size = int(model.cfg.d_vocab)
    device = next(model.parameters()).device

    torch.manual_seed(17)
    clean = torch.randint(0, vocab_size, (4, 6), device=device)
    corrupted = clean.clone()
    corrupted[:, -2] = (corrupted[:, -2] + 1) % vocab_size
    correct = torch.randint(0, vocab_size, (4,), device=device)
    incorrect = (correct + 1) % vocab_size

    auditor = PatchingAuditor(model, metric_fn=metric, hook_type="attn_head")
    result = auditor.run_patching_audit(
        clean_tokens=clean,
        corrupted_tokens=corrupted,
        correct_tokens=correct,
        incorrect_tokens=incorrect,
        method="none",
    )

    assert result.raw_patching_effects.shape == (
        model.cfg.n_layers,
        model.cfg.n_heads,
    )
    assert torch.isfinite(result.raw_patching_effects).all()
    print(
        f"{MODEL_ID} ({MODEL_DTYPE_NAME}): shape={tuple(result.raw_patching_effects.shape)}, "
        f"finite=True, calibrated={result.calibrated}"
    )
