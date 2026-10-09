import torch
import time
from transformer_lens import TransformerBridge
from transformerlens_fdr import PatchingAuditor, logit_difference, HOOK_CONFIGS

def run_ood_transfer_verification(model, baseline_results):
    """
    Tests if the circuit discovered on the baseline IOI prompt generalizes
    to an Out-Of-Distribution (OOD) prompt (different names, different context).
    """
    print("\n" + "="*60)
    print("🧪 TEST 2: Out-of-Distribution (OOD) Transfer Verification")
    print("="*60)
    
    # New prompt with entirely different tokens to test algorithmic generalization
    clean_prompt = "After Oliver and Charlotte finished reading, Oliver handed a book to"
    corrupted_prompt = "After Oliver and Charlotte finished reading, Charlotte handed a book to"
    
    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)
    
    correct_token_id = model.to_single_token(" Charlotte")
    incorrect_token_id = model.to_single_token(" Oliver")
    
    correct_tokens = torch.tensor([correct_token_id], device=model.cfg.device)
    incorrect_tokens = torch.tensor([incorrect_token_id], device=model.cfg.device)
    
    auditor = PatchingAuditor(
        model=model,
        metric_fn=logit_difference,
        fdr_threshold=0.05,
        fdr_method="fdr_by",
        hook_type="attn_head",
    )
    
    print("Running audit on OOD prompt...", flush=True)
    results = auditor.run_patching_audit(
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        correct_tokens=correct_tokens,
        incorrect_tokens=incorrect_tokens,
    )
    
    print(f"OOD Circuit Localization Index (CLI): {results.cli_score:.4f}")
    if baseline_results.significant_mask is None or results.significant_mask is None:
        print("Both runs are exploratory only; no cross-prompt FDR conclusion is available.")
    else:
        overlap = baseline_results.significant_mask & results.significant_mask
        print(f"Heads significant in both calibrated runs: {int(overlap.sum().item())}")

def run_multi_dimensional_testing(model, clean_tokens, corrupted_tokens, correct_tokens, incorrect_tokens):
    """
    Demonstrates auditing other component dimensions, like MLP layers.
    """
    print("\n" + "="*60)
    print("🧪 TEST 3: Multi-Dimensional Testing (MLP Outputs)")
    print("="*60)
    
    auditor = PatchingAuditor(
        model=model,
        metric_fn=logit_difference,
        fdr_threshold=0.05,
        fdr_method="fdr_bh",        # Valid only when p-values are calibrated and assumptions hold
        hook_type="mlp_out",        # Targeting the Multi-Layer Perceptron!
    )
    
    print("Running audit on MLP layers...", flush=True)
    results = auditor.run_patching_audit(
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        correct_tokens=correct_tokens,
        incorrect_tokens=incorrect_tokens,
    )
    
    print(f"MLP Circuit Localization Index (CLI): {results.cli_score:.4f}")
    if results.significant_mask is None:
        print("MLP results are exploratory outlier scores; no FDR mask was computed.")
        top = results.summary_df.sort_values("outlier_score", key=abs, ascending=False)
        print(top[['layer', 'raw_effect', 'outlier_score']].to_string(index=False))
    elif results.significant_mask.any():
        print("\nSignificant MLP Layers:")
        top = results.summary_df[results.summary_df['significant'] == True].sort_values("raw_effect", ascending=False)
        print(top[['layer', 'raw_effect', 'q_value']].to_string(index=False))

def main():
    print("Loading model...", flush=True)
    model = TransformerBridge.boot_transformers("gpt2")
    model.enable_compatibility_mode()
    
    print("\n" + "="*60)
    print("🧪 TEST 1: Baseline Attention Head Audit")
    print("="*60)
    
    clean_prompt = "When John and Mary went to the store, John gave a drink to"
    corrupted_prompt = "When John and Mary went to the store, Mary gave a drink to"
    
    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)
    
    correct_tokens = torch.tensor([model.to_single_token(" Mary")], device=model.cfg.device)
    incorrect_tokens = torch.tensor([model.to_single_token(" John")], device=model.cfg.device)
    
    auditor = PatchingAuditor(
        model=model,
        metric_fn=logit_difference,
        fdr_threshold=0.05,
        fdr_method="fdr_by",
        hook_type="attn_head",
    )
    
    print("Running baseline audit...", flush=True)
    baseline_results = auditor.run_patching_audit(
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        correct_tokens=correct_tokens,
        incorrect_tokens=incorrect_tokens,
    )
    
    print(f"Baseline CLI: {baseline_results.cli_score:.4f}")
    print(f"Inference calibrated: {baseline_results.calibrated}")
    if not baseline_results.calibrated:
        print("No null supplied: head values are exploratory outlier scores only.")
    
    # Test 2: OOD Transfer
    run_ood_transfer_verification(model, baseline_results)
    
    # Test 3: Multi-Dimensional (MLP)
    run_multi_dimensional_testing(model, clean_tokens, corrupted_tokens, correct_tokens, incorrect_tokens)
    
    print("\n" + "="*60)
    print("🎉 Advanced testing complete!")
    print("="*60)

if __name__ == "__main__":
    main()
