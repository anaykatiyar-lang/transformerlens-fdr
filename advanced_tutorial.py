import torch
import time
from transformer_lens import TransformerBridge
from transformerlens_fdr import PatchingAuditor, logit_difference, plot_fdr_heatmap, HOOK_CONFIGS

def run_ood_transfer_verification(model, baseline_heads_mask):
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
        n_control_samples=10, 
        hook_type="attn_head",
    )
    
    print("Running audit on OOD prompt...", flush=True)
    results = auditor.run_patching_audit(
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        correct_tokens=correct_tokens,
        incorrect_tokens=incorrect_tokens,
    )
    
    # Compare which heads survived in BOTH prompts
    overlap_mask = baseline_heads_mask & results.passed_fdr_mask
    overlap_count = overlap_mask.sum().item()
    
    print(f"OOD Circuit Localization Index (CLI): {results.cli_score:.4f}")
    print(f"Heads passing FDR on OOD prompt: {int(results.passed_fdr_mask.sum().item())}")
    print(f"✅ Robust Algorithmic Heads (passed BOTH baseline and OOD): {int(overlap_count)}")
    
    if overlap_count > 0:
        print("\nRobust Heads (Layer, Head):")
        indices = overlap_mask.nonzero()
        for idx in indices:
            print(f"  -> Layer {idx[0].item()}, Head {idx[1].item()}")
    else:
        print("\nNo heads transferred. The original circuit may be memorization/noise.")

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
        fdr_method="fdr_bh",        # MLPs are independent per layer, BH is fine
        n_control_samples=10, 
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
    passed_mlps = results.passed_fdr_mask.sum().item()
    print(f"MLP Layers passing FDR: {int(passed_mlps)} / {results.passed_fdr_mask.numel()}")
    
    if passed_mlps > 0:
        print("\nSignificant MLP Layers:")
        top = results.summary_df[results.summary_df['passed_fdr'] == True].sort_values("raw_effect", ascending=False)
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
        n_control_samples=10,
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
    print(f"Heads passing FDR: {int(baseline_results.passed_fdr_mask.sum().item())}")
    
    # Test 2: OOD Transfer
    run_ood_transfer_verification(model, baseline_results.passed_fdr_mask)
    
    # Test 3: Multi-Dimensional (MLP)
    run_multi_dimensional_testing(model, clean_tokens, corrupted_tokens, correct_tokens, incorrect_tokens)
    
    print("\n" + "="*60)
    print("🎉 Advanced testing complete!")
    print("="*60)

if __name__ == "__main__":
    main()
