import torch
import time
from transformer_lens import TransformerBridge
from transformerlens_fdr import PatchingAuditor, logit_difference, plot_fdr_heatmap, HOOK_CONFIGS

def main():
    # 1. Load a small HookedTransformer model for testing
    print("Loading model (this may take a minute on first run)...", flush=True)
    t0 = time.time()
    model = TransformerBridge.boot_transformers("gpt2")
    model.enable_compatibility_mode()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)
    
    # 2. Setup your prompts
    # For a typical Indirect Object Identification (IOI) style task
    clean_prompt = "When John and Mary went to the store, John gave a drink to"
    corrupted_prompt = "When John and Mary went to the store, Mary gave a drink to"
    
    # Tokenize prompts
    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)
    
    # Identify the token IDs for the correct and incorrect predictions
    # In clean_prompt, the correct answer is ' Mary', incorrect is ' John'
    correct_token_id = model.to_single_token(" Mary")
    incorrect_token_id = model.to_single_token(" John")
    
    # Create tensors for the auditor (needs a batch dimension)
    correct_tokens = torch.tensor([correct_token_id], device=model.cfg.device)
    incorrect_tokens = torch.tensor([incorrect_token_id], device=model.cfg.device)
    
    # 3. Show available hook types
    print("\nAvailable hook types:", flush=True)
    for key, cfg in HOOK_CONFIGS.items():
        print(f"  {key:15s} -> {cfg['description']} ({cfg['hook_suffix']})", flush=True)
    
    # 4. Initialize the Statistical Auditor
    print("\nInitializing auditor...", flush=True)
    auditor = PatchingAuditor(
        model=model,
        metric_fn=logit_difference,
        fdr_threshold=0.05,
        fdr_method="fdr_by",        # Safer for correlated heads
        n_control_samples=10,       # Small number for quick CPU demo
        hook_type="attn_head",
    )
    
    # 5. Run the Patching Audit
    print("Running patching audit (this may take a few minutes on CPU)...", flush=True)
    t0 = time.time()
    results = auditor.run_patching_audit(
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        correct_tokens=correct_tokens,
        incorrect_tokens=incorrect_tokens,
    )
    print(f"Audit completed in {time.time() - t0:.1f}s", flush=True)
    
    # 6. Review the Results
    print(f"\n{'='*50}", flush=True)
    print(f"Circuit Localization Index (CLI): {results.cli_score:.4f}", flush=True)
    print(f"Hook type audited: {results.hook_type}", flush=True)
    print(f"\nTop 10 heads by raw patching effect:", flush=True)
    top = results.summary_df.sort_values("raw_effect", ascending=False).head(10)
    print(top.to_string(index=False), flush=True)
    
    passed_heads = results.passed_fdr_mask.sum().item()
    total_heads = results.passed_fdr_mask.numel()
    print(f"\nHeads passing FDR: {int(passed_heads)} / {total_heads}", flush=True)
    
    # 7. Validation: Joint Knockout / Cumulative Patching
    if passed_heads > 0:
        print(f"\nRunning Joint Patching on the {int(passed_heads)} surviving heads...", flush=True)
        p_joint = auditor.run_joint_patching(
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
            correct_tokens=correct_tokens,
            incorrect_tokens=incorrect_tokens,
            patching_mask=results.passed_fdr_mask
        )
        print(f"Joint Knockout Restoration (P_joint): {p_joint:.4f}", flush=True)
        
    print(f"{'='*50}", flush=True)

if __name__ == "__main__":
    main()
