import torch
import time
from transformer_lens import TransformerBridge
from transformerlens_fdr import PatchingAuditor, logit_difference, HOOK_CONFIGS

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
        fdr_method="fdr_by",        # Requires valid p-values; BY cannot calibrate them
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
    
    if results.calibrated:
        passed_heads = results.significant_mask.sum().item()
        total_heads = results.significant_mask.numel()
        print(f"\nCalibrated heads passing FDR: {int(passed_heads)} / {int(total_heads)}", flush=True)
    else:
        print("\nExploratory only: no p-values or FDR mask were computed.", flush=True)
        print("Top heads by robust outlier score:", flush=True)
        print(results.summary_df.sort_values("outlier_score", key=abs, ascending=False).head(10).to_string(index=False), flush=True)
        
    print(f"{'='*50}", flush=True)

if __name__ == "__main__":
    main()
