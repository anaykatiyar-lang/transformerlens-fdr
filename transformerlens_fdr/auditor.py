import torch
import pandas as pd
from typing import Callable, Dict, Any, Optional, List
# HookedTransformer was removed in v4.0.0, we use Any for the model type hint
from dataclasses import dataclass
from .stats import compute_robust_baseline, calculate_p_values, apply_fdr_adjustment, calculate_empirical_p_values
from .metrics import normalized_patching_effect, circuit_localization_index

# Supported hook types and their configurations
HOOK_CONFIGS = {
    "attn_head": {
        "hook_suffix": "attn.hook_z",       # shape [B, S, H, D]
        "has_head_dim": True,
        "description": "Attention head outputs",
    },
    "mlp_out": {
        "hook_suffix": "hook_mlp_out",       # shape [B, S, D]
        "has_head_dim": False,
        "description": "MLP layer outputs",
    },
    "resid_mid": {
        "hook_suffix": "hook_resid_mid",     # shape [B, S, D]
        "has_head_dim": False,
        "description": "Mid-layer residual stream",
    },
}


def _build_hook_name(layer: int, hook_suffix: str) -> str:
    """Constructs the full hook name for a given layer and suffix."""
    return f"blocks.{layer}.{hook_suffix}"


@dataclass
class AuditResults:
    raw_patching_effects: torch.Tensor
    adjusted_effects: torch.Tensor
    p_values: torch.Tensor
    q_values: torch.Tensor
    passed_fdr_mask: torch.Tensor
    cli_score: float
    summary_df: pd.DataFrame
    hook_type: str = "attn_head"


class PatchingAuditor:
    def __init__(
        self,
        model: Any,
        metric_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        fdr_threshold: float = 0.05,
        fdr_method: str = 'fdr_bh',
        n_control_samples: int = 50,
        hook_type: str = "attn_head"
    ):
        """
        Args:
            model: TransformerLens model instance (e.g., TransformerBridge).
            metric_fn: Function taking (logits, correct_tokens, incorrect_tokens) -> scalar tensor.
            fdr_threshold: Target FDR q-value threshold (default: 0.05).
            fdr_method: FDR correction method — 'fdr_bh' (Benjamini-Hochberg) for
                independent / positively-dependent tests, or 'fdr_by'
                (Benjamini-Yekutieli) for arbitrarily dependent tests (safer for
                correlated attention heads, but more conservative).
            hook_type: Component type to audit. One of 'attn_head', 'mlp_out', 'resid_mid'.
        """
        if hook_type not in HOOK_CONFIGS:
            raise ValueError(
                f"Unsupported hook_type '{hook_type}'. "
                f"Choose from: {list(HOOK_CONFIGS.keys())}"
            )

        self.model = model
        self.metric_fn = metric_fn
        self.fdr_threshold = fdr_threshold
        self.fdr_method = fdr_method
        self.hook_type = hook_type
        self._hook_cfg = HOOK_CONFIGS[hook_type]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_audit_shape(self) -> tuple:
        """Returns the (n_layers, n_components) shape for the current hook type."""
        n_layers = self.model.cfg.n_layers
        if self._hook_cfg["has_head_dim"]:
            return (n_layers, self.model.cfg.n_heads)
        else:
            # MLP / residual — one scalar per layer
            return (n_layers, 1)

    def _patch_head_in_layer(
        self,
        clean_cache,
        layer: int,
        head_idx: int,
    ):
        """Returns a hook function that patches a single head in the layer."""
        hook_name = _build_hook_name(layer, self._hook_cfg["hook_suffix"])

        def hook_fn(activations, hook, _h=head_idx):
            activations[:, :, _h, :] = clean_cache[hook.name][:, :, _h, :]
            return activations

        return hook_name, hook_fn

    def _patch_whole_component(
        self,
        clean_cache,
        layer: int,
    ):
        """Returns a hook function that patches the full activation tensor for a
        layer (used for MLP / residual stream where there is no head dim)."""
        hook_name = _build_hook_name(layer, self._hook_cfg["hook_suffix"])

        def hook_fn(activations, hook):
            activations[:] = clean_cache[hook.name]
            return activations

        return hook_name, hook_fn

    # ------------------------------------------------------------------
    # Batched layer-level patching  (Fix #2)
    # ------------------------------------------------------------------

    def _patch_layer_batched(
        self,
        clean_cache,
        corrupted_tokens: torch.Tensor,
        correct_tokens: torch.Tensor,
        incorrect_tokens: torch.Tensor,
        layer: int,
        clean_metric: torch.Tensor,
        corrupted_metric: torch.Tensor,
    ) -> torch.Tensor:
        """Patches every head in *layer* using a single batched forward pass.

        The corrupted tokens are replicated n_heads times along the batch
        dimension.  A hook swaps in the clean activation for the corresponding
        head index based on the sample's position in the batch.  This reduces
        the number of forward passes from (n_layers × n_heads) to n_layers.

        Returns a Tensor of shape [n_heads] with normalised patching effects.
        """
        n_heads = self.model.cfg.n_heads
        hook_name = _build_hook_name(layer, self._hook_cfg["hook_suffix"])

        # Replicate inputs: [B, S] → [B * n_heads, S]
        batch_size = corrupted_tokens.shape[0]
        batched_tokens = corrupted_tokens.repeat(n_heads, 1)                 # [B*H, S]
        batched_correct = correct_tokens.repeat(n_heads)                     # [B*H]
        batched_incorrect = incorrect_tokens.repeat(n_heads)                 # [B*H]

        def batched_patch_hook(activations, hook):
            # activations: [B*H, S, n_heads, D]
            clean_act = clean_cache[hook.name]                               # [B, S, H, D]
            for h in range(n_heads):
                start = h * batch_size
                end = start + batch_size
                activations[start:end, :, h, :] = clean_act[:, :, h, :]
            return activations

        patched_logits = self.model.run_with_hooks(
            batched_tokens,
            return_type="logits",
            fwd_hooks=[(hook_name, batched_patch_hook)],
        )

        # Compute per-head metrics
        effects = torch.zeros(n_heads, device=self.model.cfg.device)
        for h in range(n_heads):
            start = h * batch_size
            end = start + batch_size
            head_logits = patched_logits[start:end]
            head_correct = batched_correct[start:end]
            head_incorrect = batched_incorrect[start:end]
            patched_metric = self.metric_fn(head_logits, head_correct, head_incorrect)
            effects[h] = normalized_patching_effect(patched_metric, clean_metric, corrupted_metric)

        return effects

    def _patch_layer_sequential(
        self,
        clean_cache,
        corrupted_tokens: torch.Tensor,
        correct_tokens: torch.Tensor,
        incorrect_tokens: torch.Tensor,
        layer: int,
        clean_metric: torch.Tensor,
        corrupted_metric: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback: sequential per-component patching for non-head hooks."""
        hook_name, hook_fn = self._patch_whole_component(clean_cache, layer)

        patched_logits = self.model.run_with_hooks(
            corrupted_tokens,
            return_type="logits",
            fwd_hooks=[(hook_name, hook_fn)],
        )
        patched_metric = self.metric_fn(patched_logits, correct_tokens, incorrect_tokens)
        effect = normalized_patching_effect(patched_metric, clean_metric, corrupted_metric)
        return effect.unsqueeze(0)  # [1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_patching_audit(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
        correct_tokens: torch.Tensor,
        incorrect_tokens: torch.Tensor,
        null_distribution: Optional[torch.Tensor] = None,
    ) -> AuditResults:
        """
        Executes activation patching, computes p-values,
        applies FDR correction, and calculates CLI.
        
        Args:
            null_distribution: An optional tensor of empirical null patching effects 
                from repeated null trials. If provided, empirical p-values are computed. 
                If not provided, a robust baseline (Median & Median Absolute Deviation) 
                is estimated from the observed component effects themselves.

        For attention heads (hook_type='attn_head') this uses batched
        layer-level forward passes — one pass per layer instead of one per
        head — which is dramatically faster on GPU.

        Returns an AuditResults dataclass.
        """
        n_layers, n_components = self._get_audit_shape()
        device = self.model.cfg.device

        # Forward passes for clean / corrupted baselines
        clean_logits, clean_cache = self.model.run_with_cache(clean_tokens)
        corrupted_logits = self.model(corrupted_tokens)

        clean_metric = self.metric_fn(clean_logits, correct_tokens, incorrect_tokens)
        corrupted_metric = self.metric_fn(corrupted_logits, correct_tokens, incorrect_tokens)

        raw_patching_effects = torch.zeros((n_layers, n_components), device=device)

        # ---------- Patching (batched when possible) ----------
        for l in range(n_layers):
            if self._hook_cfg["has_head_dim"]:
                raw_patching_effects[l] = self._patch_layer_batched(
                    clean_cache, corrupted_tokens, correct_tokens, incorrect_tokens,
                    l, clean_metric, corrupted_metric,
                )
            else:
                raw_patching_effects[l] = self._patch_layer_sequential(
                    clean_cache, corrupted_tokens, correct_tokens, incorrect_tokens,
                    l, clean_metric, corrupted_metric,
                )

        # ---------- Statistical audit ----------
        if null_distribution is not None:
            adjusted_effects = raw_patching_effects - torch.median(null_distribution)
            p_values = calculate_empirical_p_values(raw_patching_effects, null_distribution)
        else:
            # Fallback to robust estimation (treating sparse signals as outliers)
            mu_ctrl, sigma_ctrl = compute_robust_baseline(raw_patching_effects)
            adjusted_effects = raw_patching_effects - mu_ctrl
            p_values = calculate_p_values(raw_patching_effects, mu_ctrl, sigma_ctrl)
            
        q_values, passed_fdr_mask = apply_fdr_adjustment(
            p_values, alpha=self.fdr_threshold, method=self.fdr_method,
        )

        cli_score = circuit_localization_index(raw_patching_effects)

        # ---------- Summary DataFrame ----------
        records = []
        component_label = "head" if self._hook_cfg["has_head_dim"] else "component"
        for l in range(n_layers):
            for c in range(n_components):
                records.append({
                    "layer": l,
                    component_label: c,
                    "raw_effect": raw_patching_effects[l, c].item(),
                    "adjusted_effect": adjusted_effects[l, c].item(),
                    "p_value": p_values[l, c].item(),
                    "q_value": q_values[l, c].item(),
                    "passed_fdr": passed_fdr_mask[l, c].item(),
                })
        summary_df = pd.DataFrame(records)

        return AuditResults(
            raw_patching_effects=raw_patching_effects,
            adjusted_effects=adjusted_effects,
            p_values=p_values,
            q_values=q_values,
            passed_fdr_mask=passed_fdr_mask,
            cli_score=cli_score,
            summary_df=summary_df,
            hook_type=self.hook_type,
        )

    def run_joint_patching(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
        correct_tokens: torch.Tensor,
        incorrect_tokens: torch.Tensor,
        patching_mask: torch.Tensor
    ) -> float:
        """
        Executes a single forward pass where ALL components indicated in the 
        patching_mask are patched simultaneously.
        
        Args:
            patching_mask: Boolean tensor of shape [n_layers, n_components] (e.g., passed_fdr_mask)
            
        Returns:
            The normalized patching effect (P_joint) for the combined knockout.
        """
        clean_logits, clean_cache = self.model.run_with_cache(clean_tokens)
        corrupted_logits = self.model(corrupted_tokens)
        
        clean_metric = self.metric_fn(clean_logits, correct_tokens, incorrect_tokens)
        corrupted_metric = self.metric_fn(corrupted_logits, correct_tokens, incorrect_tokens)
        
        fwd_hooks = []
        n_layers = self.model.cfg.n_layers
        
        for l in range(n_layers):
            layer_mask = patching_mask[l]
            if not layer_mask.any():
                continue
                
            hook_name = _build_hook_name(l, self._hook_cfg["hook_suffix"])
            
            if self._hook_cfg["has_head_dim"]:
                heads_to_patch = torch.where(layer_mask)[0].tolist()
                
                # We need a closure to capture the specific heads for this layer
                def make_hook(heads):
                    def joint_head_hook(activations, hook):
                        clean_act = clean_cache[hook.name]
                        for h in heads:
                            activations[:, :, h, :] = clean_act[:, :, h, :]
                        return activations
                    return joint_head_hook
                
                fwd_hooks.append((hook_name, make_hook(heads_to_patch)))
            else:
                def make_hook():
                    def joint_comp_hook(activations, hook):
                        activations[:] = clean_cache[hook.name]
                        return activations
                    return joint_comp_hook
                    
                fwd_hooks.append((hook_name, make_hook()))
                
        patched_logits = self.model.run_with_hooks(
            corrupted_tokens,
            return_type="logits",
            fwd_hooks=fwd_hooks
        )
        
        patched_metric = self.metric_fn(patched_logits, correct_tokens, incorrect_tokens)
        return normalized_patching_effect(patched_metric, clean_metric, corrupted_metric).item()

    # Backwards-compatible alias for the original method name
    def run_head_patching_audit(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
        correct_tokens: torch.Tensor,
        incorrect_tokens: torch.Tensor,
    ) -> AuditResults:
        """Alias for run_patching_audit (backward compatibility)."""
        return self.run_patching_audit(
            clean_tokens, corrupted_tokens, correct_tokens, incorrect_tokens,
        )
