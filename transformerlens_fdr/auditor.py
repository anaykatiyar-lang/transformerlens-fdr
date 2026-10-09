"""Activation-patching audit with explicitly calibrated inference options."""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, List

import pandas as pd
import torch

from .stats import (
    apply_fdr_adjustment,
    calculate_empirical_p_values,
    compute_robust_baseline,
    signflip_p_values,
)
from .metrics import normalized_patching_effect, circuit_localization_index


HOOK_CONFIGS = {
    "attn_head": {
        "hook_suffix": "attn.hook_z",
        "has_head_dim": True,
        "has_position_dim": False,
        "description": "Attention head outputs",
    },
    "mlp_out": {
        "hook_suffix": "hook_mlp_out",
        "has_head_dim": False,
        "has_position_dim": False,
        "description": "MLP layer outputs",
    },
    "resid_mid": {
        "hook_suffix": "hook_resid_mid",
        "has_head_dim": False,
        "has_position_dim": True,
        "description": "Position-specific mid-layer residual stream",
    },
}


def _build_hook_name(layer: int, hook_suffix: str) -> str:
    return f"blocks.{layer}.{hook_suffix}"


@dataclass
class AuditResults:
    raw_patching_effects: torch.Tensor
    per_prompt_effects: torch.Tensor
    adjusted_effects: torch.Tensor
    p_values: Optional[torch.Tensor]
    q_values: Optional[torch.Tensor]
    significant_mask: Optional[torch.Tensor]
    outlier_scores: torch.Tensor
    calibrated: bool
    null_description: str
    cli_score: float
    summary_df: pd.DataFrame
    hook_type: str = "attn_head"


class PatchingAuditor:
    def __init__(
        self,
        model: Any,
        metric_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        fdr_threshold: float = 0.05,
        fdr_method: str = "fdr_bh",
        n_perm: int = 9999,
        hook_type: str = "attn_head",
    ):
        """Create an auditor.

        ``metric_fn`` must return one scalar per prompt, with shape ``[B]``.
        BH assumes valid p-values and independence or positive regression
        dependence. BY is valid under arbitrary dependence, but neither method
        can repair invalid p-values.
        """
        if hook_type not in HOOK_CONFIGS:
            raise ValueError(
                f"Unsupported hook_type '{hook_type}'. Choose from: {list(HOOK_CONFIGS)}"
            )
        if n_perm < 1:
            raise ValueError("n_perm must be positive.")

        self.model = model
        self.metric_fn = metric_fn
        self.fdr_threshold = fdr_threshold
        self.fdr_method = fdr_method
        self.n_perm = n_perm
        self.hook_type = hook_type
        self._hook_cfg = HOOK_CONFIGS[hook_type]

    def _get_audit_shape(self, seq_len: int) -> tuple:
        n_layers = self.model.cfg.n_layers
        if self._hook_cfg["has_head_dim"]:
            return n_layers, self.model.cfg.n_heads
        if self._hook_cfg["has_position_dim"]:
            return n_layers, seq_len
        return n_layers, 1

    def _metric_per_prompt(self, logits, correct_tokens, incorrect_tokens, batch_size):
        values = self.metric_fn(logits, correct_tokens, incorrect_tokens)
        if not isinstance(values, torch.Tensor) or values.shape != (batch_size,):
            raise ValueError(
                "metric_fn must return one value per prompt with shape [batch]."
            )
        if not torch.isfinite(values).all():
            raise ValueError("metric_fn returned non-finite values.")
        return values

    def _shared_denominator(self, clean_metric, corrupted_metric):
        denominator = (clean_metric - corrupted_metric).mean()
        if torch.abs(denominator).item() < 1e-8:
            raise ValueError(
                "Undefined normalized effect: mean clean and corrupted metrics "
                "are too close for a shared denominator."
            )
        return denominator

    def _patch_head_in_layer(self, clean_cache, layer: int, head_idx: int):
        hook_name = _build_hook_name(layer, self._hook_cfg["hook_suffix"])

        def hook_fn(activations, hook, _h=head_idx):
            activations[:, :, _h, :] = clean_cache[hook.name][:, :, _h, :]
            return activations

        return hook_name, hook_fn

    def _patch_whole_component(self, clean_cache, layer: int):
        hook_name = _build_hook_name(layer, self._hook_cfg["hook_suffix"])

        def hook_fn(activations, hook):
            activations[:] = clean_cache[hook.name]
            return activations

        return hook_name, hook_fn

    def _patch_position(self, clean_cache, layer: int, position: int):
        hook_name = _build_hook_name(layer, self._hook_cfg["hook_suffix"])

        def hook_fn(activations, hook, _position=position):
            activations[:, _position, :] = clean_cache[hook.name][:, _position, :]
            return activations

        return hook_name, hook_fn

    def _patch_layer_batched(
        self,
        clean_cache,
        corrupted_tokens,
        correct_tokens,
        incorrect_tokens,
        layer,
        corrupted_metric,
        denominator,
    ):
        """Patch each attention head in one replicated-batch forward pass."""
        n_heads = self.model.cfg.n_heads
        hook_name = _build_hook_name(layer, self._hook_cfg["hook_suffix"])
        batch_size = corrupted_tokens.shape[0]
        batched_tokens = corrupted_tokens.repeat(n_heads, 1)
        batched_correct = correct_tokens.repeat(n_heads)
        batched_incorrect = incorrect_tokens.repeat(n_heads)

        def batched_patch_hook(activations, hook):
            clean_act = clean_cache[hook.name]
            for head in range(n_heads):
                start, end = head * batch_size, (head + 1) * batch_size
                activations[start:end, :, head, :] = clean_act[:, :, head, :]
            return activations

        logits = self.model.run_with_hooks(
            batched_tokens,
            return_type="logits",
            fwd_hooks=[(hook_name, batched_patch_hook)],
        )
        effects = torch.zeros(
            (n_heads, batch_size), device=logits.device, dtype=logits.dtype
        )
        for head in range(n_heads):
            start, end = head * batch_size, (head + 1) * batch_size
            patched_metric = self._metric_per_prompt(
                logits[start:end], batched_correct[start:end],
                batched_incorrect[start:end], batch_size,
            )
            effects[head] = (patched_metric - corrupted_metric) / denominator
        return effects

    def _patch_layer_sequential(
        self,
        clean_cache,
        corrupted_tokens,
        correct_tokens,
        incorrect_tokens,
        layer,
        corrupted_metric,
        denominator,
    ):
        """Return per-prompt effects for MLP output or each residual position."""
        batch_size = corrupted_tokens.shape[0]
        if self._hook_cfg["has_position_dim"]:
            component_count = corrupted_tokens.shape[1]
            effects = torch.zeros(
                (component_count, batch_size),
                device=corrupted_tokens.device,
                dtype=torch.float32,
            )
            for position in range(component_count):
                hook_name, hook_fn = self._patch_position(clean_cache, layer, position)
                logits = self.model.run_with_hooks(
                    corrupted_tokens,
                    return_type="logits",
                    fwd_hooks=[(hook_name, hook_fn)],
                )
                patched_metric = self._metric_per_prompt(
                    logits, correct_tokens, incorrect_tokens, batch_size
                )
                effects[position] = (patched_metric - corrupted_metric) / denominator
            return effects

        hook_name, hook_fn = self._patch_whole_component(clean_cache, layer)
        logits = self.model.run_with_hooks(
            corrupted_tokens,
            return_type="logits",
            fwd_hooks=[(hook_name, hook_fn)],
        )
        patched_metric = self._metric_per_prompt(
            logits, correct_tokens, incorrect_tokens, batch_size
        )
        return ((patched_metric - corrupted_metric) / denominator).unsqueeze(0)

    def run_patching_audit(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
        correct_tokens: torch.Tensor,
        incorrect_tokens: torch.Tensor,
        null_distribution: Optional[torch.Tensor] = None,
        method: str = "none",
        pooled_null: bool = False,
    ) -> AuditResults:
        """Run patching and return exploratory scores or calibrated inference.

        ``null_distribution`` must be shaped ``[n_null, layers, components]``.
        A one-dimensional pooled null is accepted only with ``pooled_null=True``
        and is broadcast to each component with a warning.

        ``method='signflip'`` tests the assumption that per-prompt effects are
        symmetric about zero. With no valid null method, inferential fields are
        ``None`` and only robust exploratory outlier scores are returned.
        """
        if method not in {"none", "signflip"}:
            raise ValueError("method must be 'none' or 'signflip'.")
        if clean_tokens.shape != corrupted_tokens.shape:
            raise ValueError("clean_tokens and corrupted_tokens must have matching shapes.")
        batch_size, seq_len = corrupted_tokens.shape
        if correct_tokens.reshape(-1).numel() != batch_size or incorrect_tokens.reshape(-1).numel() != batch_size:
            raise ValueError("Token labels must contain one id per prompt.")

        n_layers, n_components = self._get_audit_shape(seq_len)
        clean_logits, clean_cache = self.model.run_with_cache(clean_tokens)
        corrupted_logits = self.model(corrupted_tokens)
        clean_metric = self._metric_per_prompt(
            clean_logits, correct_tokens, incorrect_tokens, batch_size
        )
        corrupted_metric = self._metric_per_prompt(
            corrupted_logits, correct_tokens, incorrect_tokens, batch_size
        )
        denominator = self._shared_denominator(clean_metric, corrupted_metric)

        per_prompt = torch.zeros(
            (n_layers, n_components, batch_size),
            device=corrupted_tokens.device,
            dtype=torch.float32,
        )
        for layer in range(n_layers):
            if self._hook_cfg["has_head_dim"]:
                per_prompt[layer] = self._patch_layer_batched(
                    clean_cache, corrupted_tokens, correct_tokens, incorrect_tokens,
                    layer, corrupted_metric, denominator,
                )
            else:
                per_prompt[layer] = self._patch_layer_sequential(
                    clean_cache, corrupted_tokens, correct_tokens, incorrect_tokens,
                    layer, corrupted_metric, denominator,
                )

        mean_effects = per_prompt.mean(dim=-1)
        baseline, scale = compute_robust_baseline(mean_effects.reshape(-1))
        outlier_scores = (mean_effects - baseline) / scale

        calibrated = False
        null_description = "none (robust outlier heuristic; exploratory only)"
        adjusted_effects = mean_effects - baseline
        p_values = q_values = significant_mask = None

        if null_distribution is not None:
            null_distribution = null_distribution.to(
                device=mean_effects.device, dtype=mean_effects.dtype
            )
            if null_distribution.ndim == 1:
                if not pooled_null:
                    raise ValueError(
                        "A pooled 1-D null requires pooled_null=True."
                    )
                warnings.warn(
                    "Broadcasting a pooled null assumes the same null distribution "
                    "for every component.", UserWarning, stacklevel=2,
                )
                null_distribution = null_distribution[:, None, None].expand(
                    -1, n_layers, n_components
                )
                null_description = "caller-supplied pooled empirical null (broadcast)"
            else:
                null_description = "caller-supplied per-component empirical null"

            if null_distribution.ndim != 3 or null_distribution.shape[1:] != mean_effects.shape:
                raise ValueError(
                    "null_distribution must have shape [n_null, layers, components]."
                )
            null_center = torch.median(null_distribution, dim=0).values
            adjusted_effects = mean_effects - null_center
            p_values = calculate_empirical_p_values(mean_effects, null_distribution)
            calibrated = True
        elif method == "signflip":
            p_values = signflip_p_values(per_prompt, n_perm=self.n_perm)
            adjusted_effects = mean_effects
            calibrated = True
            null_description = (
                "sign-flip over prompts (requires independent/exchangeable effects "
                "symmetric about zero; "
                "see skew robustness diagnostics)"
            )
        else:
            warnings.warn(
                "No null supplied: results are exploratory outlier scores, not FDR-controlled.",
                UserWarning,
                stacklevel=2,
            )

        if p_values is not None:
            q_values, significant_mask = apply_fdr_adjustment(
                p_values, alpha=self.fdr_threshold, method=self.fdr_method
            )

        raw_effects = mean_effects
        cli_score = circuit_localization_index(raw_effects)
        component_label = (
            "head" if self._hook_cfg["has_head_dim"]
            else "position" if self._hook_cfg["has_position_dim"]
            else "component"
        )
        records = []
        for layer in range(n_layers):
            for component in range(n_components):
                row = {
                    "layer": layer,
                    component_label: component,
                    "raw_effect": raw_effects[layer, component].item(),
                    "adjusted_effect": adjusted_effects[layer, component].item(),
                    "outlier_score": outlier_scores[layer, component].item(),
                    "calibrated": calibrated,
                    "null_description": null_description,
                }
                row["p_value"] = None if p_values is None else p_values[layer, component].item()
                row["q_value"] = None if q_values is None else q_values[layer, component].item()
                row["significant"] = (
                    None if significant_mask is None
                    else significant_mask[layer, component].item()
                )
                records.append(row)

        return AuditResults(
            raw_patching_effects=raw_effects,
            per_prompt_effects=per_prompt,
            adjusted_effects=adjusted_effects,
            p_values=p_values,
            q_values=q_values,
            significant_mask=significant_mask,
            outlier_scores=outlier_scores,
            calibrated=calibrated,
            null_description=null_description,
            cli_score=cli_score,
            summary_df=pd.DataFrame(records),
            hook_type=self.hook_type,
        )

    def run_joint_patching(
        self,
        clean_tokens,
        corrupted_tokens,
        correct_tokens,
        incorrect_tokens,
        patching_mask,
    ) -> float:
        """Patch all selected heads/components/positions in one forward pass."""
        n_layers, n_components = self._get_audit_shape(corrupted_tokens.shape[1])
        if tuple(patching_mask.shape) != (n_layers, n_components):
            raise ValueError(f"patching_mask must have shape [{n_layers}, {n_components}].")
        clean_logits, clean_cache = self.model.run_with_cache(clean_tokens)
        corrupted_logits = self.model(corrupted_tokens)
        batch_size = corrupted_tokens.shape[0]
        clean_metric = self._metric_per_prompt(
            clean_logits, correct_tokens, incorrect_tokens, batch_size
        )
        corrupted_metric = self._metric_per_prompt(
            corrupted_logits, correct_tokens, incorrect_tokens, batch_size
        )
        denominator = self._shared_denominator(clean_metric, corrupted_metric)
        fwd_hooks = []

        for layer in range(n_layers):
            selected = torch.where(patching_mask[layer])[0].tolist()
            if not selected:
                continue
            hook_name = _build_hook_name(layer, self._hook_cfg["hook_suffix"])
            if self._hook_cfg["has_head_dim"]:
                def make_head_hook(heads):
                    def hook_fn(activations, hook):
                        clean_act = clean_cache[hook.name]
                        for head in heads:
                            activations[:, :, head, :] = clean_act[:, :, head, :]
                        return activations
                    return hook_fn
                fwd_hooks.append((hook_name, make_head_hook(selected)))
            elif self._hook_cfg["has_position_dim"]:
                def make_position_hook(positions):
                    def hook_fn(activations, hook):
                        clean_act = clean_cache[hook.name]
                        for position in positions:
                            activations[:, position, :] = clean_act[:, position, :]
                        return activations
                    return hook_fn
                fwd_hooks.append((hook_name, make_position_hook(selected)))
            else:
                def make_component_hook():
                    def hook_fn(activations, hook):
                        activations[:] = clean_cache[hook.name]
                        return activations
                    return hook_fn
                fwd_hooks.append((hook_name, make_component_hook()))

        patched_logits = self.model.run_with_hooks(
            corrupted_tokens, return_type="logits", fwd_hooks=fwd_hooks
        )
        patched_metric = self._metric_per_prompt(
            patched_logits, correct_tokens, incorrect_tokens, batch_size
        )
        return ((patched_metric - corrupted_metric).mean() / denominator).item()

    def run_head_patching_audit(self, *args, **kwargs) -> AuditResults:
        """Backward-compatible alias for :meth:`run_patching_audit`."""
        return self.run_patching_audit(*args, **kwargs)
