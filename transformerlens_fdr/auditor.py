"""Activation-patching audit with explicitly calibrated inference options."""

import warnings
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, List

import pandas as pd
import torch

from .stats import (
    apply_fdr_adjustment,
    calculate_empirical_p_values,
    compute_robust_baseline,
    resolution_report,
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


def _format_methods_statement(
    calibrated: bool,
    null_description: str,
    fdr_method: Optional[str],
    alpha: Optional[float],
    family_size: int,
    family_note: Optional[str] = None,
) -> str:
    if not calibrated:
        return (
            "Exploratory analysis only: no null-hypothesis p-values or FDR "
            "correction were computed. Outlier scores and CLI are descriptive."
        )

    correction = {
        "fdr_bh": "Benjamini–Hochberg",
        "fdr_by": "Benjamini–Yekutieli",
    }.get(fdr_method, str(fdr_method))
    correction_claim = (
        "The BH correction also assumes independent tests or positive regression dependence."
        if fdr_method == "fdr_bh"
        else "The BY correction allows arbitrary dependence when the input p-values are valid."
    )
    if null_description.startswith("sign-flip"):
        null_claim = (
            "Sign-flip p-values assume prompt effects are independent or "
            "exchangeable and symmetric about zero under the null."
        )
    else:
        null_claim = (
            "Empirical-null p-values assume the caller's controls represent "
            "the same no-effect process, metric, normalization, and components. "
            "The package does not validate that design."
        )
    family_description = family_note or "within this audit"
    return (
        f"{correction} FDR at alpha={alpha:g} over {family_size} tests "
        f"{family_description}. {null_claim} {correction_claim} These assumptions were not "
        "validated automatically; significance is not evidence of practical "
        "importance or causality."
    )


@dataclass
class NullDistribution:
    """Caller-provided empirical null values and provenance labels.

    Values must contain one null replicate per row and one mean normalized
    effect per component, with shape ``[n_null, layers, components]``.
    Metadata is recorded as provenance; matching labels do not establish that
    the experimental null is scientifically appropriate.
    """

    values: torch.Tensor
    description: str = "caller-supplied empirical null"
    metric_name: Optional[str] = None
    normalization: Optional[str] = None
    hook_type: Optional[str] = None
    n_prompts: Optional[int] = None


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
    family_size: Optional[int] = None
    resolution_report: Optional[dict] = None
    minimum_attainable_p: Optional[float] = None
    fdr_method: Optional[str] = None
    fdr_threshold: Optional[float] = None
    clean_corrupted_gap_mean: Optional[float] = None
    clean_corrupted_gap_std: Optional[float] = None
    zero_effect_tolerance: float = 1e-8
    exact_zero_components: int = 0
    numerically_zero_components: int = 0
    null_metadata: Optional[dict] = None
    methods_statement: str = ""

    def format_summary(self) -> str:
        """Return a concise, copyable report of the method and key diagnostics."""
        lines = [f"Methods: {self.methods_statement}"]
        lines.append(f"Testing-family size: {self.family_size}")
        if self.clean_corrupted_gap_mean is not None:
            gap = f"Mean clean–corrupted gap: {self.clean_corrupted_gap_mean:.6g}"
            if self.clean_corrupted_gap_std is not None:
                gap += f" (prompt SD {self.clean_corrupted_gap_std:.6g})"
            else:
                gap += " (one prompt; spread unavailable)"
            lines.append(gap)
        lines.append(
            "Zero-effect components: "
            f"{self.exact_zero_components} exact, "
            f"{self.numerically_zero_components} within tolerance "
            f"({self.zero_effect_tolerance:g})"
        )
        if self.resolution_report is not None:
            report = self.resolution_report
            lines.append(
                f"P-value resolution: minimum {report['p_min']:.6g}; "
                f"first-step threshold {report['first_step_threshold']:.6g}; "
                f"minimum rank {report['min_rank_for_any_rejection']} "
                f"({report['status']})"
            )
        lines.append(f"Descriptive CLI: {self.cli_score:.4f}")
        return "\n".join(lines)


@dataclass
class AuditPreflight:
    """Cheap setup report made before running per-component patching."""

    n_prompts: int
    family_size: int
    method: str
    gap_mean: float
    gap_std: Optional[float]
    gap_usable: bool
    relative_gap_tolerance: float
    resolution_report: Optional[dict]
    messages: tuple

    def format_summary(self) -> str:
        lines = [
            f"Preflight: {self.n_prompts} prompts; "
            f"{self.family_size} components in the family; method={self.method}.",
            f"Mean clean–corrupted gap: {self.gap_mean:.6g} "
            f"(relative tolerance {self.relative_gap_tolerance:.3g}; "
            f"{'usable' if self.gap_usable else 'too small for stable normalization'}).",
        ]
        if self.gap_std is not None:
            lines.append(f"Prompt-to-prompt gap SD: {self.gap_std:.6g}.")
        else:
            lines.append("One prompt: gap spread cannot be estimated.")
        if self.resolution_report is not None:
            report = self.resolution_report
            lines.append(
                f"P-value resolution: minimum {report['p_min']:.6g}; "
                f"first-step threshold {report['first_step_threshold']:.6g}; "
                f"minimum rank {report['min_rank_for_any_rejection']} "
                f"({report['status']})."
            )
        lines.extend(self.messages)
        lines.append("Zero-effect counts are available only after component patching.")
        return "\n".join(lines)


@dataclass
class HeldOutResults:
    """Selection and disjoint held-out summaries for joint patch validation."""

    selection_audit: AuditResults
    evaluation_audit: AuditResults
    joint_effect: float
    selection_group_ids: tuple
    evaluation_group_ids: tuple


def adjust_audits_together(
    results: List[AuditResults],
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> int:
    """Apply one BH/BY correction across multiple calibrated audit results.

    This explicitly defines a larger family by pooling all supplied p-values.
    The result objects are updated in place so their masks and q-values refer
    to the joint family. Uncalibrated results cannot be included.

    Returns the total number of hypotheses in the pooled family.
    """
    if not results:
        raise ValueError("At least one AuditResults object is required.")
    if len({id(result) for result in results}) != len(results):
        raise ValueError("Each AuditResults object can appear only once in a family.")
    if any(result.p_values is None for result in results):
        raise ValueError("Every audit must contain calibrated p_values to pool.")
    if any(result.minimum_attainable_p is None for result in results):
        raise ValueError("Every audit must record its minimum attainable p-value.")
    if any(not isinstance(result.p_values, torch.Tensor) for result in results):
        raise TypeError("Every audit's p_values must be a torch.Tensor.")
    if any(len(result.summary_df) != result.p_values.numel() for result in results):
        raise ValueError("Each summary_df must have one row per p-value.")

    pooled = torch.cat([
        result.p_values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        for result in results
    ])
    q_values, significant = apply_fdr_adjustment(pooled, alpha=alpha, method=method)
    family_size = pooled.numel()
    p_min = min(result.minimum_attainable_p for result in results)
    resolution = resolution_report(family_size, alpha, method, p_min)
    PatchingAuditor._resolution_warning(resolution)

    cursor = 0
    for result in results:
        count = result.p_values.numel()
        q = q_values[cursor:cursor + count].reshape(result.p_values.shape)
        mask = significant[cursor:cursor + count].reshape(result.p_values.shape)
        result.q_values = q.to(device=result.p_values.device, dtype=result.p_values.dtype)
        result.significant_mask = mask.to(device=result.p_values.device)
        result.family_size = family_size
        result.resolution_report = dict(resolution)
        result.fdr_method = method
        result.fdr_threshold = alpha
        result.methods_statement = _format_methods_statement(
            True,
            result.null_description,
            method,
            alpha,
            family_size,
            family_note=f"jointly across {len(results)} audits",
        )
        result.summary_df["q_value"] = result.q_values.detach().cpu().reshape(-1).tolist()
        result.summary_df["significant"] = result.significant_mask.detach().cpu().reshape(-1).tolist()
        result.summary_df["family_size"] = family_size
        cursor += count
    return family_size


class PatchingAuditor:
    def __init__(
        self,
        model: Any,
        metric_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        fdr_threshold: float = 0.05,
        fdr_method: str = "fdr_bh",
        n_perm: int = 9999,
        hook_type: str = "attn_head",
        zero_effect_tolerance: float = 1e-8,
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
        if not isinstance(n_perm, int) or n_perm < 1:
            raise ValueError("n_perm must be positive.")
        if not 0 < fdr_threshold < 1:
            raise ValueError("fdr_threshold must be strictly between 0 and 1.")
        if fdr_method not in {"fdr_bh", "fdr_by"}:
            raise ValueError("fdr_method must be 'fdr_bh' or 'fdr_by'.")
        if not math.isfinite(zero_effect_tolerance) or zero_effect_tolerance < 0:
            raise ValueError("zero_effect_tolerance must be finite and non-negative.")

        self.model = model
        self.metric_fn = metric_fn
        self.fdr_threshold = fdr_threshold
        self.fdr_method = fdr_method
        self.n_perm = n_perm
        self.hook_type = hook_type
        self.zero_effect_tolerance = float(zero_effect_tolerance)
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
        if not torch.is_floating_point(values):
            raise TypeError("metric_fn must return floating-point values.")
        if not torch.isfinite(values).all():
            raise ValueError("metric_fn returned non-finite values.")
        return values

    def _shared_denominator(self, clean_metric, corrupted_metric):
        gaps = clean_metric - corrupted_metric
        denominator = gaps.mean()
        metric_scale = max(
            clean_metric.abs().mean().item(),
            corrupted_metric.abs().mean().item(),
            torch.finfo(gaps.dtype).tiny,
        )
        relative_tolerance = 1e-8 * metric_scale
        if abs(denominator.item()) <= relative_tolerance:
            raise ValueError(
                "Undefined normalized effect: mean clean and corrupted metrics "
                f"are too close for a shared denominator (gap={denominator.item():.3g}, "
                f"relative tolerance={relative_tolerance:.3g})."
            )
        return denominator

    def _validate_inputs(self, clean_tokens, corrupted_tokens, correct_tokens, incorrect_tokens):
        if not all(isinstance(value, torch.Tensor) for value in (
            clean_tokens, corrupted_tokens, correct_tokens, incorrect_tokens
        )):
            raise TypeError("Token inputs must be torch.Tensor instances.")
        if clean_tokens.ndim != 2 or corrupted_tokens.ndim != 2:
            raise ValueError("clean_tokens and corrupted_tokens must have shape [batch, sequence].")
        if clean_tokens.shape != corrupted_tokens.shape:
            raise ValueError("clean_tokens and corrupted_tokens must have matching shapes.")
        batch_size, sequence_length = corrupted_tokens.shape
        if batch_size == 0 or sequence_length == 0:
            raise ValueError("Token batches and sequences must be non-empty.")
        if clean_tokens.device != corrupted_tokens.device:
            raise ValueError("clean_tokens and corrupted_tokens must be on the same device.")
        tokenizer = getattr(self.model, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            if not torch.equal(clean_tokens == pad_token_id, corrupted_tokens == pad_token_id):
                raise ValueError(
                    "clean_tokens and corrupted_tokens must use matching padding positions."
                )
        if clean_tokens.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("Token ids must use an integer tensor dtype.")
        if corrupted_tokens.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("Token ids must use an integer tensor dtype.")
        for name, values in (("correct_tokens", correct_tokens), ("incorrect_tokens", incorrect_tokens)):
            if values.numel() != batch_size:
                raise ValueError(f"{name} must contain one token id per prompt.")
            if values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise TypeError(f"{name} must use an integer tensor dtype.")
        return batch_size, sequence_length

    @staticmethod
    def _resolution_warning(report):
        if report["min_rank_for_any_rejection"] is None:
            warnings.warn(
                "Permutation resolution is too coarse for any rejection in this "
                f"{report['n_tests']}-test {report['method']} family at alpha="
                f"{report['alpha']}; increase the number of permutations or null samples.",
                UserWarning,
                stacklevel=3,
            )
        elif report["p_min"] > report["first_step_threshold"]:
            warnings.warn(
                "A result at the minimum attainable p-value cannot pass the first "
                f"{report['method']} step alone. Rejection requires at least "
                f"{report['min_rank_for_any_rejection']} tests at or below their "
                "rank-specific threshold; increase permutations/null samples for finer resolution.",
                UserWarning,
                stacklevel=3,
            )

    def check_setup(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
        correct_tokens: torch.Tensor,
        incorrect_tokens: torch.Tensor,
        method: str = "none",
        null_distribution: Optional[torch.Tensor | NullDistribution] = None,
        pooled_null: bool = False,
    ) -> AuditPreflight:
        """Check prompt count, metric gap, and p-value resolution before patching.

        This runs only the clean and corrupted baseline forwards (two model
        passes), not the per-component patching loop. It cannot estimate how
        many components will be zero; those counts are reported after an audit.
        """
        if method not in {"none", "signflip"}:
            raise ValueError("method must be 'none' or 'signflip'.")
        batch_size, seq_len = self._validate_inputs(
            clean_tokens, corrupted_tokens, correct_tokens, incorrect_tokens
        )
        n_layers, n_components = self._get_audit_shape(seq_len)
        family_size = n_layers * n_components
        messages = []

        p_min = None
        effective_method = method
        if isinstance(null_distribution, NullDistribution):
            if null_distribution.n_prompts is not None and null_distribution.n_prompts != batch_size:
                raise ValueError(
                    "NullDistribution.n_prompts must match the audit prompt count "
                    "when it is recorded."
                )
            null_values = null_distribution.values
        else:
            null_values = null_distribution

        if null_values is not None:
            if not isinstance(null_values, torch.Tensor):
                raise TypeError("null_distribution must be a torch.Tensor or NullDistribution.")
            if not torch.is_floating_point(null_values):
                raise TypeError("null_distribution values must be floating-point.")
            valid_pooled = pooled_null and null_values.ndim == 1 and null_values.numel() > 0
            valid_componentwise = (
                null_values.ndim == 3
                and null_values.shape[0] > 0
                and tuple(null_values.shape[1:]) == (n_layers, n_components)
            )
            if not (valid_pooled or valid_componentwise):
                raise ValueError(
                    "null_distribution must have shape [n_null, layers, components] "
                    "or be one-dimensional with pooled_null=True."
                )
            if not torch.isfinite(null_values).all():
                raise ValueError("null_distribution must contain only finite values.")
            effective_method = "empirical_null"
            p_min = 1.0 / (null_values.shape[0] + 1)
        elif method == "signflip":
            p_min = 1.0 / (self.n_perm + 1)
            if batch_size < 2:
                messages.append("Sign-flip testing requires at least two prompt effects.")
        else:
            messages.append(
                "No null method selected: the audit will report descriptive scores only, "
                "with no p-values or FDR mask."
            )

        clean_logits = self.model(clean_tokens)
        corrupted_logits = self.model(corrupted_tokens)
        clean_metric = self._metric_per_prompt(
            clean_logits, correct_tokens, incorrect_tokens, batch_size
        )
        corrupted_metric = self._metric_per_prompt(
            corrupted_logits, correct_tokens, incorrect_tokens, batch_size
        )
        gaps = clean_metric - corrupted_metric
        gap_mean = gaps.mean()
        gap_std = gaps.std(unbiased=True) if batch_size > 1 else None
        metric_scale = max(
            clean_metric.abs().mean().item(),
            corrupted_metric.abs().mean().item(),
            torch.finfo(gaps.dtype).tiny,
        )
        relative_tolerance = 1e-8 * metric_scale
        gap_usable = abs(gap_mean.item()) > relative_tolerance
        if not gap_usable:
            messages.append(
                "The mean metric gap is too small relative to the metric scale; "
                "normalized effects are undefined or unstable."
            )

        resolution = None
        if p_min is not None:
            resolution = resolution_report(
                family_size, self.fdr_threshold, self.fdr_method, p_min
            )
            if resolution["min_rank_for_any_rejection"] is None:
                messages.append(
                    "No rejection is attainable at this p-value resolution for the selected family."
                )
            elif p_min > resolution["first_step_threshold"]:
                messages.append(
                    "A minimum p-value cannot pass alone; at least "
                    f"{resolution['min_rank_for_any_rejection']} tests must meet rank-specific thresholds."
                )

        return AuditPreflight(
            n_prompts=batch_size,
            family_size=family_size,
            method=effective_method,
            gap_mean=float(gap_mean.item()),
            gap_std=None if gap_std is None else float(gap_std.item()),
            gap_usable=gap_usable,
            relative_gap_tolerance=relative_tolerance,
            resolution_report=resolution,
            messages=tuple(messages),
        )

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
        null_distribution: Optional[torch.Tensor | NullDistribution] = None,
        method: str = "none",
        pooled_null: bool = False,
    ) -> AuditResults:
        """Run patching and return exploratory scores or calibrated inference.

        ``null_distribution`` must be shaped ``[n_null, layers, components]``.
        Each entry must be a replicate of the mean normalized effect for the
        same metric, normalization, and component mapping as this audit. A
        ``NullDistribution`` can record that provenance; labels are bookkeeping
        and do not validate the scientific null design. If its ``n_prompts``
        is supplied, it must match this audit's prompt count. A one-dimensional
        pooled null is accepted only with ``pooled_null=True`` and is broadcast
        to each component with a warning. Clean/corrupted tensors must use the
        same sequence shape and padding positions; padding is checked when the
        model exposes a tokenizer pad ID.

        ``method='signflip'`` tests the assumption that per-prompt effects are
        symmetric about zero. With no valid null method, inferential fields are
        ``None`` and only robust exploratory outlier scores are returned.
        """
        if method not in {"none", "signflip"}:
            raise ValueError("method must be 'none' or 'signflip'.")
        batch_size, seq_len = self._validate_inputs(
            clean_tokens, corrupted_tokens, correct_tokens, incorrect_tokens
        )

        null_metadata = None
        null_description_override = None
        null_prompt_count = None
        if isinstance(null_distribution, NullDistribution):
            metadata = null_distribution
            null_distribution = metadata.values
            null_description_override = metadata.description
            null_prompt_count = metadata.n_prompts
            null_metadata = {
                "description": metadata.description,
                "metric_name": metadata.metric_name,
                "normalization": metadata.normalization,
                "hook_type": metadata.hook_type,
                "n_prompts": metadata.n_prompts,
            }
            if null_prompt_count is not None and null_prompt_count != batch_size:
                raise ValueError(
                    "NullDistribution.n_prompts must match the audit prompt count "
                    "when it is recorded."
                )

        n_layers, n_components = self._get_audit_shape(seq_len)
        if null_distribution is not None:
            if not isinstance(null_distribution, torch.Tensor):
                raise TypeError("null_distribution must be a torch.Tensor or NullDistribution.")
            if not torch.is_floating_point(null_distribution):
                raise TypeError("null_distribution values must be floating-point.")
            valid_pooled = pooled_null and null_distribution.ndim == 1 and null_distribution.numel() > 0
            valid_componentwise = (
                null_distribution.ndim == 3
                and null_distribution.shape[0] > 0
                and tuple(null_distribution.shape[1:]) == (n_layers, n_components)
            )
            if not (valid_pooled or valid_componentwise):
                raise ValueError(
                    "null_distribution must have shape [n_null, layers, components] "
                    "or be one-dimensional with pooled_null=True."
                )
            if not torch.isfinite(null_distribution).all():
                raise ValueError("null_distribution must contain only finite values.")

        clean_logits, clean_cache = self.model.run_with_cache(clean_tokens)
        corrupted_logits = self.model(corrupted_tokens)
        clean_metric = self._metric_per_prompt(
            clean_logits, correct_tokens, incorrect_tokens, batch_size
        )
        corrupted_metric = self._metric_per_prompt(
            corrupted_logits, correct_tokens, incorrect_tokens, batch_size
        )
        denominator = self._shared_denominator(clean_metric, corrupted_metric)
        gaps = clean_metric - corrupted_metric
        gap_mean = gaps.mean()
        gap_std = gaps.std(unbiased=True) if batch_size > 1 else None

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
        exact_zero_mask = (per_prompt == 0).all(dim=-1)
        numerical_zero_mask = (
            (per_prompt.abs() <= self.zero_effect_tolerance).all(dim=-1)
            & ~exact_zero_mask
        )
        baseline, scale = compute_robust_baseline(mean_effects.reshape(-1))
        outlier_scores = (mean_effects - baseline) / scale

        calibrated = False
        null_description = "none (robust outlier heuristic; exploratory only)"
        adjusted_effects = mean_effects - baseline
        p_values = q_values = significant_mask = None
        minimum_attainable_p = None

        if null_distribution is not None:
            if not isinstance(null_distribution, torch.Tensor):
                raise TypeError("null_distribution must be a torch.Tensor or NullDistribution.")
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
            if null_distribution.shape[0] == 0:
                raise ValueError("null_distribution must contain at least one null replicate.")
            if not torch.isfinite(null_distribution).all():
                raise ValueError("null_distribution must contain only finite values.")
            null_center = torch.median(null_distribution, dim=0).values
            adjusted_effects = mean_effects - null_center
            p_values = calculate_empirical_p_values(mean_effects, null_distribution)
            calibrated = True
            minimum_attainable_p = 1.0 / (null_distribution.shape[0] + 1)
            if null_description_override is not None:
                null_description = (
                    f"{null_description_override} (pooled null broadcast)"
                    if pooled_null else null_description_override
                )
        elif method == "signflip":
            p_values = signflip_p_values(
                per_prompt,
                n_perm=self.n_perm,
                zero_effect_tolerance=self.zero_effect_tolerance,
            )
            adjusted_effects = mean_effects
            calibrated = True
            minimum_attainable_p = 1.0 / (self.n_perm + 1)
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
            family_size = p_values.numel()
            resolution = resolution_report(
                family_size, self.fdr_threshold, self.fdr_method,
                minimum_attainable_p,
            )
            self._resolution_warning(resolution)
        else:
            family_size = n_layers * n_components
            resolution = None

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
                row["family_size"] = family_size
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
            family_size=family_size,
            resolution_report=resolution,
            minimum_attainable_p=minimum_attainable_p,
            fdr_method=self.fdr_method if calibrated else None,
            fdr_threshold=self.fdr_threshold if calibrated else None,
            clean_corrupted_gap_mean=float(gap_mean.item()),
            clean_corrupted_gap_std=(
                None if gap_std is None else float(gap_std.item())
            ),
            zero_effect_tolerance=self.zero_effect_tolerance,
            exact_zero_components=int(exact_zero_mask.sum().item()),
            numerically_zero_components=int(numerical_zero_mask.sum().item()),
            null_metadata=null_metadata,
            methods_statement=_format_methods_statement(
                calibrated=calibrated,
                null_description=null_description,
                fdr_method=self.fdr_method if calibrated else None,
                alpha=self.fdr_threshold if calibrated else None,
                family_size=family_size,
            ),
        )

    def run_joint_patching(
        self,
        clean_tokens,
        corrupted_tokens,
        correct_tokens,
        incorrect_tokens,
        patching_mask,
    ) -> float:
        """Patch selected components in one forward pass and return their effect.

        For an unbiased evaluation of a mask selected by an audit, pass a
        disjoint held-out prompt set here. The method cannot infer which
        prompts were used to choose the mask, so the caller must keep the
        selection and evaluation sets separate. This is denoising patching,
        not a knockout or a stand-alone causal validation.
        """
        self._validate_inputs(
            clean_tokens, corrupted_tokens, correct_tokens, incorrect_tokens
        )
        n_layers, n_components = self._get_audit_shape(corrupted_tokens.shape[1])
        if not isinstance(patching_mask, torch.Tensor):
            raise TypeError("patching_mask must be a boolean torch.Tensor.")
        if patching_mask.dtype != torch.bool:
            raise TypeError("patching_mask must use torch.bool dtype.")
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

    def run_heldout_validation(
        self,
        selection_inputs: dict,
        evaluation_inputs: dict,
        selection_group_ids,
        evaluation_group_ids,
    ) -> HeldOutResults:
        """Select components on one split and evaluate them on a disjoint split.

        ``selection_group_ids`` and ``evaluation_group_ids`` should identify
        independent units (for example, prompt templates), not merely rows if
        several rows share a template. The selection inputs may include a null
        method or empirical null. The held-out audit is descriptive and its
        CLI is calculated on the evaluation prompts; the joint effect uses the
        selection mask on those same held-out prompts.
        """
        selection_ids = tuple(selection_group_ids)
        evaluation_ids = tuple(evaluation_group_ids)
        if not selection_ids or not evaluation_ids:
            raise ValueError("Both selection and evaluation group IDs are required.")
        try:
            overlap = set(selection_ids).intersection(evaluation_ids)
        except TypeError as error:
            raise TypeError("Prompt group IDs must be hashable.") from error
        if overlap:
            raise ValueError("Selection and evaluation groups must be disjoint.")

        selection_kwargs = dict(selection_inputs)
        evaluation_kwargs = dict(evaluation_inputs)
        required = {"clean_tokens", "corrupted_tokens", "correct_tokens", "incorrect_tokens"}
        if not required.issubset(selection_kwargs) or not required.issubset(evaluation_kwargs):
            raise ValueError("Each split must provide clean/corrupted tokens and correct/incorrect labels.")
        if selection_kwargs["clean_tokens"].shape[0] != len(selection_ids):
            raise ValueError("selection_group_ids must match the selection batch size.")
        if evaluation_kwargs["clean_tokens"].shape[0] != len(evaluation_ids):
            raise ValueError("evaluation_group_ids must match the evaluation batch size.")

        evaluation_kwargs.pop("null_distribution", None)
        evaluation_kwargs.pop("method", None)
        evaluation_kwargs.pop("pooled_null", None)
        selected = self.run_patching_audit(**selection_kwargs)
        if selected.significant_mask is None:
            raise ValueError(
                "Held-out joint validation requires calibrated selection results "
                "with a significant_mask."
            )
        evaluated = self.run_patching_audit(**evaluation_kwargs)
        joint_effect = self.run_joint_patching(
            **{key: evaluation_kwargs[key] for key in required},
            patching_mask=selected.significant_mask,
        )
        return HeldOutResults(
            selection_audit=selected,
            evaluation_audit=evaluated,
            joint_effect=joint_effect,
            selection_group_ids=selection_ids,
            evaluation_group_ids=evaluation_ids,
        )

    def run_head_patching_audit(self, *args, **kwargs) -> AuditResults:
        """Backward-compatible alias for :meth:`run_patching_audit`."""
        return self.run_patching_audit(*args, **kwargs)
