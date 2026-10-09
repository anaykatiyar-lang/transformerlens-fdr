import torch
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from typing import Tuple

def compute_control_baseline(
    control_samples: torch.Tensor,
    epsilon: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes mean and std of control baseline samples.
    """
    if control_samples.numel() < 2:
        raise ValueError("Insufficient control samples to estimate baseline variance. Need at least 2.")
        
    mu_ctrl = torch.mean(control_samples)
    sigma_ctrl = torch.std(control_samples)
    
    if not torch.isfinite(sigma_ctrl) or sigma_ctrl < epsilon:
        raise ValueError(
            f"Invalid control baseline variance (sigma={sigma_ctrl.item():.2e}). "
            "Ensure control samples have actual variation."
        )
        
    return mu_ctrl, sigma_ctrl

def compute_robust_baseline(
    effects: torch.Tensor,
    epsilon: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a robust baseline (Median and Median Absolute Deviation) 
    from the full distribution of effects, assuming true signals are sparse outliers.
    """
    if effects.numel() < 2:
        raise ValueError("Insufficient components to estimate baseline variance. Need at least 2.")
        
    median = torch.median(effects)
    mad = torch.median(torch.abs(effects - median))
    # Convert MAD to standard deviation equivalent for normal distribution
    sigma = mad * 1.4826
    
    if not torch.isfinite(sigma):
        raise ValueError("Robust baseline scale must be finite.")

    # A flat effect map has no outliers; return a finite scale so exploratory
    # scores remain available (and equal zero) instead of aborting the audit.
    return median, sigma.clamp_min(epsilon)

def calculate_p_values(
    patching_effects: torch.Tensor,
    mu_ctrl: torch.Tensor,
    sigma_ctrl: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Calculates two-tailed p-values using Z-scores against the control distribution.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if mu_ctrl.numel() != 1 or sigma_ctrl.numel() != 1:
        raise ValueError("Control mean and standard deviation must be scalars.")
    if not torch.isfinite(mu_ctrl).all() or not torch.isfinite(sigma_ctrl).all():
        raise ValueError("Control mean and standard deviation must be finite.")
    if not torch.isfinite(patching_effects).all():
        raise ValueError("Patching effects must be finite.")
    if sigma_ctrl.item() <= epsilon:
        raise ValueError(
            "Control standard deviation is zero or too small to estimate "
            "meaningful p-values. Collect more varied control samples."
        )

    z_scores = (patching_effects - mu_ctrl) / (sigma_ctrl + epsilon)

    # erfc computes the upper tail directly and avoids cancellation in
    # 2 * (1 - CDF(|z|)) for large z-scores.
    p_values = torch.special.erfc(torch.abs(z_scores) / (2.0 ** 0.5))
    # Floating-point tail underflow must not be reported as an exact p-value 0.
    return p_values.clamp_min(torch.finfo(p_values.dtype).tiny)

def calculate_empirical_p_values(
    patching_effects: torch.Tensor,
    null_distribution: torch.Tensor,
    two_sided: bool = True,
) -> torch.Tensor:
    """
    Calculates two-tailed empirical p-values by comparing patching effects
    against a provided empirical null distribution of effects.
    """
    if null_distribution.ndim != 3 or null_distribution.shape[0] == 0:
        raise ValueError("null_distribution must have shape [n_null, layers, components].")
    if patching_effects.ndim != 2 or null_distribution.shape[1:] != patching_effects.shape:
        raise ValueError("Null distribution component dimensions must match patching_effects.")
    if not torch.isfinite(null_distribution).all() or not torch.isfinite(patching_effects).all():
        raise ValueError("Observed effects and null distribution must be finite.")

    null_center = torch.median(null_distribution, dim=0).values
    observed_delta = patching_effects - null_center
    null_delta = null_distribution - null_center.unsqueeze(0)
    if two_sided:
        observed_stat = observed_delta.abs().unsqueeze(0)
        null_stat = null_delta.abs()
    else:
        observed_stat = observed_delta.unsqueeze(0)
        null_stat = null_delta

    exceedances = (null_stat >= observed_stat).sum(dim=0)
    # Plus-one correction keeps Monte Carlo/permutation p-values above zero.
    p_values = (exceedances + 1).to(patching_effects.dtype) / (null_distribution.shape[0] + 1)
    return p_values


def signflip_p_values(
    effects: torch.Tensor,
    n_perm: int = 9999,
    two_sided: bool = True,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Paired sign-flip p-values for effects shaped [layers, components, prompts].

    The null hypothesis is that independent/exchangeable prompt effects are
    symmetric about zero. The same sign pattern is applied to every component
    to preserve their dependence structure under the joint null.
    """
    if effects.ndim != 3:
        raise ValueError("effects must have shape [layers, components, prompts].")
    if n_perm < 1:
        raise ValueError("n_perm must be positive.")
    if effects.shape[-1] < 2:
        raise ValueError("Sign-flip testing requires at least two prompt effects.")
    if not torch.isfinite(effects).all():
        raise ValueError("effects must be finite.")

    n_layers, n_components, n_prompts = effects.shape
    x = effects.reshape(n_layers * n_components, n_prompts)

    def studentized(values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=-1)
        std = values.std(dim=-1, unbiased=True).clamp_min(1e-12)
        return mean / (std / (n_prompts ** 0.5))

    observed = studentized(x)
    # Generate permutations in chunks to limit memory while sharing each
    # prompt-sign pattern across all tested components.
    exceedances = torch.zeros_like(observed, dtype=torch.long)
    chunk_size = min(256, n_perm)
    generator_device = generator.device if generator is not None else effects.device
    for start in range(0, n_perm, chunk_size):
        count = min(chunk_size, n_perm - start)
        signs = torch.randint(
            0, 2, (count, n_prompts), generator=generator,
            device=generator_device,
        ).mul_(2).sub_(1).to(device=effects.device, dtype=effects.dtype)
        null_stats = studentized(signs[:, None, :] * x[None, :, :])
        if two_sided:
            exceedances += (null_stats.abs() >= observed.abs()).sum(dim=0)
        else:
            exceedances += (null_stats >= observed).sum(dim=0)

    p_values = (exceedances + 1).to(effects.dtype) / (n_perm + 1)
    return p_values.reshape(n_layers, n_components)

def apply_fdr_adjustment(
    p_values: torch.Tensor,
    alpha: float = 0.05,
    method: str = 'fdr_bh'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies FDR adjustment using statsmodels.
    method can be 'fdr_bh' (Benjamini-Hochberg) or 'fdr_by' (Benjamini-Yekutieli).
    Returns:
        q_values: The adjusted p-values.
        passed_fdr: Boolean mask of components that passed.
    """
    if not isinstance(p_values, torch.Tensor) or not torch.is_floating_point(p_values):
        raise TypeError("p_values must be a floating-point torch.Tensor.")
    if p_values.numel() == 0:
        raise ValueError("p_values must contain at least one value.")
    if not torch.isfinite(p_values).all():
        raise ValueError("p_values must contain only finite values.")
    if ((p_values < 0) | (p_values > 1)).any():
        raise ValueError("p_values must be between 0 and 1 inclusive.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")
    if method not in {"fdr_bh", "fdr_by"}:
        raise ValueError("method must be 'fdr_bh' or 'fdr_by'.")

    p_np = p_values.detach().cpu().numpy().flatten()
    
    reject, pvals_corrected, _, _ = multipletests(p_np, alpha=alpha, method=method)
    
    q_values = torch.tensor(pvals_corrected, dtype=p_values.dtype, device=p_values.device).reshape(p_values.shape)
    passed_fdr = torch.tensor(reject, dtype=torch.bool, device=p_values.device).reshape(p_values.shape)
    
    return q_values, passed_fdr
