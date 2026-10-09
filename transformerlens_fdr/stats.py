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
    
    if not torch.isfinite(sigma) or sigma < epsilon:
        raise ValueError(
            f"Invalid robust baseline variance (sigma={sigma.item():.2e}). "
            "Distribution of effects is entirely flat."
        )
        
    return median, sigma

def calculate_p_values(
    patching_effects: torch.Tensor,
    mu_ctrl: torch.Tensor,
    sigma_ctrl: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Calculates two-tailed p-values using Z-scores against the control distribution.
    """
    z_scores = (patching_effects - mu_ctrl) / (sigma_ctrl + epsilon)
    
    # Normal distribution standard:
    normal = torch.distributions.Normal(0, 1)
    # p-value = 2 * (1 - CDF(|Z|))
    p_values = 2 * (1 - normal.cdf(torch.abs(z_scores)))
    return p_values

def calculate_empirical_p_values(
    patching_effects: torch.Tensor,
    null_distribution: torch.Tensor
) -> torch.Tensor:
    """
    Calculates two-tailed empirical p-values by comparing patching effects
    against a provided empirical null distribution of effects.
    """
    if null_distribution.numel() == 0:
        raise ValueError("Provided null_distribution is empty.")
        
    abs_effects = torch.abs(patching_effects).unsqueeze(-1)  # [..., 1]
    abs_null = torch.abs(null_distribution)                  # [N_null]
    
    # p-value is the proportion of null trials that exceeded the observed effect
    # Add +1 pseudo-count (standard in permutation tests)
    exceedances = (abs_null >= abs_effects).float().sum(dim=-1)
    p_values = (exceedances + 1.0) / (null_distribution.numel() + 1.0)
    return p_values

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
    p_np = p_values.detach().cpu().numpy().flatten()
    
    reject, pvals_corrected, _, _ = multipletests(p_np, alpha=alpha, method=method)
    
    q_values = torch.tensor(pvals_corrected, dtype=p_values.dtype, device=p_values.device).reshape(p_values.shape)
    passed_fdr = torch.tensor(reject, dtype=torch.bool, device=p_values.device).reshape(p_values.shape)
    
    return q_values, passed_fdr
