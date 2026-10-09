import torch
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from typing import Tuple

def compute_control_baseline(
    control_samples: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes mean and std of control baseline samples.
    """
    mu_ctrl = torch.mean(control_samples)
    sigma_ctrl = torch.std(control_samples)
    return mu_ctrl, sigma_ctrl

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
