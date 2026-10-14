import numpy as np
import torch
from statsmodels.stats.multitest import multipletests
from typing import Tuple
import math

def compute_control_baseline(
    control_samples: torch.Tensor,
    epsilon: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the mean and sample standard deviation of control values.

    These moments alone do not establish a valid null model. They support the
    normal-null route only when the control values are representative
    and approximately normal; callers should prefer a validated empirical or
    sign-flip null when those assumptions are not defensible.
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
    """Compute a median/MAD baseline for descriptive outlier scores.

    This is a robust descriptive heuristic. It is not a Gaussian null, does
    not produce inferential p-values, and assumes signals are sparse relative
    to the full set of effects. The scaled MAD is floored at 10% of the
    population standard deviation; a flat map's scale is floored at
    ``epsilon``. This remains descriptive only.
    """
    median, scale, _ = _compute_robust_baseline_details(effects, epsilon)
    return median, scale


def _compute_robust_baseline_details(effects, epsilon=1e-8):
    """Internal baseline calculation with a reportable scale-method label."""
    if effects.numel() < 2:
        raise ValueError("Insufficient components to estimate baseline variance. Need at least 2.")
        
    median = torch.median(effects)
    mad = torch.median(torch.abs(effects - median))
    # Convert MAD to standard deviation equivalent for normal distribution
    sigma = mad * 1.4826
    
    if not torch.isfinite(sigma):
        raise ValueError("Robust baseline scale must be finite.")

    scale_method = "scaled MAD"
    relative_floor = 0.1 * effects.std(unbiased=False)
    if relative_floor > sigma:
        sigma = relative_floor
        scale_method = "scaled MAD with 0.1 population-standard-deviation floor"
    if sigma <= epsilon:
        scale_method += " with epsilon floor"
    return median, sigma.clamp_min(epsilon), scale_method

def calculate_p_values(
    patching_effects: torch.Tensor,
    mu_ctrl: torch.Tensor,
    sigma_ctrl: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """Calculate two-sided normal-approximation p-values from control moments.

    This normal-null option assumes a representative control sample and a normal
    null with the supplied mean and standard deviation. It remains available
    for callers whose analysis explicitly uses those assumptions. The robust
    descriptive baseline does not make this Gaussian inference valid.
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
    result_dtype = torch.float32 if patching_effects.device.type == "mps" else torch.float64
    p_values = (exceedances + 1).to(result_dtype) / (null_distribution.shape[0] + 1)
    return p_values


def signflip_p_values(
    effects: torch.Tensor,
    n_perm: int = 9999,
    two_sided: bool = True,
    generator: torch.Generator = None,
    zero_effect_tolerance: float = 1e-8,
) -> torch.Tensor:
    """Paired sign-flip p-values for effects shaped [layers, components, prompts].

    The null hypothesis is that prompt effects are independent and each is
    symmetric about zero. The same sign pattern is applied to every component
    to preserve their dependence structure under the joint null. Values whose
    absolute magnitude is at most ``zero_effect_tolerance`` are set to zero
    before computing the statistic; use a tolerance in normalized-effect units.
    The routine enumerates every unique sign pattern when its count is no
    larger than ``n_perm``; otherwise it uses the requested Monte Carlo draws.
    """
    if effects.ndim != 3:
        raise ValueError("effects must have shape [layers, components, prompts].")
    if not isinstance(n_perm, int) or n_perm < 1:
        raise ValueError("n_perm must be positive.")
    if not math.isfinite(zero_effect_tolerance) or zero_effect_tolerance < 0:
        raise ValueError("zero_effect_tolerance must be finite and non-negative.")
    if effects.shape[-1] < 2:
        raise ValueError("Sign-flip testing requires at least two prompt effects.")
    if not torch.isfinite(effects).all():
        raise ValueError("effects must be finite.")

    n_layers, n_components, n_prompts = effects.shape
    result_dtype = torch.float32 if effects.device.type == "mps" else torch.float64
    x = effects.reshape(n_layers * n_components, n_prompts).to(result_dtype).clone()
    x[x.abs() <= zero_effect_tolerance] = 0

    def studentized(values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=-1)
        std = values.std(dim=-1, unbiased=True).clamp_min(1e-12)
        return mean / (std / (n_prompts ** 0.5))

    observed = studentized(x)
    # Enumerate the unique sign patterns when feasible. For a two-sided test,
    # patterns s and -s give the same absolute statistic, so fix the first
    # sign to +1 and enumerate 2**(B-1) patterns.
    exponent = n_prompts - 1 if two_sided else n_prompts
    exact_count = (1 << exponent) if exponent <= 62 else None
    exact = exact_count is not None and exact_count <= n_perm
    total = exact_count if exact else n_perm
    exceedances = torch.zeros_like(observed, dtype=torch.long)
    chunk_size = min(256, total)
    generator_device = generator.device if generator is not None else effects.device
    tie_tolerance = (
        16 * torch.finfo(result_dtype).eps
        * observed.abs().clamp_min(1.0)
    )
    for start in range(0, total, chunk_size):
        count = min(chunk_size, total - start)
        if exact:
            indices = torch.arange(start, start + count, device=effects.device)
            bit_positions = torch.arange(exponent, device=effects.device)
            signs = (((indices[:, None] >> bit_positions) & 1) * 2 - 1).to(result_dtype)
            if two_sided:
                signs = torch.cat((torch.ones((count, 1), device=effects.device, dtype=result_dtype), signs), dim=1)
        else:
            signs = torch.randint(
                0, 2, (count, n_prompts), generator=generator,
                device=generator_device,
            ).mul_(2).sub_(1).to(device=effects.device, dtype=result_dtype)
        # Sign patterns preserve sum(x**2), so calculate studentized means
        # from the matrix product rather than allocating [permutations, tests,
        # prompts]. This keeps memory proportional to [permutations, tests].
        signed_sums = signs @ x.T
        sum_squares = (x * x).sum(dim=-1)
        variance_numerator = (n_prompts * sum_squares.unsqueeze(0) - signed_sums.square()).clamp_min(0)
        denominator = variance_numerator.sqrt().clamp_min(
            1e-12 * math.sqrt(n_prompts * (n_prompts - 1))
        )
        null_stats = signed_sums * math.sqrt(n_prompts - 1) / denominator
        if exact and start <= total - 1 < start + count:
            # The all-positive pattern is the observed data. Set it to the
            # separately computed observed statistic so cancellation in the
            # variance identity cannot lose this guaranteed exceedance.
            null_stats[total - 1 - start] = observed
        if two_sided:
            exceedances += (
                null_stats.abs() >= observed.abs() - tie_tolerance
            ).sum(dim=0)
        else:
            exceedances += (null_stats >= observed - tie_tolerance).sum(dim=0)

    p_values = (
        exceedances.to(result_dtype) / total if exact
        else (exceedances + 1).to(result_dtype) / (n_perm + 1)
    )
    return p_values.reshape(n_layers, n_components)


def signflip_permutation_info(
    n_prompts: int, n_perm: int, two_sided: bool = True
) -> dict:
    """Describe whether the sign-flip calculation enumerates or samples."""
    if not isinstance(n_prompts, int) or n_prompts < 2:
        raise ValueError("n_prompts must be an integer of at least 2.")
    if not isinstance(n_perm, int) or n_perm < 1:
        raise ValueError("n_perm must be a positive integer.")
    exponent = n_prompts - 1 if two_sided else n_prompts
    pattern_count = (1 << exponent) if exponent <= 62 else None
    if pattern_count is not None and pattern_count <= n_perm:
        return {
            "mode": "exact",
            "n_patterns": pattern_count,
            "minimum_p": 1.0 / pattern_count,
        }
    return {
        "mode": "monte_carlo",
        "n_patterns": n_perm,
        "minimum_p": 1.0 / (n_perm + 1),
    }


def signflip_minimum_p(n_prompts: int, n_perm: int, two_sided: bool = True) -> float:
    """Return the attainable p-value floor for the chosen sign-flip design."""
    return signflip_permutation_info(n_prompts, n_perm, two_sided)["minimum_p"]

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

    # Move to CPU before float64 conversion (MPS has no float64 support).
    p_np = p_values.detach().cpu().to(torch.float64).numpy().flatten()

    _, pvals_corrected, _, _ = multipletests(p_np, alpha=alpha, method=method)
    order = p_np.argsort(kind="mergesort")
    sorted_p = p_np[order]
    n_tests = len(p_np)
    dependence_factor = (
        1.0 if method == "fdr_bh"
        else sum(1.0 / rank for rank in range(1, n_tests + 1))
    )
    thresholds = alpha * np.arange(1, n_tests + 1, dtype=np.float64) / (n_tests * dependence_factor)
    # Equality is part of the step-up rule. The tiny relative allowance only
    # absorbs floating-point evaluation error at a mathematically exact tie.
    passing = sorted_p <= thresholds * (1.0 + 1e-12)
    reject = np.zeros(n_tests, dtype=bool)
    if passing.any():
        reject[order[:int(passing.nonzero()[0][-1]) + 1]] = True
    # Keep the adjusted-value and rejection-mask interfaces consistent at a
    # floating-point boundary accepted by the tiny step-up tolerance.
    pvals_corrected[reject] = np.minimum(pvals_corrected[reject], alpha)
    
    q_dtype = torch.float32 if p_values.device.type == "mps" else torch.float64
    q_values = torch.tensor(pvals_corrected, dtype=q_dtype, device=p_values.device).reshape(p_values.shape)
    passed_fdr = torch.tensor(reject, dtype=torch.bool, device=p_values.device).reshape(p_values.shape)
    
    return q_values, passed_fdr


def resolution_report(
    n_tests: int,
    alpha: float,
    method: str,
    p_min: float,
) -> dict:
    """Describe the rank needed for a discrete p-value to cross BH/BY.

    ``n_tests`` is the number of p-values in the correction family, not a
    user-supplied estimate. ``min_rank_for_any_rejection`` is a necessary
    rank condition; it is not sufficient unless the ordered p-values meet
    the step-up threshold at that rank.
    """
    if not isinstance(n_tests, int) or n_tests < 1:
        raise ValueError("n_tests must be a positive integer.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")
    if method not in {"fdr_bh", "fdr_by"}:
        raise ValueError("method must be 'fdr_bh' or 'fdr_by'.")
    if not math.isfinite(p_min) or not 0 < p_min <= 1:
        raise ValueError("p_min must be finite and in (0, 1].")

    dependence_factor = (
        1.0 if method == "fdr_bh"
        else sum(1.0 / rank for rank in range(1, n_tests + 1))
    )
    first_threshold = alpha / (n_tests * dependence_factor)
    min_rank = max(1, math.ceil(p_min * n_tests * dependence_factor / alpha))
    attainable_rank = min_rank if min_rank <= n_tests else None
    return {
        "n_tests": n_tests,
        "alpha": alpha,
        "method": method,
        "p_min": p_min,
        "first_step_threshold": first_threshold,
        "min_rank_for_any_rejection": attainable_rank,
        "status": "attainable" if attainable_rank is not None else "no attainable rejection",
    }
