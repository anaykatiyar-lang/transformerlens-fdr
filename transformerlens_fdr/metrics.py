import torch

def logit_difference(
    logits: torch.Tensor,
    correct_tokens: torch.Tensor,
    incorrect_tokens: torch.Tensor
) -> torch.Tensor:
    """
    Computes Logit Difference (ΔL) = Logit(correct) - Logit(incorrect).
    Assuming logits has shape [batch, seq_len, d_vocab] or [batch, d_vocab]
    """
    if logits.ndim == 3:
        # Assuming last token is the one we care about
        logits = logits[:, -1, :]
    
    batch_indices = torch.arange(logits.size(0))
    correct_logits = logits[batch_indices, correct_tokens]
    incorrect_logits = logits[batch_indices, incorrect_tokens]
    
    return (correct_logits - incorrect_logits).mean()

def normalized_patching_effect(
    patched_effect: torch.Tensor,
    clean_effect: torch.Tensor,
    corrupted_effect: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Computes Normalized Patching Effect (P).
    """
    denominator = clean_effect - corrupted_effect
    if torch.abs(denominator) < epsilon:
        raise ValueError(
            f"Undefined normalized effect: clean and corrupted metrics are too close "
            f"(diff={denominator.item():.2e}). The null design might be flawed or task is trivial."
        )
    return (patched_effect - corrupted_effect) / denominator

def circuit_localization_index(
    normalized_effects: torch.Tensor,
    epsilon: float = 1e-8
) -> float:
    """
    Computes Circuit Localization Index (CLI).
    """
    # p(l, h) = |P(l, h)| / sum(|P(l', h')| + eps)
    abs_effects = torch.abs(normalized_effects)
    sum_effects = torch.sum(abs_effects)
    
    if sum_effects < epsilon:
        # No signal at all, return 0.0 (completely un-localized)
        return 0.0
        
    p = abs_effects / (sum_effects + epsilon)
    
    # H_causal = -sum(p * log2(p + eps))
    h_causal = -torch.sum(p * torch.log2(p + epsilon))
    
    # CLI = 1 - H_causal / log2(N)
    N = normalized_effects.numel()
    if N <= 1:
        return 1.0
    
    cli = 1.0 - (h_causal / torch.log2(torch.tensor(float(N))))
    
    return cli.item()
