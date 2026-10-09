import plotly.express as px
import numpy as np
from .auditor import AuditResults

def plot_fdr_heatmap(results: AuditResults, title: str = "Activation Patching (FDR Masked)"):
    """
    Render calibrated significant effects. Exploratory results have no mask and
    cannot be presented as FDR-filtered discoveries.
    """
    if not results.calibrated or results.significant_mask is None:
        raise ValueError(
            "No calibrated significance mask is available. Supply a valid null "
            "or run with method='signflip' before plotting an FDR heatmap."
        )
    adj_effects = results.adjusted_effects.detach().cpu().numpy()
    passed_mask = results.significant_mask.detach().cpu().numpy()
    
    # Mask out components that didn't pass FDR
    masked_effects = np.where(passed_mask, adj_effects, 0.0)
    
    x_label = {
        "attn_head": "Head",
        "resid_mid": "Position",
        "mlp_out": "Component",
    }.get(results.hook_type, "Component")
    fig = px.imshow(
        masked_effects,
        labels=dict(x=x_label, y="Layer", color="Adjusted Effect"),
        title=title,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0
    )
    
    return fig
