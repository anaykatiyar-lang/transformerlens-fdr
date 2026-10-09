import plotly.express as px
import numpy as np
from .auditor import AuditResults

def plot_fdr_heatmap(results: AuditResults, title: str = "Activation Patching (FDR Masked)"):
    """
    Renders a heatmap of adjusted effects, highlighting only those that passed FDR.
    """
    adj_effects = results.adjusted_effects.detach().cpu().numpy()
    passed_mask = results.passed_fdr_mask.detach().cpu().numpy()
    
    # Mask out components that didn't pass FDR
    masked_effects = np.where(passed_mask, adj_effects, 0.0)
    
    fig = px.imshow(
        masked_effects,
        labels=dict(x="Head", y="Layer", color="Adjusted Effect"),
        title=title,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0
    )
    
    return fig
