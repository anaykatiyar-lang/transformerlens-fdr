# TransformerLens-FDR 📊

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TransformerLens v4.0.0+](https://img.shields.io/badge/transformer__lens-4.0.0+-orange.svg)](https://github.com/TransformerLensOrg/TransformerLens)

**Statistical Audit Add-on for TransformerLens.**

This library wraps standard `TransformerLens` activation patching workflows to enforce rigorous statistical auditing. It prevents "false-positive circuit discoveries" (where random model noise is mistaken for meaningful algorithmic structure) by applying false discovery rate (FDR) corrections.

---

## 🚀 Why Use FDR?

Standard activation patching measures a normalized effect, often calling any score $> 0.05$ a "component of the circuit". But when evaluating 144 attention heads, statistical noise guarantees false positives! 

**Our pipeline fixes this by:**
1. Running activation patching across components.
2. Sampling a "null distribution" using random control noise.
3. Calculating $Z$-scores and $p$-values against the noise.
4. Applying **Benjamini-Hochberg** (or **Benjamini-Yekutieli**) FDR adjustments.
5. Computing a holistic **Circuit Localization Index (CLI)** to quantify sparsity.

---

## 🧮 The Mathematics

### 1. Normalized Patching Effect ($P$)
How much does patching a clean activation recover the clean metric?

$$ P = \frac{M_{\text{patched}} - M_{\text{corrupted}}}{M_{\text{clean}} - M_{\text{corrupted}}} $$

### 2. $Z$-Score Calculation
We estimate control noise parameters $\mu_{\text{ctrl}}$ and $\sigma_{\text{ctrl}}$ from a random sample of control components.

$$ Z = \frac{P - \mu_{\text{ctrl}}}{\sigma_{\text{ctrl}} + \epsilon} $$
$$ p = 2 \cdot (1 - \Phi(|Z|)) $$

*(where $\Phi$ is the standard normal cumulative distribution function)*

### 3. Circuit Localization Index ($\text{CLI}$)
Quantifies how sparse/localized a circuit is based on causal entropy ($H_{\text{causal}}$). A score near $1.0$ means extreme localization (a tight bottleneck). A score near $0.0$ means distributed representation.

$$ p_i = \frac{|P_i|}{\sum_j |P_j| + \epsilon} $$
$$ H_{\text{causal}} = - \sum p_i \log_2 (p_i + \epsilon) $$
$$ \text{CLI} = 1 - \frac{H_{\text{causal}}}{\log_2(N)} $$

---

## 🛠️ Plain & Simple Function List

The API is intentionally simple. You will mostly interact with `PatchingAuditor`.

### `PatchingAuditor` (Class)
The main controller for your statistical audit. 
- **`__init__(model, metric_fn, fdr_threshold=0.05, fdr_method='fdr_bh', hook_type='attn_head')`**  
  Creates the auditor. `fdr_method` can be `'fdr_bh'` (standard) or `'fdr_by'` (Benjamini-Yekutieli, safer for highly correlated attention heads).
- **`run_patching_audit(...)`**  
  Automatically patches all components of `hook_type`, computes stats, and applies FDR. Returns an `AuditResults` object.
- **`run_joint_patching(..., patching_mask)`**  
  Performs a "Joint Knockout". Patches multiple components simultaneously (e.g. all heads that passed the FDR check) to see their combined causal effect ($P_{\text{joint}}$).

### `AuditResults` (Dataclass)
Returned by the auditor. Contains:
- `raw_patching_effects`: The raw $P$ scores tensor.
- `adjusted_effects`: $P$ scores minus the baseline noise $\mu$.
- `p_values` / `q_values`: Tensors of statistical significance.
- `passed_fdr_mask`: A boolean tensor masking out false positives.
- `cli_score`: The Circuit Localization Index float.
- `summary_df`: A neat Pandas DataFrame of all results.

### Metrics & Stats (Functions)
- **`logit_difference(logits, correct_tokens, incorrect_tokens)`**  
  A standard interpretability metric calculating the difference between the correct and incorrect prediction logits.
- **`normalized_patching_effect(patched, clean, corrupted)`**  
  Calculates $P$.
- **`circuit_localization_index(normalized_effects)`**  
  Calculates the CLI.
- **`apply_fdr_adjustment(p_values, alpha, method)`**  
  Runs the statsmodels FDR logic.

### Visualization (Functions)
- **`plot_fdr_heatmap(results, title)`**  
  Renders a Plotly heatmap that visually masks (hides) all components that failed the FDR check, leaving only true causal mechanisms visible.

---

## 💻 Quick Start

To install and run a full GPT-2 IOI audit script:

```bash
# Set up environment
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Install editable with tests
pip install -e .[test]

# Run the provided example script!
python example_usage.py
```
