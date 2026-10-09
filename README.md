# TransformerLens-FDR 📊

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TransformerLens v4.0.0+](https://img.shields.io/badge/transformer__lens-4.0.0+-orange.svg)](https://github.com/TransformerLensOrg/TransformerLens)

**Activation-patching analysis utilities for TransformerLens.** I built this package to make component patching effects easier to inspect and to provide an explicit, assumption-aware path from those effects to multiple-testing results.

The package computes per-prompt patching effects and descriptive robust outlier scores. It returns FDR-adjusted significance results only when a caller supplies an empirical null or explicitly requests a sign-flip test. Those results are valid only when the null assumptions are appropriate and have been checked for the task.

---

## 🚀 Why Use FDR?

Standard activation patching measures a normalized effect. I do not treat a fixed effect-size cutoff as proof that a component belongs to a circuit. When many components are tested, false discoveries need to be considered explicitly.

The default audit does not invent a null from the observed components. It reports exploratory outlier scores and leaves p-values, q-values, and the significance mask unset. For inference, use a per-component empirical null or request sign-flip testing over independent prompt pairs.

---

## 🧮 The Mathematics

### 1. Normalized Patching Effect ($P$)
How much does patching a clean activation recover the clean metric?

$$ P = \frac{M_{\text{patched}} - M_{\text{corrupted}}}{M_{\text{clean}} - M_{\text{corrupted}}} $$

### 2. Inference and its assumptions
`method="signflip"` uses per-prompt effects and requires independent/exchangeable prompt effects symmetric about zero under the null. Sign-flip validity depends on that assumption; the calibration harness includes symmetric null tests and separate skew diagnostics. Alternatively, pass a caller-generated null with shape `[n_null, layers, components]` so each component is compared to its own null distribution. A pooled null is available only by explicit opt-in and emits a warning.

Benjamini-Hochberg and Benjamini-Yekutieli adjust valid p-values for the complete family of tests being reported. BH assumes independent or positively dependent tests; BY handles arbitrary dependence but is more conservative. Neither procedure fixes invalid p-values or a misspecified null. The smallest sign-flip p-value is `1 / (n_perm + 1)`. Choose enough permutations to resolve the correction threshold for the number of tests; for 144 tests at alpha 0.05, 3,000 permutations resolve the first BH threshold, while BY requires more.

For an empirical null, build the null from independent, task-appropriate negative controls and evaluate it on held-out controls. The null controls must represent the no-effect condition for the analysis. A shuffled or unrelated pair is useful only when that shuffle actually breaks the relationship being tested. Do not reuse the same controls to both estimate the null and claim its calibration.

### 3. Circuit Localization Index ($\text{CLI}$)
Describes how concentrated the absolute point estimates are. It is descriptive, not a significance test or evidence that a circuit is causal. A score near $1.0$ means concentrated effects; a score near $0.0$ means diffuse effects.

$$ p_i = \frac{|P_i|}{\sum_j |P_j| + \epsilon} $$
$$ H_{\text{causal}} = - \sum p_i \log_2 (p_i + \epsilon) $$
$$ \text{CLI} = 1 - \frac{H_{\text{causal}}}{\log_2(N)} $$

---

## 🛠️ Plain & Simple Function List

The API is intentionally simple. You will mostly interact with `PatchingAuditor`.

### `PatchingAuditor` (Class)
The main controller for your statistical audit. 
- **`__init__(model, metric_fn, fdr_threshold=0.05, fdr_method='fdr_bh', n_perm=9999, hook_type='attn_head')`**
  `metric_fn` must return one metric value per prompt (`[B]`). BH requires valid p-values and independence or positive regression dependence; BY handles arbitrary dependence but still requires valid p-values.
- **`run_patching_audit(...)`**  
  Patches all components. Pass `method="signflip"` to use the symmetry-about-zero sign-flip test, or provide a per-component `null_distribution`. With neither, results are exploratory and inferential fields are `None`.
- **`run_joint_patching(..., patching_mask)`**  
  Patches selected components simultaneously and reports their joint normalized effect.

### `AuditResults` (Dataclass)
Returned by the auditor. Contains:
- `raw_patching_effects`: Mean point estimate per component `[layers, components]`.
- `per_prompt_effects`: Per-prompt effects `[layers, components, prompts]`.
- `adjusted_effects`: Point estimates centered by the same null used for inference; robust-centered when exploratory.
- `p_values` / `q_values`: Optional tensors, present only when a null method was requested.
- `significant_mask`: Optional boolean mask, present only when p-values were produced.
- `outlier_scores`: Robust descriptive scores; not p-values.
- `calibrated` / `null_description`: State whether inference was attempted and which null assumption was used.
- `cli_score`: The Circuit Localization Index float.
- `summary_df`: A neat Pandas DataFrame of all results.

### Metrics & Stats (Functions)
- **`logit_difference(logits, correct_tokens, incorrect_tokens)`**  
  Returns the correct-minus-incorrect logit difference for each prompt.
- **`normalized_patching_effect(patched, clean, corrupted)`**  
  Calculates $P$.
- **`circuit_localization_index(normalized_effects)`**  
  Calculates the CLI.
- **`apply_fdr_adjustment(p_values, alpha, method)`**  
  Runs the statsmodels FDR logic.
- **`signflip_p_values(effects, n_perm, two_sided)`**
  Computes Monte Carlo sign-flip p-values for `[layers, components, prompts]` under a symmetry-about-zero null.

### Visualization (Functions)
- **`plot_fdr_heatmap(results, title)`**  
  Renders a heatmap only when calibrated significance results are available. The plot does not establish causal validity by itself.

---

## 📚 Published Benchmark

I check the Benjamini–Hochberg adjustment against the published 15-hypothesis example in Benjamini and Hochberg (1995). At a target FDR of 0.05, the implementation rejects the same four hypotheses as the paper's BH step-up procedure.

The regression test uses the paper's ordered p-values and checks the rejected hypotheses directly:

```bash
python -m pytest -q tests/test_published_benchmark.py
```

Reference: Benjamini, Y. and Hochberg, Y. (1995), “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing,” *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300. [DOI: 10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x). The paper's p-values and result are also available in its [full text](https://www.stat.cmu.edu/~ryantibs/journalclub/benjamini_1995.pdf).

---

## 💻 Quick Start

### Option 1: Install Directly from GitHub
Anyone can install this directly into their environment without needing to clone the repo:
```bash
pip install git+https://github.com/anaykatiyar-lang/transformerlens-fdr.git
```

### Option 2: Clone for Local Development
If you want to edit the code or run the tutorials:
```bash
git clone https://github.com/anaykatiyar-lang/transformerlens-fdr.git
cd transformerlens-fdr

# Install editable with tests
pip install -e .[test]

# Run the provided example scripts!
python example_usage.py
python advanced_tutorial.py

# Reproduce the published BH reference benchmark
python -m pytest -q tests/test_published_benchmark.py
```

---

## ⚠️ Limitations

This published benchmark checks the multiple-testing adjustment on one reference example. It does not validate the p-values supplied to that adjustment, establish that a chosen empirical null fits a particular model or dataset, or guarantee calibrated results for every TransformerLens model. Inference still depends on valid input p-values and the assumptions of the selected null and correction method. The localization index and exploratory outlier scores are descriptive; they are not significance tests or evidence of causality.
