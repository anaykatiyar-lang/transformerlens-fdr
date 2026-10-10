# TransformerLens-FDR 📊

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TransformerLens v4.0.0+](https://img.shields.io/badge/transformer__lens-4.0.0+-orange.svg)](https://github.com/TransformerLensOrg/TransformerLens)
[![PyPI version](https://img.shields.io/pypi/v/transformerlens-fdr)](https://pypi.org/project/transformerlens-fdr/)


**Activation-patching analysis utilities for TransformerLens.** I built this package to make component patching effects easier to inspect and to provide an explicit, assumption-aware path from those effects to multiple-testing results.

The package computes per-prompt patching effects and descriptive robust outlier scores. It returns FDR-adjusted significance results only when a caller supplies an empirical null or explicitly requests a sign-flip test. Those results are valid only when the null assumptions are appropriate and have been checked for the task.

---

## 🚀 Why Use FDR?

Standard activation patching measures a normalized effect. I do not treat a fixed effect-size cutoff as proof that a component belongs to a circuit. When many components are tested, false discoveries need to be considered explicitly.

The default audit does not invent a null from the observed components. It reports exploratory outlier scores and leaves p-values, q-values, and the significance mask unset. For inference, use a per-component empirical null or request sign-flip testing over independent prompt pairs.

### Which path should I use?

| Your setup | Choose | What the result means |
| --- | --- | --- |
| You are exploring, or cannot defend a no-effect comparison | Default `method="none"` | Descriptive effects and outlier scores only; no p-values or FDR claim |
| Prompt effects are independent or exchangeable and symmetric around zero when there is no effect | `method="signflip"` | FDR-adjusted results under those sign-flip assumptions; the package cannot verify them |
| You have task-specific control runs that preserve the task structure while removing the relation of interest | `null_distribution=...` | Results relative to those controls, if the controls are a valid null for the stated metric and component family |
| Effects are clustered or skewed, and neither sign-flip assumptions nor a defensible control design fit | Do not treat the output as calibrated inference | Use descriptive mode or design a task-appropriate null and validate it separately |

Sign-flip is an explicit opt-in; it is not the default. No path can turn an unsuitable null into valid p-values.

---

## 🧮 The Mathematics

### 1. Normalized Patching Effect ($P$)
How much does patching a clean activation recover the clean metric?

$$ P = \frac{M_{\text{patched}} - M_{\text{corrupted}}}{M_{\text{clean}} - M_{\text{corrupted}}} $$

### 2. Inference: choose a null before interpreting results

> **Quick check:** What would your data look like if the effect were absent? If you cannot give a defensible answer, keep the default `method="none"`. You will get exploratory effect and outlier summaries, but no p-values, q-values, or significance mask.

<details>
<summary>Option 1 · Sign-flip test for matched prompts</summary>

Use `method="signflip"` when per-prompt effects are independent (or exchangeable) and the no-effect distribution is symmetric around zero. Repeated or closely related prompts can break independence; skew can break symmetry. The package cannot check these assumptions for you.

The smallest possible p-value is `1 / (n_perm + 1)`. Set `n_perm` when creating `PatchingAuditor`. For example, with 144 tests and target level 0.05, 3,000 permutations resolve the first Benjamini–Hochberg threshold. Benjamini–Yekutieli needs finer p-value resolution, so it requires more permutations.

For sign-flip inference, effects whose absolute normalized value is at most `zero_effect_tolerance` (default `1e-8`) are treated as zero. This is a numerical-zero safeguard, not a practical-effect test. The result records exact and numerical zero-component counts.

</details>

<details>
<summary>Option 2 · Empirical null from control runs</summary>

Pass caller-built control **mean normalized effects** with shape `[n_null, layers, components]`. They must use the same metric, normalization, component mapping, and prompt count as the observed audit. A shuffled or unrelated pair is a suitable control only if it really removes the relationship you are testing. The package does not build or certify a null for you.

**Worked design example:** for an indirect-object identification task, suppose the question is whether components respond to the name-to-role binding. A possible control set would preserve prompt length, token positions, name frequency, and the scoring metric while changing the prompt construction so the binding relation is absent. Each control replicate must then go through the same patching and normalization pipeline as the observed prompts. This rationale is an experimental choice: the library can check shapes and recorded prompt counts, but cannot establish exchangeability or certify that the relation was removed. Do not use arbitrary shuffling unless it preserves the relevant structure and breaks the specific relation being tested.

`NullDistribution` can record the control design and these matching details. The metadata is provenance and a bookkeeping check; it cannot prove that the control is scientifically valid. If `n_prompts` is recorded, the library checks it against the observed batch size.

```python
null = NullDistribution(
    values=control_mean_effects,  # [n_null, layers, components]
    description="Matched controls with the tested relation removed",
    metric_name="logit_difference",
    normalization="mean normalized patching effect",
    hook_type="attn_head",
    n_prompts=clean_tokens.shape[0],
)
results = auditor.run_patching_audit(
    clean_tokens, corrupted_tokens, correct_tokens, incorrect_tokens,
    null_distribution=null,
)
```

Where possible, estimate the null from one set of controls and check calibration on separate, held-out controls. Do not use the same controls both to build the null and to claim that it is calibrated.

</details>

### Match the correction to the full search

An audit corrects the p-values for the components in that audit `family_size` reports the number of hypotheses, and `resolution_report` shows whether the smallest attainable p-value can cross the correction threshold. For example, 144 tests with BY and 9,999 sign flips have a minimum p-value of `0.0001`. One result at that minimum cannot pass the first step; at least two results must meet their rank-specific thresholds. That rank condition is necessary, not sufficient.

If you searched several component types or runs as one family, combine their valid p-values before interpretation:

```python
family_size = adjust_audits_together(
    [attention_results, mlp_results],
    alpha=0.05,
    method="fdr_by",
)
```

This updates the q-values and significance masks on both results. Separate calls only control FDR within each call.

<details>
<summary>After valid p-values · Choose an FDR correction</summary>

FDR describes the long-run expected share of false results among those called significant:

$$ \mathrm{FDR} = \mathbb{E}\left[\frac{V}{\max(R, 1)}\right] \leq \alpha $$

`R` is the number called significant; `V` is the false ones among them. It is not a guarantee about the false-result share in one run.

- **Benjamini–Hochberg (`fdr_bh`)** is less conservative and assumes independent tests or certain positive dependence.
- **Benjamini–Yekutieli (`fdr_by`)** allows arbitrary dependence when the individual p-values are valid, but is more conservative.

Neither correction can make invalid p-values or a poorly chosen null trustworthy. Report the null and correction you used.

</details>

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
- **`__init__(model, metric_fn, fdr_threshold=0.05, fdr_method='fdr_bh', n_perm=9999, hook_type='attn_head', zero_effect_tolerance=1e-8)`**
  `metric_fn` must return one metric value per prompt (`[B]`). BH requires valid p-values and independence or positive regression dependence; BY handles arbitrary dependence but still requires valid p-values.
- **`run_patching_audit(...)`**  
  Patches all components. Pass `method="signflip"` to use the symmetry-about-zero sign-flip test, or provide a per-component `null_distribution`. With neither, results are exploratory and inferential fields are `None`.
- **`check_setup(...)`**
  Runs the clean and corrupted baseline forwards and reports prompt count, metric-gap size, family size, and p-value resolution before per-component patching. It costs two model forwards. It cannot predict zero-effect component counts; those are reported after the full audit.
- **`run_joint_patching(..., patching_mask)`**  
  Patches selected components simultaneously and reports their joint normalized effect. When evaluating a mask chosen from an audit, use disjoint held-out prompts.
- **`run_heldout_validation(selection_inputs, evaluation_inputs, selection_group_ids, evaluation_group_ids)`**
  Selects components on one prompt split, then reports joint patching and descriptive CLI on disjoint evaluation groups. Group IDs should keep related prompt templates together.

### `AuditResults` (Dataclass)
Returned by the auditor. Contains:
- `raw_patching_effects`: Mean point estimate per component `[layers, components]`.
- `per_prompt_effects`: Per-prompt effects `[layers, components, prompts]`.
- `adjusted_effects`: Point estimates centered by the same null used for inference; robust-centered when exploratory.
- `p_values` / `q_values`: Optional tensors, present only when a null method was requested.
- `significant_mask`: Optional boolean mask, present only when p-values were produced.
- `outlier_scores`: Robust descriptive scores; not p-values.
- `calibrated` / `null_description`: State whether inference was attempted and which null assumption was used.
- `family_size` / `resolution_report`: Number of components in the declared correction family; the rank report is present when calibrated p-values were adjusted.
- `clean_corrupted_gap_mean` / `clean_corrupted_gap_std`: Mean clean-to-corrupted metric gap and its prompt spread; spread is omitted for one prompt.
- `exact_zero_components` / `numerically_zero_components`: Components with exactly zero per-prompt effects or effects all within the configured zero tolerance.
- `cli_score`: The Circuit Localization Index float.
- `summary_df`: A neat Pandas DataFrame of all results.
- `methods_statement`: Plain-language description of the correction family and assumptions. For a copyable report, use `print(results.format_summary())`.

Before a long audit, the same inputs can be checked with:

```python
preflight = auditor.check_setup(
    clean_tokens, corrupted_tokens, correct_tokens, incorrect_tokens,
    method="signflip",  # omit this for the descriptive default
)
print(preflight.format_summary())
```

`AuditPreflight` is also importable from `transformerlens_fdr`. If you plan to use an empirical null, pass the same `null_distribution` and `pooled_null` settings to `check_setup` that you will use for the audit. A preflight reports setup facts; it does not certify the statistical assumptions.

For example, after preparing non-overlapping prompt-template groups:

```python
heldout = auditor.run_heldout_validation(
    selection_inputs={**selection_inputs, "method": "signflip"},
    evaluation_inputs=evaluation_inputs,
    selection_group_ids=selection_template_ids,
    evaluation_group_ids=evaluation_template_ids,
)
print(heldout.joint_effect)
print(heldout.evaluation_audit.cli_score)  # descriptive only
```

The helper checks that the group IDs do not overlap. It cannot establish that
the groups are independent or that the selected mask is causal.

### Metrics & Stats (Functions)
- **`logit_difference(logits, correct_tokens, incorrect_tokens)`**  
  Returns the correct-minus-incorrect logit difference for each prompt.
- **`normalized_patching_effect(patched, clean, corrupted)`**  
  Calculates $P$.
- **`circuit_localization_index(normalized_effects)`**  
  Calculates the CLI.
- **`apply_fdr_adjustment(p_values, alpha, method)`**  
  Runs the statsmodels FDR logic.
- **`adjust_audits_together(results, alpha, method)`**
  Applies one correction across several calibrated audit results, mutating their q-values and masks to reflect the pooled family.
- **`resolution_report(n_tests, alpha, method, p_min)`**
  Reports the first-step threshold and the necessary minimum rank for a rejection at the available p-value resolution.
- **`signflip_p_values(effects, n_perm, two_sided)`**
  Computes Monte Carlo sign-flip p-values for `[layers, components, prompts]` under a symmetry-about-zero null.

### Visualization (Functions)
- **`plot_fdr_heatmap(results, title)`**  
  Renders a heatmap only when calibrated significance results are available. The plot does not establish causal validity by itself.

---

## 📚 Published Benchmark

I checked the Benjamini–Hochberg adjustment against the published 15-hypothesis example in Benjamini and Hochberg (1995). At a target FDR of 0.05, the implementation rejects the same four hypotheses as the paper's BH step-up procedure.

The regression test uses the paper's ordered p-values and checks the rejected hypotheses directly:

```bash
python -m pytest -q tests/test_published_benchmark.py
```

Reference: Benjamini, Y. and Hochberg, Y. (1995), “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing,” *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300. [DOI: 10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x). The paper's p-values and result are also available in its [full text](https://www.stat.cmu.edu/~ryantibs/journalclub/benjamini_1995.pdf).

---

## 💻 Quick Start
### Option 1. Install from PyPI:

```bash
pip install transformerlens-fdr
```

### Option 2: Install Directly from GitHub
Anyone can install this directly into their environment without needing to clone the repo:
```bash
pip install git+https://github.com/anaykatiyar-lang/transformerlens-fdr.git
```

### Option 3: Clone for Local Development
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

The exported `calculate_p_values` helper is a separate normal-null option: it assumes a representative control sample and an approximately normal null. `compute_control_baseline` only returns control moments; it does not validate normality. `compute_robust_baseline` is used for descriptive outlier scores and is not a Gaussian inference method.

The numerical-zero tolerance does not decide whether a statistically detectable effect is large enough to matter. A separate minimum-effect analysis needs its own explicit hypothesis and validation; it is not part of the current FDR test.
