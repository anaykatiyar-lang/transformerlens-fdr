# TransformerLens-FDR 📊

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TransformerLens v4.0.0+](https://img.shields.io/badge/transformer__lens-4.0.0+-orange.svg)](https://github.com/TransformerLensOrg/TransformerLens)
[![PyPI version](https://img.shields.io/pypi/v/transformerlens-fdr)](https://pypi.org/project/transformerlens-fdr/)


**Activation-patching analysis utilities for TransformerLens.** I built this package to make component patching effects easier to inspect and to provide an explicit, assumption-aware path from those effects to multiple-testing results.

The package computes per-prompt patching effects and descriptive robust outlier scores. It returns FDR-adjusted significance results only when a caller supplies an empirical null or explicitly requests a sign-flip test. Those results are valid only when the null assumptions are appropriate and have been checked for the task.

## 🚀 Why Use FDR?


This audit patches clean activations into a corrupted run. It asks whether those clean activations are sufficient to recover the chosen metric in that context; it does not test whether a component is necessary. Noising or ablating a component asks a different question. I do not treat a fixed effect-size cutoff as proof that a component belongs to a circuit. When many components are tested, false discoveries need to be considered explicitly.

The default audit does not invent a null from the observed components. It reports exploratory outlier scores and leaves p-values, q-values, and the significance mask unset. For inference, use a per-component empirical null or request sign-flip testing over independent prompt pairs.

<details>
<summary>Which inference path should I use?</summary>

- **Exploring, or no defensible no-effect comparison:** use the default `method="none"`. You get descriptive effects and outlier scores, not p-values or an FDR claim.
- **Independent prompt effects symmetric about zero under the null:** use `method="signflip"`. Results rely on both assumptions; the package cannot verify them.
- **Task-specific control runs that remove the relation being tested:** provide `null_distribution=...`. It must use the same metric, normalization, component mapping, and prompt count as the observed effects.
- **Clustered or skewed effects without a defensible control design:** stay in descriptive mode until you can design and validate a suitable null.

</details>

Sign-flip is an explicit opt-in; it is not the default. It tests symmetry about zero, not merely a zero mean: a skewed, mean-zero null can still produce invalid p-values. The code applies shared random sign patterns across components to preserve their dependence in each permutation; this does not remove the need for independent prompt effects and valid marginal symmetry. No path can turn an unsuitable null into valid p-values.

</details>

---

## 🧮 The Mathematics

### 1. Normalized Patching Effect ($P$)
How much does patching a clean activation recover the clean metric?

$$
\large\displaystyle P = \frac{M_{\text{patched}} - M_{\text{corrupted}}}{M_{\text{clean}} - M_{\text{corrupted}}}
$$

### 2. Inference: choose a null before interpreting results

> **Quick check:** What would your data look like if the effect were absent? If you cannot give a defensible answer, keep the default `method="none"`. You will get exploratory effect and outlier summaries, but no p-values, q-values, or significance mask.

<details>
<summary>Option 1 · Sign-flip test for matched prompts</summary>

Use `method="signflip"` when prompt-level effects are independent and each effect is symmetric about zero under its null. Exchangeability alone is not enough: sign-flip validity comes from sign symmetry, not from permutation exchangeability. Repeated or closely related prompts can break independence; a skewed distribution with mean zero can break symmetry. The test checks symmetry about zero, not just whether the mean is zero. The package cannot check these assumptions for you.

The smallest possible p-value is `1 / (n_perm + 1)`. Set `n_perm` when creating `PatchingAuditor`. For example, with 144 tests and target level 0.05, 3,000 permutations resolve the first Benjamini–Hochberg threshold. Benjamini–Yekutieli needs finer p-value resolution, so it requires more permutations.

The sign-flip test is two-sided by default. A significant negative effect means patching the clean activation lowered the supplied metric relative to the corrupted run; interpret that direction according to how your metric is defined. Effects whose absolute normalized value is at most `zero_effect_tolerance` (default `1e-8`) are treated as zero. This is a numerical-zero safeguard, not a practical-effect test. The result records exact and numerical zero-component counts.

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

### What a hook represents

| `hook_type` | What is patched | How to read it |
| --- | --- | --- |
| `attn_head` (default) | One attention head's `attn.hook_z` output at a time | Head-specific sufficiency effects. |
| `mlp_out` | The whole MLP output at each layer | A layer-level effect, not an individual-neuron score. Architectures with a post-MLP normalization or other intervening operation may interpret this hook differently; inspect the model's hook placement and validate it for your architecture. |
| `resid_mid` | The residual stream at each token position and layer | A whole-stream intervention, not a component-specific attribution. In TransformerLens 4 bridges that expose `set_use_hook_mlp_in`, it resolves to `blocks.{layer}.hook_mlp_in` (the MLP-branch entry); older or other architectures may use a different hook. These locations may not be numerically comparable. Effects near 1 can occur because this restores a broad activation and does not identify individual heads or neurons. |

These are activation-patching locations, not claims that all architectures implement the same computation at each location. The tests exercise native TransformerLens 4 bridges and a cached GPT-2 integration path; that is not validation of every supported architecture or of every hook's scientific interpretation.

<details>
<summary>After valid p-values · Choose an FDR correction</summary>

FDR describes the long-run expected share of false results among those called significant:

$$
\large\displaystyle \mathrm{FDR} = \mathbb{E}\left[\frac{V}{\max(R, 1)}\right] \leq \alpha
$$

`R` is the number called significant; `V` is the false ones among them. It is not a guarantee about the false-result share in one run.

- **Benjamini–Hochberg (`fdr_bh`)** is less conservative and assumes independent tests or certain positive dependence.
- **Benjamini–Yekutieli (`fdr_by`)** allows arbitrary dependence when the individual p-values are valid, but is more conservative.

Neither correction can make invalid p-values or a poorly chosen null trustworthy. Report the null and correction you used.

</details>

### 3. Circuit Localization Index ($\text{CLI}$)
Describes how concentrated the absolute point estimates are. It is descriptive, not a significance test or evidence that a circuit is causal. A score near $1.0$ means concentrated effects; a score near $0.0$ means diffuse effects.

$$
\large\displaystyle p_i = \frac{|P_i|}{\sum_j |P_j| + \epsilon}
$$

$$
\large\displaystyle H_{\text{causal}} = - \sum_i p_i \log_2 (p_i + \epsilon)
$$

$$
\large\displaystyle \text{CLI} = 1 - \frac{H_{\text{causal}}}{\log_2(N)}
$$

---

## 🛠️ API Reference

The API is intentionally simple. You will mostly interact with `PatchingAuditor`.

### `PatchingAuditor` (Class)
The main controller for your statistical audit.
- **`__init__(model, metric_fn, fdr_threshold=0.05, fdr_method='fdr_bh', n_perm=9999, hook_type='attn_head', zero_effect_tolerance=1e-8)`**
  `metric_fn` must return one metric value per prompt (`[B]`). The default is BH: it requires valid p-values and independence or positive regression dependence among tests. BY handles arbitrary dependence among valid p-values but is more conservative. The simulations in this repository cover specific dependence settings; they do not establish BH's assumptions for every model or task. Choose BY when you cannot defend BH's dependence condition.
- **`run_patching_audit(...)`**<br>
  Patches all components. Pass `method="signflip"` to use the symmetry-about-zero sign-flip test, or provide a per-component `null_distribution`. With neither, results are exploratory and inferential fields are `None`.
- **`check_setup(...)`**
  Runs the clean and corrupted baseline forwards and reports prompt count, metric-gap size, family size, and p-value resolution before per-component patching. It costs two model forwards. It cannot predict zero-effect component counts; those are reported after the full audit.
- **`run_joint_patching(..., patching_mask)`**<br>
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
- `outlier_scale_method`: The scale used for these scores (`scaled MAD`, or population standard deviation when the MAD is degenerate; flat maps use an epsilon floor).
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

### TransformerLens 4 example

This short example uses two matched prompt pairs only to show the API; real inference needs enough independent prompt pairs for the task and testing family. Supply aligned clean/corrupted prompts, target token IDs, and a metric that returns one value per prompt.

```python
import torch
from transformer_lens import TransformerBridge
from transformerlens_fdr import PatchingAuditor, logit_difference

model = TransformerBridge.boot_transformers("gpt2")
model.enable_compatibility_mode()  # HookedTransformer-equivalent numerics
clean = model.to_tokens([
    "When John and Mary went to the store, John gave a drink to",
    "When Tom and Alice went to the store, Tom gave a drink to",
])
corrupt = model.to_tokens([
    "When John and Mary went to the store, Mary gave a drink to",
    "When Tom and Alice went to the store, Alice gave a drink to",
])
correct = torch.tensor([model.to_single_token(" Mary"), model.to_single_token(" Alice")], device=model.cfg.device)
incorrect = torch.tensor([model.to_single_token(" John"), model.to_single_token(" Tom")], device=model.cfg.device)

auditor = PatchingAuditor(model, metric_fn=logit_difference, fdr_method="fdr_by")
result = auditor.run_patching_audit(
    clean, corrupt, correct, incorrect, method="signflip"
)
print(result.format_summary())
print(result.summary_df.sort_values("q_value").head())
```

`TransformerBridge` defaults to raw Hugging Face numerics; `enable_compatibility_mode()` requests HookedTransformer-equivalent numerics. Keep the mode fixed when comparing results and report it. The repository's GPT-2 example and small-model auditor tests use compatibility mode; optional real-model integration tests exercise the bridge separately. Compatibility mode changes the numerical convention, so it can change measured effects.

For reproducibility, record the exact prompt pairs and model ID/revision, hook type, metric, correction family and method, `n_perm`, compatibility mode, and the package/TransformerLens versions. `AuditResults` records the correction and resolution details, whether sign flips used exact enumeration or Monte Carlo sampling, the number of patterns evaluated, and the resolved hook name for each layer. It does **not** store the RNG seed, exact `n_perm`, model revision, or package versions. The auditor currently uses PyTorch's global random generator for sign flips; set and record `torch.manual_seed(...)` before the audit if you need a repeatable permutation stream.

### Metrics & Stats (Functions)
- **`logit_difference(logits, correct_tokens, incorrect_tokens)`**<br>
  Returns the correct-minus-incorrect logit difference for each prompt using the final sequence position. With right-padded inputs, that can be a padding position; use a custom metric that selects each prompt's last non-padding position.
- **`normalized_patching_effect(patched, clean, corrupted)`**<br>
  Calculates $P$.
- **`circuit_localization_index(normalized_effects)`**<br>
  Calculates the CLI.
- **`apply_fdr_adjustment(p_values, alpha, method)`**<br>
  Runs the statsmodels FDR logic.
- **`adjust_audits_together(results, alpha, method)`**
  Applies one correction across several calibrated audit results, mutating their q-values and masks to reflect the pooled family.
- **`resolution_report(n_tests, alpha, method, p_min)`**
  Reports the first-step threshold and the necessary minimum rank for a rejection at the available p-value resolution.
- **`signflip_p_values(effects, n_perm, two_sided)`**
  Computes sign-flip p-values for `[layers, components, prompts]` under a symmetry-about-zero null. It enumerates all unique patterns when feasible and otherwise uses Monte Carlo sampling.

### Visualization (Functions)
- **`plot_fdr_heatmap(results, title)`**<br>
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
pip install git+https://github.com/anaykatiyar-lang/transformerlens_fdr.git
```

### Option 3: Clone for Local Development
If you want to edit the code or run the tutorials:
```bash
git clone https://github.com/anaykatiyar-lang/transformerlens_fdr.git
cd transformerlens-fdr

# Install editable with tests
pip install -e .[test]

# Run the provided example scripts!
python example_usage.py
python advanced_tutorial.py

```

---

## ⚠️ Limitations

The published benchmark checks the multiple-testing adjustment on one reference example. Other synthetic and bridge tests provide software evidence, but do not validate p-values for a particular research dataset, establish that a chosen empirical null fits a particular model or dataset, or guarantee calibrated results for every TransformerLens model. Inference still depends on valid input p-values and the assumptions of the selected null and correction method. The localization index and exploratory outlier scores are descriptive; they are not significance tests or evidence of causality.

The exported `calculate_p_values` helper is a separate normal-null option: it assumes a representative control sample and an approximately normal null. `compute_control_baseline` only returns control moments; it does not validate normality. `compute_robust_baseline` is used for descriptive outlier scores and is not a Gaussian inference method.

The numerical-zero tolerance does not decide whether a statistically detectable effect is large enough to matter. A separate minimum-effect analysis needs its own explicit hypothesis and validation; it is not part of the current FDR test. The empirical-null snippet above is a design illustration, not a runnable control-generation recipe: a defensible control depends on the research question and must be implemented and validated by the researcher.
