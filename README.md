# TransformerLens-FDR

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TransformerLens v4.0.0+](https://img.shields.io/badge/transformer__lens-4.0.0+-orange.svg)](https://github.com/TransformerLensOrg/TransformerLens)
[![PyPI version](https://img.shields.io/pypi/v/transformerlens-fdr)](https://pypi.org/project/transformerlens-fdr/)

**Multiple-testing-aware activation patching for TransformerLens.**
Patch every head or layer, then ask which effects survive false-discovery-rate control, with the assumptions stated rather than hidden.

- Per-prompt **normalized patching effects** for attention heads, MLP outputs, or the residual stream
- **Descriptive outlier scores by default**. P-values, q-values, and a significance mask appear only when you opt in (sign-flip test, or your own empirical null)
- **Benjamini–Hochberg or Benjamini–Yekutieli** correction, with a resolution report showing whether your permutation count can reach significance at all
- **Held-out validation** helper for checking a selected component set on disjoint prompt groups

> Activation patching tests **sufficiency**, not necessity, and this package cannot verify your null hypothesis. Read [Assumptions](#assumptions-checklist) before interpreting any p-value.

**Contents:** [Quick start](#quick-start) · [Which path should I use?](#which-path-should-i-use) · [Assumptions](#assumptions-checklist) · [Method](#method) · [API](#api-reference) · [Validation results](VALIDATION.md) · [FAQ](#faq) · [Limitations](#limitations) · [References](#references) · [Citing](#citing)

---

## Quick start

### Install

```bash
pip install transformerlens-fdr
```

Or directly from GitHub:

```bash
pip install git+https://github.com/anaykatiyar-lang/transformerlens_fdr.git
```

Or for local development:

```bash
git clone https://github.com/anaykatiyar-lang/transformerlens_fdr.git
cd transformerlens_fdr
pip install -e .[test]

python example_usage.py
python advanced_tutorial.py
```

### Minimal example (TransformerLens 4)

Supply aligned clean/corrupted prompts, target token IDs, and a metric that returns one value per prompt.

> **Note:** this example uses two prompt pairs only to show the API. With two pairs there are just four distinct sign patterns, so no result can be significant. Real inference needs enough independent prompt pairs for the task and the testing family.

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
correct = torch.tensor(
    [model.to_single_token(" Mary"), model.to_single_token(" Alice")],
    device=model.cfg.device,
)
incorrect = torch.tensor(
    [model.to_single_token(" John"), model.to_single_token(" Tom")],
    device=model.cfg.device,
)

auditor = PatchingAuditor(model, metric_fn=logit_difference, fdr_method="fdr_by")
result = auditor.run_patching_audit(
    clean, corrupt, correct, incorrect, method="signflip"
)
print(result.format_summary())
print(result.summary_df.sort_values("q_value").head())
```

`TransformerBridge` defaults to raw Hugging Face numerics; `enable_compatibility_mode()` requests HookedTransformer-equivalent numerics. Keep the mode fixed when comparing results and report it. Compatibility mode changes the numerical convention, so it can change measured effects.

### Check your setup before a long audit

```python
preflight = auditor.check_setup(
    clean, corrupt, correct, incorrect,
    method="signflip",  # omit this for the descriptive default
)
print(preflight.format_summary())
```

`check_setup` runs the clean and corrupted baseline forwards and reports prompt count, metric-gap size, family size, and p-value resolution. It costs two model forwards. If you plan to use an empirical null, pass the same `null_distribution` and `pooled_null` settings you will use for the audit. A preflight reports setup facts; it does not certify the statistical assumptions.

---

## Which path should I use?

This audit patches clean activations into a corrupted run. It asks whether those clean activations are **sufficient** to recover the chosen metric in that context. It does not test whether a component is necessary; noising or ablating a component asks a different question. A fixed effect-size cutoff is not treated as proof that a component belongs to a circuit.

The default audit does not invent a null from the observed components. It reports exploratory outlier scores and leaves p-values, q-values, and the significance mask unset.

| Your setup | Choose | What the result means |
| --- | --- | --- |
| You are exploring, or cannot defend a no-effect comparison | Default `method="none"` | Descriptive effects and outlier scores only; no p-values or FDR claim |
| Prompt effects are independent, and each is symmetric about zero under its null | `method="signflip"` | FDR-adjusted results under those sign-flip assumptions; the package cannot verify them |
| You have task-specific control runs that preserve the task structure while removing the relation of interest | `null_distribution=...` | Results relative to those controls, if they are a valid null for the stated metric and component family |
| Effects are clustered or skewed, and neither sign-flip assumptions nor a defensible control design fit | Do not treat output as calibrated inference | Use descriptive mode, or design a task-appropriate null and validate it separately |

Sign-flip is an explicit opt-in; it is not the default. It tests symmetry about zero, not merely a zero mean: a skewed, mean-zero null can still produce invalid p-values. Shared random sign patterns are applied across components to preserve their dependence within each permutation. This does not remove the need for independent prompt effects and valid marginal symmetry. No path can turn an unsuitable null into valid p-values.

<details>
<summary><b>Option 1 · Sign-flip test for matched prompts</b></summary>

Use `method="signflip"` when prompt-level effects are independent and each is symmetric about zero under its null. Exchangeability alone is not enough: validity comes from sign symmetry, not permutation exchangeability. Repeated or closely related prompts can break independence; a skewed distribution with mean zero can break symmetry. The package cannot check these assumptions for you.

The smallest possible p-value is `1 / (n_perm + 1)`. Set `n_perm` when creating `PatchingAuditor`. For example, with 144 tests and target level 0.05, 3,000 permutations resolve the first Benjamini–Hochberg threshold. Benjamini–Yekutieli needs finer p-value resolution, so it requires more permutations.

The test is two-sided by default. A significant negative effect means patching the clean activation lowered the supplied metric relative to the corrupted run; interpret direction according to how your metric is defined. Effects whose absolute normalized value is at most `zero_effect_tolerance` (default `1e-8`) are treated as zero. This is a numerical-zero safeguard, not a practical-effect test. The result records exact and numerical zero-component counts.

</details>

<details>
<summary><b>Option 2 · Empirical null from control runs</b></summary>

Pass caller-built control **mean normalized effects** with shape `[n_null, layers, components]`. They must use the same metric, normalization, component mapping, and prompt count as the observed audit. A shuffled or unrelated pair is a suitable control only if it really removes the relationship you are testing. The package does not build or certify a null for you.

**Worked design example:** for an indirect-object identification task, suppose the question is whether components respond to the name-to-role binding. A possible control set would preserve prompt length, token positions, name frequency, and the scoring metric while changing the prompt construction so the binding relation is absent. Each control replicate must go through the same patching and normalization pipeline as the observed prompts. This is an experimental design choice: the library can check shapes and recorded prompt counts, but cannot establish exchangeability or certify that the relation was removed. Do not use arbitrary shuffling unless it preserves the relevant structure and breaks the specific relation being tested.

`NullDistribution` can record the control design and matching details. The metadata is provenance and a bookkeeping check; it cannot prove the control is scientifically valid. If `n_prompts` is recorded, the library checks it against the observed batch size.

```python
null = NullDistribution(
    values=control_mean_effects,  # [n_null, layers, components]
    description="Matched controls with the tested relation removed",
    metric_name="logit_difference",
    normalization="mean normalized patching effect",
    hook_type="attn_head",
    n_prompts=clean.shape[0],
)
results = auditor.run_patching_audit(
    clean, corrupt, correct, incorrect,
    null_distribution=null,
)
```

Where possible, estimate the null from one set of controls and check calibration on separate, held-out controls. Do not use the same controls both to build the null and to claim it is calibrated.

</details>

---

## Assumptions checklist

Before reporting a q-value from this package, you should be able to say yes to each item that applies to your path.

**All inference paths**
- [ ] My metric returns one finite value per prompt, and I can explain what a positive effect means.
- [ ] The component family I tested is the family I searched. I pooled related audits with `adjust_audits_together` if they form one search.
- [ ] I recorded the model ID and revision, hook type, metric, correction method, `n_perm`, compatibility mode, and package versions.

**Sign-flip**
- [ ] Prompt pairs are independent: no near-duplicate templates, no repeated prompts with trivial word changes.
- [ ] Per-prompt effects are plausibly symmetric about zero under no effect. Skew, even with mean zero, breaks this.
- [ ] I have enough prompts and enough permutations for the p-value resolution my correction needs (see `resolution_report`).

**Empirical null**
- [ ] Control runs preserve the task structure but remove the specific relation under test.
- [ ] Controls went through the identical patching and normalization pipeline.
- [ ] Calibration was checked on controls separate from those used to build the null.

**BH vs BY**
- [ ] I use BH only if I can defend independence or positive dependence among tests. Otherwise I use BY.

None of these can be verified by the package. They are your responsibility.

---

## Method

### Normalized patching effect ($P$)

How much does patching a clean activation recover the clean metric?

$$ P = \frac{M_{\text{patched}} - M_{\text{corrupted}}}{M_{\text{clean}} - M_{\text{corrupted}}} $$

### Match the correction to the full search

An audit corrects p-values for the components in that audit. `family_size` reports the number of hypotheses, and `resolution_report` shows whether the smallest attainable p-value can cross the correction threshold. For example, 144 tests with BY and 9,999 sign flips have a minimum p-value of `0.0001`. One result at that minimum cannot pass the first step; at least two results must meet their rank-specific thresholds. That rank condition is necessary, not sufficient.

If you searched several component types or runs as one family, combine their valid p-values before interpretation:

```python
family_size = adjust_audits_together(
    [attention_results, mlp_results],
    alpha=0.05,
    method="fdr_by",
)
```

This updates the q-values and significance masks on both results. Separate calls only control FDR within each call.

### Choosing an FDR correction

FDR describes the long-run expected share of false results among those called significant:

$$ \mathrm{FDR} = \mathbb{E}\left[\frac{V}{\max(R, 1)}\right] \leq \alpha $$

`R` is the number called significant; `V` is the false ones among them. It is not a guarantee about the false-result share in one run.

- **Benjamini–Hochberg (`fdr_bh`)** is less conservative and assumes independent tests or certain forms of positive dependence.
- **Benjamini–Yekutieli (`fdr_by`)** allows arbitrary dependence when the individual p-values are valid, but is more conservative.

Neither correction can make invalid p-values or a poorly chosen null trustworthy. Report the null and correction you used.

### What a hook represents

| `hook_type` | What is patched | How to read it |
| --- | --- | --- |
| `attn_head` (default) | One attention head's `attn.hook_z` output at a time | Head-specific sufficiency effects |
| `mlp_out` | The whole MLP output at each layer | A layer-level effect, not an individual-neuron score. Architectures with a post-MLP normalization or other intervening operation may interpret this hook differently; inspect hook placement and validate for your architecture |
| `resid_mid` | The residual stream at each token position and layer | A whole-stream intervention, not component-specific attribution. Effects near 1 across layers can be expected and do not identify individual heads or neurons |

These are activation-patching locations, not claims that all architectures implement the same computation at each location. The tests exercise native TransformerLens 4 bridges and a cached GPT-2 integration path; that is not validation of every supported architecture or every hook's scientific interpretation.

### Circuit Localization Index ($\text{CLI}$)

Describes how concentrated the absolute point estimates are. It is descriptive, not a significance test or evidence that a circuit is causal. A score near $1.0$ means concentrated effects; near $0.0$ means diffuse effects.

$$ p_i = \frac{|P_i|}{\sum_j |P_j| + \epsilon} $$
$$ H_{\text{causal}} = - \sum p_i \log_2 (p_i + \epsilon) $$
$$ \text{CLI} = 1 - \frac{H_{\text{causal}}}{\log_2(N)} $$

---

## API reference

You will mostly use `PatchingAuditor`.

### `PatchingAuditor`

**`__init__(model, metric_fn, fdr_threshold=0.05, fdr_method='fdr_bh', n_perm=9999, hook_type='attn_head', zero_effect_tolerance=1e-8)`**
`metric_fn` must return one metric value per prompt (`[B]`). The default is BH: it requires valid p-values and independence or positive regression dependence among tests. BY handles arbitrary dependence among valid p-values but is more conservative. The simulations in this repository cover specific dependence settings; they do not establish BH's assumptions for every model or task. Choose BY when you cannot defend BH's dependence condition.

**`run_patching_audit(...)`**
Patches all components. Pass `method="signflip"` for the symmetry-about-zero sign-flip test, or provide a per-component `null_distribution`. With neither, results are exploratory and inferential fields are `None`.

**`check_setup(...)`**
Runs the clean and corrupted baseline forwards and reports prompt count, metric-gap size, family size, and p-value resolution before per-component patching. Costs two model forwards. It cannot predict zero-effect component counts; those are reported after the full audit.

**`run_joint_patching(..., patching_mask)`**
Patches selected components simultaneously and reports their joint normalized effect. When evaluating a mask chosen from an audit, use disjoint held-out prompts.

**`run_heldout_validation(selection_inputs, evaluation_inputs, selection_group_ids, evaluation_group_ids)`**
Selects components on one prompt split, then reports joint patching and descriptive CLI on disjoint evaluation groups. Group IDs should keep related prompt templates together.

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

The helper checks that group IDs do not overlap. It cannot establish that the groups are independent or that the selected mask is causal.

### `AuditResults`

| Field | Contents |
| --- | --- |
| `raw_patching_effects` | Mean point estimate per component `[layers, components]` |
| `per_prompt_effects` | Per-prompt effects `[layers, components, prompts]` |
| `adjusted_effects` | Point estimates centered by the same null used for inference; robust-centered when exploratory |
| `p_values` / `q_values` | Optional tensors, present only when a null method was requested |
| `significant_mask` | Optional boolean mask, present only when p-values were produced |
| `outlier_scores` | Robust descriptive scores; not p-values |
| `calibrated` / `null_description` | Whether inference was attempted and which null assumption was used |
| `family_size` / `resolution_report` | Number of components in the declared correction family; rank report when calibrated p-values were adjusted |
| `clean_corrupted_gap_mean` / `clean_corrupted_gap_std` | Mean clean-to-corrupted metric gap and its prompt spread (spread omitted for one prompt) |
| `exact_zero_components` / `numerically_zero_components` | Components with exactly zero per-prompt effects, or all within the zero tolerance |
| `cli_score` | The Circuit Localization Index |
| `summary_df` | Pandas DataFrame of all results |
| `methods_statement` | Plain-language description of the correction family and assumptions; use `print(results.format_summary())` for a copyable report |

`AuditPreflight` is also importable from `transformerlens_fdr`.

### Metrics and statistics functions

- **`logit_difference(logits, correct_tokens, incorrect_tokens)`**: correct-minus-incorrect logit difference per prompt
- **`normalized_patching_effect(patched, clean, corrupted)`**: calculates $P$
- **`circuit_localization_index(normalized_effects)`**: calculates the CLI
- **`apply_fdr_adjustment(p_values, alpha, method)`**: runs the statsmodels FDR logic
- **`adjust_audits_together(results, alpha, method)`**: applies one correction across several calibrated audit results, mutating their q-values and masks to reflect the pooled family
- **`resolution_report(n_tests, alpha, method, p_min)`**: reports the first-step threshold and the necessary minimum rank for a rejection at the available p-value resolution
- **`signflip_p_values(effects, n_perm, two_sided)`**: Monte Carlo sign-flip p-values for `[layers, components, prompts]` under a symmetry-about-zero null

### Visualization

- **`plot_fdr_heatmap(results, title)`**: renders a heatmap only when calibrated significance results are available. The plot does not establish causal validity by itself.

### Reproducibility

Record the exact prompt pairs and model ID/revision, hook type, metric, correction family and method, `n_perm`, compatibility mode, and the package and TransformerLens versions. `AuditResults` records the correction and resolution details, but does **not** store the RNG seed, exact `n_perm`, model revision, or package versions. The auditor currently uses PyTorch's global random generator for sign flips; call `torch.manual_seed(...)` and record the seed before the audit if you need a repeatable permutation stream.

---

## Validation

**Headline result:** the implementation reproduces the published Benjamini–Hochberg example: **4 of 15 hypotheses are rejected** at an FDR target of 0.05.

I also ran calibration simulations and implementation tests. The conditions, results, and limits are in [VALIDATION.md](VALIDATION.md).

---
## FAQ

**Why are there no p-values by default?**
Because the package will not invent a null from the observed components. Without a defensible no-effect comparison, a p-value would be meaningless. Choose `method="signflip"` or supply a `null_distribution` when you can defend the assumptions.

**Why is nothing significant in my audit?**
Check `resolution_report` and `check_setup` first. With few prompts the sign-flip p-values cannot get small enough, and BY needs finer resolution than BH. Raise `n_perm`, add independent prompt pairs, or both. A small family or a genuinely weak effect can also explain it.

**Should I use BH or BY?**
BH is less conservative but needs independence or positive dependence among tests. If you cannot defend that for your components, use BY.

**Does a significant result mean the component is necessary for the behavior?**
No. Patching shows sufficiency in the tested context. Necessity needs ablation or noising experiments.

**Can I use this with my own prompts?**
Yes. Keep prompts independent and diverse. Near-duplicate templates with small word changes are not independent samples.

---

## Limitations

The published benchmark checks the multiple-testing adjustment on one reference example. The synthetic and bridge tests provide additional software evidence, but do not validate p-values for a particular research dataset, establish that a chosen empirical null fits a particular model or dataset, or guarantee calibrated results for every TransformerLens model. Inference depends on valid input p-values and the assumptions of the selected null and correction method. The localization index and exploratory outlier scores are descriptive; they are not significance tests or evidence of causality.

Specifically, the package cannot:
- design valid control runs for your task (a human must do this),
- turn a sufficiency measure into a necessity claim,
- guarantee that your prompts are independent.

The exported `calculate_p_values` helper is a separate normal-null option: it assumes a representative control sample and an approximately normal null. `compute_control_baseline` only returns control moments; it does not validate normality. `compute_robust_baseline` is used for descriptive outlier scores and is not a Gaussian inference method.

The numerical-zero tolerance does not decide whether a statistically detectable effect is large enough to matter. A separate minimum-effect analysis would need its own null hypothesis and is not provided by this package.

---

## References

- Benjamini, Y. and Hochberg, Y. (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300. [DOI](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
- Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4), 1165–1188.
- Zhang, F. and Nanda, N. (2023). Towards best practices of activation patching in language models. [arXiv:2309.16042](https://arxiv.org/abs/2309.16042)
- Conmy, A. et al. (2023). Towards automated circuit discovery for mechanistic interpretability. [arXiv:2304.14997](https://arxiv.org/abs/2304.14997)
- Hanna, M., Pezzelle, S. and Belinkov, Y. (2024). Have faith in faithfulness. [arXiv:2403.17806](https://arxiv.org/abs/2403.17806)

---

## Roadmap

Planned, **not yet implemented**: runtime warnings for skewed prompt effects and near-duplicate prompts; an optional `seed` argument, with `n_perm` and the seed recorded in `AuditResults`. A minimum-effect analysis is under consideration, but it would change the null hypothesis, so it would be a separate, clearly labeled option.

---

## Citing

If you use this package, please cite it. <!-- TODO: add a CITATION.cff or BibTeX block with your name, title, year, and repository URL. -->

## License

MIT. See [LICENSE](LICENSE).
