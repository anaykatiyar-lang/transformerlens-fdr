# Purpose and usage

TransformerLens-FDR helps compare activation-patching effects across model components and, when a defensible null model is available, adjust valid p-values for multiple testing.

## Choose your inference path

**Which best describes your experiment? Select a card to see what to do.**

| Situation | Recommended path | Interpretation |
| --- | --- | --- |
| Exploring or no defensible no-effect comparison | `method="none"` (default) | Descriptive effect and outlier summaries only |
| Independent or exchangeable prompt effects; plausible symmetry around zero under no effect | `method="signflip"` | Inference depends on those assumptions; they are not checked by the package |
| Task-specific controls preserve the experiment while removing the relation being tested | `null_distribution=...` | Inference is relative to the control design, which the researcher must justify |
| Neither null design is defensible for the data | Stay exploratory or develop and validate a suitable null | Do not read descriptive rankings as significance |

<details>
<summary>I'm exploring, or I don't have a defensible no-effect comparison</summary>

Run the default `method="none"`. You get effect estimates and descriptive outlier scores, but no p-values, q-values, or significance mask. This is the right choice when you are still exploring or cannot explain what “no effect” should look like.

</details>

<details>
<summary>I have matched prompts and their effects are independent and symmetric around zero</summary>

Use `method="signflip"`. The test changes the signs of per-prompt effects to build a null distribution. It relies on independent prompt effects and symmetry around zero; the package cannot verify those assumptions for you. Repeated or closely related prompts can break independence.

```python
auditor = PatchingAuditor(model, metric_fn, n_perm=10_000)
results = auditor.run_patching_audit(
    clean_tokens=clean_tokens,
    corrupted_tokens=corrupted_tokens,
    correct_tokens=correct_tokens,
    incorrect_tokens=incorrect_tokens,
    method="signflip",
)
```

The smallest possible p-value is `1 / (n_perm + 1)`. Use enough permutations to resolve the significance levels you care about.

For sign-flip inference, per-prompt normalized effects with absolute value at most `zero_effect_tolerance` (default `1e-8`) are treated as zero to avoid calling floating-point residue a consistent effect. This is not a minimum-effect test: a small but nonzero effect above that tolerance can still be statistically significant, and practical importance needs a separate analysis.

</details>

<details>
<summary>I have separate control runs that represent “no effect”</summary>

Pass a distribution of control **mean normalized effects** as `null_distribution`, with shape `[n_controls, layers, components]`. Each entry must come from the same metric, normalized-effect definition, component mapping, and prompt count as the observed audit. Controls must preserve the experiment's relevant structure while removing the effect you are testing. Shuffling is only a good control if it actually breaks the relationship of interest; this package does not build or certify a null for you.

**Example rationale:** in an indirect-object identification experiment about name-to-role binding, a candidate control might preserve prompt length, positions, name frequency, and the scoring metric while changing how the prompt is constructed so the binding relation is absent. Run those controls through the same patching and normalization steps. This is only a defensible null if the control prompts really preserve other relevant sources of component effects while removing the target relation. The package checks recorded metadata and array structure; it cannot verify that scientific argument.

You can record how you built the control values with `NullDistribution`. The labels are provenance notes and shape/prompt-count checks; matching labels do not prove that the control is a valid null.

```python
null = NullDistribution(
    values=control_mean_effects,  # [n_controls, layers, components]
    description="Matched control pairs with the tested relation removed",
    metric_name="logit_difference",
    normalization="mean normalized patching effect",
    hook_type="attn_head",
    n_prompts=clean_tokens.shape[0],
)
results = auditor.run_patching_audit(
    clean_tokens=clean_tokens,
    corrupted_tokens=corrupted_tokens,
    correct_tokens=correct_tokens,
    incorrect_tokens=incorrect_tokens,
    null_distribution=null,
)
```

The null values must be control replicates of the *component-wise mean effect*, not a collection of raw activations or individual prompt effects. Ideally, use separate controls to check calibration.

</details>

## Once you have p-values: account for the number of components

A search across many heads or other components can produce small p-values by chance. FDR adjustment accounts for that search. The target is a long-run expected proportion of false discoveries among the results called significant:

$$ \mathrm{FDR} = \mathbb{E}\left[\frac{V}{\max(R, 1)}\right] \leq \alpha $$

Here, `R` is the number of results called significant and `V` is the false ones among them. It is not a promise about the false-discovery fraction in one run. Adjustment cannot fix p-values made from an unsuitable null.

<details>
<summary>Which adjustment should I use?</summary>

- **Benjamini–Hochberg (`fdr_bh`)** is less conservative; it assumes independent tests or certain positive dependence.
- **Benjamini–Yekutieli (`fdr_by`)** allows arbitrary dependence when each p-value is valid, but is usually more conservative.

When unsure, report which method you chose and why. Neither method validates the null model or the p-values supplied to it.

</details>

## Match the correction family to the question

When inference runs, an ordinary audit adjusts across the components in that result. Its `family_size` records the number of components in the family, and `resolution_report` shows whether the smallest attainable p-value can cross the first BH/BY threshold. With no null, `family_size` still describes the search size but no correction is performed and the resolution report is unset. For example, 144 tests with BY and 9,999 sign flips have minimum p-value `0.0001`; an isolated minimum p-value cannot pass the first step, and at least two tests must be small enough to meet their rank-specific thresholds. That rank condition is necessary, not sufficient.

If several audits belong to one scientific search (for example, attention heads and MLP layers searched together), pool them before interpreting significance:

```python
family_size = adjust_audits_together(
    [attention_results, mlp_results],
    alpha=0.05,
    method="fdr_by",
)
```

This updates both results' q-values and masks to use one shared family. Separate audit calls without this pooling control FDR only within each call.

## Check a selected mask on new prompts

Selecting a mask and measuring its joint effect on the same prompts can make the result look stronger than it is. Split prompts into selection and evaluation groups before examining effects, and keep related templates in the same group. The held-out helper checks that the group IDs do not overlap:

```python
heldout = auditor.run_heldout_validation(
    selection_inputs={
        **selection_tokens_and_labels,
        "method": "signflip",
    },
    evaluation_inputs=evaluation_tokens_and_labels,
    selection_group_ids=selection_template_ids,
    evaluation_group_ids=evaluation_template_ids,
)
print(heldout.joint_effect)
print(heldout.evaluation_audit.cli_score)  # descriptive, computed on evaluation prompts
```

The selection split supplies the significant-component mask. The joint effect and evaluation CLI are computed on the held-out split. This helps avoid reusing the same prompts for selection and evaluation; it does not fix a misspecified null or establish causality.

## Define the effect before looking at results

The metric says what counts as an effect. For next-token preference, `logit_difference(logits, correct_tokens, incorrect_tokens)` subtracts the alternative token's logit from the target token's logit. For another question, pass a `metric_fn` that returns one finite value per prompt, with shape `[B]`. Use the same score and outcome definition for clean, corrupted, and patched runs.

The normalized patching effect is the share of the clean-to-corrupted score gap recovered by replacing an activation:

$$ P = \frac{M_{\text{patched}} - M_{\text{corrupted}}}{M_{\text{clean}} - M_{\text{corrupted}}} $$

A value near 0 means little recovery; 1 means the full gap was recovered; above 1 means overshoot. The ratio is undefined if clean and corrupted scores are equal.

## Read the results

For a quick preflight before component patching, call `auditor.check_setup(...)`. It runs the clean and corrupted baseline forwards (two model passes) and reports prompt count, component-family size, the clean-to-corrupted metric gap, and p-value resolution. It does not run per-component patching and cannot know zero-effect counts ahead of time.

```python
preflight = auditor.check_setup(
    clean_tokens, corrupted_tokens, correct_tokens, incorrect_tokens,
    method="signflip",  # omit for the descriptive default
)
print(preflight.format_summary())
```

After the full audit, `print(results.format_summary())` gives a short, copyable methods statement and key diagnostics. The statement records the selected null assumptions and correction family; it does not claim that the assumptions were validated.

| Result | What it tells you |
| --- | --- |
| `raw_patching_effects` | Average normalized effect for each component |
| `per_prompt_effects` | How effects vary across prompts |
| `p_values` / `q_values` | Unadjusted / multiple-testing-adjusted values, when inference ran |
| `significant_mask` | Components passing the selected FDR threshold |
| `outlier_scores` | Descriptive ranking; not significance tests |
| `cli_score` | Whether absolute effects are concentrated in a few components; not significance or causality |
| `clean_corrupted_gap_mean` / `clean_corrupted_gap_std` | Mean clean-to-corrupted metric gap and its prompt-to-prompt spread; the spread is omitted for one prompt |
| `exact_zero_components` / `numerically_zero_components` | Components whose per-prompt effects are exactly zero or all within the configured sign-flip zero tolerance |
| `family_size` / `resolution_report` | Number of hypotheses corrected and the attainable p-value/rank information |
| `methods_statement` | Plain-language summary of the correction and null assumptions |

Treat selected components as candidates for follow-up patching or ablation. A result applies to the chosen metric, prompts, null, and correction method; it does not establish a general causal role.

## Run a published reference check

The published-reference test checks the Benjamini–Hochberg implementation against Example 3.2 of Benjamini and Hochberg (1995): at level 0.05, the same first four hypotheses should be rejected.

```bash
python -m pytest -q tests/test_published_benchmark.py
```

Benjamini, Y. and Hochberg, Y. (1995), “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing,” *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300. [DOI](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) · [Full text](https://www.stat.cmu.edu/~ryantibs/journalclub/benjamini_1995.pdf).

## Scope and limitations

This reference check validates one published multiple-testing example. It does not validate every way of generating p-values, every control design, or calibration across all models and datasets. Calibration depends on the metric, prompts, null model, and dependence assumptions. The example in [`example_usage.py`](example_usage.py) uses GPT-2 and one prompt, so it demonstrates the workflow but does not make significance claims. The CLI and outlier scores are descriptive summaries, not significance tests or evidence of causality.
