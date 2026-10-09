# Purpose and usage

TransformerLens-FDR helps compare activation-patching effects across model components and, when a defensible null model is available, adjust valid p-values for multiple testing.

## Choose your inference path

**Which best describes your experiment? Select a card to see what to do.**

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

</details>

<details>
<summary>I have separate control runs that represent “no effect”</summary>

Pass the control effects as `null_distribution`, with shape `[n_controls, layers, components]`. Controls must preserve the experiment's relevant structure while removing the effect you are testing. For example, shuffling is only a good control if it actually breaks the relationship of interest. If possible, check the control design on separate data.

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

## Define the effect before looking at results

The metric says what counts as an effect. For next-token preference, `logit_difference(logits, correct_tokens, incorrect_tokens)` subtracts the alternative token's logit from the target token's logit. For another question, pass a `metric_fn` that returns one finite value per prompt, with shape `[B]`. Use the same score and outcome definition for clean, corrupted, and patched runs.

The normalized patching effect is the share of the clean-to-corrupted score gap recovered by replacing an activation:

$$ P = \frac{M_{\text{patched}} - M_{\text{corrupted}}}{M_{\text{clean}} - M_{\text{corrupted}}} $$

A value near 0 means little recovery; 1 means the full gap was recovered; above 1 means overshoot. The ratio is undefined if clean and corrupted scores are equal.

## Read the results

| Result | What it tells you |
| --- | --- |
| `raw_patching_effects` | Average normalized effect for each component |
| `per_prompt_effects` | How effects vary across prompts |
| `p_values` / `q_values` | Unadjusted / multiple-testing-adjusted values, when inference ran |
| `significant_mask` | Components passing the selected FDR threshold |
| `outlier_scores` | Descriptive ranking; not significance tests |
| `cli_score` | Whether absolute effects are concentrated in a few components; not significance or causality |

Treat selected components as candidates for follow-up patching or ablation. A result applies to the chosen metric, prompts, null, and correction method; it does not establish a general causal role.

## Run a published reference check

The published-reference test checks the Benjamini–Hochberg implementation against Example 3.2 of Benjamini and Hochberg (1995): at level 0.05, the same first four hypotheses should be rejected.

```bash
python -m pytest -q tests/test_published_benchmark.py
```

Benjamini, Y. and Hochberg, Y. (1995), “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing,” *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300. [DOI](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) · [Full text](https://www.stat.cmu.edu/~ryantibs/journalclub/benjamini_1995.pdf).

## Scope and limitations

This reference check validates one published multiple-testing example. It does not validate every way of generating p-values, every control design, or calibration across all models and datasets. Calibration depends on the metric, prompts, null model, and dependence assumptions. The example in [`example_usage.py`](example_usage.py) uses GPT-2 and one prompt, so it demonstrates the workflow but does not make significance claims. The CLI and outlier scores are descriptive summaries, not significance tests or evidence of causality.
