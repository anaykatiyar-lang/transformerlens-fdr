# Purpose and usage

TransformerLens-FDR is a research add-on for analyzing activation-patching experiments in TransformerLens. I built it to make component-level effects easier to compare and to provide an explicit route from those measurements to multiple-testing results when the experiment supports valid inference.

This guide explains the problem the package addresses, how to run an audit, how to select a metric and a null model, and what conclusions the outputs do and do not support.

## Why adjust for multiple testing?

An activation-patching study can test many attention heads, MLP layers, or residual-stream positions. Even if every component has no real effect, some measurements can look large by chance. A fixed magnitude cutoff or a top-*k* list does not account for the number of components searched.

False Discovery Rate (FDR) procedures adjust valid p-values across a family of tests. At level $\alpha$, the target is to control the expected fraction of false discoveries among the hypotheses rejected:

$$ \mathrm{FDR} = \mathbb{E}\left[\frac{V}{\max(R, 1)}\right] \leq \alpha $$

Here, $R$ is the number of rejected hypotheses and $V$ is the number of those rejections that are false. This is a long-run guarantee under the method's assumptions; it does not promise a particular false-discovery fraction in one analysis. FDR adjustment also cannot repair invalid p-values or an unsuitable null model.

## A practical workflow

1. **Define the behavior.** Choose a score that represents the model behavior you want to study. For next-token preference, the built-in `logit_difference` subtracts the alternative token's logit from the target token's logit for each prompt.
2. **Specify the comparison.** Prepare matched clean and corrupted prompts, decide the target and alternative outcomes before inspecting results, and use the same metric on clean, corrupted, and patched runs.
3. **Inspect effect estimates.** Run the example in [`example_usage.py`](example_usage.py). It loads GPT-2 and uses one prompt, so it is a descriptive demonstration; it does not estimate prompt-to-prompt variability or make significance claims.
4. **Choose whether inference is justified.** If you have a defensible no-effect comparison, select one of the null methods below. Otherwise leave the default method (`"none"`) and report the measurements as exploratory.
5. **Interpret discoveries as candidates.** An FDR-significant component is a candidate under the chosen metric, prompt set, null, and correction method. It is not proof that the component has a general causal role across tasks or prompts.

## Choosing a metric

The metric defines what “an effect” means in the analysis.

- For next-token preference, use `logit_difference(logits, correct_tokens, incorrect_tokens)`. A positive value means the model assigns a higher logit to the target token than to the alternative.
- For another research question, provide a `metric_fn` that returns exactly one finite value per prompt, with shape `[B]`.
- Use the same metric and outcome definition in every condition. Choose these before looking at which components rank highest.

The package checks the output shape and finiteness of the metric. It cannot determine whether the metric is a good measure of the scientific outcome.

## Choosing a null model

A **null model** describes the measurements expected if the effect under study were absent. The package does not invent a null from the observed components.

### No null: descriptive analysis

With the default `method="none"`, the audit returns patching effects and descriptive outlier scores. It leaves p-values, q-values, and the significance mask unset. Use this mode for exploration or when you cannot justify a no-effect comparison.

### Sign-flip test

Set `method="signflip"` to compare per-prompt effects after changing their signs. This requires independent prompt effects and a null distribution symmetric around zero. Prompt examples that are duplicated, strongly related, or systematically skewed can violate those assumptions. The attainable p-value resolution is `1 / (n_perm + 1)`; use enough permutations for the number of hypotheses being tested.

```python
results = auditor.run_patching_audit(
    clean_tokens=clean_tokens,
    corrupted_tokens=corrupted_tokens,
    correct_tokens=correct_tokens,
    incorrect_tokens=incorrect_tokens,
    method="signflip",
)
```

Use this only when the prompt sampling and effect distribution make the symmetry assumption credible. The code cannot verify that assumption from tensor shapes alone.

### Caller-supplied empirical null

An empirical null is built from control runs designed to represent the no-effect condition. Pass per-component control effects shaped `[n_controls, layers, components]` through `null_distribution`. Controls should reflect the specific relationship or behavior being tested. A shuffle is useful only if it truly removes that relationship. Where possible, validate the control design using separate data not used to build the null.

## Correcting across components

After p-values are computed, the auditor adjusts them across the tested family:

- **Benjamini–Hochberg (`fdr_bh`)** controls FDR under independent tests and certain positive-dependence conditions.
- **Benjamini–Yekutieli (`fdr_by`)** controls FDR under arbitrary dependence between tests when the individual p-values are valid. It is usually more conservative.

Neither correction method makes an invalid p-value valid. If the null is misspecified, or if the p-values are not calibrated, the corrected results are not trustworthy evidence.

## Reading the outputs

- `raw_patching_effects` are average normalized patching effects for each component.
- `per_prompt_effects` show how those effects vary across prompts.
- `p_values` are unadjusted results from the selected null method; `q_values` adjust for testing multiple components.
- `significant_mask` marks components passing the selected FDR threshold. It is only present when a null method was used.
- `outlier_scores` rank unusual effects for description. They are not p-values.
- `cli_score` is the Circuit Localization Index (CLI): a summary of whether absolute effects are concentrated in a few components or spread across many. It is not a significance test.
- `calibrated` and `null_description` report whether a null method ran and which method was selected; they do not independently establish that the assumptions hold.

## Mathematical definitions

The normalized patching effect measures the part of the clean-to-corrupted score gap recovered by replacing an activation:

$$ P = \frac{M_{\text{patched}} - M_{\text{corrupted}}}{M_{\text{clean}} - M_{\text{corrupted}}} $$

A value near $0$ indicates little recovery; $1$ indicates recovery of the clean–corrupted gap; values above $1$ indicate overshoot. The ratio is undefined when clean and corrupted scores are equal.

For the CLI, $w_i$ is component $i$'s share of the total absolute patching effect; it is not a statistical p-value:

$$ w_i = \frac{|P_i|}{\sum_j |P_j| + \epsilon} $$
$$ H = -\sum_i w_i \log_2(w_i + \epsilon) $$
$$ \mathrm{CLI} = 1 - \frac{H}{\log_2(N)} $$

The CLI describes concentration in the observed effect map. It does not show that the effects differ from a null or that the same components matter in another setting.

## Published reference check

The package's published-reference test uses the 15 ordered p-values in Example 3.2 of Benjamini and Hochberg (1995). At FDR level 0.05, it checks that the implementation rejects the same first four hypotheses as the paper's procedure.

Run it with:

```bash
python -m pytest -q tests/test_published_benchmark.py
```

Reference: Benjamini, Y. and Hochberg, Y. (1995), “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing,” *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300. [DOI: 10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x). [Full text](https://www.stat.cmu.edu/~ryantibs/journalclub/benjamini_1995.pdf).

## Scope and limitations

This reference check validates the multiple-testing adjustment on one published example. It does not validate every method of generating p-values, establish that a particular empirical null fits every task, or demonstrate calibration across all TransformerLens models and datasets. Calibration depends on the metric, prompt sampling, null model, and dependence assumptions. Report those choices with results, and treat the CLI and outlier scores as descriptive summaries rather than significance or causal evidence.
