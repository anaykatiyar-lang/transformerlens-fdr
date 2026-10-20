# ABC versus name-swap corruption in GPT-2 Small IOI

**Pre-specified analysis plan**  
[Research discussion, Issue #1](https://github.com/anaykatiyar-lang/transformerlens-fdr/issues/1)

## Research question

How does the choice of corruption strategy affect attention-head activation-patching results on the indirect object identification (IOI) task in GPT-2 Small? We compare the benchmark's ABC corruption with a global swap of the two names. Because these interventions alter different features of the prompt, differences in their results will be interpreted in the context of the respective corruptions. Layer-specific patterns, including possible early-layer sensitivity, are exploratory.

## Experimental setup

The comparison follows `benchmarks/ioi_gpt2_small.py` at source commit `6ffe7add646d030c890e07eddd40d8c2d8e041e3`.

| Setting | Protocol |
| --- | --- |
| Model | GPT-2 Small; `TransformerBridge.boot_transformers("gpt2")` with compatibility mode |
| Components | All 144 attention heads (12 layers × 12 heads) |
| Intervention | Denoising: patch clean activations into corrupted runs at `attn.hook_z`, all positions, one head at a time |
| Metric | Final-position logit difference: original clean IO name minus original clean S name |
| Prompts | Existing templates, ABBA/BABA orders, and eligible single-token names, places, and objects |
| Sampling | Six prompts per retained template; seed `2026` |
| Statistical settings | Template-cluster sign-flip; 9,999 permutations; BH primary, BY sensitivity; α = 0.05 |

Generate both corruption conditions from the **same clean examples**, keeping names, template, order, place, and object matched:

- **ABC corruption:** Replace the three name occurrences with three different names, as in the existing benchmark.
- **Name-swap corruption:** Exchange the original IO and S names at every occurrence. For example, `John and Mary ... Mary gave ...` becomes `Mary and John ... John gave ...`. This preserves the repetition pattern while reversing the expected completion.

Use the **original clean IO and S tokens** as the positive and negative metric labels for both conditions. Check the slot substitutions, name repetition, ABBA/BABA cases, token lengths, and alignment of clean and corrupted sequences under the benchmark's left-padding scheme before collecting patching results.

## Template split

The tokenizer-only check (`python benchmarks/ioi_gpt2_small.py --dry-run`) retained 20 of 32 templates, with a target length of 16 tokens excluding BOS and at most three left-padded BOS tokens. It identified 99 eligible names, 17 places, and 16 objects.

All indices below are zero-based positions in `TEMPLATES` in the source benchmark.

| Group | Template indices |
| --- | --- |
| Retained (20) | `0, 1, 3, 7, 9, 10, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 31` |
| Selection (16) | `0, 1, 3, 9, 10, 14, 15, 18, 20, 21, 22, 23, 24, 25, 27, 31` |
| Evaluation (4) | `7, 13, 19, 29` |

To reproduce the split, shuffle the sorted retained indices using `random.Random(20261019).shuffle(...)`, assign the first 16 to selection and the remaining four to evaluation, and sort each group for reporting. Keep all six examples belonging to a template in the same group; use the same split for both corruptions.

The original ABC Phase 0 analysis has already used all 20 templates. The four-template subset therefore serves as a **descriptive evaluation**, not independent confirmation of the overall comparison. Independent validation would require new templates.

## Effect measures

Let `M` denote the IO-minus-S logit difference. For head `h` and corruption condition `c`, define the mean **unnormalized patching effect** on the relevant prompt subset as

`D[h,c] = mean(M(patched[h,c]) - M(corrupted[c]))`.

Define the corresponding mean clean–corrupted gap as

`G[c] = mean(M(clean) - M(corrupted[c]))`.

When `G[c]` is usable, the normalized effect is `N[h,c] = D[h,c] / G[c]`. The package field `raw_patching_effects` already contains this normalized value; recover `D[h,c]` by multiplying it by the gap measured on the **same condition and prompt subset**.

As the denominator is shared across heads within a condition, nonzero normalization leaves absolute-effect rankings unchanged. Report each gap's mean, standard deviation, sign, and fraction of positive prompt-level gaps. If a gap is unusable, record the failed preflight rather than forcing a normalized audit.

## Primary analysis

On the **16 selection templates**:

1. Rank all 144 heads by the absolute value of `D[h,c]` for each corruption. For top-K selection, break exact ties by ascending layer and head index.
2. Compare the absolute-effect rankings with Spearman's rank correlation, assigning average ranks to ties.
3. Compute Jaccard overlap of the top-9 and top-26 head sets.
4. Report the signed unnormalized effects and the normalized effects where defined, alongside the clean–corrupted gap diagnostics.

## Exploratory analysis

- **Statistical discoveries:** Apply template-cluster sign-flip testing separately to each condition on the 16 selection templates. Apply BH and BY **within each condition's 144-head family**, with BH primary and BY as a sensitivity analysis. Compare discovery sets descriptively. These corrections do not constitute a combined 288-test FDR analysis or a statistical test of differences between conditions.
- **Inference assumptions:** Assess the plausibility of independence and sign symmetry at the template-group level, and report p-value resolution. Without a defensible null, report the results as exploratory rather than claiming calibrated FDR control.
- **Layer patterns:** Inspect layer-wise effect differences, including early layers, without assuming a particular outcome.
- **Evaluation templates:** Run `run_heldout_validation` for each corruption, selecting heads on the 16-template group using cluster sign-flip with BH and evaluating the resulting mask's joint patching effect on the four-template group. Report empty selections and unusable gaps where applicable. This is a descriptive assessment because the templates were previously used in Phase 0.

The all-position denoising intervention measures restoration of the chosen metric in the corrupted context; it does not establish that a head is necessary or reproduce the published IOI path-patching experiment.

## Implementation and records

Keep `benchmarks/PHASE0_PLAN.md` and the original results unchanged. Reproduce the existing ABC baseline before running the new comparison. Implement the comparison separately from the original Phase 0 analysis, and add tests for paired prompt construction, swapped IO/S labels, ABBA/BABA variants, token alignment, fixed template splitting, separate correction families, and evaluation with empty and nonempty selected-head masks.

Record the source and analysis commit hashes, model and library versions, seeds, template assignments, per-head and per-template effects, baseline gaps, diagnostics, and deviations. Store smoke-test results separately. Report the primary comparison and exploratory findings separately, including null or inconclusive outcomes.

## Deviations

- **2026-10-20, post-run provenance:** After the full model-scored comparison had completed and its results had been inspected, the installed TransformerLens and `transformerlens-fdr` versions were checked in the Colab environment. The original run summary recorded both as `unknown`; the subsequent metadata check returned `4.2.0` and `0.1.0`. This was a departure from the plan to capture all library versions with the run. No prompts, model outputs, pre-specified settings, or inference results were changed. The original run's package provenance remains incomplete because continuity with the later Colab environment cannot be established from the saved records. The updated runner now captures the installed distribution versions before loading the model and stores the capture stage in the output. A fresh run would be required to verify the complete run-time environment; the original findings are retained as recorded.
