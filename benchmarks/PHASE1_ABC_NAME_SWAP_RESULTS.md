# ABC versus name-swap corruption in GPT-2 Small IOI

## Study setup

Protocol: [PHASE1_ABC_NAME_SWAP_PLAN.md](PHASE1_ABC_NAME_SWAP_PLAN.md)  
Analysis code at run time: `d216c21530eb27052f7711ce5a199f423f351fbd`  
Phase 0 benchmark source: `6ffe7add646d030c890e07eddd40d8c2d8e041e3`

The experiment used GPT-2 Small (12 layers, 12 attention heads per layer), with clean-to-corrupted activation patching at `attn.hook_z` across all positions. The outcome was the final-token logit difference between the original clean indirect-object (IO) and subject (S) names, with the same answer-token labels for both corruption strategies.

The ABC baseline included 120 prompts from 20 templates. The primary comparison used 96 paired prompts from 16 selection templates; four additional templates (24 prompts) were reserved for descriptive evaluation. Each retained template contributed six prompts. Prompt sampling used seed 2026. Per-condition exploratory inference used template-cluster sign-flip testing with 9,999 permutations, followed by BH and BY adjustments separately for each set of 144 heads at α = 0.05.

## Primary comparison

| Measure | Result |
| --- | ---: |
| Spearman correlation of absolute, mean unnormalized head effects | 0.6224 |
| Top-9 head intersection / Jaccard | 6 heads / 0.5000 |
| Top-26 head intersection / Jaccard | 17 heads / 0.4857 |
| Mean clean-corrupted logit gap, ABC | 3.0673 |
| Mean clean-corrupted logit gap, name-swap | 7.1335 |
| Positive prompt-level gaps, ABC / name-swap | 91.67% / 97.92% |

The head-effect rankings show moderate positive agreement, with different members in the highest-ranked sets. Name-swap also produces a larger clean-corrupted gap. Mean unnormalized effects were computed by multiplying the library's normalized `raw_patching_effects` values by the mean gap from the same condition and prompt subset.

### Selected head effects

Values are signed, unnormalized changes in the original IO-minus-S logit difference.

| Head (layer.head) | ABC | Name-swap |
| --- | ---: | ---: |
| 9.9 | +2.516 | +1.524 |
| 10.7 | −1.589 | −2.764 |
| 11.10 | −0.805 | −1.723 |
| 9.6 | +1.052 | +0.233 |
| 4.11 | −0.005 | +0.836 |
| 8.10 | +0.189 | +0.792 |
| 0.4 | −0.203 | +0.074 |

Heads 9.9, 10.7, and 11.10 have relatively large effects under both strategies. Head 4.11 has a larger positive effect under name-swap than under ABC.

## Exploratory head discoveries

| Adjustment | ABC | Name-swap | Shared | ABC only | Name-swap only | Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BH | 28 | 49 | 19 | 9 | 30 | 0.3276 |
| BY | 21 | 37 | 15 | 6 | 22 | 0.3488 |

Name-swap produces more discoveries under both corrections. On layers 0–6, the BH counts are 6 under ABC and 19 under name-swap (BY: 4 and 13). The concentration of several name-swap-only discoveries in layers 3–6 is an exploratory observation. These discovery counts do not provide a formal test of between-strategy effect differences.

## Descriptive evaluation on four templates

| Outcome | ABC | Name-swap |
| --- | ---: | ---: |
| Selection-set BH heads used in joint patching | 28 | 49 |
| Evaluation clean-corrupted logit gap | 3.4075 | 6.5298 |
| Joint normalized restoration | 0.9579 | 1.0440 |
| Joint unnormalized patching effect | 3.2640 | 6.8173 |

Both selected masks restore most of the corresponding mean logit gap. The name-swap normalized restoration exceeds one slightly, indicating an overshoot of that metric rather than an improvement in predictive accuracy. These four templates were included in the earlier Phase 0 work, so this evaluation is not independent validation of the overall comparison.

## Numerical checks

The saved per-prompt effects and per-head tables reproduce the normalized and unnormalized means, Spearman correlation, top-K sets, discovery-set overlaps, and BH/BY adjusted values. All 120 paired prompt substitutions were checked against the intended name-swap rule, and no nonfinite effect arrays were found.

An additional post-hoc sensitivity calculation enumerated all 32,768 distinct two-sided sign patterns across the 16 selection-template groups. It recovered 28 ABC and 49 name-swap BH discoveries, matching the planned Monte Carlo run. ABC BY remained at 21 discoveries; name-swap BY increased from 37 to 38, with head 1.5 crossing the adjustment threshold. The pre-specified 9,999-permutation results are reported above without replacement.

## Interpretation and limitations

Within this experiment, the corruption strategy affects which GPT-2 Small IOI heads receive large activation-patching effects, although several late-layer heads rank highly under both conditions. The two corruptions change different aspects of the prompt and have different clean-corrupted gaps, limiting a direct mechanistic comparison.

The sign-flip tests rely on independence and sign symmetry of template-level effects, assumptions that are difficult to establish with related prompt templates. No calibrated between-strategy contrast test was run. The four evaluation templates do not supply independent confirmation of the overall comparison. Finally, clean-to-corrupted activation restoration does not establish head necessity or replicate the published IOI path-patching circuit.

The full-run archive includes `summary.json`, paired prompts, head-level tables, a layer comparison, and normalized and unnormalized per-prompt effects. Its run metadata records code commit `d216c21530eb27052f7711ce5a199f423f351fbd` and available library versions; TransformerLens and the project package versions were recorded as `unknown`.
