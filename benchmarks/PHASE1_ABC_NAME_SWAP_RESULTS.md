# GPT-2 Small IOI: ABC versus name-swap corruption

**Full-run results — descriptive primary comparison and exploratory inference**

Protocol: [PHASE1_ABC_NAME_SWAP_PLAN.md](PHASE1_ABC_NAME_SWAP_PLAN.md)  
Executed research commit: `d216c21530eb27052f7711ce5a199f423f351fbd`  
Original Phase 0 reference commit: `6ffe7add646d030c890e07eddd40d8c2d8e041e3`

## Setup

GPT-2 Small; 12 layers × 12 attention heads; all-position clean-to-corrupted activation patching at `attn.hook_z`; original clean IO-minus-S logit difference as the metric in both conditions. The ABC baseline used 20 retained templates (120 prompts). The paired primary comparison used 16 selection templates (96 clean prompts paired across both corruptions); the remaining four templates (24 prompts) were used for descriptive evaluation. Sampling seed: 2026; six prompts per template; 9,999 Monte Carlo template-cluster sign-flip permutations; BH primary and BY sensitivity, each within a separate 144-head family per corruption at α = 0.05.

## Primary comparison: selection templates

| Measure | Result |
| --- | ---: |
| Spearman correlation, absolute unnormalized head effects | **0.6224** |
| Top-9 intersection / Jaccard | **6 heads / 0.5000** |
| Top-26 intersection / Jaccard | **17 heads / 0.4857** |
| Mean clean–corrupted metric gap: ABC | 3.0673 |
| Mean clean–corrupted metric gap: name-swap | 7.1335 |
| Fraction of positive prompt-level gaps: ABC / name-swap | 91.67% / 97.92% |

These strategies produce moderately associated yet nonidentical absolute head-effect rankings. Their different corruption semantics and gap magnitudes are important interpretive confounds. Unnormalized effects were recovered from the package's normalized `raw_patching_effects` by multiplying by the mean gap for the **same condition and prompt subset**.

## Illustrative head effects

Signed mean unnormalized effects, in original IO-minus-S logit-difference units:

| Head (layer.head) | ABC | Name-swap |
| --- | ---: | ---: |
| 9.9 | +2.516 | +1.524 |
| 10.7 | −1.589 | −2.764 |
| 11.10 | −0.805 | −1.723 |
| 9.6 | +1.052 | +0.233 |
| 4.11 | −0.005 | +0.836 |
| 8.10 | +0.189 | +0.792 |
| 0.4 | −0.203 | +0.074 |

Heads 9.9, 10.7 and 11.10 remain prominent in both settings. Head 4.11 stands out descriptively under name-swap. These values describe denoising effects under the specified corrupted contexts, not head necessity.

## Exploratory discovery sets

| Correction | ABC | Name-swap | Shared | ABC-only | Name-swap-only | Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BH | 28 | 49 | 19 | 9 | 30 | 0.3276 |
| BY | 21 | 37 | 15 | 6 | 22 | 0.3488 |

Name-swap yields more discoveries under both procedures; however, **the difference in counts is not a statistical test of differences between corruption effects**, and the two 144-head testing families were not pooled into a 288-test correction. Valid interpretation of head discoveries depends on template-group independence and sign symmetry; similarities among prompt templates make those assumptions uncertain.

In layers 0–6, the BH discovery counts were 6 (ABC) and 19 (name-swap); BY counts were 4 and 13. This is an **exploratory layer-pattern observation**, not a tested mechanistic conclusion.

## Descriptive four-template evaluation

| Output | ABC | Name-swap |
| --- | ---: | ---: |
| Selection-set BH heads used in joint patch | 28 | 49 |
| Evaluation mean clean–corrupted gap | 3.4075 | 6.5298 |
| Joint normalized restoration | 0.9579 | 1.0440 |
| Joint unnormalized patching effect | 3.2640 | 6.8173 |

The name-swap joint patch slightly exceeds its mean clean–corrupted gap; this is an overshoot of the measured logit-difference restoration, **not** a claim of superior accuracy. These evaluation templates already featured in earlier Phase 0 work, so the assessment is **not an independent confirmation** of the overall comparison. Mask sizes and corruption severity also differ.

## Verification and post-hoc sensitivity

Independent checks against the saved per-prompt NPZ arrays and per-head CSVs reproduced the signed and normalized head means, Spearman correlation, top-K rankings, BH/BY q-values and selection masks, and discovery-set intersections. All 120 clean/name-swap prompt substitutions were verified. The original 9,999-permutation results above are retained.

As an explicitly **post-hoc sensitivity check**, exhaustive two-sided sign-flip enumeration over all 32,768 distinct patterns for 16 template groups reproduced the reported BH counts of 28 and 49, and the ABC BY count of 21. Name-swap BY yielded 38 instead of 37, adding head 1.5 at the threshold. This does not replace the pre-specified Monte Carlo result or validate the sign-flip assumptions.

## Interpretation and limitations

The data support a descriptive **corruption-strategy dependence** of GPT-2 Small IOI activation-patching rankings and head-discovery sets. The overlap of prominent late-layer heads is accompanied by distinct effects in several earlier layers. There is no calibrated between-condition contrast test, no validation on previously unseen templates, and no evidence here that head-level denoising establishes necessity or reconstructs the published IOI path-patching circuit.

The archived full run contains `summary.json`, paired prompts, per-head CSVs, layer comparison, and normalized/unnormalized per-prompt arrays; its recorded code commit is `d216c21530eb27052f7711ce5a199f423f351fbd`. Python, PyTorch, NumPy, SciPy and Transformers versions are in `summary.json`; TransformerLens and project package versions were recorded as `unknown`. A separate derived review package includes a 144-head comparison and per-template head-effect table.

**Status:** Numerical results checked and documented. No upstream pull request has been opened, and the pre-specified plan remains unchanged.
