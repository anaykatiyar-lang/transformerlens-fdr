# Phase 0 plan: reference-recovery test on GPT-2 small IOI

**Freeze this plan and `ioi_gpt2_small.py` (commit both, record the hash) before the smoke
run or the full run.** Everything under "Fixed in advance" and "Pre-registered criteria" must
not change after you see results. If you must deviate, add a dated "Deviations" section at
the bottom and say why; do not edit the criteria in place. Keep smoke results separate
(the script writes them to `<out>_smoke`).

## Question

Does the FDR audit recover known structure in a real model, and do its p-values behave
sensibly on real activations rather than simulated ones?

This is a **reference-recovery test, not a reproduction of the published experiment.**
Wang et al. identified heads with specific path-patching analyses. This script patches each
head's `attn.hook_z` at **all positions** from a clean prompt into an ABC-corrupted prompt, a
different intervention. A miss against the published circuit therefore does not contradict
the paper, and the paper's role labels do not guarantee the sign of each head's effect here.

## Fixed in advance

| Item | Choice |
|---|---|
| Model | GPT-2 small via `TransformerBridge.boot_transformers("gpt2")` + `enable_compatibility_mode()` |
| Components | 144 attention heads (12 x 12), `hook_type="attn_head"` (`attn.hook_z`, all positions) |
| Direction | Denoising: patch clean activations into the corrupted run |
| Metric | Logit difference, IO name minus S name, last position |
| Corrupted prompts | ABC: all three name slots replaced by three new distinct names |
| Prompts | 32 templates (in the script); keep those whose token length fits the padding window (up to 3 left-padded BOS tokens); 6 prompts per template; random names/place/object; ABBA and BABA orders |
| Cluster unit | Template (`group_ids` = template index) |
| Level / test | alpha 0.05, BH (primary) and BY (secondary), `n_perm=9999` |
| Reference set | The 26 published heads in the script. A reviewer reports the list matches the head set in Wang et al., Appendix K; it was originally written from memory, so check it yourself before freezing. |
| Seeds | 2026 (main), derived seeds for placebo and subsets |

Run order: `--dry-run` (tokenizer only, no model weights; may download tokenizer files),
`--smoke` (pipeline check, separate output directory), then the full run once.

## Pre-registered criteria

Statuses are PASS, FAIL, INCONCLUSIVE (needed evidence missing) or INFO (non-gating).

Design (must hold or the run is not interpretable):
- **A1** at least 14 templates retained. At 14 the exact two-sided floor is 1/8192 = 1.2e-4, below the first BH cutoff for 144 tests (alpha/144 = 3.5e-4), so a single head at the floor can be rejected. The 14 requirement is a conservative margin, not a hard limit: with fewer templates a rejection still happens if enough heads reach the floor together (13 templates: 1 head; 12: 2; 10: 6; 8: 23). The run reports the actual minimum rank from the preflight resolution report.
- **A2** at least 80 prompts.
- **A3** the preflight clean-corrupted gap is usable and at least 90% of prompts have a positive gap (clean metric above corrupted). The shared-denominator normalization is only meaningful if the gap is stable; the script reports gap mean, SD and the positive fraction.

Recovery (cluster sign-flip, BH, alpha 0.05). These are **pre-registered hypotheses about this
intervention**, not claims about the paper. A failure means the hypothesis was not supported
here.
- **B1** all 3 name-mover heads (9.9, 9.6, 10.0) are discovered. They have the largest direct effect on the metric; if the method cannot find them under this setup, treat that as informative about the setup and the method alike.
- **B2** among discoveries inside the published circuit, at least 90% have the expected sign (positive, except negative name movers 10.7 and 11.10, expected negative). INCONCLUSIVE if there are no in-circuit discoveries.

Negative control (placebo: patching between two independent ABC samples). This is a
**gross-failure detector, not a calibration proof.**
- **C1** at most 3 of 20 repetitions produce any BH rejection (cluster sign-flip), and **all 20 repetitions must complete**; otherwise C1 is INCONCLUSIVE. Under a true global null BH rejects something in at most 5% of repetitions; 3 of 20 is about the 98th percentile. Even 0 of 20 leaves roughly a 14% upper 95% bound on the true rejection rate, so a pass can only rule out an obvious problem. If you have a GPU, raise `--placebo-reps` (for example 100) before freezing.
- Interpretation of a failure: either the test is miscalibrated, or the placebo effects are not symmetric about zero (patching between independently sampled ABC prompts does not by itself guarantee that). The two cannot be separated from this experiment alone.

Stability (**non-gating**, report only):
- **D1** median Jaccard of the discovery set under leave-two-templates-out. 0.5 is a rough reference, not a pass/fail line.

## Report-only analyses

- Per-prompt vs cluster sign-flip: discoveries and placebo rejections side by side (the real-data version of the simulation result).
- ICC of per-prompt effects within templates (median over the top-10 heads and over all heads). High ICC means the cluster test matters; near zero means prompt-level and cluster results should agree.
- **Subsampling curve:** discoveries when this dataset is thinned to fewer templates (6 to 24). This reuses the same measured effects, so it shows how discoveries depend on template count here; it is not a power estimate from repeated experiments.
- BH vs BY.
- Every discovery outside the published circuit, by layer and head.
- Descriptive top-9 and top-26 by absolute effect, as the baseline the inference layer has to beat.

## Outcomes and what they mean

- **All gating criteria pass:** on one model, one task, one corruption, the audit recovers the strongest known structure and a coarse negative control behaves. Proceed to Phase 1 and state exactly that scope.
- **B1 fails, C1 passes:** calibrated but underpowered, or single-head all-position denoising is weaker than expected because of redundancy. Check the gap, padding and signs. Report honestly; consider path patching as a better effect definition. Do not lower the criterion afterwards.
- **C1 fails:** the placebo contradicts either the test's calibration or the symmetry of placebo effects. Inspect which heads reject, and investigate before building on this.
- **C1 INCONCLUSIVE:** repetitions were skipped (for example a degenerate gap); fix the cause and rerun the placebo.
- **A1 fails:** add templates before the run (not after) or raise `--max-pad` and say so.

## Threats to validity

- The published circuit is a reference, not ground truth. Discoveries outside it are not necessarily false, and misses inside it may reflect redundancy (backup heads hide each other when patched one at a time).
- Left-padding shorter templates with BOS tokens shifts absolute positions. The script prints the clean logit difference and warns below 1.0, but the gap diagnostic (A3) is the relevant normalization check.
- Patching `attn.hook_z` at all positions is a coarse intervention compared with position- or path-specific patching.
- The placebo is a negative-control diagnostic only.
- 32 hand-written templates from one family is a narrow design; independence across templates is itself an assumption.
- Compatibility mode folds LayerNorm; results may differ slightly from native Hugging Face numerics.

## Protocol

1. Check `PUBLISHED_CIRCUIT` in the script against the paper.
2. Freeze: commit this plan and the script, record the hash.
3. `python benchmarks/ioi_gpt2_small.py --dry-run`
4. `python benchmarks/ioi_gpt2_small.py --smoke` (writes to a `_smoke` directory)
5. `python benchmarks/ioi_gpt2_small.py --out benchmarks/results/ioi` (once)
6. Commit `ioi_phase0.json`, `.md` and the heatmap unedited.

Rough cost (an estimate, not measured): GPT-2 small, about 100 prompts, 36 batched forward
passes per sweep; seconds per sweep on a GPU, minutes on a CPU. The run does one main sweep
plus the placebo sweeps.

## Deviations

- 2026-10-19 (after the first full run): the script was changed only to also write `per_head.csv`
  (effect, role, expected sign, p and q values and discovery flags per head) and
  `per_prompt_effects.npz`. This is output only. Criteria, analyses, seeds and thresholds are
  unchanged, and the first run's results stand as recorded (including the B2 FAIL). Rerunning
  with the same seed should reproduce the first run's counts; `--placebo-reps 0` reruns the
  sweep alone. Anything learned from the new files is exploratory, not pre-registered.
