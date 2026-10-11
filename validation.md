# Validation results

## Published benchmark

The Benjamini–Hochberg implementation reproduces the 15-hypothesis example from Benjamini and Hochberg (1995): **4 of 15 hypotheses are rejected** at a target FDR of 0.05.

```bash
python -m pytest -q tests/test_published_benchmark.py
```

This verifies the correction arithmetic against a published example. It is not a model experiment or a calibration result for model-generated p-values.

## Calibration simulations

The simulation report used seed `20261009`, 1,000 repetitions per main setting, 30 prompts per repetition, 12 tested components, 999 sign flips, and Benjamini–Yekutieli at 0.05. No TransformerLens model or downloaded weights were used.

| Symmetric null | KS distance from uniform | KS p-value |
| --- | ---: | ---: |
| Normal | 0.031 | 0.286 |
| Heavy-tailed *t* (3 df) | 0.018 | 0.896 |
| Positively correlated normal effects | 0.032 | 0.252 |
| Symmetric mixture | 0.030 | 0.323 |

The tests did not detect non-uniformity in these simulations; this does not prove uniformity in all settings.

| True-null share | Observed FDR | Monte Carlo SE | Power |
| ---: | ---: | ---: | ---: |
| 50% | 0.79% | 0.11 percentage points | 99.32% |
| 90% | 0.89% | 0.23 percentage points | 91.10% |
| 100% | 0.70% | 0.26 percentage points | Not applicable |

In the all-null simulation, at least one rejection occurred in 0.70% of repetitions against the nominal 5% target. These results apply only to the specified simulation design and test family.

Mean-zero skew diagnostics used 250 repetitions per setting and showed rejection rates from 0% to 2.4%. They are diagnostics, **not validation**: sign-flip inference requires symmetry about zero.

## Implementation checks

- Calibration tests: **6 passed**.
- Published benchmark test: **1 passed**.
- The full calibration report completed and produced the results above.

Run the recorded checks with:

```bash
python -m pytest -q tests/test_published_benchmark.py
python -m pytest -q tests/test_calibration.py
python -m tests.test_calibration
```

## What this evidence supports

These checks support the implementation of the tested procedures. They do not establish that a chosen null is appropriate for a research dataset, validate every TransformerLens model or hook, or guarantee calibration for arbitrary prompt dependence. Sign-flip results require independent prompt effects symmetric about zero; the FDR correction also requires valid input p-values.
