# Validation results

## Validation

> **Published benchmark:** the BH calculation reproduces the 15-hypothesis example in Benjamini and Hochberg (1995): **4 of 15 hypotheses are rejected** at a target FDR of 0.05.

```bash
python -m pytest -q tests/test_published_benchmark.py
```

The rest of the test suite checks the implementation:

- **Sign-flip math:** compared with exact results on small examples; simulations check symmetric nulls.
- **Known-answer examples:** toy models check whether planted component effects are recovered.
- **TransformerLens 4 integration:** native bridge tests check hook integration. An optional GPT-2 test uses cached model weights.
- **Skewed data:** reported as a diagnostic, because sign-flip validity requires symmetry about zero.

```bash
python -m pytest -q tests/test_stress.py tests/test_calibration.py tests/test_bridge_native.py
python -m tests.test_calibration   # longer calibration report
```

These checks support the software; they do not show that a null is right for your dataset or that results are calibrated for every model.

### Calibration results from this repository

I ran the full calibration report with seed `20261009`: 1,000 simulated repetitions per main setting, 30 prompts per repetition, 12 tested components, 999 sign flips, and Benjamini–Yekutieli (BY) at 0.05. No TransformerLens model or downloaded weights were used.

| Symmetric null simulation | KS distance from uniform | KS p-value |
| --- | ---: | ---: |
| Normal | 0.031 | 0.286 |
| Heavy-tailed *t* (3 df) | 0.018 | 0.896 |
| Positively correlated normal effects | 0.032 | 0.252 |
| Symmetric mixture | 0.030 | 0.323 |

The KS tests did not detect departures from uniformity in these runs. This is a check on generated data, not proof of uniform p-values in every setting.

| True-null share | Observed false-discovery rate | Monte Carlo SE | Power |
| ---: | ---: | ---: | ---: |
| 50% | 0.79% | 0.11 percentage points | 99.32% |
| 90% | 0.89% | 0.23 percentage points | 91.10% |
| 100% | 0.70% | 0.26 percentage points | Not applicable (no planted effects) |

With all 12 hypotheses null, at least one rejection occurred in 0.70% of repetitions (nominal target: 5%). These results apply to the simulation design above; they do not establish calibration for real model prompts or arbitrary component dependence.

Mean-zero skew diagnostics used 250 repetitions per setting. Their observed rejection rates ranged from 0% to 2.4%. These are non-gating diagnostics and do not validate the sign-flip test for skewed data, where its symmetry assumption is violated.

### Test results

- Calibration test file: **6 passed**.
- Published BH reference test: **1 passed**.
- Full simulation report: completed and printed the results above.

Reproduce them with:

```bash
python -m pytest -q tests/test_published_benchmark.py
python -m pytest -q tests/test_calibration.py
python -m tests.test_calibration
```

---
