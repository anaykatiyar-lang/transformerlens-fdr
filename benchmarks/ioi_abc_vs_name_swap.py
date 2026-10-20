"""Compare ABC and name-swap corruptions on GPT-2 Small IOI.

The pre-specified settings are in benchmarks/PHASE0B_ABC_NAME_SWAP_PLAN.md.
The original Phase 0 benchmark and results are left unchanged.

    python benchmarks/ioi_abc_vs_name_swap.py --dry-run
    python benchmarks/ioi_abc_vs_name_swap.py --smoke
    python benchmarks/ioi_abc_vs_name_swap.py

The default run first audits ABC on all 20 retained templates, then compares
both corruptions on the frozen 16-template selection set and descriptively
assesses selected heads on the four evaluation templates.
"""

import argparse
import csv
import json
from importlib.metadata import PackageNotFoundError, version as installed_version
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.ioi_gpt2_small import (  # noqa: E402
    NAMES, OBJECTS, PLACES, TEMPLATES, choose_window,
    render, single_token_filter, tokenize, validate_templates,
)
from benchmarks.ioi_corruption_pairs import (  # noqa: E402
    RETAINED_TEMPLATES, build_paired_examples, condition_batch,
    fixed_template_split, validate_token_alignment,
)


SEED = 2026
ALPHA = 0.05
N_PERM = 9999
PER_TEMPLATE = 6
MAX_PAD = 3


def package_version(distribution_name):
    """Read the installed distribution version, including editable installs."""
    try:
        return installed_version(distribution_name)
    except PackageNotFoundError:
        return "unknown"


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def select_examples(examples, template_ids):
    """Retain complete template groups without resampling clean prompts."""
    selected = set(template_ids)
    result = [e for e in examples if e["template"] in selected]
    if {e["template"] for e in result} != selected:
        raise ValueError("Missing one or more templates in the paired examples.")
    return result


def rank_heads(unnormalized, k):
    """Top-K absolute effects; exact ties resolved by layer and head index."""
    effects = np.asarray(unnormalized, dtype=np.float64)
    if effects.ndim != 2 or not np.isfinite(effects).all():
        raise ValueError("Head effects must be a finite two-dimensional array.")
    if not (1 <= k <= effects.size):
        raise ValueError("Invalid top-K head count.")
    heads = [(layer, head) for layer in range(effects.shape[0])
             for head in range(effects.shape[1])]
    return sorted(heads, key=lambda pair: (-abs(effects[pair]), pair[0], pair[1]))[:k]


def compare_effects(abc, swapped):
    """Absolute-effect Spearman correlation and pre-specified Jaccard scores."""
    a = np.asarray(abc, dtype=np.float64)
    b = np.asarray(swapped, dtype=np.float64)
    if a.shape != (12, 12) or b.shape != a.shape:
        raise ValueError("Expected two 12 x 12 GPT-2 Small head-effect arrays.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Comparison requires finite patching effects.")
    # Spearman is undefined for a constant ranking.
    if np.ptp(np.abs(a)) == 0 or np.ptp(np.abs(b)) == 0:
        rho = None
    else:
        coefficient = float(spearmanr(np.abs(a).ravel(), np.abs(b).ravel()).statistic)
        rho = coefficient if np.isfinite(coefficient) else None
    result = {"spearman_absolute_effects": rho}
    for k in (9, 26):
        left = set(rank_heads(a, k))
        right = set(rank_heads(b, k))
        result[f"jaccard_top_{k}"] = len(left & right) / len(left | right)
        result[f"top_{k}_abc"] = [list(head) for head in rank_heads(a, k)]
        result[f"top_{k}_name_swap"] = [list(head) for head in rank_heads(b, k)]
    return result


def write_layer_table(path, abc, swapped):
    """Exploratory layer-level summaries; no layer-wise hypothesis tests."""
    abc, swapped = np.asarray(abc), np.asarray(swapped)
    if abc.shape != (12, 12) or swapped.shape != (12, 12):
        raise ValueError("Expected effects for 12 layers and 12 heads.")
    with Path(path).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=[
            "layer", "abc_signed_mean", "name_swap_signed_mean",
            "abc_abs_mean", "name_swap_abs_mean", "abs_mean_difference",
        ])
        writer.writeheader()
        for layer in range(12):
            writer.writerow({
                "layer": layer,
                "abc_signed_mean": float(abc[layer].mean()),
                "name_swap_signed_mean": float(swapped[layer].mean()),
                "abc_abs_mean": float(np.abs(abc[layer]).mean()),
                "name_swap_abs_mean": float(np.abs(swapped[layer]).mean()),
                "abs_mean_difference": float(np.abs(swapped[layer]).mean() - np.abs(abc[layer]).mean()),
            })


def fdr_families(p_values, alpha=ALPHA):
    """Apply BH and BY to one condition's 144-head family, never pooled."""
    if tuple(p_values.shape) != (12, 12):
        raise ValueError("Each correction family must contain exactly 144 heads.")
    from transformerlens_fdr.stats import apply_fdr_adjustment
    results = {}
    for name, method in (("bh", "fdr_bh"), ("by", "fdr_by")):
        q_values, mask = apply_fdr_adjustment(p_values, alpha=alpha, method=method)
        results[name] = (q_values.detach().cpu().numpy(), mask.detach().cpu().numpy())
    return results


def compare_discoveries(left, right):
    """Descriptive overlap of BH/BY discovery sets across corruptions."""
    comparison = {}
    for method in ("bh", "by"):
        a = set(map(tuple, np.argwhere(left[method][1])))
        b = set(map(tuple, np.argwhere(right[method][1])))
        union = a | b
        comparison[method] = {
            "jaccard": len(a & b) / len(union) if union else None,
            "shared_heads": [[int(value) for value in head] for head in sorted(a & b)],
            "abc_only": [[int(value) for value in head] for head in sorted(a - b)],
            "name_swap_only": [[int(value) for value in head] for head in sorted(b - a)],
        }
    return comparison


def gap_diagnostics(model, inputs):
    """Measure clean–corrupted gaps with the same labels as the patching audit."""
    import torch
    from transformerlens_fdr import logit_difference
    clean, corrupted, correct, incorrect, _ = inputs
    with torch.no_grad():
        clean_scores = logit_difference(model(clean), correct, incorrect)
        corrupted_scores = logit_difference(model(corrupted), correct, incorrect)
    gaps = (clean_scores - corrupted_scores).detach().cpu().numpy()
    return {
        "mean": float(np.mean(gaps)),
        "std": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else None,
        "fraction_positive": float(np.mean(gaps > 0)),
        "minimum": float(np.min(gaps)),
        "maximum": float(np.max(gaps)),
        "n_prompts": len(gaps),
    }


def write_head_table(path, normalized, gap_mean, p_values, corrected):
    """Write the complete per-head results for one condition and prompt set."""
    normalized = np.asarray(normalized, dtype=float)
    if normalized.shape != (12, 12):
        raise ValueError("Expected normalized effects for 144 attention heads.")
    raw = normalized * gap_mean
    p = p_values.detach().cpu().numpy()
    rows = []
    for layer in range(12):
        for head in range(12):
            rows.append({
                "layer": layer,
                "head": head,
                "unnormalized_effect": float(raw[layer, head]),
                "normalized_effect": float(normalized[layer, head]),
                "p_cluster": float(p[layer, head]),
                "q_bh": float(corrected["bh"][0][layer, head]),
                "selected_bh": bool(corrected["bh"][1][layer, head]),
                "q_by": float(corrected["by"][0][layer, head]),
                "selected_by": bool(corrected["by"][1][layer, head]),
            })
    with Path(path).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return raw


def dry_run(model_name="gpt2"):
    """Check the frozen template set without downloading model weights."""
    from transformers import AutoTokenizer
    validate_templates(TEMPLATES)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    usable = lambda words: [
        word for word in words
        if len(tokenizer.encode(" " + word, add_special_tokens=False)) == 1
    ]
    names, places, objects = usable(NAMES), usable(PLACES), usable(OBJECTS)
    if not names or not places or not objects:
        raise ValueError("No eligible single-token names, places, or objects.")
    lengths = {
        i: len(tokenizer.encode(
            render(t, "ABBA", ["John", "Mary", "Mary"], places[0], objects[0]),
            add_special_tokens=False,
        ))
        for i, t in enumerate(TEMPLATES)
    }
    target, retained = choose_window(lengths, MAX_PAD)
    selection, evaluation = fixed_template_split(retained)
    print(f"Tokenizer check: target={target} excluding BOS; {len(retained)}/32 templates")
    print("Selection:", selection)
    print("Evaluation:", evaluation)
    print(f"Single-token words: {len(names)} names, {len(places)} places, {len(objects)} objects")
    return retained


def prepare_prompts(model, per_template):
    """Sample once across all 20 templates, then partition by frozen IDs."""
    names = single_token_filter(model, NAMES)
    places = single_token_filter(model, PLACES)
    objects = single_token_filter(model, OBJECTS)
    if len(names) < 12 or not places or not objects:
        raise ValueError("Insufficient eligible single-token words for the benchmark.")
    probes = {
        i: len(tokenize(model, render(t, "ABBA", ["John", "Mary", "Mary"], places[0], objects[0])))
        for i, t in enumerate(TEMPLATES)
    }
    target, retained = choose_window(probes, MAX_PAD)
    selection, evaluation = fixed_template_split(retained)
    examples = build_paired_examples(
        retained, TEMPLATES, per_template, names, places, objects, random.Random(SEED)
    )
    if len(examples) != len(RETAINED_TEMPLATES) * per_template:
        raise AssertionError("Incorrect number of paired examples.")
    validate_token_alignment(model, examples, target, MAX_PAD)
    return target, examples, selection, evaluation


def audit_one(model, auditor, inputs, output_dir, condition, tag):
    """Preflight, denoising sweep, and separate BH/BY summaries."""
    _, _, _, _, groups = inputs
    preflight = auditor.check_setup(
        *inputs[:4], method="signflip", group_ids=groups
    )
    diagnostics = gap_diagnostics(model, inputs)
    record = {
        "condition": condition,
        "subset": tag,
        "templates": sorted(set(groups)),
        "n_prompts": len(groups),
        "gap": diagnostics,
        "gap_usable": bool(preflight.gap_usable),
        "relative_gap_tolerance": preflight.relative_gap_tolerance,
        "minimum_attainable_p": preflight.resolution_report["p_min"] if preflight.resolution_report else None,
        "status": "complete" if preflight.gap_usable else "unusable_gap",
    }
    print(f"{condition} {tag}: {preflight.format_summary()}", flush=True)
    if not preflight.gap_usable:
        return record, None, None, None

    import torch
    # Use the same Monte Carlo sign patterns for each condition and the
    # subsequent held-out selection audit.
    torch.manual_seed(SEED)
    with torch.no_grad():
        result = auditor.run_patching_audit(
            *inputs[:4], method="signflip", group_ids=groups
        )
    if result.p_values is None:
        raise RuntimeError("Cluster sign-flip audit did not yield p-values.")
    corrected = fdr_families(result.p_values)
    normalized = result.raw_patching_effects.detach().cpu().numpy()
    raw = write_head_table(
        output_dir / f"{tag}_{condition}_heads.csv",
        normalized, diagnostics["mean"], result.p_values, corrected,
    )
    np.savez_compressed(
        output_dir / f"{tag}_{condition}_effects.npz",
        normalized_per_prompt=result.per_prompt_effects.detach().cpu().numpy(),
        unnormalized_per_prompt=(result.per_prompt_effects * diagnostics["mean"]).detach().cpu().numpy(),
        normalized_mean=normalized,
        unnormalized_mean=raw,
        template_ids=np.asarray(groups),
    )
    record["bh_discoveries"] = int(corrected["bh"][1].sum())
    record["by_discoveries"] = int(corrected["by"][1].sum())
    record["signflip_mode"] = result.signflip_mode
    record["signflip_pattern_count"] = result.signflip_pattern_count
    record["zero_effect_components"] = result.exact_zero_components
    return record, result, raw, corrected


def heldout_one(model, auditor, selection, evaluation, target_length, condition):
    """Use the library's disjoint-template helper for descriptive joint patching."""
    import torch

    selected_inputs = condition_batch(model, selection, condition, target_length, torch)
    eval_inputs = condition_batch(model, evaluation, condition, target_length, torch)
    select_check = auditor.check_setup(
        *selected_inputs[:4], method="signflip", group_ids=selected_inputs[4]
    )
    eval_check = auditor.check_setup(*eval_inputs[:4])
    info = {
        "condition": condition,
        "status": "complete" if select_check.gap_usable and eval_check.gap_usable else "unusable_gap",
        "selection_gap_usable": bool(select_check.gap_usable),
        "evaluation_gap_usable": bool(eval_check.gap_usable),
    }
    if info["status"] != "complete":
        return info

    torch.manual_seed(SEED)
    with torch.no_grad():
        held = auditor.run_heldout_validation(
            selection_inputs=dict(
                clean_tokens=selected_inputs[0], corrupted_tokens=selected_inputs[1],
                correct_tokens=selected_inputs[2], incorrect_tokens=selected_inputs[3],
                method="signflip", group_ids=selected_inputs[4],
            ),
            evaluation_inputs=dict(
                clean_tokens=eval_inputs[0], corrupted_tokens=eval_inputs[1],
                correct_tokens=eval_inputs[2], incorrect_tokens=eval_inputs[3],
            ),
            selection_group_ids=selected_inputs[4],
            evaluation_group_ids=eval_inputs[4],
        )
    if held.selection_audit.significant_mask is None:
        raise RuntimeError("Selection mask is missing from held-out audit.")
    info["selection_heads_bh"] = int(held.selection_audit.significant_mask.sum().item())
    info["joint_normalized_effect"] = float(held.joint_effect)
    info["evaluation_gap"] = gap_diagnostics(model, eval_inputs)
    info["joint_unnormalized_effect"] = info["joint_normalized_effect"] * info["evaluation_gap"]["mean"]
    info["selection_templates"] = sorted(set(selected_inputs[4]))
    info["evaluation_templates"] = sorted(set(eval_inputs[4]))
    return info


def run(args):
    import torch
    from transformer_lens import TransformerBridge
    from transformerlens_fdr import PatchingAuditor, logit_difference
    import scipy
    import transformers

    validate_templates(TEMPLATES)
    torch.manual_seed(SEED)
    model = TransformerBridge.boot_transformers("gpt2")
    model.enable_compatibility_mode()
    model.eval()
    target, pairs, selection_ids, evaluation_ids = prepare_prompts(model, args.per_template)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paired_prompts.jsonl").open("w") as output:
        for example in pairs:
            output.write(json.dumps(example, sort_keys=True) + "\n")
    records = {
        "mode": "smoke" if args.smoke else "full",
        "source_commit": git_sha(),
        "model": "gpt2",
        "seed": SEED,
        "signflip_seed": SEED,
        "n_perm": args.n_perm,
        "per_template": args.per_template,
        "alpha": ALPHA,
        "selection_template_ids": list(selection_ids),
        "evaluation_template_ids": list(evaluation_ids),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "transformers": transformers.__version__,
            "transformer_lens": package_version("transformer-lens"),
            "transformerlens_fdr": package_version("transformerlens-fdr"),
        },
        "analyses": {},
    }
    auditor = PatchingAuditor(
        model, logit_difference, hook_type="attn_head",
        fdr_threshold=ALPHA, fdr_method="fdr_bh",
        n_perm=args.n_perm, head_batch_size=args.head_batch_size,
    )

    # Reproduce the original 20-template ABC analysis before the comparison.
    full_abc_inputs = condition_batch(model, pairs, "abc", target, torch)
    baseline, _, _, _ = audit_one(
        model, auditor, full_abc_inputs, output_dir, "abc", "baseline20"
    )
    records["analyses"]["abc_baseline_20"] = baseline
    write_summary(output_dir, records)
    if not baseline["gap_usable"]:
        raise RuntimeError("ABC baseline has an unusable gap; comparison not run.")
    if args.baseline_only:
        print(f"ABC baseline saved to {output_dir}")
        return

    selected = select_examples(pairs, selection_ids)
    evaluated = select_examples(pairs, evaluation_ids)
    effects = {}
    discoveries = {}
    for condition in ("abc", "name_swap"):
        batch = condition_batch(model, selected, condition, target, torch)
        record, _, raw, corrected = audit_one(
            model, auditor, batch, output_dir, condition, "selection16"
        )
        records["analyses"][f"selection_{condition}"] = record
        if raw is not None:
            effects[condition] = raw
            discoveries[condition] = corrected
        write_summary(output_dir, records)

    if set(effects) == {"abc", "name_swap"}:
        records["comparison"] = compare_effects(effects["abc"], effects["name_swap"])
        records["comparison"]["discovery_sets"] = compare_discoveries(
            discoveries["abc"], discoveries["name_swap"]
        )
        write_layer_table(output_dir / "layer_comparison.csv", effects["abc"], effects["name_swap"])
    else:
        records["comparison"] = {"status": "inconclusive_unusable_gap"}
    write_summary(output_dir, records)

    for condition in ("abc", "name_swap"):
        records["analyses"][f"evaluation_{condition}"] = heldout_one(
            model, auditor, selected, evaluated, target, condition
        )
        write_summary(output_dir, records)
    print(f"Results saved under {output_dir}")


def _json_numpy(value):
    """Convert NumPy data to JSON-native numbers, booleans, and lists."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def write_summary(output_dir, records):
    """Validate JSON before writing; do not leave a partial summary on error."""
    content = json.dumps(records, indent=2, allow_nan=False, default=_json_numpy)
    (output_dir / "summary.json").write_text(content + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Tokenizer-only template check")
    parser.add_argument("--smoke", action="store_true", help="Short pipeline check; results stored separately")
    parser.add_argument("--baseline-only", action="store_true", help="Run the full-template ABC baseline only")
    parser.add_argument("--head-batch-size", type=int, default=4, help="Heads patched per replicated forward pass")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/ioi_abc_vs_name_swap"))
    args = parser.parse_args()
    if args.head_batch_size < 1:
        parser.error("--head-batch-size must be positive")
    if args.dry_run:
        dry_run()
        return
    args.per_template = 2 if args.smoke else PER_TEMPLATE
    args.n_perm = 999 if args.smoke else N_PERM
    args.output = Path(str(args.out) + "_smoke") if args.smoke else args.out
    run(args)


if __name__ == "__main__":
    main()
