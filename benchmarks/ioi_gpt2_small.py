"""Phase 0: does the FDR audit recover the published IOI circuit in GPT-2 small,
and is it calibrated on real activations?

Read benchmarks/PHASE0_PLAN.md first and commit it BEFORE running this, so the
pass/fail criteria are fixed before you see results.

What this runs (attention heads, denoising patch of ``attn.hook_z`` from clean
into ABC-corrupted prompts, metric = logit difference IO - S at the last token):

1. Prompts from many templates; each template is one cluster (``group_ids``).
2. One patching sweep, then three readouts from the same per-prompt effects:
   descriptive top-K, per-prompt sign-flip, cluster sign-flip (BH and BY).
3. Recovery against the published circuit (Wang et al. 2022; the head list in
   PUBLISHED_CIRCUIT is written from memory: verify it against the paper).
4. Placebo control: patch between two independent ABC samples (no IOI
   structure). A valid test should rarely reject anything. This is a
   diagnostic, not a formal null.
5. Template effect (ICC), a discoveries-vs-templates subsampling curve, and leave-two-
   templates-out stability.

Caveat: the published circuit is not exhaustive ground truth. A discovery
outside it is "outside the published circuit", not a "false discovery".

    python benchmarks/ioi_gpt2_small.py --dry-run          # tokenization/template check
    python benchmarks/ioi_gpt2_small.py --smoke            # quick pipeline check
    python benchmarks/ioi_gpt2_small.py --out benchmarks/results/ioi
"""

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Published circuit (verify against Wang et al. 2022 before publishing)
# ---------------------------------------------------------------------------
PUBLISHED_CIRCUIT = {
    "name mover": [(9, 9), (9, 6), (10, 0)],
    "negative name mover": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}
CORE_ROLES = ("name mover", "negative name mover", "s2 inhibition")
ALL_HEADS = {head for heads in PUBLISHED_CIRCUIT.values() for head in heads}
CORE_HEADS = {head for role in CORE_ROLES for head in PUBLISHED_CIRCUIT[role]}
NAME_MOVERS = set(PUBLISHED_CIRCUIT["name mover"])
NEGATIVE_MOVERS = set(PUBLISHED_CIRCUIT["negative name mover"])

NAMES = [
    "Michael", "Christopher", "Jessica", "Matthew", "Ashley", "Jennifer", "Joshua", "Amanda",
    "Daniel", "David", "James", "Robert", "John", "Joseph", "Andrew", "Ryan", "Brandon", "Jason",
    "Justin", "Sarah", "William", "Jonathan", "Stephanie", "Brian", "Nicole", "Nicholas",
    "Anthony", "Heather", "Eric", "Elizabeth", "Adam", "Megan", "Melissa", "Kevin", "Steven",
    "Thomas", "Timothy", "Christina", "Kyle", "Rachel", "Laura", "Lauren", "Amber", "Brittany",
    "Danielle", "Richard", "Kimberly", "Jeffrey", "Amy", "Crystal", "Michelle", "Tiffany",
    "Jeremy", "Benjamin", "Mark", "Emily", "Aaron", "Charles", "Rebecca", "Jacob", "Stephen",
    "Patrick", "Sean", "Erin", "Zachary", "Jamie", "Kelly", "Samantha", "Nathan", "Sara",
    "Dustin", "Paul", "Angela", "Tyler", "Scott", "Katherine", "Andrea", "Gregory", "Erica",
    "Mary", "Travis", "Lisa", "Kenneth", "Bryan", "Lindsey", "Kristen", "Jose", "Alexander",
    "Jesse", "Katie", "Lindsay", "Shannon", "Vanessa", "Courtney", "Christine", "Alicia",
    "Cody", "Allison", "Bradley", "Samuel",
]
PLACES = ["store", "park", "school", "office", "hospital", "garden", "restaurant", "station",
          "library", "beach", "market", "church", "farm", "bank", "house", "village", "club"]
OBJECTS = ["drink", "book", "ring", "bone", "letter", "kiss", "necklace", "snack", "basketball",
           "ball", "gift", "toy", "flower", "bag", "phone", "card"]

# Every template has exactly one "{A} and {B}" (the first mention pair) and one
# later "{B}" (the repeated subject, S2). The answer is always {A} (the IO).
TEMPLATES = [
    "Then, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to",
    "Afterwards, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to",
    "When {A} and {B} got a {OBJECT} at the {PLACE}, {B} decided to give it to",
    "After {A} and {B} went to the {PLACE}, {B} gave a {OBJECT} to",
    "Then, {A} and {B} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to",
    "Then, {A} and {B} were working at the {PLACE}. {B} decided to give a {OBJECT} to",
    "Then, {A} and {B} were thinking about going to the {PLACE}. {B} wanted to give a {OBJECT} to",
    "Then, {A} and {B} had a long argument, and afterwards {B} said to",
    "After the lunch, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to",
    "Then, {A} and {B} had a long argument. Afterwards {B} said to",
    "The {PLACE} {A} and {B} went to had a {OBJECT}. {B} gave it to",
    "Friends {A} and {B} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to",
    "While {A} and {B} were at the {PLACE}, {B} found a {OBJECT} and gave it to",
    "Yesterday {A} and {B} visited the {PLACE}. {B} handed a {OBJECT} to",
    "This morning {A} and {B} walked to the {PLACE}. {B} passed a {OBJECT} to",
    "Last week {A} and {B} drove to the {PLACE}. {B} sold a {OBJECT} to",
    "Later that day, {A} and {B} met at the {PLACE}. {B} lent a {OBJECT} to",
    "At the {PLACE}, {A} and {B} talked for hours. In the end, {B} offered a {OBJECT} to",
    "Today {A} and {B} cleaned the {PLACE}. Afterwards {B} showed a {OBJECT} to",
    "On Monday, {A} and {B} arrived at the {PLACE}. {B} brought a {OBJECT} for",
    "Soon after, {A} and {B} entered the {PLACE}. {B} threw a {OBJECT} to",
    "Since {A} and {B} liked the {PLACE}, {B} bought a {OBJECT} and gave it to",
    "Once {A} and {B} reached the {PLACE}, {B} sent a {OBJECT} to",
    "Although {A} and {B} were tired, {B} still gave a {OBJECT} to",
    "As {A} and {B} left the {PLACE}, {B} gave the {OBJECT} to",
    "Before {A} and {B} closed the {PLACE}, {B} gave a {OBJECT} to",
    "Because {A} and {B} were friends, {B} gave a {OBJECT} to",
    "In the evening {A} and {B} sat in the {PLACE}. {B} said hello to",
    "Then {A} and {B} left the {PLACE} together. {B} waved at",
    "Early on, {A} and {B} worked at the {PLACE}. {B} explained the {OBJECT} to",
    "Every day {A} and {B} go to the {PLACE}. Today {B} will give a {OBJECT} to",
    "Meanwhile, {A} and {B} waited at the {PLACE}. Then {B} gave a {OBJECT} to",
]

SLOT = re.compile(r"\{(A|B)\}")


def validate_templates(templates):
    for template in templates:
        if template.count("{A} and {B}") != 1 or template.count("{A}") != 1:
            raise ValueError(f"Template needs exactly one '{{A}} and {{B}}': {template!r}")
        if template.count("{B}") != 2:
            raise ValueError(f"Template needs exactly one extra '{{B}}' (S2): {template!r}")
        if template.startswith("{"):
            raise ValueError(f"Template must not start with a name slot: {template!r}")


# ---------------------------------------------------------------------------
# Prompt construction (tokenizer-agnostic, testable without a model)
# ---------------------------------------------------------------------------
def render(template, order, slot_names, place, obj):
    """Fill a template. ``slot_names`` are the three names in order of appearance."""
    text = template.replace("{A} and {B}", "{B} and {A}") if order == "BABA" else template
    text = text.replace("{PLACE}", place).replace("{OBJECT}", obj)
    names = iter(slot_names)
    return SLOT.sub(lambda _match: next(names), text)


def slot_keys(template, order):
    text = template.replace("{A} and {B}", "{B} and {A}") if order == "BABA" else template
    return SLOT.findall(text)


def build_examples(template_indices, templates, per_template, names, places, objects, rng):
    """Clean (IO=a, S=b) and ABC-corrupted (three new distinct names) prompt strings."""
    examples = []
    for t_index in template_indices:
        template = templates[t_index]
        for _ in range(per_template):
            a, b, c1, c2, c3 = rng.sample(names, 5)
            order = rng.choice(["ABBA", "BABA"])
            place, obj = rng.choice(places), rng.choice(objects)
            keys = slot_keys(template, order)
            clean_names = [{"A": a, "B": b}[key] for key in keys]
            examples.append(
                {
                    "template": t_index,
                    "order": order,
                    "io": a,
                    "s": b,
                    "clean": render(template, order, clean_names, place, obj),
                    "corrupted": render(template, order, [c1, c2, c3], place, obj),
                }
            )
    return examples


def build_placebo_examples(template_indices, templates, per_template, names, places, objects, rng):
    """Two independent ABC samples (no IOI structure) plus an unrelated name pair for the metric."""
    examples = []
    for t_index in template_indices:
        template = templates[t_index]
        for _ in range(per_template):
            picks = rng.sample(names, 8)
            order = rng.choice(["ABBA", "BABA"])
            place, obj = rng.choice(places), rng.choice(objects)
            examples.append(
                {
                    "template": t_index,
                    "io": picks[6],
                    "s": picks[7],
                    "clean": render(template, order, picks[0:3], place, obj),
                    "corrupted": render(template, order, picks[3:6], place, obj),
                }
            )
    return examples


def choose_window(lengths, max_pad):
    """Pick the target length whose [target-max_pad, target] window keeps most templates."""
    best = None
    for target in sorted(set(lengths.values())):
        kept = sorted(i for i, length in lengths.items() if target - max_pad <= length <= target)
        if best is None or len(kept) > len(best[1]):
            best = (target, kept)
    return best


def left_pad(ids, target):
    return [ids[0]] * (target - len(ids)) + list(ids)


# ---------------------------------------------------------------------------
# Scoring and diagnostics (NumPy only)
# ---------------------------------------------------------------------------
def heads_from_mask(mask):
    rows, cols = np.nonzero(np.asarray(mask))
    return {(int(layer), int(head)) for layer, head in zip(rows, cols)}


def top_k_heads(effects, k):
    flat = np.argsort(-np.abs(np.asarray(effects)).ravel(), kind="stable")[:k]
    return {(int(i // effects.shape[1]), int(i % effects.shape[1])) for i in flat}


def score_set(found, effects):
    """Recovery against the published circuit, with sign agreement for discoveries inside it."""
    inside = found & ALL_HEADS
    agree = 0
    for layer, head in inside:
        expected = -1.0 if (layer, head) in NEGATIVE_MOVERS else 1.0
        agree += int(np.sign(effects[layer, head]) == expected)
    return {
        "n_discoveries": len(found),
        "name_movers_found": len(found & NAME_MOVERS),
        "core_found": len(found & CORE_HEADS),
        "full_found": len(inside),
        "outside_published": len(found - ALL_HEADS),
        "fraction_outside": (len(found - ALL_HEADS) / len(found)) if found else float("nan"),
        "sign_agreement_inside": (agree / len(inside)) if inside else float("nan"),
    }


SCRIPT_VERSION = "2026-10-19 per-head dump"
HEAD_ROLE = {head: role for role, heads in PUBLISHED_CIRCUIT.items() for head in heads}
ARM_KEYS = ("cluster_bh", "cluster_by", "prompt_bh", "prompt_by")


def per_head_rows(raw, p_values, q_values, passed):
    """One row per head. ``p_values`` maps 'cluster'/'prompt' to [L,H] arrays; ``q_values`` and
    ``passed`` map the keys in ARM_KEYS to [L,H] arrays. Sorted by absolute effect."""
    n_layers, n_heads = raw.shape
    order = np.argsort(-np.abs(raw).ravel(), kind="stable")
    rank = np.empty(raw.size, dtype=int)
    rank[order] = np.arange(1, raw.size + 1)
    rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            role = HEAD_ROLE.get((layer, head), "outside published")
            expected = "" if role == "outside published" else (-1 if (layer, head) in NEGATIVE_MOVERS else 1)
            effect = float(raw[layer, head])
            row = {"layer": layer, "head": head, "role": role, "effect": effect,
                   "abs_effect_rank": int(rank[layer * n_heads + head]), "expected_sign": expected,
                   "sign_agrees": "" if expected == "" else int(np.sign(effect) == expected),
                   "p_cluster": float(p_values["cluster"][layer, head]),
                   "p_prompt": float(p_values["prompt"][layer, head])}
            for key in ARM_KEYS:
                row[f"q_{key}"] = float(q_values[key][layer, head])
                row[f"discovered_{key}"] = int(bool(passed[key][layer, head]))
            rows.append(row)
    rows.sort(key=lambda r: r["abs_effect_rank"])
    return rows


def jaccard(a, b):
    union = a | b
    return float("nan") if not union else len(a & b) / len(union)


def icc1(values, codes):
    """One-way random-effects intraclass correlation of prompt effects within templates."""
    values = np.asarray(values, dtype=float)
    codes = np.asarray(codes)
    groups = int(codes.max()) + 1
    n = len(values)
    counts = np.bincount(codes, minlength=groups)
    means = np.array([values[codes == g].mean() for g in range(groups)])
    grand = values.mean()
    between = (counts * (means - grand) ** 2).sum() / (groups - 1)
    within = sum(((values[codes == g] - means[g]) ** 2).sum() for g in range(groups)) / (n - groups)
    n0 = (n - (counts**2).sum() / n) / (groups - 1)
    denom = between + (n0 - 1) * within
    return float((between - within) / denom) if denom > 0 else float("nan")


def evaluate_criteria(report, args=None):
    """Pre-registered criteria from PHASE0_PLAN.md. Keep in sync with that file.

    Each entry has a status of PASS, FAIL, INCONCLUSIVE (required evidence is
    missing) or INFO (report-only, never gates). B1/B2/C1 are pre-registered
    *hypotheses* about this particular intervention, not claims about the paper.
    """
    def entry(name, status, detail, gating=True):
        return {"name": name, "status": status, "detail": detail, "gating": gating}

    out = []
    design = report["design"]
    out.append(entry("A1 templates retained >= 14", "PASS" if design["n_templates"] >= 14 else "FAIL",
                     f"{design['n_templates']} templates; min BH rank for any rejection "
                     f"{design.get('min_rank_for_any_rejection')}"))
    out.append(entry("A2 prompts >= 80", "PASS" if design["n_prompts"] >= 80 else "FAIL",
                     f"{design['n_prompts']} prompts"))
    gap_ok = design.get("gap_usable")
    frac = design.get("fraction_positive_gap")
    if gap_ok is None or frac is None:
        out.append(entry("A3 clean-corrupted gap usable and >= 90% of prompts positive", "INCONCLUSIVE", "not measured"))
    else:
        out.append(entry("A3 clean-corrupted gap usable and >= 90% of prompts positive",
                         "PASS" if (gap_ok and frac >= 0.9) else "FAIL",
                         f"gap mean {design['gap_mean']:.3g}, SD {design['gap_std']:.3g}, "
                         f"{frac:.0%} of prompts positive, usable={gap_ok}"))
    main = report["arms"]["cluster_signflip_bh"]
    out.append(entry("B1 all 3 name movers discovered (cluster, BH)",
                     "PASS" if main["name_movers_found"] == 3 else "FAIL", f"{main['name_movers_found']}/3"))
    sign = main["sign_agreement_inside"]
    if math.isnan(sign):
        out.append(entry("B2 sign agreement inside circuit >= 0.9 (cluster, BH)", "INCONCLUSIVE", "no in-circuit discoveries"))
    else:
        out.append(entry("B2 sign agreement inside circuit >= 0.9 (cluster, BH)",
                         "PASS" if sign >= 0.9 else "FAIL", f"{sign:.2f}"))
    placebo = report.get("placebo")
    if not placebo:
        out.append(entry("C1 placebo negative control", "INCONCLUSIVE", "placebo not run"))
    else:
        requested, done = placebo["reps_requested"], placebo["reps_completed"]
        limit = int(math.floor(0.05 * requested + 2))
        count = placebo["cluster_bh_reps_with_rejection"]
        if done < requested:
            out.append(entry(f"C1 placebo: <= {limit}/{requested} reps with any BH rejection (cluster)",
                             "INCONCLUSIVE", f"only {done}/{requested} repetitions completed"))
        else:
            out.append(entry(f"C1 placebo: <= {limit}/{requested} reps with any BH rejection (cluster)",
                             "PASS" if count <= limit else "FAIL",
                             f"{count}/{requested}; gross-failure detector only: 0/{requested} still leaves a "
                             f"~{100 * (1 - 0.05 ** (1 / requested)):.0f}% upper 95% bound on the rejection rate"))
    stability = report.get("stability")
    if stability and not math.isnan(stability["median_jaccard"]):
        out.append(entry("D1 leave-two-templates-out median Jaccard (report-only; 0.5 is a rough reference)",
                         "INFO", f"{stability['median_jaccard']:.2f}", gating=False))
    return out


# ---------------------------------------------------------------------------
# Model-dependent pipeline
# ---------------------------------------------------------------------------
def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def single_token_filter(model, words, prefix=" "):
    kept = []
    for word in words:
        try:
            model.to_single_token(prefix + word)
            kept.append(word)
        except Exception:  # noqa: BLE001
            continue
    return kept


def tokenize(model, text):
    return model.to_tokens(text)[0].tolist()


def to_batch(model, examples, target_length, torch):
    device = model.cfg.device
    clean = torch.tensor([left_pad(tokenize(model, e["clean"]), target_length) for e in examples], device=device)
    corrupted = torch.tensor([left_pad(tokenize(model, e["corrupted"]), target_length) for e in examples], device=device)
    correct = torch.tensor([model.to_single_token(" " + e["io"]) for e in examples], device=device)
    incorrect = torch.tensor([model.to_single_token(" " + e["s"]) for e in examples], device=device)
    groups = [e["template"] for e in examples]
    return clean, corrupted, correct, incorrect, groups


def dry_run(args):
    """Template/tokenization check using the tokenizer only (no model weights are loaded).

    Lengths exclude the BOS token; the keep/drop window is relative, so the kept set
    is the same as in the real run.
    """
    from transformers import AutoTokenizer

    validate_templates(TEMPLATES)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def usable(words):
        return [w for w in words if len(tokenizer.encode(" " + w)) == 1]

    names, places, objects = usable(NAMES), usable(PLACES), usable(OBJECTS)
    print(f"Single-token words: {len(names)} names, {len(places)} places, {len(objects)} objects.")
    probe = {i: len(tokenizer.encode(render(t, "ABBA", ["John", "Mary", "Mary"], places[0], objects[0])))
             for i, t in enumerate(TEMPLATES)}
    target, kept = choose_window(probe, args.max_pad)
    for i in range(len(TEMPLATES)):
        print(f"  [{'keep' if i in kept else 'drop'}] len={probe[i]:2d}  {TEMPLATES[i]}")
    verdict = "OK" if len(kept) >= args.min_templates else f"TOO FEW (need >= {args.min_templates})"
    print(f"Target length {target} (excl. BOS); keeps {len(kept)}/{len(TEMPLATES)} templates: {verdict}.")


def check_package(module):
    """Capabilities this script needs from the transformerlens_fdr copy it imported."""
    import inspect

    auditor = module.PatchingAuditor
    init_params = inspect.signature(auditor.__init__).parameters
    missing = [f"run_patching_audit(group_ids=...)" for _ in [0]
               if "group_ids" not in inspect.signature(auditor.run_patching_audit).parameters]
    missing += [f"check_setup(group_ids=...)" for _ in [0]
                if "group_ids" not in inspect.signature(auditor.check_setup).parameters]
    return {"has_head_batch_size": "head_batch_size" in init_params, "missing": missing}


def run(args):
    import torch
    from transformer_lens import TransformerBridge

    # Use the checkout this script lives in, not whatever copy is installed in the environment:
    # a script in benchmarks/ puts benchmarks/ (not the repo root) first on the import path.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(repo_root, "transformerlens_fdr")):
        sys.path.insert(0, repo_root)
    import transformerlens_fdr
    from transformerlens_fdr import PatchingAuditor, apply_fdr_adjustment, logit_difference, signflip_p_values

    capabilities = check_package(transformerlens_fdr)
    print(f"transformerlens_fdr imported from: {transformerlens_fdr.__file__}")
    if capabilities["missing"]:
        sys.exit("This transformerlens_fdr copy is too old for the benchmark (missing: "
                 + ", ".join(capabilities["missing"]) + "). Run `pip install -e .` from the repo root "
                 "and check `git log -1 --oneline` (expect commit 77b3761 or later).")

    print(f"ioi_gpt2_small.py version: {SCRIPT_VERSION}")
    validate_templates(TEMPLATES)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    model = TransformerBridge.boot_transformers(args.model)
    model.enable_compatibility_mode()
    names = single_token_filter(model, NAMES)
    places = single_token_filter(model, PLACES)
    objects = single_token_filter(model, OBJECTS)
    if len(names) < 12 or not places or not objects:
        sys.exit(f"Too few single-token words (names={len(names)}, places={len(places)}, objects={len(objects)}).")

    probe = {i: len(tokenize(model, render(t, "ABBA", ["John", "Mary", "Mary"], places[0], objects[0])))
             for i, t in enumerate(TEMPLATES)}
    target, kept = choose_window(probe, args.max_pad)
    print(f"Template token lengths: {sorted(set(probe.values()))}; target length {target} "
          f"keeps {len(kept)}/{len(TEMPLATES)} templates (left-pad up to {args.max_pad} BOS tokens).")
    if len(kept) < args.min_templates:
        sys.exit(f"Only {len(kept)} templates retained; need >= {args.min_templates}. "
                 "Raise --max-pad or add templates (see PHASE0_PLAN.md).")

    per_template = args.per_template
    examples = build_examples(kept, TEMPLATES, per_template, names, places, objects, rng)
    clean, corrupted, correct, incorrect, groups = to_batch(model, examples, target, torch)
    n_prompts = len(examples)
    with torch.no_grad():
        logits = model(clean)
        clean_ld = logit_difference(logits, correct, incorrect).mean().item()
    print(f"{n_prompts} prompts in {len(kept)} templates; mean clean logit difference {clean_ld:.2f}.")
    if clean_ld < 1.0:
        print("WARNING: clean logit difference is low; padding or template choice may be hurting the task.")

    auditor_kwargs = dict(fdr_threshold=args.alpha, fdr_method="fdr_bh", n_perm=args.n_perm, hook_type="attn_head")
    if capabilities["has_head_batch_size"]:
        auditor_kwargs["head_batch_size"] = args.head_batch_size
    else:
        print("NOTE: this transformerlens_fdr copy has no head_batch_size; using its default patching batch.")
    auditor = PatchingAuditor(model, logit_difference, **auditor_kwargs)
    preflight = auditor.check_setup(clean, corrupted, correct, incorrect, method="signflip", group_ids=groups)
    print(preflight.format_summary())

    with torch.no_grad():
        corrupted_ld = logit_difference(model(corrupted), correct, incorrect)
        clean_ld_each = logit_difference(model(clean), correct, incorrect)
    fraction_positive_gap = float(((clean_ld_each - corrupted_ld) > 0).float().mean().item())

    started = time.time()
    results = auditor.run_patching_audit(clean, corrupted, correct, incorrect, method="signflip", group_ids=groups)
    print(f"Patching sweep done in {time.time() - started:.0f}s.\n{results.format_summary()}")

    per = results.per_prompt_effects
    raw = results.raw_patching_effects.detach().cpu().numpy()
    n_layers, n_heads = raw.shape

    def discoveries(p_values, method):
        _, passed = apply_fdr_adjustment(p_values, alpha=args.alpha, method=method)
        return heads_from_mask(passed.detach().cpu().numpy())

    p_cluster = results.p_values
    p_prompt = signflip_p_values(per, n_perm=args.n_perm, zero_effect_tolerance=auditor.zero_effect_tolerance)
    arms = {}
    for label, p in (("cluster", p_cluster), ("prompt", p_prompt)):
        for method, tag in (("fdr_bh", "bh"), ("fdr_by", "by")):
            found = discoveries(p, method)
            arms[f"{label}_signflip_{tag}"] = {**score_set(found, raw), "heads": sorted(found)}
    for k in (len(CORE_HEADS), len(ALL_HEADS)):
        found = top_k_heads(raw, k)
        arms[f"descriptive_top{k}"] = {**score_set(found, raw), "heads": sorted(found)}

    # --- per-head table and per-prompt effects (for post-hoc diagnosis; output only) ---
    adjusted_q, adjusted_pass = {}, {}
    for label, p in (("cluster", p_cluster), ("prompt", p_prompt)):
        for method, tag in (("fdr_bh", "bh"), ("fdr_by", "by")):
            q_tensor, passed_tensor = apply_fdr_adjustment(p, alpha=args.alpha, method=method)
            adjusted_q[f"{label}_{tag}"] = q_tensor.detach().cpu().numpy()
            adjusted_pass[f"{label}_{tag}"] = passed_tensor.detach().cpu().numpy()
    table = per_head_rows(raw, {"cluster": p_cluster.detach().cpu().numpy(), "prompt": p_prompt.detach().cpu().numpy()},
                          adjusted_q, adjusted_pass)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "per_head.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    np.savez_compressed(os.path.join(args.out, "per_prompt_effects.npz"),
                        effects=per.detach().cpu().numpy(), template_codes=np.array([kept.index(g) for g in groups]),
                        templates=np.array([TEMPLATES[i] for i in kept]))
    print(f"Wrote {os.path.join(args.out, 'per_head.csv')} and {os.path.join(args.out, 'per_prompt_effects.npz')}")

    # --- template effect --------------------------------------------------
    codes = np.array([kept.index(g) for g in groups])
    per_np = per.detach().cpu().numpy()
    strongest = top_k_heads(raw, 10)
    icc_top = [icc1(per_np[layer, head], codes) for layer, head in strongest]
    icc_all = [icc1(per_np[l, h], codes) for l in range(n_layers) for h in range(n_heads)]
    template_effect = {"median_icc_top10_heads": float(np.nanmedian(icc_top)),
                       "median_icc_all_heads": float(np.nanmedian(icc_all))}

    # --- discoveries vs number of templates (subsampling the SAME effects; not a power estimate) ---
    unique_templates = list(range(len(kept)))
    group_codes = torch.tensor(codes, device=per.device)
    subsample_rows = []
    sizes = sorted({g for g in (6, 8, 10, 12, 14, 16, 20, 24) if g < len(kept)} | {len(kept)})
    sub_rng = random.Random(args.seed + 1)
    for size in sizes:
        found_counts, mover_counts = [], []
        for _ in range(1 if size == len(kept) else args.subsets):
            chosen = sub_rng.sample(unique_templates, size)
            idx = torch.nonzero(torch.isin(group_codes, torch.tensor(chosen, device=per.device))).squeeze(-1)
            p = signflip_p_values(per[..., idx], n_perm=args.n_perm,
                                  zero_effect_tolerance=auditor.zero_effect_tolerance,
                                  group_ids=codes[idx.cpu().numpy()].tolist())
            found = discoveries(p, "fdr_bh")
            found_counts.append(len(found))
            mover_counts.append(len(found & NAME_MOVERS))
        subsample_rows.append({"templates": size, "mean_discoveries": float(np.mean(found_counts)),
                           "mean_name_movers_found": float(np.mean(mover_counts))})

    # --- leave-two-templates-out stability ---------------------------------
    full_found = {tuple(h) for h in arms["cluster_signflip_bh"]["heads"]}
    jaccards = []
    stab_rng = random.Random(args.seed + 2)
    if len(kept) >= 8 and full_found:
        for _ in range(args.stability_reps):
            chosen = stab_rng.sample(unique_templates, len(kept) - 2)
            idx = torch.nonzero(torch.isin(group_codes, torch.tensor(chosen, device=per.device))).squeeze(-1)
            p = signflip_p_values(per[..., idx], n_perm=args.n_perm,
                                  zero_effect_tolerance=auditor.zero_effect_tolerance,
                                  group_ids=codes[idx.cpu().numpy()].tolist())
            jaccards.append(jaccard(full_found, discoveries(p, "fdr_bh")))
    stability = {"reps": len(jaccards), "median_jaccard": float(np.nanmedian(jaccards)) if jaccards else float("nan")}

    # --- placebo control ----------------------------------------------------
    placebo = None
    if args.placebo_reps > 0:
        counts = {"cluster_bh": 0, "cluster_by": 0, "prompt_bh": 0, "prompt_by": 0}
        done = 0
        for rep in range(args.placebo_reps):
            ex = build_placebo_examples(kept, TEMPLATES, per_template, names, places, objects,
                                        random.Random(args.seed + 1000 + rep))
            c, k, cor, inc, g = to_batch(model, ex, target, torch)
            try:
                res = auditor.run_patching_audit(c, k, cor, inc, method="signflip", group_ids=g)
            except ValueError as error:
                print(f"placebo rep {rep}: skipped ({error})")
                continue
            done += 1
            p_prompt_rep = signflip_p_values(res.per_prompt_effects, n_perm=args.n_perm,
                                             zero_effect_tolerance=auditor.zero_effect_tolerance)
            for label, p in (("cluster", res.p_values), ("prompt", p_prompt_rep)):
                for method, tag in (("fdr_bh", "bh"), ("fdr_by", "by")):
                    counts[f"{label}_{tag}"] += int(len(discoveries(p, method)) > 0)
            print(f"placebo rep {rep + 1}/{args.placebo_reps} done", flush=True)
        placebo = {"reps_requested": args.placebo_reps, "reps_completed": done,
                   **{f"{key}_reps_with_rejection": value for key, value in counts.items()}}

    report = {
        "design": {"model": args.model, "n_templates": len(kept), "n_prompts": n_prompts,
                   "prompts_per_template": per_template, "target_length": target, "max_pad": args.max_pad,
                   "mean_clean_logit_diff": clean_ld, "alpha": args.alpha, "n_perm": args.n_perm,
                   "gap_mean": preflight.gap_mean, "gap_std": preflight.gap_std, "gap_usable": preflight.gap_usable,
                   "fraction_positive_gap": fraction_positive_gap,
                   "min_rank_for_any_rejection": (preflight.resolution_report or {}).get("min_rank_for_any_rejection"),
                   "seed": args.seed, "family_size": n_layers * n_heads,
                   "p_value_floor_cluster": results.minimum_attainable_p,
                   "templates": [TEMPLATES[i] for i in kept]},
        "arms": arms, "template_effect": template_effect, "subsample_curve": subsample_rows,
        "stability": stability, "placebo": placebo,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "package_commit": git_commit()},
    }
    checks = evaluate_criteria(report, args)
    report["criteria"] = checks

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ioi_phase0.json"), "w") as handle:
        json.dump(report, handle, indent=1, default=float)
    with open(os.path.join(args.out, "ioi_phase0.md"), "w") as handle:
        handle.write(to_markdown(report))
    try:
        make_heatmap(results, raw, args, os.path.join(args.out, "ioi_phase0_heatmap.png"))
    except ImportError:
        print("matplotlib not installed; skipped the heatmap.")
    print(to_markdown(report))


def to_markdown(report):
    d = report["design"]
    lines = [f"# IOI Phase 0 ({d['model']})", "",
             f"{d['n_prompts']} prompts, {d['n_templates']} templates, family {d['family_size']} heads, "
             f"alpha {d['alpha']}, cluster p-value floor {d['p_value_floor_cluster']:.2e}, "
             f"mean clean logit difference {d['mean_clean_logit_diff']:.2f}; clean-corrupted gap {d['gap_mean']:.3g} "
             f"(SD {d['gap_std']:.3g}, {d['fraction_positive_gap']:.0%} of prompts positive); "
             f"BH needs {d['min_rank_for_any_rejection']} head(s) at the p-value floor to reject anything.", "",
             "| arm | discoveries | name movers /3 | core /9 | published /26 | outside published | sign agreement |",
             "|:---|---:|---:|---:|---:|---:|---:|"]
    for name, a in report["arms"].items():
        lines.append(f"| {name} | {a['n_discoveries']} | {a['name_movers_found']} | {a['core_found']} | "
                     f"{a['full_found']} | {a['outside_published']} | {a['sign_agreement_inside']:.2f} |")
    t = report["template_effect"]
    lines += ["", f"Template effect: median ICC {t['median_icc_top10_heads']:.2f} (top-10 heads), "
                  f"{t['median_icc_all_heads']:.2f} (all heads).", "",
              "Discoveries when thinning THIS dataset to fewer templates (subsampling curve, not a power estimate):", "",
              "| templates | mean discoveries | mean name movers found |", "|---:|---:|---:|"]
    lines += [f"| {r['templates']} | {r['mean_discoveries']:.1f} | {r['mean_name_movers_found']:.2f} |"
              for r in report["subsample_curve"]]
    if report["placebo"]:
        lines += ["", f"Placebo ({report['placebo']['reps_completed']} reps, reps with any rejection): "
                      + ", ".join(f"{k.replace('_reps_with_rejection', '')}={v}"
                                  for k, v in report["placebo"].items() if k.endswith("rejection"))]
    lines += ["", "## Pre-registered criteria"]
    lines += [f"- [{c['status']}{'' if c['gating'] else ', non-gating'}] {c['name']}: {c['detail']}"
              for c in report["criteria"]]
    return "\n".join(lines) + "\n"


def make_heatmap(results, raw, args, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = results.q_values.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(-np.log10(np.clip(q, 1e-12, 1.0)), cmap="viridis", origin="lower")
    for layer, head in ALL_HEADS:
        ax.plot(head, layer, marker="o", markersize=9, markerfacecolor="none", markeredgecolor="white")
    for layer, head in heads_from_mask(results.significant_mask.detach().cpu().numpy()):
        ax.plot(head, layer, marker="x", markersize=7, color="red")
    ax.set_xlabel("head"), ax.set_ylabel("layer")
    ax.set_title("-log10 q (cluster sign-flip, BH)\nwhite circle = published, red x = discovered")
    fig.colorbar(image, label="-log10 q")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--per-template", type=int, default=6)
    parser.add_argument("--min-templates", type=int, default=14)
    parser.add_argument("--max-pad", type=int, default=3, help="max BOS tokens to left-pad shorter templates")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-perm", type=int, default=9999)
    parser.add_argument("--head-batch-size", type=int, default=4)
    parser.add_argument("--placebo-reps", type=int, default=20)
    parser.add_argument("--subsets", type=int, default=10, help="random template subsets per size in the subsampling curve")
    parser.add_argument("--stability-reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="benchmarks/results/ioi")
    parser.add_argument("--dry-run", action="store_true",
                        help="tokenizer-only template check; downloads tokenizer files if not cached, no model weights")
    parser.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline")
    args = parser.parse_args()
    if args.smoke:
        args.per_template, args.placebo_reps, args.subsets, args.stability_reps = 2, 2, 2, 3
        args.n_perm, args.min_templates = 999, 6
        args.out = args.out.rstrip("/\\") + "_smoke"  # never mix smoke artifacts with real results
    if args.dry_run:
        dry_run(args)
        return
    run(args)


if __name__ == "__main__":
    main()
