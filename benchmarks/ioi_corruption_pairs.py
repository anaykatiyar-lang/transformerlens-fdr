"""Matched ABC and name-swap corruptions for the GPT-2 Small IOI comparison.

The original Phase 0 benchmark supplies prompt sampling, tokenization, and
left-padding. This module keeps those conventions without modifying Phase 0.
"""

import random
import re

from benchmarks.ioi_gpt2_small import build_examples, to_batch


SPLIT_SEED = 20261019
RETAINED_TEMPLATES = (
    0, 1, 3, 7, 9, 10, 13, 14, 15, 18,
    19, 20, 21, 22, 23, 24, 25, 27, 29, 31,
)
SELECTION_TEMPLATES = (
    0, 1, 3, 9, 10, 14, 15, 18, 20, 21, 22, 23, 24, 25, 27, 31,
)
EVALUATION_TEMPLATES = (7, 13, 19, 29)


def fixed_template_split(retained_indices):
    """Return the frozen selection/evaluation split, checking the retained set."""
    retained = tuple(sorted(retained_indices))
    if retained != RETAINED_TEMPLATES:
        raise ValueError("Retained templates differ from the committed analysis plan.")

    shuffled = list(retained)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    selection = tuple(sorted(shuffled[:16]))
    evaluation = tuple(sorted(shuffled[16:]))
    if (selection, evaluation) != (SELECTION_TEMPLATES, EVALUATION_TEMPLATES):
        raise AssertionError("Frozen template split does not match the recorded seed.")
    return selection, evaluation


def swap_io_s_names(prompt, io_name, s_name):
    """Exchange the two complete name tokens without cascading replacements.

    IO and S labels refer to the original clean prompt, not the swapped prompt.
    The input must be a benchmark-style IOI prompt with one IO and two S mentions.
    """
    if not io_name or not s_name or io_name == s_name:
        raise ValueError("IO and S names must be distinct nonempty strings.")
    pattern = re.compile(
        r"(?<!\w)(?:" + re.escape(io_name) + "|" + re.escape(s_name) + r")(?!\w)"
    )
    matches = pattern.findall(prompt)
    if matches.count(io_name) != 1 or matches.count(s_name) != 2:
        raise ValueError("Expected exactly one IO and two S mentions in the clean prompt.")
    return pattern.sub(
        lambda match: s_name if match.group() == io_name else io_name,
        prompt,
    )


def build_paired_examples(template_indices, templates, per_template, names, places, objects, rng):
    """Generate the two corruptions from identical clean examples.

    Each result keeps Phase 0's ``corrupted`` field for ABC and adds ``name_swap``.
    The ``io`` and ``s`` fields always retain their original clean-task roles.
    """
    baseline = build_examples(
        template_indices, templates, per_template, names, places, objects, rng
    )
    paired = []
    for example in baseline:
        item = dict(example)
        item["name_swap"] = swap_io_s_names(
            item["clean"], item["io"], item["s"]
        )
        paired.append(item)
    return paired


def validate_token_alignment(model, examples, target_length, max_pad=3):
    """Check that only the three name-token positions change in each corruption.

    ``target_length`` includes BOS, matching ``to_batch`` in Phase 0. This
    validates unpadded token IDs; left padding therefore preserves positions.
    """
    for index, example in enumerate(examples):
        clean = model.to_tokens(example["clean"])[0].tolist()
        abc = model.to_tokens(example["corrupted"])[0].tolist()
        swap = model.to_tokens(example["name_swap"])[0].tolist()
        lengths = {len(clean), len(abc), len(swap)}
        if len(lengths) != 1:
            raise ValueError(f"Example {index}: clean and corruption lengths differ.")
        if not (target_length - max_pad <= len(clean) <= target_length):
            raise ValueError(f"Example {index}: sequence falls outside padding window.")
        abc_changes = {i for i, (a, b) in enumerate(zip(clean, abc)) if a != b}
        swap_changes = {i for i, (a, b) in enumerate(zip(clean, swap)) if a != b}
        if len(abc_changes) != 3 or abc_changes != swap_changes:
            raise ValueError(f"Example {index}: name-token positions are not aligned.")
    return len(examples)


def condition_batch(model, examples, condition, target_length, torch):
    """Prepare audit inputs while preserving the original IO and S labels.

    Returns Phase 0's ``(clean, corrupted, correct, incorrect, groups)`` tuple.
    """
    if condition not in ("abc", "name_swap"):
        raise ValueError("Condition must be 'abc' or 'name_swap'.")
    if condition == "abc":
        selected = examples
    else:
        selected = [dict(example, corrupted=example["name_swap"]) for example in examples]
    return to_batch(model, selected, target_length, torch)
