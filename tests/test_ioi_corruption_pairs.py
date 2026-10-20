"""Construction and alignment tests for the ABC/name-swap IOI comparison."""

import random
from types import SimpleNamespace

import pytest
import torch

from benchmarks.ioi_gpt2_small import TEMPLATES, build_examples, slot_keys
from benchmarks.ioi_corruption_pairs import (
    EVALUATION_TEMPLATES,
    RETAINED_TEMPLATES,
    SELECTION_TEMPLATES,
    build_paired_examples,
    condition_batch,
    fixed_template_split,
    swap_io_s_names,
    validate_token_alignment,
)


NAMES = ["John", "Mary", "Alice", "Bob", "Carol", "David", "Sarah", "James"]


@pytest.fixture
def examples():
    return build_paired_examples(
        [0, 3], TEMPLATES, 12,
        NAMES, ["park"], ["book"], random.Random(2026),
    )


def test_split_matches_prespecified_template_ids():
    selection, evaluation = fixed_template_split(RETAINED_TEMPLATES)
    assert selection == SELECTION_TEMPLATES
    assert evaluation == EVALUATION_TEMPLATES
    assert len(selection) == 16 and len(evaluation) == 4
    assert set(selection).isdisjoint(evaluation)
    assert set(selection) | set(evaluation) == set(RETAINED_TEMPLATES)


def test_split_rejects_different_retained_set():
    with pytest.raises(ValueError, match="Retained templates"):
        fixed_template_split(RETAINED_TEMPLATES[:-1])


def test_swap_changes_each_occurrence_without_cascading():
    clean = "Then, John and Mary went to the park. Mary gave a book to"
    assert swap_io_s_names(clean, "John", "Mary") == (
        "Then, Mary and John went to the park. John gave a book to"
    )


def test_swap_rejects_missing_or_ambiguous_names():
    with pytest.raises(ValueError, match="distinct"):
        swap_io_s_names("John John John", "John", "John")
    with pytest.raises(ValueError, match="exactly one IO"):
        swap_io_s_names("John and Mary", "John", "Mary")


def test_sampling_matches_existing_abc_benchmark(examples):
    baseline = build_examples(
        [0, 3], TEMPLATES, 12,
        NAMES, ["park"], ["book"], random.Random(2026),
    )
    for paired, original in zip(examples, baseline):
        assert {key: paired[key] for key in original} == original


def test_paired_corruptions_preserve_origins_and_both_orders(examples):
    assert len(examples) == 24
    assert {e["order"] for e in examples} == {"ABBA", "BABA"}
    for example in examples:
        io, s = example["io"], example["s"]
        assert example["clean"] != example["corrupted"]
        assert example["clean"] != example["name_swap"]
        expected = [s if key == "A" else io for key in slot_keys(
            TEMPLATES[example["template"]], example["order"]
        )]
        assert [word for word in example["name_swap"].split()
                if word in {io, s}] == expected
        assert example["name_swap"].count(io) == 2
        assert example["name_swap"].count(s) == 1


class ToyTokenizerModel:
    """One-token-per-word tokenizer for checking batch alignment without a model."""

    def __init__(self):
        self.cfg = SimpleNamespace(device="cpu")
        self.vocab = {"<bos>": 0}

    def token_id(self, word):
        return self.vocab.setdefault(word, len(self.vocab))

    def to_tokens(self, text):
        return torch.tensor([[0] + [self.token_id(word) for word in text.split()]])

    def to_single_token(self, text):
        return self.token_id(text.strip())


def test_alignment_and_original_answer_labels(examples):
    model = ToyTokenizerModel()
    max_length = max(len(model.to_tokens(e["clean"])[0]) for e in examples)
    assert validate_token_alignment(model, examples, max_length, max_pad=3) == len(examples)

    clean_a, corrupted_a, correct_a, incorrect_a, groups_a = condition_batch(
        model, examples, "abc", max_length, torch
    )
    clean_s, corrupted_s, correct_s, incorrect_s, groups_s = condition_batch(
        model, examples, "name_swap", max_length, torch
    )
    assert torch.equal(clean_a, clean_s)
    assert torch.equal(correct_a, correct_s)
    assert torch.equal(incorrect_a, incorrect_s)
    assert groups_a == groups_s
    assert not torch.equal(corrupted_a, corrupted_s)
    assert all(example["io"] != example["s"] for example in examples)


def test_alignment_rejects_extra_nonname_change(examples):
    model = ToyTokenizerModel()
    max_length = max(len(model.to_tokens(e["clean"])[0]) for e in examples)
    modified = [dict(e) for e in examples]
    modified[0]["name_swap"] = modified[0]["name_swap"].replace("park", "school")
    with pytest.raises(ValueError, match="name-token positions"):
        validate_token_alignment(model, modified, max_length, max_pad=3)


def test_alignment_rejects_token_length_change(examples):
    model = ToyTokenizerModel()
    max_length = max(len(model.to_tokens(e["clean"])[0]) for e in examples)
    modified = [dict(e) for e in examples]
    modified[0]["name_swap"] += " tomorrow"
    with pytest.raises(ValueError, match="lengths differ"):
        validate_token_alignment(model, modified, max_length, max_pad=3)


def test_condition_batch_rejects_unknown_condition(examples):
    with pytest.raises(ValueError, match="Condition"):
        condition_batch(ToyTokenizerModel(), examples, "other", 30, torch)
