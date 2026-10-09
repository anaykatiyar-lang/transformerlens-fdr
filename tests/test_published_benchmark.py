"""Reference-data regression for the published Benjamini-Hochberg example."""

import torch

from transformerlens_fdr.stats import apply_fdr_adjustment


def test_bh_1995_published_example_rejects_first_four_hypotheses():
    """The package matches BH's published decision at target FDR 0.05.

    Benjamini & Hochberg, "Controlling the False Discovery Rate: A Practical
    and Powerful Approach to Multiple Testing," JRSS B 57(1), 289–300 (1995),
    doi:10.1111/j.2517-6161.1995.tb02031.x, Example 3.2.
    """
    ordered_p_values = torch.tensor(
        [
            0.0001, 0.0004, 0.0019, 0.0095, 0.0201,
            0.0278, 0.0298, 0.0344, 0.0459, 0.3240,
            0.4262, 0.5719, 0.6528, 0.7590, 1.0000,
        ],
        dtype=torch.float64,
    )

    _, rejected = apply_fdr_adjustment(
        ordered_p_values, alpha=0.05, method="fdr_bh"
    )

    # The paper identifies p_(4)=0.0095 as the step-up cutoff.
    assert torch.where(rejected)[0].tolist() == [0, 1, 2, 3]
