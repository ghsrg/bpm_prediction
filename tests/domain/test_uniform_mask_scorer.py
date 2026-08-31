import pytest
import torch

from src.domain.services.uniform_mask_scorer import UniformMaskScorer


def test_score_is_uniform_without_inventing_a_top_one_prediction():
    result = UniformMaskScorer().score(
        allowed_mask=torch.tensor([[False, True, True]]),
        candidate_keys=("z", "b", "a"),
    )
    assert torch.equal(result.probabilities, torch.tensor([[0.0, 0.5, 0.5]]))
    assert result.mask_cardinality.tolist() == [2]
    assert result.invalid_rows.tolist() == [False]


def test_seeded_draw_is_reproducible_and_candidate_order_invariant():
    scorer = UniformMaskScorer()
    first_idx = scorer.sample_prediction(
        allowed_mask=torch.tensor([[True, False, True]]),
        candidate_keys=("node_B", "node_C", "node_A"),
        evaluation_seed=41,
        draw_index=7,
        sample_key="v2/trace-4/prefix-2",
    )
    second_idx = scorer.sample_prediction(
        allowed_mask=torch.tensor([[True, True, False]]),
        candidate_keys=("node_A", "node_B", "node_C"),
        evaluation_seed=41,
        draw_index=7,
        sample_key="v2/trace-4/prefix-2",
    )
    assert ("node_B", "node_C", "node_A")[first_idx] == ("node_A", "node_B", "node_C")[second_idx]


def test_seeded_draws_are_uniform_over_allowed_candidates():
    scorer = UniformMaskScorer()
    choices = [
        scorer.sample_prediction(
            allowed_mask=torch.tensor([[True, True, True]]),
            candidate_keys=("node_A", "node_B", "node_C"),
            evaluation_seed=20260831,
            draw_index=draw_index,
            sample_key="v2/trace-4/prefix-2",
        )
        for draw_index in range(3000)
    ]
    counts = [choices.count(candidate_index) for candidate_index in range(3)]
    assert all(abs(count / len(choices) - (1.0 / 3.0)) < 0.04 for count in counts)


def test_score_raises_for_empty_row_in_research_policy():
    with pytest.raises(ValueError, match="Uniform mask contains an empty candidate row\\."):
        UniformMaskScorer().score(
            allowed_mask=torch.tensor([[False, False]]),
            candidate_keys=("A", "B"),
        )


def test_score_records_empty_row_without_candidate_space_fallback():
    result = UniformMaskScorer(empty_mask_policy="record_invalid").score(
        allowed_mask=torch.tensor([[False, False], [True, False]]),
        candidate_keys=("A", "B"),
    )
    assert result.invalid_rows.tolist() == [True, False]
    assert torch.equal(result.probabilities, torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
