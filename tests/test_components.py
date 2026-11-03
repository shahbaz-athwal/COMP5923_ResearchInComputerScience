"""
Checks that each component does what its paper says it does, and that each
condition changes only the one thing it is supposed to change.

    python -m pytest tests
"""

import torch

from ablation.model import GPT, LayerNorm, ModelConfig, RMSNorm, RotaryPositionEmbedding
from ablation.muon import orthogonalize
from ablation.settings import CONDITIONS, DEFAULT_SETTINGS
from ablation.train import build_optimizers, learning_rate_multiplier

TINY_MODEL = dict(
    context_length=32, vocabulary_size=65, number_of_layers=2, number_of_heads=2, embedding_size=32, dropout=0.0
)


def make_tiny_model(**changes):
    torch.manual_seed(0)
    return GPT(ModelConfig(**{**TINY_MODEL, **changes}))


def model_changes_for(condition):
    # The muon condition changes the optimizer, not the model, so leave that key out.
    return {name: value for name, value in CONDITIONS[condition].items() if name != "optimizer"}


def test_rope_scores_depend_only_on_relative_position():
    rotary = RotaryPositionEmbedding(head_size=16, context_length=64, base=10000.0)
    query, key = torch.randn(16), torch.randn(16)
    # Put the same query and the same key at every position, then rotate them.
    queries = rotary(query.expand(1, 1, 64, 16))[0, 0]
    keys = rotary(key.expand(1, 1, 64, 16))[0, 0]

    # Positions (10, 7) and (40, 37) are both 3 apart, so the scores should match.
    assert torch.allclose(queries[10] @ keys[7], queries[40] @ keys[37], atol=1e-5)
    # Rotating a vector never changes its length.
    assert torch.allclose(queries.norm(dim=-1), query.norm().expand(64), atol=1e-5)


def test_rmsnorm_rescales_without_centering():
    inputs = torch.randn(4, 10, 32) * 3 + 5  # deliberately far from zero mean
    rmsnorm_output = RMSNorm(32)(inputs)
    layernorm_output = LayerNorm(32, use_bias=False)(inputs)

    # RMSNorm makes the root mean square 1 ...
    assert torch.allclose(rmsnorm_output.pow(2).mean(-1), torch.ones(4, 10), atol=1e-3)
    # ... but leaves the mean where it was, unlike LayerNorm which moves it to 0.
    assert rmsnorm_output.mean(-1).abs().min() > 0.5
    assert layernorm_output.mean(-1).abs().max() < 1e-4


def test_newton_schulz_brings_singular_values_close_to_one():
    torch.manual_seed(0)
    for shape in [(64, 64), (128, 32), (32, 128)]:
        singular_values = torch.linalg.svdvals(orthogonalize(torch.randn(shape)).float())
        # Five steps land the singular values roughly in [0.7, 1.2]. A random square matrix
        # often has one tiny singular value that five steps can't lift, so I check the bulk
        # of them (95%) rather than the very smallest one.
        assert singular_values.max() < 1.3, shape
        assert torch.quantile(singular_values, 0.05) > 0.6, shape


def test_models_cannot_see_the_future():
    for condition in ["baseline", "rope", "rmsnorm"]:
        model = make_tiny_model(**model_changes_for(condition)).eval()
        token_ids = torch.randint(0, 65, (1, 20))
        changed_future = token_ids.clone()
        changed_future[0, 15:] = (changed_future[0, 15:] + 1) % 65

        original_logits, _ = model(token_ids)
        changed_logits, _ = model(changed_future)
        # Changing characters 15+ must not change the predictions at positions 0-14.
        assert torch.allclose(original_logits[0, :15], changed_logits[0, :15], atol=1e-5), condition


def test_each_condition_changes_only_its_component():
    baseline = make_tiny_model()
    rope = make_tiny_model(position_encoding="rope")
    rmsnorm = make_tiny_model(normalization="rmsnorm")

    def names_and_shapes(model):
        return {(name, tuple(parameter.shape)) for name, parameter in model.named_parameters()}

    # RoPE: exactly the baseline's parameters, minus the learned position table.
    position_table = ("position_embedding.weight", (32, 32))
    assert names_and_shapes(rope) == names_and_shapes(baseline) - {position_table}
    # RMSNorm: identical parameters, because the baseline LayerNorm has no bias either.
    assert names_and_shapes(rmsnorm) == names_and_shapes(baseline)
    assert baseline.count_parameters() == rope.count_parameters() == rmsnorm.count_parameters()


def test_every_parameter_goes_to_exactly_one_optimizer():
    model = make_tiny_model()
    hidden_matrices, other_matrices, no_decay = model.parameter_groups()
    grouped_ids = [id(parameter) for parameter in hidden_matrices + other_matrices + no_decay]
    all_ids = {id(parameter) for parameter in model.parameters()}
    assert len(grouped_ids) == len(set(grouped_ids)) == len(all_ids)
    assert all(parameter.dim() == 2 for parameter in hidden_matrices)
    assert all(parameter.dim() == 1 for parameter in no_decay)
    # The embedding (tied with the output head) must never be given to Muon.
    assert any(parameter is model.token_embedding.weight for parameter in other_matrices)

    adamw, muon = build_optimizers(model, {**DEFAULT_SETTINGS, "optimizer": "muon"}, "cpu")
    assert sum(len(group["params"]) for group in muon.param_groups) == len(hidden_matrices)
    assert sum(len(group["params"]) for group in adamw.param_groups) == len(other_matrices) + len(no_decay)


def test_learning_rate_schedule_matches_nanogpt():
    settings = {**DEFAULT_SETTINGS, "decay_steps": 5000}
    assert learning_rate_multiplier(0, settings) < 0.02  # starts near zero
    assert abs(learning_rate_multiplier(100, settings) - 1.0) < 1e-9  # peak right after warmup
    assert abs(learning_rate_multiplier(5000, settings) - 0.1) < 1e-9  # ends at a tenth of the peak
