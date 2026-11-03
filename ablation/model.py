"""
The GPT model, rebuilt from Karpathy's nanoGPT (model.py) with readable names.

With the default settings this is the same network as nanoGPT's Shakespeare-char model:
learned position embeddings, pre-norm LayerNorm, GELU feed-forward layers and an output
layer that shares its weights with the token embedding.

One structural difference from nanoGPT: the query, key and value projections are three
separate Linear layers instead of one fused layer. Under AdamW this changes nothing,
since every weight is initialised the same way and updated on its own. It matters for
Muon, which treats each weight matrix as a whole and should see Q, K and V separately.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    context_length: int = 256
    vocabulary_size: int = 65
    number_of_layers: int = 6
    number_of_heads: int = 6
    embedding_size: int = 384
    dropout: float = 0.2
    use_bias: bool = False
    normalization: str = "layernorm"  # "layernorm" (baseline) or "rmsnorm"
    position_encoding: str = "learned"  # "learned" (baseline) or "rope"
    rope_base: float = 10000.0


class LayerNorm(nn.Module):
    """
    LayerNorm with an optional bias. nanoGPT uses this because older PyTorch versions
    didn't allow turning the bias off. Subtracts the mean, divides by the standard
    deviation, then multiplies by a learned gain.
    """

    def __init__(self, size, use_bias, epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if use_bias else None
        self.epsilon = epsilon

    def forward(self, inputs):
        return F.layer_norm(inputs, self.weight.shape, self.weight, self.bias, self.epsilon)


class RMSNorm(nn.Module):
    """
    RMSNorm (Zhang and Sennrich, 2019). Divides by the root mean square of the input and
    multiplies by a learned gain. Compared with LayerNorm it skips subtracting the mean,
    so the only thing it controls is the overall size of the vector, not where it's centred.

        output = input / sqrt(mean(input^2) + epsilon) * gain

    I use the same epsilon as the LayerNorm above so that isn't a second difference.
    """

    def __init__(self, size, epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.epsilon = epsilon

    def forward(self, inputs):
        # Compute in float32 even under mixed precision, since squaring small numbers in
        # bfloat16 loses a lot of accuracy.
        inputs_float = inputs.float()
        inverse_rms = torch.rsqrt(inputs_float.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return (inputs_float * inverse_rms).type_as(inputs) * self.weight


def make_norm(config):
    if config.normalization == "layernorm":
        return LayerNorm(config.embedding_size, config.use_bias)
    if config.normalization == "rmsnorm":
        return RMSNorm(config.embedding_size)
    raise ValueError(f"unknown normalization: {config.normalization}")


class RotaryPositionEmbedding(nn.Module):
    """
    RoPE (Su et al., RoFormer). Instead of adding a learned position vector to the input,
    each query and key vector is rotated by an angle that grows with its position.
    The head dimensions are treated as pairs, and each pair rotates at its own speed:
    the first pairs turn quickly, the last ones very slowly.

    The useful part is that the dot product between a query at position m and a key at
    position n only depends on m - n, because rotating both and comparing them leaves
    just the difference of the two angles. So attention sees relative distance.

    I pair dimension i with dimension i + head_size/2 ("rotate half", as in GPT-NeoX and
    LLaMA) instead of neighbouring dimensions like the paper. That is just a fixed
    reordering of the dimensions inside a head and gives the same attention scores.
    """

    def __init__(self, head_size, context_length, base):
        super().__init__()
        assert head_size % 2 == 0, "RoPE needs an even head size"
        pair_index = torch.arange(0, head_size, 2).float()
        rotation_speeds = 1.0 / (base ** (pair_index / head_size))
        positions = torch.arange(context_length).float()
        angles = torch.outer(positions, rotation_speeds)  # (context_length, head_size / 2)
        # Precompute once. The two leading 1s let these broadcast over batch and heads.
        self.register_buffer("cosine", angles.cos()[None, None], persistent=False)
        self.register_buffer("sine", angles.sin()[None, None], persistent=False)

    def forward(self, vectors):
        # vectors: (batch, heads, sequence, head_size)
        sequence_length = vectors.size(2)
        cosine = self.cosine[:, :, :sequence_length].to(vectors.dtype)
        sine = self.sine[:, :, :sequence_length].to(vectors.dtype)
        first_half, second_half = vectors.chunk(2, dim=-1)
        # The usual 2D rotation, applied to every (first_half[i], second_half[i]) pair.
        rotated_first = first_half * cosine - second_half * sine
        rotated_second = first_half * sine + second_half * cosine
        return torch.cat([rotated_first, rotated_second], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_size % config.number_of_heads == 0
        size = config.embedding_size
        self.query = nn.Linear(size, size, bias=config.use_bias)
        self.key = nn.Linear(size, size, bias=config.use_bias)
        self.value = nn.Linear(size, size, bias=config.use_bias)
        self.output_projection = nn.Linear(size, size, bias=config.use_bias)
        self.attention_dropout = config.dropout
        self.output_dropout = nn.Dropout(config.dropout)
        self.number_of_heads = config.number_of_heads
        self.head_size = size // config.number_of_heads
        self.rotary = None
        if config.position_encoding == "rope":
            self.rotary = RotaryPositionEmbedding(self.head_size, config.context_length, config.rope_base)

    def split_into_heads(self, tensor):
        # (batch, sequence, embedding) -> (batch, heads, sequence, head_size)
        batch_size, sequence_length, _ = tensor.shape
        tensor = tensor.view(batch_size, sequence_length, self.number_of_heads, self.head_size)
        return tensor.transpose(1, 2)

    def forward(self, inputs):
        batch_size, sequence_length, embedding_size = inputs.shape
        queries = self.split_into_heads(self.query(inputs))
        keys = self.split_into_heads(self.key(inputs))
        values = self.split_into_heads(self.value(inputs))
        if self.rotary is not None:
            # Only queries and keys get rotated. The attention scores come from them,
            # while the values are the content being passed along.
            queries = self.rotary(queries)
            keys = self.rotary(keys)

        # is_causal=True applies the mask, so position t can only look at positions <= t.
        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )

        # Put the heads back side by side and mix them with the output projection.
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size)
        return self.output_dropout(self.output_projection(attended))


class FeedForward(nn.Module):
    """The MLP in each block: widen to 4x the embedding size, apply GELU, project back."""

    def __init__(self, config):
        super().__init__()
        size = config.embedding_size
        self.input_projection = nn.Linear(size, 4 * size, bias=config.use_bias)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(4 * size, size, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        hidden = self.activation(self.input_projection(inputs))
        return self.dropout(self.output_projection(hidden))


class TransformerBlock(nn.Module):
    """Pre-norm block: normalise, then attention or feed-forward, then add back to the residual stream."""

    def __init__(self, config):
        super().__init__()
        self.attention_norm = make_norm(config)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = make_norm(config)
        self.feed_forward = FeedForward(config)

    def forward(self, residual):
        residual = residual + self.attention(self.attention_norm(residual))
        residual = residual + self.feed_forward(self.feed_forward_norm(residual))
        return residual


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.position_encoding not in ("learned", "rope"):
            raise ValueError(f"unknown position encoding: {config.position_encoding}")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.embedding_size)
        # With RoPE the position information lives inside attention, so there is no table.
        self.position_embedding = None
        if config.position_encoding == "learned":
            self.position_embedding = nn.Embedding(config.context_length, config.embedding_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.number_of_layers)])
        self.final_norm = make_norm(config)
        self.output_head = nn.Linear(config.embedding_size, config.vocabulary_size, bias=False)

        # Weight tying, as in GPT-2 and nanoGPT: the same matrix turns ids into vectors
        # at the input and vectors back into scores over characters at the output.
        self.token_embedding.weight = self.output_head.weight

        self.apply(self.initialise_weights)
        # GPT-2 scales down the layers that write into the residual stream, so the stream
        # doesn't grow with depth. nanoGPT does the same thing for its c_proj layers.
        for name, parameter in self.named_parameters():
            if name.endswith("output_projection.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.number_of_layers))

    def initialise_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self, include_position_embedding=False):
        """Like nanoGPT, the position table is left out of the count by default."""
        total = sum(parameter.numel() for parameter in self.parameters())
        if self.position_embedding is not None and not include_position_embedding:
            total -= self.position_embedding.weight.numel()
        return total

    def forward(self, token_ids, targets=None):
        batch_size, sequence_length = token_ids.shape
        assert sequence_length <= self.config.context_length, "sequence is longer than the context length"

        hidden = self.token_embedding(token_ids)
        if self.position_embedding is not None:
            positions = torch.arange(sequence_length, device=token_ids.device)
            hidden = hidden + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.output_head(self.final_norm(hidden))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def parameter_groups(self):
        """
        Split the parameters into the groups the optimizers need:

          hidden_matrices  2D weights inside the transformer blocks (the ones Muon can take over)
          other_matrices   token embedding / tied output head (always AdamW, with weight decay)
          no_decay         norm gains and biases (always AdamW, no weight decay)

        Under AdamW the first two groups get the same settings, so this is the same
        decay / no-decay split that nanoGPT uses.
        """
        hidden_matrices, other_matrices, no_decay = [], [], []
        already_seen = set()
        for name, parameter in self.named_parameters():
            # named_parameters() skips the tied output head, but check anyway so that
            # a parameter can never end up in two groups.
            if id(parameter) in already_seen:
                continue
            already_seen.add(id(parameter))
            if parameter.dim() < 2:
                no_decay.append(parameter)
            elif name.startswith("blocks."):
                hidden_matrices.append(parameter)
            else:
                other_matrices.append(parameter)
        return hidden_matrices, other_matrices, no_decay

    @torch.no_grad()
    def generate(self, token_ids, number_of_new_tokens, temperature=1.0, top_k=None):
        """Sample characters one at a time, feeding each one back in as input."""
        for _ in range(number_of_new_tokens):
            context = token_ids[:, -self.config.context_length :]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                kth_largest = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1))).values[:, [-1]]
                next_token_logits[next_token_logits < kth_largest] = -float("inf")
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids
