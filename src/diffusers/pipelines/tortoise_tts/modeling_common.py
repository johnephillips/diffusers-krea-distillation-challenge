import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin
from ...models.attention_processor import AttnProcessor
from ...utils import BaseOutput, logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TortoiseTTSAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        n_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=True,
        out_bias: bool = True,
        scale_qk: bool = True,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
    ):
        super().__init__()
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.query_dim = query_dim
        self.key_value_proj_dim = dim_head
        self.n_heads = n_heads
        self.dropout = dropout
        self.scale_qk = scale_qk
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        # bias set to True for
        self.q = nn.Linear(self.query_dim, self.inner_dim, bias=bias)
        self.k = nn.Linear(self.query_dim, self.inner_dim, bias=bias)
        self.v = nn.Linear(self.query_dim, self.inner_dim, bias=bias)
        self.o = nn.Linear(self.inner_dim, self.query_dim, bias=out_bias)

        self.norm = nn.GroupNorm(num_groups=32, num_channels=self.query_dim)

        self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.norm(hidden_states).transpose(1, 2)

        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        scale = 1 / math.sqrt(self.query_dim // self.n_heads) if self.scale_qk else 1.0

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states * scale, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        # scores += position_bias_masked
        scores += (
            position_bias_masked * 8
        )  # its actually root under the dimension of each attn head will be updated in the final version

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs


class TortoiseTTSSelfAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        n_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        out_bias: bool = True,
        scale_qk: bool = True,
        norm_num_groups: int = 32,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention = TortoiseTTSAttention(
            query_dim=query_dim,
            n_heads=n_heads,
            dim_head=dim_head,
            dropout=dropout,
            bias=bias,
            out_bias=out_bias,
            scale_qk=scale_qk,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
        )
        self.layer_norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        normed_hidden_states = torch.permute(normed_hidden_states, (0, 2, 1))

        attention_output = self.attention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        hidden_states = torch.permute(hidden_states, (0, 2, 1))
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# From tortoise.models.random_latent_generator.fused_leaky_relu
# https://github.com/152334H/tortoise-tts-fast/blob/main/tortoise/models/random_latent_generator.py#L8
def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2**0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim),
                negative_slope=negative_slope,
            )
            * scale
        )
    else:
        return F.leaky_relu(input, negative_slope=0.2) * scale


# From tortoise.models.random_latent_generator.EqualLinear
# https://github.com/152334H/tortoise-tts-fast/blob/main/tortoise/models/random_latent_generator.py#L22
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale)
        out = fused_leaky_relu(out, self.bias * self.lr_mul)
        return out


@dataclass
class RandomLatentConverterOutput(BaseOutput):
    """
    The output of [`RandomLatentConverter`].

    Args:
        TODO: fix
        latents (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    latents: torch.FloatTensor


# Based on tortoise.models.random_latent_generator.RandomLatentConverter
# https://github.com/152334H/tortoise-tts-fast/blob/main/tortoise/models/random_latent_generator.py#L39
class RandomLatentConverter(ModelMixin, ConfigMixin):
    """
    Converts standard Gaussian noise to random latents suitable for use as conditioning embeddings in place of output
    from a [`ConditioningEncoder`] class, when no conditioning audio is available.

    Parameters:
        channels (`int`):
            The number of input channels of the incoming Gaussian noise tensors.
        num_equallinear_layers (`int`, *optional*, defaults to 5):
            The number of `EqualLinear` layers to use (before the final linear layer).
        lr_mul (`float`, *optional*, defaults to 0.1):
            TODO
    """

    @register_to_config
    def __init__(self, channels: int, num_equallinear_layers: int = 5, lr_mul: float = 0.1):
        super().__init__()

        self.equallinear = nn.ModuleList(
            [EqualLinear(channels, channels, lr_mul=lr_mul) for _ in range(num_equallinear_layers)]
        )
        self.linear = nn.Linear(channels, channels)

    def forward(self, noise: torch.FloatTensor, return_dict: bool = True):
        """
        Converts standard Gaussian noise into latents.

        Args:
            noise (`torch.FloatTensor`):
                A tensor of standard Gaussian noise (e.g. from `torch.randn`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`RandomLatentConverterOutput`] instead of a plain tuple.

        Returns:
            [`RandomLatentConverterOutput`] or `tuple`:
            [`RandomLatentConverterOutput`] if `return_dict` is `True`, otherwise a `tuple`.
            When returning a tuple the first element is the rnadom latents.
        """
        assert noise.shape[-1] == self.config.channels, "The last dim of `noise` must match `self.config.channels`."

        for equallinear_layer in self.equallinear:
            noise = equallinear_layer(noise)
        latents = self.linear(noise)

        if not return_dict:
            return (latents,)

        return RandomLatentConverterOutput(latents=latents)
