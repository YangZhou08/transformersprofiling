# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)

from .configuration_llama import LlamaConfig 
from termcolor import colored 

import torch
import matplotlib.pyplot as plt
import numpy as np

from ...utils.import_utils import is_torch_fx_available
from .configuration_llama import LlamaConfig 


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils.AttentionMaskConverter._prepare_4d_attention_mask"
    )
    # return AttentionMaskConverter._prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)
    return AttentionMaskConverter._expand_mask(mask = mask, dtype = dtype, tgt_len = tgt_len) 


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.llama.modeling_llama.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    ) 

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_legacy(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None): 
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min) 


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        ) 

class LlamaRotaryEmbeddingqksep(nn.Module): 
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device = None, key_scaling_factor = 2.0): 
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype(), key_scaling_factor = key_scaling_factor, 
        ) 

    def _set_cos_sin_cache(self, seq_len, device, dtype, key_scaling_factor): 
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype) 

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        # self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False) 
        # self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False) 
        self.register_buffer("cosq_cached", emb.cos().to(dtype), persistent = False) 
        self.register_buffer("sinq_cached", emb.sin().to(dtype), persistent = False) 

        t_two = t / key_scaling_factor 

        freqs_two = torch.einsum("i,j->ij", t_two, self.inv_freq) 
        emb_two = torch.cat((freqs_two, freqs_two), dim = -1) 
        self.register_buffer("cosk_cached", emb_two.cos().to(dtype), persistent = False) 
        self.register_buffer("sink_cached", emb_two.sin().to(dtype), persistent = False) 

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            # self.cos_cached[:seq_len].to(dtype=x.dtype), 
            # self.sin_cached[:seq_len].to(dtype=x.dtype), 
            self.cosq_cached[:seq_len].to(dtype = x.dtype), 
            self.sinq_cached[:seq_len].to(dtype = x.dtype), 
            self.cosk_cached[:seq_len].to(dtype = x.dtype), 
            self.sink_cached[:seq_len].to(dtype = x.dtype), 
        ) 

class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed 

def apply_differently_rotary_pos_emb(q, k, cosq, sinq, cosk, sink, position_ids, unsqueeze_dim = 1): 
    cosq = cosq[position_ids].unsqueeze(unsqueeze_dim) 
    sinq = sinq[position_ids].unsqueeze(unsqueeze_dim) 

    cosk = cosk[position_ids].unsqueeze(unsqueeze_dim) 
    sink = sink[position_ids].unsqueeze(unsqueeze_dim) 

    q_embed = (q * cosq) + (rotate_half(q) * sinq) 
    k_embed = (k * cosk) + (rotate_half(k) * sink) 

    return q_embed, k_embed 

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                ) 
            elif scaling_type == "sep_q_k": # NOTE I added to separate the q and k embeddings 
                print(colored("We got hee", "red")) 
                self.rotary_emb = LlamaRotaryEmbeddingqksep( 
                    self.head_dim, 
                    max_position_embeddings = self.max_position_embeddings, 
                    base = self.rope_theta, 
                    key_scaling_factor = scaling_factor, 
                ) 
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2] 
        if isinstance(self.rotary_emb, LlamaRotaryEmbeddingqksep): 
            print(colored("We Got Here!!!", "red")) 
            cosq, sinq, cosk, sink = self.rotary_emb(value_states, seq_len = kv_seq_len) 
            query_states, key_states = apply_differently_rotary_pos_emb(query_states, key_states, cosq, sinq, cosk, sink, position_ids) 
        else: 
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len) 
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids) 
        

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        '''
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
        # Note the next line is critical, since right now the softmax of all the values -inf is a very strange number 
        if "mask_list_pos" in kwargs: 
            # print("found it") 
            mask = torch.ones((attn_weights.shape[-2], attn_weights.shape[-1]), device = attn_weights.device) 
            mask[kwargs["mask_list_pos"], :] = 0.0 
            # attn_weights[:, :, kwargs["mask_list_pos"], :] = 0.0 
            attn_weights = attn_weights * mask 
        '''

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
        # Note the next line is critical, since right now the softmax of all the values -inf is a very strange number 
        if "mask_list_pos" in kwargs: 
            # print("found it") 
            mask = torch.ones((attn_weights.shape[-2], attn_weights.shape[-1]), device = attn_weights.device) 
            mask[kwargs["mask_list_pos"], :] = 0.0 
            # attn_weights[:, :, kwargs["mask_list_pos"], :] = 0.0 
            attn_weights = attn_weights * mask 
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=self.is_causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            LlamaAttention(config=config)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else LlamaFlashAttention2(config=config)
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ) 


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```""" 

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

class LlamaModelWeirdAttentionMap(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask 
    
    # def _modify_attention_mask_in_a_weird_way(self, combined_attention_mask, dtype, mask_list_pos, start_idx = None, kernel_size = None): 
    def _modify_attention_mask_in_a_weird_way(self, combined_attention_mask, dtype, start_idx = None, kernel_size = None): 
        mask_shape = combined_attention_mask.shape # (batch_size, 1, tgt_seq_len, src_seq_len) 
        seq_len = mask_shape[-1] 
        start_idx = start_idx if start_idx is not None else self.start_idx 
        kernel_size = kernel_size if kernel_size is not None else self.sliding_window_length 
        
        # row dimensional masking 
        # row_idx_masked_out = [start_idx + i * (kernel_size + 1) for i in range((seq_len - start_idx) / (kernel_size + 1))] 
        row_mask = torch.zeros(mask_shape[-2], mask_shape[-1], device = combined_attention_mask.device) # NOTE currently, this line only works for training 
        # row_mask[row_idx_masked_out] = 1 
        # row_mask[mask_list_pos] = 1 
        mask_list_pos = [start_idx + i * kernel_size for i in range((seq_len - start_idx) // kernel_size)] 
        row_mask[mask_list_pos[0] :, 0 : mask_list_pos[0]] = 1 

        # column dimensional masking 
        # condensed_token_idx_list = row_idx_masked_out 
        condensed_token_idx_list = mask_list_pos 
        for i in range(len(condensed_token_idx_list) - 1): 
            # row_mask[:, :, condensed_token_idx_list[i + 1] :, condensed_token_idx_list[i]] = 1 
            # row_mask[condensed_token_idx_list[i + 1] :, condensed_token_idx_list[i]] = 1 
            row_mask[condensed_token_idx_list[i + 1] : , condensed_token_idx_list[i] : condensed_token_idx_list[i + 1]] = 1 
        # print("row mask shape {}".format(row_mask.shape)) 
        row_mask = row_mask[None, None, :, :].expand(mask_shape).to(torch.bool) 
        row_mask = row_mask.to(device = combined_attention_mask.device) 

        combined_attention_mask.masked_fill_(row_mask, torch.finfo(dtype).min) 

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            ) 
        self._modify_attention_mask_in_a_weird_way(attention_mask, inputs_embeds.dtype, start_idx = 64, kernel_size = 4) 

        # visualize the attention mask 

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ) 

class LlamaCausalLMWeirdTwo(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelWeirdAttentionMap(config) 
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model 
    
    @staticmethod 
    def plot_attention_map(attention_maps, layer_num, head_num, seq_length, filename):
        """
        Plots the attention map for a specific layer and head and saves it to a file.

        :param attention_maps: A nested list or array of attention maps from the transformer model.
        :param layer_num: The layer number to visualize.
        :param head_num: The head number to visualize.
        :param seq_length: The sequence length of the model.
        :param filename: The filename to save the plot.
        """

        import matplotlib.colors as mcolors

        # Extract the specific attention map
        # attention_map = attention_maps[layer_num][head_num] 
        attention_map = attention_maps[layer_num][0][head_num].cpu().detach().numpy() 
        
        # Create a mask for exact zeros
        zero_mask = attention_map == 0
        '''
        # Create a custom colormap
        colors = [(plt.cm.bwr(i)) for i in range(256)]
        # midpoint = np.where(np.linspace(-1, 1, 256) == 0)[0][0] 
        midpoint = np.abs(np.linspace(-1, 1, 256)).argmin() 
        colors[midpoint] = (0, 0, 0, 1)
        new_colormap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)
        ''' 
        # Define a new color dictionary
        cdict = {
            'red':   ((0.0, 0.0, 0.0),   # Black
                    (0.25, 1.0, 1.0),  # Red
                    (0.5, 1.0, 1.0),   # Yellow (1.0, 1.0, 0.0) -> Red + Green
                    (0.75, 0.0, 0.0),  # Green
                    (1.0, 0.0, 0.0)),  # Blue

            'green': ((0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 1.0, 1.0),   # Yellow
                    (0.75, 1.0, 1.0),  # Green
                    (1.0, 0.0, 0.0)),

            'blue':  ((0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 0.0, 0.0),   # Yellow has no blue component
                    (0.75, 0.0, 0.0),  # Green
                    (1.0, 1.0, 1.0))   # Blue
        }

        custom_cmap = mcolors.LinearSegmentedColormap('custom_colormap', cdict)
        new_colormap = custom_cmap 

        # Normalization
        max_val = np.max(attention_map)
        norm = mcolors.Normalize(vmin=0, vmax=max_val)
        '''
        # Normalization
        max_val = np.max(np.abs(attention_map))
        norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
        ''' 
        # Create a custom colormap
        fig, ax = plt.subplots(figsize=(50, 30)) 
        '''
        colors = [(0, 0, 0)] + [(plt.cm.bwr(i)) for i in range(256)]
        new_colormap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors, N=257)
        new_colormap.set_under('black')  # for values under the min value
        
        # Replace -inf with a value smaller than the minimum of the colormap
        attention_map = np.where(attention_map == -np.inf, -np.finfo(np.float32).max, attention_map)
        ''' 
        # Plotting
        # cbar = ax.imshow(attention_map, norm = norm, cmap=new_colormap, aspect='auto', interpolation='nearest', vmin=-1, vmax=1) 
        cbar = ax.imshow(attention_map, cmap=new_colormap, norm=norm, aspect='auto', interpolation='nearest') 
        ax.imshow(zero_mask, cmap=mcolors.ListedColormap(['none', 'gold']), aspect='auto', interpolation='nearest', alpha=0.5) 
        ax.set_title(f'Attention Map: Layer {layer_num}, Head {head_num}')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Sequence Position')
        ax.set_xticks(range(seq_length))
        ax.set_yticks(range(seq_length))

        plt.colorbar(cbar, orientation = "vertical") 

        # Save to file
        plt.savefig(filename, bbox_inches='tight')
        plt.close() 

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```""" 

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

class LlamaForCausalLMWeird(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, adding_mode = "average"): 
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
        if adding_mode in ["average", "concatenate"]: 
            self.adding_mode = adding_mode 
        
        # self.adding_mode = "concatenate" 
        # self.adding_mode = "average" 
        self.adding_mode = adding_mode 
        if self.adding_mode == "concatenate": 
            self.lm_head_different = nn.Linear(config.hidden_size * 2, config.vocab_size, bias = False) 
        else: 
            self.lm_head_different = nn.Linear(config.hidden_size, config.vocab_size, bias = False) 
        self.target_model_dim = 4096 
        self.embed_projection = nn.Linear(self.target_model_dim, config.hidden_size, bias = False) 

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        added_condensed_token: Optional[torch.Tensor] = None, 
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else: 
            if self.adding_mode == "average": 
                hidden_states[:, -1, :] += self.embed_projection(added_condensed_token) 
                hidden_states[:, -1, :] /= 2.0 
                # logits = self.lm_head(hidden_states) 
                logits = self.lm_head_different(hidden_states) 
            else: 
                hidden_states_needed = torch.cat((hidden_states[:, -1, :], self.embed_projection(added_condensed_token)), dim = -1) 
                logits = self.lm_head_different(hidden_states_needed) 
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        ) 
    
class SimpleSmallModel(LlamaPreTrainedModel): 
    _tied_weights_keys = ["lm_head.weight"] 
    
    def __init__(self, config, sliding_window_length = 4, hostname = None): 
        super().__init__(config) 
        # copied from LlamaModel 
        self.padding_idx = config.pad_token_id 
        self.vocab_size = config.vocab_size 
        
        # cross model projection of the hidden states dimension 
        self.target_model_dim = 4096 
        self.embed_projection = nn.Linear(self.target_model_dim, config.hidden_size, bias = False) 
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx) 
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) 
        
        self.gradient_checkpointing = False 
        
        # copied from LlamaForCausalLM 
        self.vocab_size = config.vocab_size 
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 

        # needed to be used for the interleaving embeddings 
        self.sliding_window_length = sliding_window_length 
        
        # Initialize weights and apply final processing
        self.post_init() 

        # add an evaluation mode 
        self.eval_mode = False 
        self.condensed_fashion = "projection_mode" 
        self.all_list_condensed = ["projection_mode", "ground_truth"] 

        self.criticalpath = None 
        # determine something 
        if hostname is not None: 
            if "lovelace" in hostname: 
                self.criticalpath = "/home/yangzho6/" 
            elif "ada" in hostname: 
                self.criticalpath = "/home/beidic/yangzho6/" 

        if self.criticalpath is None or hostname is None: 
            raise ValueError("critical path is not set") 
    
    # input embeddings 
    def get_input_embeddings(self):
        # return self.model.embed_tokens 
        return self.embed_tokens 

    def set_input_embeddings(self, value):
        # self.model.embed_tokens = value 
        self.embed_tokens = value 

    # output embeddings 
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # def set_decoder(self, decoder):
    #     self.model = decoder

    # def get_decoder(self):
    #     return self.model 

    # this function happens after inputs_embeds has been made, so there shouldn't be problem related to condensed tokens 
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length): 
        combined_attention_mask = None 
        if input_shape[-1] > 1: 
            combined_attention_mask = _make_causal_mask(
                input_shape, 
                inputs_embeds.dtype, 
                device = inputs_embeds.device, 
                past_key_values_length = past_key_values_length, 
            ) 
            # print("combined attention mask shape {}".format(combined_attention_mask.shape)) 
        
        if attention_mask is not None: 

            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len = input_shape[-1]).to( #008000 
                inputs_embeds.device 
            ) 
            # print("expanded_attn_mask has shape {}".format(expanded_attn_mask.shape)) 
            # print("combined_attention_mask has shape {}".format(combined_attention_mask.shape)) 
            # print("expanded attention mask shape {}".format(expanded_attn_mask.shape)) 
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask 
            ) 
        
        return combined_attention_mask 

    def _convert_to_normal_attention_mask(self, combined_attention_mask, dtype, mask_list_pos, start_idx = None, kernel_size = None): 
        mask_shape = combined_attention_mask.shape # (batch_size, 1, tgt_seq_len, src_seq_len) 
        seq_len = mask_shape[-1] 
        start_idx = start_idx if start_idx is not None else self.start_idx 
        kernel_size = kernel_size if kernel_size is not None else self.sliding_window_length 
        
        # row dimensional masking 
        # row_idx_masked_out = [start_idx + i * (kernel_size + 1) for i in range((seq_len - start_idx) / (kernel_size + 1))] 
        row_mask = torch.zeros(mask_shape[-2], mask_shape[-1], device = combined_attention_mask.device) # NOTE currently, this line only works for training 
        # row_mask[row_idx_masked_out] = 1 
        row_mask[mask_list_pos] = 1 

        condensed_token_idx_list = mask_list_pos 
        for i in range(len(condensed_token_idx_list)): 
            row_mask[condensed_token_idx_list[i]: , condensed_token_idx_list[i]] = 1 
        row_mask = row_mask[None, None, :, :].expand(mask_shape).to(torch.bool) 
        row_mask = row_mask.to(device = combined_attention_mask.device) 

        combined_attention_mask.masked_fill_(row_mask, torch.finfo(dtype).min) 

    # def _modify_decoder_attention_mask(self, combined_attention_mask, dtype, start_idx = None, kernel_size = None): 
    def _modify_decoder_attention_mask(self, combined_attention_mask, dtype, mask_list_pos, start_idx = None, kernel_size = None): 
        mask_shape = combined_attention_mask.shape # (batch_size, 1, tgt_seq_len, src_seq_len) 
        seq_len = mask_shape[-1] 
        start_idx = start_idx if start_idx is not None else self.start_idx 
        kernel_size = kernel_size if kernel_size is not None else self.sliding_window_length 
        
        # row dimensional masking 
        # row_idx_masked_out = [start_idx + i * (kernel_size + 1) for i in range((seq_len - start_idx) / (kernel_size + 1))] 
        row_mask = torch.zeros(mask_shape[-2], mask_shape[-1], device = combined_attention_mask.device) # NOTE currently, this line only works for training 
        # row_mask[row_idx_masked_out] = 1 
        # row_mask[mask_list_pos] = 1 
        row_mask[mask_list_pos, :] = 1 

        # column dimensional masking 
        # condensed_token_idx_list = row_idx_masked_out 
        condensed_token_idx_list = mask_list_pos 
        for i in range(len(condensed_token_idx_list) - 1): 
            # row_mask[:, :, condensed_token_idx_list[i + 1] :, condensed_token_idx_list[i]] = 1 
            row_mask[condensed_token_idx_list[i + 1] :, condensed_token_idx_list[i]] = 1 
        # print("row mask shape {}".format(row_mask.shape)) 
        row_mask = row_mask[None, None, :, :].expand(mask_shape).to(torch.bool) 
        row_mask = row_mask.to(device = combined_attention_mask.device) 

        combined_attention_mask.masked_fill_(row_mask, torch.finfo(dtype).min) 
    
    def _modify_decoder_attention_mask_for_harder(self, combined_attention_mask, dtype, mask_list_pos, start_idx = None, kernel_size = None): 
        mask_shape = combined_attention_mask.shape # (batch_size, 1, tgt_seq_len, src_seq_len) 
        seq_len = mask_shape[-1] 
        start_idx = start_idx if start_idx is not None else self.start_idx 
        kernel_size = kernel_size if kernel_size is not None else self.sliding_window_length 
        
        # row dimensional masking 
        # row_idx_masked_out = [start_idx + i * (kernel_size + 1) for i in range((seq_len - start_idx) / (kernel_size + 1))] 
        row_mask = torch.zeros(mask_shape[-2], mask_shape[-1], device = combined_attention_mask.device) # NOTE currently, this line only works for training 
        # row_mask[row_idx_masked_out] = 1 
        row_mask[mask_list_pos] = 1 

        # column dimensional masking 
        # condensed_token_idx_list = row_idx_masked_out 
        condensed_token_idx_list = mask_list_pos 
        for i in range(len(condensed_token_idx_list) - 2): 
            # row_mask[:, :, condensed_token_idx_list[i + 1] :, condensed_token_idx_list[i]] = 1 
            row_mask[condensed_token_idx_list[i + 2] :, condensed_token_idx_list[i]] = 1 
        
        # adding blocking to force attention 
        for i in range(len(condensed_token_idx_list) - 1): 
            if i < len(condensed_token_idx_list) - 2: 
                row_mask[condensed_token_idx_list[i + 1] + 1: condensed_token_idx_list[i + 2], condensed_token_idx_list[i] + 1 : condensed_token_idx_list[i + 1]] = 1 
            else: 
                row_mask[condensed_token_idx_list[i + 1] + 1 :, condensed_token_idx_list[i] + 1 : condensed_token_idx_list[i + 1]] = 1 

        # print("row mask shape {}".format(row_mask.shape)) 
        row_mask = row_mask[None, None, :, :].expand(mask_shape).to(torch.bool) 
        row_mask = row_mask.to(device = combined_attention_mask.device) 

        combined_attention_mask.masked_fill_(row_mask, torch.finfo(dtype).min) 
    
    def _modify_decoder_attention_mask_for_harder2(self, combined_attention_mask, dtype, mask_list_pos, start_idx = None, kernel_size = None): 
        mask_shape = combined_attention_mask.shape # (batch_size, 1, tgt_seq_len, src_seq_len) 
        seq_len = mask_shape[-1] 
        start_idx = start_idx if start_idx is not None else self.start_idx 
        kernel_size = kernel_size if kernel_size is not None else self.sliding_window_length 
        
        # row dimensional masking 
        # row_idx_masked_out = [start_idx + i * (kernel_size + 1) for i in range((seq_len - start_idx) / (kernel_size + 1))] 
        row_mask = torch.zeros(mask_shape[-2], mask_shape[-1], device = combined_attention_mask.device) # NOTE currently, this line only works for training 
        # row_mask[row_idx_masked_out] = 1 
        row_mask[mask_list_pos] = 1 

        # column dimensional masking 
        # condensed_token_idx_list = row_idx_masked_out 
        condensed_token_idx_list = mask_list_pos 
        for i in range(len(condensed_token_idx_list) - 2): 
            # row_mask[:, :, condensed_token_idx_list[i + 1] :, condensed_token_idx_list[i]] = 1 
            row_mask[condensed_token_idx_list[i + 2] :, condensed_token_idx_list[i]] = 1 
        
        # adding blocking to force attention 
        for i in range(len(condensed_token_idx_list) - 1): 
            if i < len(condensed_token_idx_list) - 2: 
                row_mask[condensed_token_idx_list[i + 1] + 1: condensed_token_idx_list[i + 2], condensed_token_idx_list[i] + 1 : condensed_token_idx_list[i + 1] + 1] = 1 
            else: 
                row_mask[condensed_token_idx_list[i + 1] + 1 :, condensed_token_idx_list[i] + 1 : condensed_token_idx_list[i + 1] + 1] = 1 

        # print("row mask shape {}".format(row_mask.shape)) 
        row_mask = row_mask[None, None, :, :].expand(mask_shape).to(torch.bool) 
        row_mask = row_mask.to(device = combined_attention_mask.device) 

        combined_attention_mask.masked_fill_(row_mask, torch.finfo(dtype).min) 
    
    def _modify_decoder_attention_mask_for_hardest(self, combined_attention_mask, dtype, mask_list_pos, start_idx = None, kernel_size = None): 
        mask_shape = combined_attention_mask.shape # (batch_size, 1, tgt_seq_len, src_seq_len) 
        seq_len = mask_shape[-1] 
        start_idx = start_idx if start_idx is not None else self.start_idx 
        kernel_size = kernel_size if kernel_size is not None else self.sliding_window_length 
        
        # row dimensional masking 
        # row_idx_masked_out = [start_idx + i * (kernel_size + 1) for i in range((seq_len - start_idx) / (kernel_size + 1))] 
        row_mask = torch.zeros(mask_shape[-2], mask_shape[-1], device = combined_attention_mask.device) # NOTE currently, this line only works for training 
        # row_mask[row_idx_masked_out] = 1 
        row_mask[mask_list_pos] = 1 

        condensed_token_idx_list = mask_list_pos 
        for i in range(len(condensed_token_idx_list)): 
            if i == 0: 
                row_mask[condensed_token_idx_list[i] : , : condensed_token_idx_list[i]] = 1 
            else: 
                row_mask[condensed_token_idx_list[i] : , condensed_token_idx_list[i - 1] : condensed_token_idx_list[i]] = 1 
        
        # print("row mask shape {}".format(row_mask.shape)) 
        row_mask = row_mask[None, None, :, :].expand(mask_shape).to(torch.bool) 
        row_mask = row_mask.to(device = combined_attention_mask.device) 

        combined_attention_mask.masked_fill_(row_mask, torch.finfo(dtype).min) 
    
    def interleaving_embeddings_inputs(self, input_embeds, condensed_embeds, kernel_size = 4, start_idx = 64): 
        assert (input_embeds.shape[1] - start_idx)/kernel_size == condensed_embeds.shape[1] 
        combined_embeds = input_embeds[:, : start_idx, :] 
        input_embeds_count = start_idx 
        # print("combined embeds shape {}".format(combined_embeds.shape)) 
        for i in range(condensed_embeds.shape[1]): 
            # print("i is {}".format(i)) 
            combined_embeds = torch.cat([combined_embeds, condensed_embeds[:, i, :].unsqueeze(1)], dim = 1) 
            combined_embeds = torch.cat([combined_embeds, input_embeds[:, input_embeds_count : input_embeds_count + kernel_size, :]], dim = 1) 
            input_embeds_count += kernel_size 
            # print("combined embeds shape {}".format(combined_embeds.shape)) 
        return combined_embeds 
    
    def visualize_position_ids(self, position_ids, mask_idx): 
        # for debugging purposes 
        position_ids = position_ids.squeeze(0) 
        outputline = "" 
        for i in range(position_ids.shape[0]): 
            if i in mask_idx: 
                outputline += colored(str(position_ids[i].item()), "red") + " "
            else: 
                outputline += str(position_ids[i].item()) + " " 
        return outputline 
    
    def visualize_attention_mask(self, sequence_length, tensor, filename): 
        # code generated by GPT-4 
        '''
        # Create a tensor for demonstration purposes
        # In your case, replace this with the actual tensor
        tensor = torch.full((sequence_length, sequence_length), float('-inf'))
        mask = torch.rand(sequence_length, sequence_length) > 0.5  # Random mask for demo
        tensor[mask] = 0.0
        ''' 
        # Convert to numpy for visualization
        tensor_np = tensor.cpu().clone().numpy() 

        # Replace -inf with 1 and 0 with 0 for visualization purposes
        # visual_tensor = np.where(tensor_np == float('-inf'), 1, 0) 
        visual_tensor = np.where(tensor_np < 0, 1, 0) 
        # print(visual_tensor) 

        # Create the plot
        fig, ax = plt.subplots(figsize=(30, 30)) 
        cmap = plt.cm.colors.ListedColormap(['black', 'lightblue'])
        bounds = [-0.5, 0.5, 1.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        # Display the data
        cbar = ax.imshow(visual_tensor, cmap=cmap, norm=norm)

        # Set the background color to white
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Add gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        # ax.set_xticks(np.arange(-0.5, sequence_length, 1)) 
        # ax.set_yticks(np.arange(-0.5, sequence_length, 1)) 
        tick_positions = np.arange(0, sequence_length, 1)
        ax.set_xticks(tick_positions - 0.5)  # Shift the tick positions to be centered between the gridlines
        ax.set_yticks(tick_positions - 0.5)  # Same shift for y-ticks

        # Label the axes
        ax.set_xticklabels(np.arange(0, sequence_length))
        ax.set_yticklabels(np.arange(0, sequence_length))

        # Set the tick labels for both axes
        plt.xticks(rotation=90)
        ax.tick_params(axis=u'both', which=u'both',length=0)

        # Set axis limits to make the grid fit the image correctly
        ax.set_xlim(-0.5, sequence_length-0.5)
        ax.set_ylim(sequence_length-0.5, -0.5)

        # Show the color bar
        plt.colorbar(cbar, ticks=[0, 1], orientation='vertical', shrink=0.8, aspect=20)

        # Save the plot
        plt.savefig(filename, format='jpg', bbox_inches='tight') 
        # print("we got here") 
        plt.close() 
    
    def downsample_vectors(self, listoflasthiddenstates, kernel_size = 4): 
        downsampled_vectors = [] 
        shape = listoflasthiddenstates[0].shape 
        device = listoflasthiddenstates[0].device 
        for i in range(len(listoflasthiddenstates)): 
            sum = torch.zeros(shape, device = device) 
            if i % kernel_size == kernel_size - 1: 
                sum += listoflasthiddenstates[i] 
                downsampled_vectors.append(sum/kernel_size) 
                sum.mul_(0.) 
            else: 
                sum += listoflasthiddenstates[i] 
        return downsampled_vectors 

    def downsample_vectors2(self, cat_tokens, kernel_size = 4): 
        # downsampled_vectors = [] 
        device = cat_tokens.device 
        assert cat_tokens.shape[1] == kernel_size 
        sum = torch.zeros((cat_tokens.shape[0], cat_tokens.shape[-1]), device = device) 
        for i in range(cat_tokens.shape[1]): 
            sum += cat_tokens[:, i, :] 
        sum /= kernel_size 
        return sum 

    @staticmethod 
    def plot_attention_map(attention_maps, layer_num, head_num, seq_length, filename):
        """
        Plots the attention map for a specific layer and head and saves it to a file.

        :param attention_maps: A nested list or array of attention maps from the transformer model.
        :param layer_num: The layer number to visualize.
        :param head_num: The head number to visualize.
        :param seq_length: The sequence length of the model.
        :param filename: The filename to save the plot.
        """

        import matplotlib.colors as mcolors

        # Extract the specific attention map
        # attention_map = attention_maps[layer_num][head_num] 
        attention_map = attention_maps[layer_num][0][head_num].cpu().detach().numpy() 
        
        # Create a mask for exact zeros
        zero_mask = attention_map == 0
        '''
        # Create a custom colormap
        colors = [(plt.cm.bwr(i)) for i in range(256)]
        # midpoint = np.where(np.linspace(-1, 1, 256) == 0)[0][0] 
        midpoint = np.abs(np.linspace(-1, 1, 256)).argmin() 
        colors[midpoint] = (0, 0, 0, 1)
        new_colormap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)
        ''' 
        # Define a new color dictionary
        cdict = {
            'red':   ((0.0, 0.0, 0.0),   # Black
                    (0.25, 1.0, 1.0),  # Red
                    (0.5, 1.0, 1.0),   # Yellow (1.0, 1.0, 0.0) -> Red + Green
                    (0.75, 0.0, 0.0),  # Green
                    (1.0, 0.0, 0.0)),  # Blue

            'green': ((0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 1.0, 1.0),   # Yellow
                    (0.75, 1.0, 1.0),  # Green
                    (1.0, 0.0, 0.0)),

            'blue':  ((0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 0.0, 0.0),   # Yellow has no blue component
                    (0.75, 0.0, 0.0),  # Green
                    (1.0, 1.0, 1.0))   # Blue
        }

        custom_cmap = mcolors.LinearSegmentedColormap('custom_colormap', cdict)
        new_colormap = custom_cmap 

        # Normalization
        max_val = np.max(attention_map)
        norm = mcolors.Normalize(vmin=0, vmax=max_val)
        '''
        # Normalization
        max_val = np.max(np.abs(attention_map))
        norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
        ''' 
        # Create a custom colormap
        fig, ax = plt.subplots(figsize=(50, 30)) 
        '''
        colors = [(0, 0, 0)] + [(plt.cm.bwr(i)) for i in range(256)]
        new_colormap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors, N=257)
        new_colormap.set_under('black')  # for values under the min value
        
        # Replace -inf with a value smaller than the minimum of the colormap
        attention_map = np.where(attention_map == -np.inf, -np.finfo(np.float32).max, attention_map)
        ''' 
        # Plotting
        # cbar = ax.imshow(attention_map, norm = norm, cmap=new_colormap, aspect='auto', interpolation='nearest', vmin=-1, vmax=1) 
        cbar = ax.imshow(attention_map, cmap=new_colormap, norm=norm, aspect='auto', interpolation='nearest') 
        ax.imshow(zero_mask, cmap=mcolors.ListedColormap(['none', 'gold']), aspect='auto', interpolation='nearest', alpha=0.5) 
        ax.set_title(f'Attention Map: Layer {layer_num}, Head {head_num}')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Sequence Position')
        ax.set_xticks(range(seq_length))
        ax.set_yticks(range(seq_length))

        plt.colorbar(cbar, orientation = "vertical") 

        # Save to file
        plt.savefig(filename, bbox_inches='tight')
        plt.close() 

# Example usage
# plot_attention_map(attention_maps, layer_num=0, head_num=0, seq_length=128, filename='attention_map.png')

    def forward(
        self,
        input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None, 
        condensed_embeds: Optional[torch.FloatTensor] = None, 
        # later_input_ids: torch.LongTensor = None, 
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        start_idx = 64, 
        eval_mode = False, 
        iteration_count = None, 
        condensed_fashion = "projection_mode", 
        experiment_setting = "setting0", 
    ) -> Union[Tuple, CausalLMOutputWithPast]: 
        
        assert condensed_fashion in self.all_list_condensed 
        self.condensed_fashion = condensed_fashion 

        self.experiment_setting = experiment_setting 

        self.eval_mode = eval_mode 
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict 
        
        # assert input_ids.shape[0] == inputs_embeds.shape[0] 
        
        batch_size = input_ids.shape[0] 
        seq_length = input_ids.shape[1] 
        if not self.eval_mode: 
            # dimension matching 
            assert input_ids.shape[0] == condensed_embeds.shape[0] # batch size has to match 
            print("input_ids shape {} condensed_embeds shape {}".format(input_ids.shape, condensed_embeds.shape)) 
            print("sliding window length {}".format(self.sliding_window_length)) 
            assert (input_ids.shape[1] - start_idx)/self.sliding_window_length == condensed_embeds.shape[1] # number of condensed tokens should have desired mapping with sequence length 

            if self.condensed_fashion == "ground_truth": 
                with torch.no_grad(): 
                    condensed_embeds = [self.embed_tokens(input_ids[:, start_idx + i * self.sliding_window_length : start_idx + (i + 1) * self.sliding_window_length]) for i in range((seq_length - start_idx)//self.sliding_window_length)] 
                    # print("shape of every entry of the condensed tokens: {}".format(condensed_embeds[0].shape)) 
                    condensed_embeds = [self.downsample_vectors2(condensed_embeds[i]) for i in range(len(condensed_embeds))] 
                    condensed_embeds = torch.stack(condensed_embeds, dim = 1) 
                    # print("shape of condensed_embeds: {}".format(condensed_embeds.shape)) 
                # exit(0) 
                assert (condensed_embeds.shape[0] == batch_size) and (condensed_embeds.shape[-1] == self.config.hidden_size) 
        else: 
            # for the eval mode we simply ignore the condensed_embeds 
            condensed_length = int((input_ids.shape[1] - start_idx)/self.sliding_window_length) 
            condensed_embeds = torch.zeros((batch_size, condensed_length, self.target_model_dim)).to(input_ids.device) 
        # seq_length = input_ids.shape[1] + inputs_embeds.shape[1] 
        # batch_size = inputs_embeds.shape[0] # NOTE inputs_embeds is the only tensor that will always be here 
        # seq_length = inputs_embeds.shape[1] 
        # if context_input_ids is not None: 
            # seq_length += context_input_ids.shape[1] 
        # if later_input_ids is not None: 
            # seq_length += later_input_ids.shape[1] 
        seq_length += condensed_embeds.shape[1] 
        # print("batch size is {} seq length is {}".format(batch_size, seq_length)) 
        seq_length_with_past = seq_length 
        past_key_values_length = 0 
        
        if past_key_values is not None: 
            past_key_values_length = past_key_values[0][0].shape[2] 
            seq_length_with_past = seq_length_with_past + past_key_values_length 
        
        # self.mask_list_pos = [self.start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - self.start_idx) // (self.sliding_window_length + 1))] 
        # mask_list_pos = [self.start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - self.start_idx) // (self.sliding_window_length + 1))] 
        mask_list_pos = [start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - start_idx) // (self.sliding_window_length + 1))] 
        if position_ids is None: 
            device = input_ids.device 
            # device = inputs_embeds.device 
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device = device 
            # ) 
            position_list = [] 
            pos_count = past_key_values_length 
            following_flag = False 
            for i in range(seq_length): 
                # if i in self.mask_list_pos: 
                if i in mask_list_pos: 
                    pos_count += 1 
                    position_list.append(pos_count) 
                    following_flag = True 
                else: 
                    if following_flag: 
                        following_flag = False 
                        position_list.append(pos_count) 
                    else: 
                        pos_count += 1 
                        position_list.append(pos_count) 
            position_ids = torch.tensor(position_list, dtype = torch.long, device = device) 
            position_ids = position_ids.unsqueeze(0) 
        # print("mask list pos : {}".format(self.mask_list_pos)) 
        # print("position ids found is {}".format(self.visualize_position_ids(position_ids, self.mask_list_pos))) 
        
        # the important part 
        # input_embeds should not be None 
        torch.set_printoptions(threshold = 500) 
        input_embeds = None 
        if condensed_embeds is not None: 
            '''
            print("this is only for debugging purposes, if you see this in the commandline output, this is not for anything else ohter than debugging") 
            self.embed_projection.weight.data.mul_(0.) # only for debugging purposes 
            ''' 
            # inputs_embeds = self.embed_projection(inputs_embeds) 
            if self.condensed_fashion == "projection_mode": 
                condensed_embeds = self.embed_projection(condensed_embeds) 
            # print("condensed_embeds first ten numbers: {}".format(condensed_embeds.view(-1)[: 100])) 
            # print("condensed_embeds has nan numbers: {}".format(torch.isnan(condensed_embeds.view(-1)).any())) 
            # ids_input_embeds = self.embed_tokens(input_ids) 
            # print("embed_tokens dtype: {}".format(self.embed_tokens.weight.dtype)) 
            input_embeds = self.embed_tokens(input_ids) 
            # print("input_embeds first ten numbers: {}".format(input_embeds.view(-1)[: 10])) 
            # print("input_embeds: {}".format(input_embeds[0, : 5, : 40])) 
            # print("attention_mask: {}".format(attention_mask)) 
            # for i in range(input_embeds.shape[1]): 
                # print("input_id for it is {} does it has nan number {}".format(input_ids[0][i], torch.isnan(input_embeds[0][i]).any())) 
            # print() 
            input_embeds = self.interleaving_embeddings_inputs(input_embeds, condensed_embeds, kernel_size = self.sliding_window_length, start_idx = start_idx) 
            # print("input_embeds first ten numbers: {}".format(input_embeds[0][0][: 200])) 
            # print("weights in embed_tokens first ten numbers: {}".format(self.embed_tokens.weight[0][: 10])) 
            # print("weights in embed_projection first ten numbers: {}".format(self.embed_projection.weight[0][: 10])) 
            '''
            # debugging only 
            for i in range(input_embeds.shape[1]): 
                if input_embeds[0][i][0] == 0: 
                    print(colored("sequence position {} first 20 of the embedding values: {}".format(i, input_embeds[0][i][: 10]), "red")) 
                else: 
                    print("sequence position {} first 20 of the embedding values: {}".format(i, input_embeds[0][i][: 10])) 
            exit(0) 
            ''' 
        else: 
            raise ValueError("We cannot have an inference or any forward propagation without the inputs_embeds") 
        # print("input_embeds has shape {}".format(input_embeds.shape)) 
        
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device = input_embeds.device 
            ) 
            padding_mask = None
        else:
            if 0 in attention_mask:
                padding_mask = attention_mask
            else:
                padding_mask = None 
        # print("attention mask shape {}".format(attention_mask.shape)) 
        
        attention_mask = self._prepare_decoder_attention_mask(
            # attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length 
            attention_mask, (batch_size, seq_length), input_embeds, past_key_values_length 
        ) 
        # working_dir = "/home/yangzho6/" 
        # working_dir = "/home/beidic/yangzho6/" 
        # self.visualize_attention_mask(seq_length, attention_mask[0][0], working_dir + "attention_mask_before_modification.jpg") 
        # print(attention_mask[0][0]) 
        # self._modify_decoder_attention_mask(attention_mask, dtype = input_embeds.dtype, start_idx = self.start_idx, kernel_size = self.sliding_window_length) 
        if self.eval_mode: 
            # the attention_mask ignores the condensed tokens 
            self._convert_to_normal_attention_mask(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
        else: 
            if self.experiment_setting == "setting0": 
                self._modify_decoder_attention_mask(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting1": 
                self._modify_decoder_attention_mask_for_harder(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting2": 
                self._modify_decoder_attention_mask_for_harder2(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting3": 
                self._modify_decoder_attention_mask_for_hardest(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            else: 
                raise ValueError("We do not have the experiment setting you are looking for") 
            
        if iteration_count is not None and iteration_count == 1: 
            working_dir = self.criticalpath 
            self.visualize_attention_mask(seq_length, attention_mask[0][0], working_dir + "attention_mask_after_modification.jpg") 
        # print(attention_mask[0][0]) 
        
        # hidden_states = inputs_embeds 
        hidden_states = input_embeds 
        
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False 
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None 
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask, 
                    mask_list_pos = mask_list_pos, 
                ) 

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        
        # if not return_dict:
        #     return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # ) 
        
        if self.config.pretraining_tp > 1: 
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1) 
        else: 
            logits = self.lm_head(hidden_states) 
        logits = logits.float() 

        loss = None 
        if labels is not None: 
            # Shift so that tokens < n predict n 
            selected_indices = list(range(start_idx)) 
            for i in range(start_idx, seq_length): 
                # if i not in self.mask_list_pos: 
                if i not in mask_list_pos: 
                    selected_indices.append(i) 
            # shift_logits = shift_logits[:, selected_indices, :] 
            logits = logits[:, selected_indices, :] 
            # print("selected indices are : {}".format(selected_indices)) 
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # expecting 143 sequence length 
            # print("shift_logits have sequence length to be {}".format(shift_logits.shape)) 
            # expecting 127 sequence length 
            # print("shift_labels have sequence length to be {}".format(shift_labels.shape)) 
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1) 
            # print("shift_logits {}".format(shift_logits)) 
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels) 
            # print(loss) 
        
        # self.iter_count += 1 
        
        if not return_dict: 
            # output = (logits,) + outputs[1:] 
            output = (logits,) + tuple(v for v in [next_cache, all_hidden_states, all_self_attns] if v is not None) 
            return (loss,) + output if loss is not None else output 

        return CausalLMOutputWithPast( 
            loss = loss, 
            logits = logits, 
            past_key_values = next_cache, 
            hidden_states = all_hidden_states, 
            attentions = all_self_attns 
        ) 
    
    @staticmethod 
    # copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
    def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
        """
        Args:
            logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
            top_k (int, optional): top_k. Defaults to 0.
            top_p (float, optional): top_p. Defaults to 0.0.

        Returns:
            torch.Tensor: a renormalized logits
        """
        if top_k > 0:
            filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
            logits[logits < filter[:, [-1]]] = float('-inf')
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1)
            filter = cumulative_probs > top_p
            filter[..., 1:] = filter[..., :-1].clone()
            filter[..., 0] = 0
            indices_to_remove = filter.scatter(1, sorted_indices, filter)
            logits[indices_to_remove] = float('-inf')
        return logits 
    
    @staticmethod 
    def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): shape (1, vocab)
            temperature (float): temperature
            top_k (float): top_k
            top_p (float): top_p

        Returns:
            torch.Tensor: next token with shape as (batch,  1)
        """
        assert logits.dim() == 2
        logits = logits / temperature
        # logits = self.top_k_top_p_filter(logits, top_k=top_k, top_p=top_p) 
        logits = SimpleSmallModel.top_k_top_p_filter(logits, top_k=top_k, top_p=top_p) 
        probs = F.softmax(logits, dim=1)
        return probs 

    @staticmethod 
    def sample(probs : torch.Tensor, num_samples: int = 1, random_seed = None):
        if random_seed:
            torch.manual_seed(random_seed)
        idx_next = torch.multinomial(probs, num_samples=num_samples)
        if (idx_next.item() == 0):
            raise RuntimeError
        return idx_next 

    @staticmethod 
    def logitsToText(logits, topk, topp, temperature, tokenizer, use_sample = True): 
        # this function goes from logits to the actual decoded text 
        seq_len = logits.shape[-2] 
        print("sequence length is {}".format(seq_len)) 
        decoded_index = [] 
        for n in range(seq_len): 
            if use_sample: 
                probs = SimpleSmallModel.norm_logits(logits[:, n, :], temperature, topk, topp) 
                idx_next = SimpleSmallModel.sample(probs) 
            else: 
                idx_next = torch.argmax(logits[:, n, :], dim = -1) 
                # dimension of idx_next is (batch_size, 1) 
        decoded_index.append(idx_next) 
        output = torch.cat(decoded_index, dim = -1) 
        text = tokenizer.batch_decode(output) 
        return text 
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None: 
            # NOTE this problem is not a concern during training (kvcache isn't used) 
            # inference would be fine because condensed token k v are also in the past_key_values 
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        '''
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        ''' 
        model_inputs = {"input_embeds": inputs_embeds} 
        model_inputs.update({"input_ids": input_ids}) 
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs 
    
    def update_cache_for_new(self, past_key_values): 
        if self.iter_count < self.decipher_threshold: 
            raise ValueError("Note expected to roll back just yet") 
        elif self.iter_count == self.decipher_threshold: 
            new_past_key_values = [] 
            for i in range(len(past_key_values)): 
                new_layer_past = [] 
                for j in range(len(past_key_values[i])): 
                    new_layer_past.append(past_key_values[i][j][:, :, : -(self.decipher_threshold + 1), :]) # remove the generated one 
                new_past_key_values.append(tuple(new_layer_past)) 
            return new_past_key_values 
        else: 
            raise ValueError("We detected an error") 

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past 
