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
from typing import TYPE_CHECKING, Any, Callable, Dict 

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast 
from ...modeling_outputs import CausalLMOutputWithPastWithOriginalOutput 
from ...modeling_outputs import CausalLMOutputWithPastLargeDistance 
from ...modeling_outputs import CausalLMOutputWithPastLargeDistance3 
from ...modeling_outputs import CausalLMOutputWithPastLargeDistance2 
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
import torch.distributed as dist 

from ...generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from ...generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
) 
from ...generation.utils import SampleEncoderDecoderOutput, SampleDecoderOnlyOutput 
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput] 

if TYPE_CHECKING: 
    from ...generation.streamers import BaseStreamer 

import math 


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa 

torch.set_printoptions(threshold=5000) 


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

        # t_two = t / key_scaling_factor 
        # t_two = torch.floor_divide(t, key_scaling_factor) 
        t_two = t.clone() 
        # t_two[4: ] = (torch.floor_divide(t, key_scaling_factor) + (self.max_seq_len_cached // 2))[4 :] 
        # print("printing t: {}".format(t.to(torch.long))) 
        # print("printing t_two: {}".format(t_two.to(torch.long))) 

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

def apply_differently_rotary_pos_emb(q, k, cosq, sinq, cosk, sink, position_ids, unsqueeze_dim = 1, inv_freq = None): 
    cosq = cosq[position_ids].unsqueeze(unsqueeze_dim) 
    sinq = sinq[position_ids].unsqueeze(unsqueeze_dim) 

    if inv_freq is not None: 
        # print(position_ids) 
        # inv_freq = inv_freq[position_ids] 
        if len(position_ids.shape) != 1: 
            position_ids = position_ids[0] 
        t = position_ids.to(inv_freq.dtype) 
        seqlen = position_ids.shape[-1] 
        # t[4: ] = (torch.floor_divide(t, 2.) + (seqlen // 2))[4 :] 
        # t[4: ] = t[4: ]/2 + (seqlen/2) 
        # print("printing t: {}".format(t.to(torch.long))) 
        freqs = torch.einsum("i,j->ij", t, inv_freq) 
        emb = torch.cat((freqs, freqs), dim = -1) 
        cosk = emb.cos().to(dtype = k.dtype) 
        sink = emb.sin().to(dtype = k.dtype) 
    else: 
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
            inv_freq = self.rotary_emb.inv_freq 
            query_states, key_states = apply_differently_rotary_pos_emb(query_states, key_states, cosq, sinq, cosk, sink, position_ids, inv_freq = inv_freq) 
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
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
        # Note the next line is critical, since right now the softmax of all the values -inf is a very strange number 

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
        # Note the next line is critical, since right now the softmax of all the values -inf is a very strange number 
        
        # using the following not the above 
        if "mask_list_pos" in kwargs and "horizontal_bar_enabled" in kwargs: 
            if kwargs["horizontal_bar_enabled"]: 
                # print("found it") 
                mask = torch.ones((attn_weights.shape[-2], attn_weights.shape[-1]), device = attn_weights.device) 
                mask[kwargs["mask_list_pos"], :] = 0.0 
                # attn_weights[:, :, kwargs["mask_list_pos"], :] = 0.0 
                attn_weights = attn_weights * mask 
                attn_weights = attn_weights.to(value_states.dtype) 
        
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
            # print(colored("position_ids: {}".format(position_ids), "red")) # TODO remove this print once this is make sure to be correct 

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            print("seq_length: {}".format(seq_length)) 
            randomtensor = torch.full((seq_length, seq_length), torch.finfo(inputs_embeds.dtype).min, device = inputs_embeds.device) 
            print("we pass through a tensor making process") 
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

class LlamaModelSpecialTokenPrediction(LlamaPreTrainedModel):
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
    
    def _modified_attention_mask_for_special_adding_soft_token(self, attention_mask, input_shape, inputs_embeds, past_key_values_length, dtype): 
        # Since we are adding one special token to the end, it is for a special purpose 
        mask_shape = attention_mask.shape 
        seq_len = mask_shape[-1] 
        
        row_mask = torch.zeros(mask_shape[-2], mask_shape[-1], device = attention_mask.device) 
        
        row_mask[:, seq_len - 1] = 1 
        
        row_mask = row_mask[None, None, :, :].expand(mask_shape).to(torch.bool) 
        row_mask = row_mask.to(device = attention_mask.device) 
        
        attention_mask.masked_fill_(row_mask, torch.finfo(dtype).min) 
    
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
        tensordtype = tensor.dtype 
        if tensordtype == torch.bfloat16: 
            tensor_np = tensor.cpu().clone().to(torch.float32).numpy() 
        else: 
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
            # print(colored("position_ids: {}".format(position_ids), "red")) # TODO remove this print once this is make sure to be correct 

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
        self._modified_attention_mask_for_special_adding_soft_token(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, inputs_embeds.dtype) 
        
        self.visualize_attention_mask(seq_length, attention_mask[0][0], "specialtokenattentionmask.jpg") 

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

class LlamaForCausalLMAddingSpecialToken(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelSpecialTokenPrediction(config) 
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
        # labels: Optional[torch.LongTensor] = None, 
        condensed_labels: Optional[torch.LongTensor] = None, 
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
        
        # TODO add the special token to the input_ids 
        input_ids = torch.cat([input_ids, torch.zeros(input_ids.shape[0], 1, dtype = torch.long).to(input_ids.device)], dim = 1) # adding the <unk> token 

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
        # logits = logits.float() 
        logits = hidden_states[:, -1, :] 

        loss = None
        if condensed_labels is not None: 
            loss_function = nn.MSELoss() 
            loss = loss_function(logits, condensed_labels) 

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

class LlamaModelHybridSequenceLength(LlamaPreTrainedModel): 
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
        self.full_sequence_length_layer_pos = 1 # this line is for identifying which line we should remove 
        self.kernel_size = 7 
        self.post_init()
        
    def set_full_sequence_length_layer_pos(self, full_sequence_length_layer_pos): 
        self.full_sequence_length_layer_pos = full_sequence_length_layer_pos 

    def set_kernel_size(self, kernel_size): 
        self.kernel_size = kernel_size 
        
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
        tensordtype = tensor.dtype 
        if tensordtype == torch.bfloat16: 
            tensor_np = tensor.cpu().clone().to(torch.float32).numpy() 
        else: 
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
    
    def _first_level_attention_mask(self, combined_attention_mask, dtype, mask_list_pos, kernel_size = None): 
        mask_shape = combined_attention_mask.shape 
        seq_len = mask_shape[-1] 
        start_idx = 0 
        kernel_size = kernel_size if kernel_size is not None else self.sliding_window_length 
        
        row_mask = torch.zeros(mask_shape[-2], mask_shape[-1], device = combined_attention_mask.device) 
        mask_list_pos = [1 + i * kernel_size for i in range((seq_len - 1) // kernel_size)] # for our example, [1, 5] 
        for i in range(1, len(mask_list_pos)): 
            row_mask[mask_list_pos[i] : , mask_list_pos[i - 1] : mask_list_pos[i]] = 1 
        
        row_mask = row_mask[None, None, :, :].expand(mask_shape).to(torch.bool) 
        row_mask = row_mask.to(device = combined_attention_mask.device) 
        
        out_attention_mask = combined_attention_mask.clone() 
        out_attention_mask.masked_fill_(row_mask, torch.finfo(dtype).min) 
        
        return out_attention_mask 
    
    def _second_level_attention_mask(self, seq_len, dtype, mask_list_pos, kernel_size = None, batch_size = None, inputs_embeds = None, past_key_values_length = None): 
        out_attention_mask = torch.ones( 
            (batch_size, (seq_len - 1) // kernel_size), dtype = torch.bool, device = inputs_embeds.device 
        ) 
        out_attention_mask = self._prepare_decoder_attention_mask( 
            out_attention_mask, (batch_size, (seq_len - 1) // kernel_size), inputs_embeds, past_key_values_length 
        ) 
        return out_attention_mask 

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
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            # )
            # position_ids = position_ids.unsqueeze(0) 
            firstlayer_position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype = torch.long, device = device 
            ) 
            firstlayer_position_ids = firstlayer_position_ids.unsqueeze(0) 
            secondlayer_position_ids = torch.arange(
                past_key_values_length, (seq_length - 1) // self.kernel_size + past_key_values_length, dtype = torch.long, device = device
            ) 
            secondlayer_position_ids = secondlayer_position_ids.unsqueeze(0) 
            # print(colored("position_ids: {}".format(position_ids), "red")) # TODO remove this print once this is make sure to be correct 

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
        
        # attention mask changes 
        first_layer_attention_mask = self._first_level_attention_mask(attention_mask, dtype = inputs_embeds.dtype, mask_list_pos = None, kernel_size = self.kernel_size) 
        self.visualize_attention_mask(seq_length, first_layer_attention_mask[0][0], "first_layer_attention_mask.jpg") 
        second_layer_attention_mask = self._second_level_attention_mask(seq_length, dtype = inputs_embeds.dtype, mask_list_pos = None, kernel_size = self.kernel_size, batch_size = batch_size, inputs_embeds = inputs_embeds, past_key_values_length = past_key_values_length) 
        self.visualize_attention_mask((seq_length - 1) // self.kernel_size, second_layer_attention_mask[0][0], "second_layer_attention_mask.jpg") 

        # embed positions
        hidden_states = inputs_embeds 
        
        print(colored("hidden_states.shape: {}".format(hidden_states.shape), "yellow")) 
        print(colored("self.kernel_size: {}".format(self.kernel_size), "yellow")) 
        # basic sanity check 
        assert (hidden_states.shape[1] - 1) % self.kernel_size == 0 

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
            
            # sampling after a few layers 
            # print(colored("idx: {}".format(idx), "red")) 
            if idx == self.full_sequence_length_layer_pos: 
                # here is an illustration of sampling, say we have a sequence of <bos> A B C D E F G H, where the kernel size is 4 
                # then, the designed downsampling or sampling on dimension reduction is as follows, we group tokens after the start 
                # of sequence token into group of four, where all tokens inside the same group are attending with causal relationship 
                # the start of sequence token will be added to the first group, so that the first group would have (<bos> A B C D) 
                # so the input would be (<bos> A B C D) (E F G H) -> (A B C D N1) (F G H N2) 
                # the hidden states vectors that are picked would be N1 and N2 
                picking_index_list = [self.kernel_size * i for i in range(1, (seq_length - 1) // self.kernel_size + 1)] 
                hidden_states = hidden_states[:, picking_index_list, :] 
                assert hidden_states.shape[1] == second_layer_attention_mask.shape[-1] 
            # print(colored("idx: {} hidden_states.shape: {}".format(idx, hidden_states.shape), "red")) 
            assert self.gradient_checkpointing == False # some lines below are deleted, if you need to recover it, please refer to full versions elsewhere in this file 
            # if self.gradient_checkpointing and self.training:
            #     layer_outputs = self._gradient_checkpointing_func(
            #         decoder_layer.__call__,
            #         hidden_states,
            #         attention_mask,
            #         position_ids,
            #         past_key_value,
            #         output_attentions,
            #         use_cache,
            #     ) 
            layer_outputs = decoder_layer(
                hidden_states,
                # attention_mask=attention_mask, 
                attention_mask = first_layer_attention_mask if idx < self.full_sequence_length_layer_pos else second_layer_attention_mask, 
                # position_ids=position_ids,
                position_ids = firstlayer_position_ids if idx < self.full_sequence_length_layer_pos else secondlayer_position_ids, 
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

class LlamaForCausalLM2(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # self.model = LlamaModelWeirdAttentionMap(config)
        self.model = LlamaModel(config) 
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init() 
    
    def reinitialize_embeddings(self, type = "xavieruniform"): 
        from torch.nn import init 
        embedding = self.model.embed_tokens 
        if type == "xavieruniform": 
            init.xavier_uniform_(embedding.weight) 
        elif type == "xaviernormal": 
            init.xavier_normal_(embedding.weight) 
        elif type == "kaimingnormal": 
            init.kaiming_normal_(embedding.weight) 
        else: 
            raise ValueError("type not recognized") 

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

class LlamaWeirdLargeIntermediate(LlamaPreTrainedModel): 
    """ 
    Inside this class, the lm_head is not used 
    We also have a groupping function call at the beginning of the forward function 
    """ 
    # almost identical to LlamaWeirdLarge3, but weird fix for some model 
    _tied_weights_keys = ["lm_head.weight"]
    '''
    def __init__(self, *args, small_config, hostname, large_dim, sliding_window_length = 7, use_mse_loss = False, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.model = LlamaModel(self.config) 
        self.vocab_size = self.config.vocab_size 
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False) 
        
        # self.addonsmallmodel = addonsmallmodel 
        # self.addonsmallmodel = SimpleSmallModel(small_config, sliding_window_length = sliding_window_length, hostname = hostname, target_model_dim = large_dim) 
        self.addonsmallmodel = None 
        self.sliding_window_length = sliding_window_length 
        # self.small_model_dtype = self.addonsmallmodel.embed_projection.weight.dtype 
        self.small_model_dtype = torch.bfloat16 
        print(colored("small_model_dtype {}".format(self.small_model_dtype), "red")) 
        self.use_mse_loss = use_mse_loss 
        self.alpha = 0.5 

        # Initialize weights and apply final processing
        self.post_init()
    ''' 
    def __init__(self, config): 
        super().__init__(config) 
        # self.model = LlamaModel(config) 
        self.model = LlamaModelHybridSequenceLength(config) 
        self.vocab_size = config.vocab_size 
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False) 
        # self.addonsmallmodel = None 
        small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m") 
        self.sliding_window_length = 7 
        self.addonsmallmodel = SimpleSmallModel(small_config, sliding_window_length = self.sliding_window_length, target_model_dim = self.config.hidden_size) 
        self.small_model_dtype = torch.bfloat16 
        self.use_mse_loss = False 
        self.ce_loss_only = False 
        self.alpha = 0.5 
        self.addonmodel_start = self.sliding_window_length + 1 
        self.inference_setting = "setting0" 
        self.use_cosinesimilarity = False 
        self.generate_iteration_count = 0 
        self.generate_model_hidden_states = torch.tensor(0) # this field saves the intermediate tensors generated by the large model 
        self.tokenizer_bos_id = 1 
        self.tokenizer_pad_id = 2 
        
        self.post_init() 

    def get_input_embeddings(self):
        return self.model.embed_tokens 
    
    def set_msece_loss(self, use_mse_loss, ce_loss_only): 
        self.use_mse_loss = use_mse_loss 
        self.ce_loss_only = ce_loss_only 
    
    def set_full_sequence_length_layer_pos(self, full_sequence_length_layer_pos): 
        self.model.set_full_sequence_length_layer_pos(full_sequence_length_layer_pos) 
        print(colored("full_sequence_length_layer_pos {}".format(full_sequence_length_layer_pos), "yellow")) 
    
    def set_cosinesimilarity(self, use_cosinesimilarity): 
        if use_cosinesimilarity: 
            self.use_cosinesimilarity = True 
    
    def set_addonsmallmodel_statedict(self, small_state_dict_for_model): 
        new_state_dict = {} 

        for key in small_state_dict_for_model.keys(): 
            new_key = key 
            if 'lm_head' in key: 
                print("got here found the following key {}".format(key)) 
            if 'model.' in key: 
                new_key = key[6 :] 
            print(new_key) 
            new_state_dict[new_key] = small_state_dict_for_model[key] 
        # if args.embedding_pretrained: 
        #     new_state_dict["embed_projection.weight"] = torch.load("linearprojectionweighttesting.pt") 
        try: 
            self.addonsmallmodel.load_state_dict(new_state_dict) 
        except RuntimeError as r: 
            print(colored(r, "yellow")) 

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value 
    
    def set_inference_setting(self, setting = "setting0"): 
        self.inference_setting = setting 
    
    def set_slidingwindowlength(self, sliding_window_length, addonmodel_start = None): 
        self.sliding_window_length = sliding_window_length 
        if addonmodel_start is not None: 
            self.addonmodel_start = addonmodel_start 
        else: 
            self.addonmodel_start = self.sliding_window_length + 1 
    
    def set_tokenizer_bos_id(self, bos_id, pad_id): 
        self.tokenizer_bos_id = bos_id 
        self.tokenizer_pad_id = pad_id 
    
    def set_walpha(self, alpha) : 
        self.alpha = alpha 

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder
        
    def reinitialize_embeddings(self, type = "xavieruniform"): 
        from torch.nn import init 
        embedding = self.model.embed_tokens 
        if type == "xavieruniform": 
            init.xavier_uniform_(embedding.weight) 
        elif type == "xaviernormal": 
            init.xavier_normal_(embedding.weight) 
        elif type == "kaimingnormal": 
            init.kaiming_normal_(embedding.weight) 
        else: 
            raise ValueError("type not recognized") 

    def get_decoder(self):
        return self.model 
    
    def naive_grouping(self, input_ids): 
        embedding_searched = self.model.embed_tokens(input_ids) 
        # print("embedding_searched shape {} {}".format(embedding_searched.shape, embedding_searched.dtype)) 
        seq_length = embedding_searched.shape[1] 
        print("seq_length {}".format(seq_length)) 
        
        # assert seq_length % 7 == 0, "seq_length is not divisible by 7" 
        # assert seq_length % self.sliding_window_length == 0, "seq_length is not divisible by sliding_window_length" 
        # added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // 7, embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
        added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // self.sliding_window_length, embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
        # for i in range(seq_length // 7): 
        for i in range(seq_length // self.sliding_window_length): 
            sum = torch.zeros((embedding_searched.shape[0], embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
            # for j in range(7): 
            for j in range(self.sliding_window_length): 
                # sum += embedding_searched[:, i * 7 + j, :] 
                sum += embedding_searched[:, i * self.sliding_window_length + j, :] 
                # sum /= 7. 
                # print("sum dtype {}".format(sum.dtype)) 
            sum /= float(self.sliding_window_length) 
            added_tensor[:, i, :] = sum 
        # print("added_tensor shape {}".format(added_tensor.shape)) 
        
        return added_tensor 
    
    def attention_mask_upper(self, input_ids): 
        sequence_length = ((input_ids.shape[1] - 1) // self.sliding_window_length) + 1 
        batch_size = input_ids.shape[0] 
        condition_mask = input_ids == self.tokenizer_bos_id # finds the index of the start of sequence token 
        start_of_sequenceidx = torch.nonzero(condition_mask)[:, 1] 
        start_of_sequenceidx //= self.sliding_window_length 
        start_of_sequenceidx = start_of_sequenceidx.to(torch.long) 
        # modified_input_bos_sequence_indices = modified_input_bos_sequence_indices[:, 1].unsqueeze(1).expand(-1, seq_length) 
        start_of_sequenceidx2 = start_of_sequenceidx.unsqueeze(1).expand(-1, sequence_length) 
        print("start_of_sequenceidx shape {}".format(start_of_sequenceidx2.shape)) 
        col_indices = torch.arange(sequence_length).expand(batch_size, -1).to(input_ids.device) 
        attention_mask = col_indices >= start_of_sequenceidx2 
        attention_mask = attention_mask.to(torch.long) 
        return start_of_sequenceidx, attention_mask 
        
    def set_addonsmallmodel(self, addonsmallmodel): 
        self.addonsmallmodel = addonsmallmodel 
    
    def set_smallmodelfull(self): 
        self.addonsmallmodel = self.addonsmallmodel.to(torch.float32) 
    
    def l2distancecompute(self, inputs, hidden_states): 
        input_used = inputs.clone().detach()[:, 1:, :]
        hidden_states_used = hidden_states.clone().detach()[:, :-1, :] 
        assert input_used.shape == hidden_states_used.shape 
        dmod = input_used.shape[-1] 
        input_used = input_used.reshape(-1, dmod) 
        hidden_states_used = hidden_states_used.reshape(-1, dmod) 
        # compute the difference 
        diff = input_used - hidden_states_used 
        
        # compute the square 
        diff = diff ** 2
        
        # sum up the square 
        diff = torch.sum(diff, dim = 1) 
        
        # take square root 
        diff = torch.sqrt(diff) 
        
        # average the l2 distance 
        diff = torch.mean(diff) 
        
        return diff 

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        # input_ids: torch.LongTensor = None, 
        large_input_ids: torch.LongTensor = None, 
        small_input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        original_attention_mask = None, 
        condensed_embed_labels = None, 
    ) -> Union[Tuple, CausalLMOutputWithPastLargeDistance2]: 
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

        # inside this function callm, input is through input_embeds 
        # now, we have a start of sequence token 
        # the attention mask should be compatible to the new input_embeds 
        # print("condensed_embeds_labels shape {}".format(condensed_embed_labels.shape)) 
        assert inputs_embeds is None, "inputs_embeds is not None" 
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn) 
        # TODO delete the following line 
        
        outputs = self.model(
            input_ids = large_input_ids, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) 

        hidden_states = outputs[0] # we don't need the lm_head 
        print("hidden_states shape {} dtype {}".format(hidden_states.shape, hidden_states.dtype)) 
        if self.small_model_dtype == torch.float32: 
            hidden_states = hidden_states.to(torch.float32) 
        elif self.small_model_dtype == torch.bfloat16: 
            hidden_states = hidden_states.to(torch.bfloat16) 
        # print(colored("small_model_type: {}".format(self.small_model_dtype), "red")) 
        # intermediate_l2_dist = self.l2distancecompute(inputs_embeds, hidden_states) 
        
        # NOTE this is very important 
        hidden_states = hidden_states[:, :-1, :] 
        if condensed_embed_labels is not None: 
            mselabels = condensed_embed_labels 
            # print("first 100 eleents of hidden_states: {}".format(hidden_states[0][0][: 100])) 
            # print("first 100 elements of mselabels: {}".format(mselabels[0][0][: 100])) 
            # hidden_states[practical_mask == 0] = 0 
            # hidden_states = hidden_states[:, :-1, :] # NOTE this is very important 
            # output 30 condensed tokens, the last one and the first one doesn't have the condensed token label, so 28 left 
            # assert labels.shape == hidden_states.shape 
            assert mselabels.shape == hidden_states.shape 
            mse_lossfunc = nn.MSELoss() 
            mse_loss = mse_lossfunc(hidden_states, mselabels) 
            cosinesimlossfunc = nn.CosineEmbeddingLoss() 
            cossim_loss = cosinesimlossfunc(hidden_states.reshape(-1, hidden_states.shape[-1]), mselabels.reshape(-1, mselabels.shape[-1]), torch.ones(hidden_states.shape[0] * hidden_states.shape[1]).to(hidden_states.device)) 
        else: 
            mse_loss = torch.tensor(0) 
            cossim_loss = torch.tensor(0) 
        # mse_loss = 0.5 * mse_loss + 0.5 * cossim_loss 
        # intermediate_l2_dist = mse_loss.clone().detach() 
        intermediate_l2_dist = mse_loss.clone().detach() 
        if self.use_cosinesimilarity: 
            mse_loss = cossim_loss 
        cossim_input = cossim_loss.clone().detach() 
        # print(colored("mse_loss {}".format(mse_loss), "red")) 
        
        # print(colored("mse_loss_input {}".format(mse_loss_input), "red")) 
        # cossim_input = F.cosine_similarity(hidden_states.reshape(-1, hidden_states.shape[-1]), inputs_embeds.reshape(-1, inputs_embeds.shape[-1]), dim = 1) 
        # print("cossim_input shape {}".format(cossim_input.shape)) 
        # cossim_input = cossim_input.mean(dim = 0) 
        # print("cossim_input {}".format(cossim_input)) 
        l2_distance_input = torch.tensor(0) 
        
        if self.use_mse_loss: 
            print(colored("mse_loss {}".format(mse_loss), "red")) 
            # still use the small model and get ce 
            hidden_states = hidden_states.detach().clone() 
            # hidden_states = torch.zeros_like(hidden_states).detach() 
            # hidden_states = condensed_embed_labels 
            '''
            return CausalLMOutputWithPastLargeDistance2(
                loss = mse_loss, 
                logits = None, 
                past_key_values = outputs.past_key_values, 
                hidden_states=outputs.hidden_states,
                attentions = outputs.attentions, 
                l2_distance = intermediate_l2_dist, 
                ce_loss = torch.tensor(0), 
            ) 
            ''' 
        # hidden_states has shape (batch_size, seq_length // 7, hidden states) 
        # hidden_states = hidden_states[:, :-1, :] 
        
        # interleave the hidden_states and the input_ids 
        # assert hidden_states.shape[1] == small_input_ids.shape[1] // 7 - 1 
        assert hidden_states.shape[1] == (small_input_ids.shape[1] - self.addonmodel_start) // self.sliding_window_length 
        addonmodeloutput = self.addonsmallmodel( 
            # input_ids = input_ids, 
            input_ids = small_input_ids, 
            attention_mask = original_attention_mask, 
            position_ids = None, 
            past_key_values = None, 
            condensed_embeds = hidden_states, 
            labels = None, 
            use_cache = None, 
            output_attentions = True, 
            output_hidden_states = None, 
            return_dict = True, 
            start_idx = self.addonmodel_start, # NOTE this is very important 
            eval_mode = False, 
            iteration_count = 1, 
            condensed_fashion = "projection_mode", 
            # experiment_setting = "setting3", 
            experiment_setting = self.inference_setting, 
        ) 
        
        logits = addonmodeloutput.logits 
        
        '''
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        ''' 
        
        # seq_length = input_ids.shape[1] + hidden_states.shape[1] 
        seq_length = small_input_ids.shape[1] + hidden_states.shape[1] 
        assert seq_length == logits.shape[1], "seq_length is not compatible to logits" 
        # mask_list_pos = [i * (self.sliding_window_length + 1) for i in range(seq_length // (self.sliding_window_length + 1))] 
        # mask_list_pos = [7 + i * (self.sliding_window_length + 1) for i in range((seq_length - 7) // (self.sliding_window_length + 1))] 
        mask_list_pos = [self.addonmodel_start + i * (self.sliding_window_length + 1) for i in range((seq_length - self.addonmodel_start) // (self.sliding_window_length + 1))] 
        mask_list_pos22 = [x - 1 for x in mask_list_pos] 
        # print(colored("mask_list_pos {}".format(mask_list_pos), "red")) 
        loss = None 
        if labels is not None: 
            # selected_indices = list(range(7)) 
            selected_indices = list(range(self.addonmodel_start - 1)) 
            # for i in range(7, seq_length): 
                # if i not in mask_list_pos: 
                    # selected_indices.append(i) 
            for i in range(self.addonmodel_start - 1, seq_length): 
                if i not in mask_list_pos22: 
                    selected_indices.append(i) 
            # print(colored("selected_indices {}".format(selected_indices), "red")) 
            # select and shift the logits 
            logits = logits[:, selected_indices, :] 
            shift_logits = logits[..., :-1, :].contiguous() 
            shift_labels = labels[..., 1:].contiguous() # shape (batch_size, seq_length - 1) 
            print("shift_logits shape {}; shift_labels shape {}".format(shift_logits.shape, shift_labels.shape)) 
            # Flatten the tokens 
            loss_fct = CrossEntropyLoss() 
            shift_logits = shift_logits.view(-1, self.config.vocab_size) 
            shift_labels = shift_labels.view(-1) 
            # Enable model parallelism 
            shift_labels = shift_labels.to(shift_logits.device) 
            ce_loss = loss_fct(shift_logits, shift_labels) 
            loss = ce_loss 
            
            first_pos_loss = torch.tensor(0) 
            second_pos_loss = torch.tensor(0) 
            # print(colored("rank {} loss {}".format(self.accelerator.state.process_index, loss), "yellow")) 
        if loss is not None and not self.use_mse_loss: 
            if self.ce_loss_only: 
                print(colored("ce_loss only", "red")) 
                loss = ce_loss 
            else: 
                print(colored("ce_loss + mse_loss", "red")) 
                # loss = self.alpha * loss + (1 - self.alpha) * mse_loss 
                loss = self.alpha * ce_loss + (1 - self.alpha) * mse_loss 
        else: 
            print(colored("mse_loss only", "red")) 
            loss = mse_loss 

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output 

        return CausalLMOutputWithPastLargeDistance2(
            loss=loss,
            logits = logits, 
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions, 
            attentions = addonmodeloutput.attentions, # delibrately using the model's attention mask with modifications 
            l2_distance = intermediate_l2_dist, 
            ce_loss = ce_loss.detach().clone(), 
            l2_distance_input = l2_distance_input, 
            cossim_input = cossim_input, 
            first_pos_loss = first_pos_loss, 
            second_pos_loss = second_pos_loss, 
        ) 
    
    def prepare_inputs_for_generation2(
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

    def prepare_inputs_for_generation( 
        self, input_ids, past_key_values = None, attention_mask = None, inputs_embeds = None, adjustment_scheme = None, **kwargs): 
        # mainly used to debug preparing inputs for generation and not using past_key_values 
        assert past_key_values is None, "past_key_values is not None" 
        batch_size, seq_length = input_ids.shape 
        print("batch_size {}; seq_length {}".format(batch_size, seq_length)) 
        
        # adjusting the inputs and mask 
        print("input_ids {}".format(input_ids[2])) 
        print("attention_mask {}".format(attention_mask[2])) 
        condition_mask = input_ids == self.tokenizer_bos_id 
        input_sequence_indices = torch.nonzero(condition_mask).to(input_ids.device).to(torch.long) 
        print("input_sequence_indices shape {}".format(input_sequence_indices.shape)) 
        print("input_sequence_indices: {}".format(input_sequence_indices[2])) 
        input_sequence_indices2 = [] 
        modified_input_bos_sequence_indices = [] 
        assert input_sequence_indices.shape[0] == input_ids.shape[0], "every row of sequences need to have an bos" 
        for i in range(input_ids.shape[0]): # iterate through the batch_size 
            # if input_sequence_indices[i] % self.sliding_window_length != 0: # we found a sequence that needs to be adjusted 
            if input_sequence_indices[i][1].data % self.sliding_window_length != 0: # we found a sequence that needs to be adjusted 
                input_sequence_indices2.append(torch.tensor([i, (input_sequence_indices[i][1])]).to(input_ids.device).view(1, -1)) 
                if adjustment_scheme == "case1": 
                    modified_input_bos_sequence_indices.append(torch.tensor([i, (input_sequence_indices[i][1] // self.sliding_window_length) * self.sliding_window_length]).to(input_ids.device).view(1, -1)) 
                elif adjustment_scheme == "case2": 
                    modified_input_bos_sequence_indices.append(torch.tensor([i, (input_sequence_indices[i][1] // self.sliding_window_length + 1) * self.sliding_window_length]).to(input_ids.device).view(1, -1)) 
                else: 
                    raise ValueError("adjustment_scheme is not recognized") 
        if len(input_sequence_indices2) != 0: 
            # adjusting the input_ids 
            input_sequence_indices2 = torch.cat(input_sequence_indices2, dim = 0).to(input_ids.device).to(torch.long) 
            modified_input_bos_sequence_indices = torch.cat(modified_input_bos_sequence_indices, dim = 0).to(input_ids.device).to(torch.long) 
            print("shape of modified_input_bos_sequence_indices {}".format(modified_input_bos_sequence_indices.shape)) 
            print(modified_input_bos_sequence_indices) 
            
            row_indices = input_sequence_indices2[:, 0] 
            col_indices = input_sequence_indices2[:, 1] 
            input_ids.index_put_((row_indices, col_indices), torch.full_like(row_indices, fill_value = self.tokenizer_pad_id, device = input_ids.device, dtype = input_ids.dtype), accumulate = False) 
            
            row_indices = modified_input_bos_sequence_indices[:, 0] 
            col_indices = modified_input_bos_sequence_indices[:, 1] 
            input_ids.index_put_((row_indices, col_indices), torch.full_like(row_indices, fill_value = self.tokenizer_bos_id, device = input_ids.device, dtype = input_ids.dtype), accumulate = False) 
            
            print("input_ids {}".format(input_ids[2])) 
        # just for checking 
        checking_indices = torch.nonzero(input_ids == self.tokenizer_bos_id) 
        print("positions of the start of sequence after modification: {}".format(checking_indices)) 
        for i in range(checking_indices.shape[0]): 
            assert checking_indices[i][1] % self.sliding_window_length == 0, "start of sequence is not at the right position" 
            
        # making attention_mask 
        modified_input_bos_sequence_indices = torch.nonzero(input_ids == self.tokenizer_bos_id).to(input_ids.device).to(torch.long) 
        modified_input_bos_sequence_indices = modified_input_bos_sequence_indices[:, 1].unsqueeze(1).expand(-1, seq_length) 
        print("modified_input_bos_sequence_indices shape {}".format(modified_input_bos_sequence_indices.shape)) 
        col_indices = torch.arange(seq_length).expand(batch_size, -1).to(input_ids.device) 
        attention_mask = col_indices >= modified_input_bos_sequence_indices 
        # attention_mask = input_ids != self.tokenizer_pad_id 
        attention_mask = attention_mask.to(torch.long) 
        print("attention_mask {}".format(attention_mask[2])) 
        # just for checking 
        for i in range(checking_indices.shape[0]): 
            if checking_indices[i][1] != 0: 
                assert torch.unique(attention_mask[i][: checking_indices[i][1]]) == 0, "attention_mask is not correct" 
            assert torch.unique(attention_mask[i][checking_indices[i][1] : ]) == 1, "attention_mask is not correct" 
            print(colored("checking attention_mask passed", "green")) 
                
        # past_key_values is not used and input_ids is not changed 
        '''
        position_ids = kwargs.get("position_ids", None) 
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :] 
        ''' 
        position_ids = None 
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"large_input_ids": input_ids, 
                            "small_input_ids": input_ids,} 

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

class LlamaWeirdLarge3(LlamaPreTrainedModel): 
    """ 
    Inside this class, the lm_head is not used 
    We also have a groupping function call at the beginning of the forward function 
    """ 
    # almost identical to LlamaWeirdLarge3, but weird fix for some model 
    _tied_weights_keys = ["lm_head.weight"]
    '''
    def __init__(self, *args, small_config, hostname, large_dim, sliding_window_length = 7, use_mse_loss = False, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.model = LlamaModel(self.config) 
        self.vocab_size = self.config.vocab_size 
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False) 
        
        # self.addonsmallmodel = addonsmallmodel 
        # self.addonsmallmodel = SimpleSmallModel(small_config, sliding_window_length = sliding_window_length, hostname = hostname, target_model_dim = large_dim) 
        self.addonsmallmodel = None 
        self.sliding_window_length = sliding_window_length 
        # self.small_model_dtype = self.addonsmallmodel.embed_projection.weight.dtype 
        self.small_model_dtype = torch.bfloat16 
        print(colored("small_model_dtype {}".format(self.small_model_dtype), "red")) 
        self.use_mse_loss = use_mse_loss 
        self.alpha = 0.5 

        # Initialize weights and apply final processing
        self.post_init()
    ''' 
    def __init__(self, config): 
        super().__init__(config) 
        self.model = LlamaModel(config) 
        self.vocab_size = config.vocab_size 
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False) 
        # self.addonsmallmodel = None 
        small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m") 
        self.sliding_window_length = 7 
        self.addonsmallmodel = SimpleSmallModel(small_config, sliding_window_length = self.sliding_window_length, target_model_dim = self.config.hidden_size) 
        self.small_model_dtype = torch.bfloat16 
        self.use_mse_loss = False 
        self.ce_loss_only = False 
        self.alpha = 0.5 
        self.addonmodel_start = self.sliding_window_length + 1 
        self.inference_setting = "setting0" 
        self.use_cosinesimilarity = False 
        self.generate_iteration_count = 0 
        self.generate_model_hidden_states = torch.tensor(0) # this field saves the intermediate tensors generated by the large model 
        self.tokenizer_bos_id = 1 
        self.tokenizer_pad_id = 2 
        
        self.post_init() 

    def get_input_embeddings(self):
        return self.model.embed_tokens 
    
    def set_msece_loss(self, use_mse_loss, ce_loss_only): 
        self.use_mse_loss = use_mse_loss 
        self.ce_loss_only = ce_loss_only 
    
    def set_cosinesimilarity(self, use_cosinesimilarity): 
        if use_cosinesimilarity: 
            self.use_cosinesimilarity = True 
    
    def set_addonsmallmodel_statedict(self, small_state_dict_for_model): 
        new_state_dict = {} 

        for key in small_state_dict_for_model.keys(): 
            new_key = key 
            if 'lm_head' in key: 
                print("got here found the following key {}".format(key)) 
            if 'model.' in key: 
                new_key = key[6 :] 
            print(new_key) 
            new_state_dict[new_key] = small_state_dict_for_model[key] 
        # if args.embedding_pretrained: 
        #     new_state_dict["embed_projection.weight"] = torch.load("linearprojectionweighttesting.pt") 
        try: 
            self.addonsmallmodel.load_state_dict(new_state_dict) 
        except RuntimeError as r: 
            print(colored(r, "yellow")) 

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value 
    
    def set_inference_setting(self, setting = "setting0"): 
        self.inference_setting = setting 
    
    def set_slidingwindowlength(self, sliding_window_length, addonmodel_start = None): 
        if self.sliding_window_length != sliding_window_length: 
            print("sliding_window_length {} sliding_window_length {}".format(self.sliding_window_length, sliding_window_length)) 
            raise Warning("Detecting sliding winodw length change, if this is not intended, please reinitialize the large model with the correct sliding window length") 
        self.sliding_window_length = sliding_window_length 
        if addonmodel_start is not None: 
            self.addonmodel_start = addonmodel_start 
        else: 
            self.addonmodel_start = self.sliding_window_length + 1 
    
    def set_tokenizer_bos_id(self, bos_id, pad_id): 
        self.tokenizer_bos_id = bos_id 
        self.tokenizer_pad_id = pad_id 
    
    def set_walpha(self, alpha) : 
        self.alpha = alpha 

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder
        
    def reinitialize_embeddings(self, type = "xavieruniform"): 
        from torch.nn import init 
        embedding = self.model.embed_tokens 
        if type == "xavieruniform": 
            init.xavier_uniform_(embedding.weight) 
        elif type == "xaviernormal": 
            init.xavier_normal_(embedding.weight) 
        elif type == "kaimingnormal": 
            init.kaiming_normal_(embedding.weight) 
        else: 
            raise ValueError("type not recognized") 

    def get_decoder(self):
        return self.model 
    
    def naive_grouping(self, input_ids): 
        embedding_searched = self.model.embed_tokens(input_ids) 
        # print("embedding_searched shape {} {}".format(embedding_searched.shape, embedding_searched.dtype)) 
        seq_length = embedding_searched.shape[1] 
        print("seq_length {}".format(seq_length)) 
        
        # assert seq_length % 7 == 0, "seq_length is not divisible by 7" 
        # assert seq_length % self.sliding_window_length == 0, "seq_length is not divisible by sliding_window_length" 
        # added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // 7, embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
        added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // self.sliding_window_length, embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
        # for i in range(seq_length // 7): 
        for i in range(seq_length // self.sliding_window_length): 
            sum = torch.zeros((embedding_searched.shape[0], embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
            # for j in range(7): 
            for j in range(self.sliding_window_length): 
                # sum += embedding_searched[:, i * 7 + j, :] 
                sum += embedding_searched[:, i * self.sliding_window_length + j, :] 
                # sum /= 7. 
                # print("sum dtype {}".format(sum.dtype)) 
            sum /= float(self.sliding_window_length) 
            added_tensor[:, i, :] = sum 
        # print("added_tensor shape {}".format(added_tensor.shape)) 
        
        return added_tensor 
    
    def attention_mask_upper(self, input_ids): 
        sequence_length = ((input_ids.shape[1] - 1) // self.sliding_window_length) + 1 
        batch_size = input_ids.shape[0] 
        condition_mask = input_ids == self.tokenizer_bos_id # finds the index of the start of sequence token 
        start_of_sequenceidx = torch.nonzero(condition_mask)[:, 1] 
        start_of_sequenceidx //= self.sliding_window_length 
        start_of_sequenceidx = start_of_sequenceidx.to(torch.long) 
        # modified_input_bos_sequence_indices = modified_input_bos_sequence_indices[:, 1].unsqueeze(1).expand(-1, seq_length) 
        start_of_sequenceidx2 = start_of_sequenceidx.unsqueeze(1).expand(-1, sequence_length) 
        print("start_of_sequenceidx shape {}".format(start_of_sequenceidx2.shape)) 
        col_indices = torch.arange(sequence_length).expand(batch_size, -1).to(input_ids.device) 
        attention_mask = col_indices >= start_of_sequenceidx2 
        attention_mask = attention_mask.to(torch.long) 
        return start_of_sequenceidx, attention_mask 
        
    def set_addonsmallmodel(self, addonsmallmodel): 
        self.addonsmallmodel = addonsmallmodel 
    
    def set_smallmodelfull(self): 
        self.addonsmallmodel = self.addonsmallmodel.to(torch.float32) 
    
    def l2distancecompute(self, inputs, hidden_states): 
        input_used = inputs.clone().detach()[:, 1:, :]
        hidden_states_used = hidden_states.clone().detach()[:, :-1, :] 
        assert input_used.shape == hidden_states_used.shape 
        dmod = input_used.shape[-1] 
        input_used = input_used.reshape(-1, dmod) 
        hidden_states_used = hidden_states_used.reshape(-1, dmod) 
        # compute the difference 
        diff = input_used - hidden_states_used 
        
        # compute the square 
        diff = diff ** 2
        
        # sum up the square 
        diff = torch.sum(diff, dim = 1) 
        
        # take square root 
        diff = torch.sqrt(diff) 
        
        # average the l2 distance 
        diff = torch.mean(diff) 
        
        return diff 

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        # input_ids: torch.LongTensor = None, 
        large_input_ids: torch.LongTensor = None, 
        small_input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        original_attention_mask = None, 
        condensed_embed_labels = None, 
    ) -> Union[Tuple, CausalLMOutputWithPastLargeDistance2]: 
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

        # inside this function callm, input is through input_embeds 
        # now, we have a start of sequence token 
        print("large_input_ids sequence first element: {}".format(large_input_ids[:, 0])) 
        start_token = self.model.embed_tokens(large_input_ids[:, 0].unsqueeze(1)) 
        extra_pass_in_embeds = self.naive_grouping(large_input_ids[:, 1: ]) 
        extra_pass_in_embeds = torch.cat((start_token, extra_pass_in_embeds), dim = 1) # concatenate at the sequence length dimension 
        # the attention mask should be compatible to the new input_embeds 
        print("attention_mask shape {} and extra_pass_in_embeds shape {}".format(attention_mask.shape, extra_pass_in_embeds.shape)) 
        # print("condensed_embeds_labels shape {}".format(condensed_embed_labels.shape)) 
        assert attention_mask.shape[1] == extra_pass_in_embeds.shape[1], "attention_mask shape is not compatible to the new input_embeds" 
        assert inputs_embeds is None, "inputs_embeds is not None" 
        inputs_embeds = extra_pass_in_embeds 
        print(colored("inputs_embeds shape {} dtype {}".format(inputs_embeds.shape, inputs_embeds.dtype), "yellow")) 
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn) 
        # TODO delete the following line 
        
        outputs = self.model(
            input_ids=None, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) 

        hidden_states = outputs[0] # we don't need the lm_head 
        print("hidden_states shape {} dtype {}".format(hidden_states.shape, hidden_states.dtype)) 
        if self.small_model_dtype == torch.float32: 
            hidden_states = hidden_states.to(torch.float32) 
        elif self.small_model_dtype == torch.bfloat16: 
            hidden_states = hidden_states.to(torch.bfloat16) 
        # print(colored("small_model_type: {}".format(self.small_model_dtype), "red")) 
        # intermediate_l2_dist = self.l2distancecompute(inputs_embeds, hidden_states) 
        
        practical_mask = attention_mask.unsqueeze(-1).expand_as(inputs_embeds) 
        hidden_states = hidden_states[:, 1:-1, :] # NOTE this is very important 
        if condensed_embed_labels is not None: 
            mselabels = condensed_embed_labels 
            # print("first 100 eleents of hidden_states: {}".format(hidden_states[0][0][: 100])) 
            # print("first 100 elements of mselabels: {}".format(mselabels[0][0][: 100])) 
            # hidden_states[practical_mask == 0] = 0 
            # hidden_states = hidden_states[:, :-1, :] # NOTE this is very important 
            # output 30 condensed tokens, the last one and the first one doesn't have the condensed token label, so 28 left 
            # assert labels.shape == hidden_states.shape 
            assert mselabels.shape == hidden_states.shape 
            mse_lossfunc = nn.MSELoss() 
            mse_loss = mse_lossfunc(hidden_states, mselabels) 
            cosinesimlossfunc = nn.CosineEmbeddingLoss() 
            cossim_loss = cosinesimlossfunc(hidden_states.reshape(-1, hidden_states.shape[-1]), mselabels.reshape(-1, mselabels.shape[-1]), torch.ones(hidden_states.shape[0] * hidden_states.shape[1]).to(hidden_states.device)) 
        else: 
            mse_loss = torch.tensor(0) 
            cossim_loss = torch.tensor(0) 
        # mse_loss = 0.5 * mse_loss + 0.5 * cossim_loss 
        # intermediate_l2_dist = mse_loss.clone().detach() 
        intermediate_l2_dist = mse_loss.clone().detach() 
        if self.use_cosinesimilarity: 
            mse_loss = cossim_loss 
        cossim_input = cossim_loss.clone().detach() 
        # print(colored("mse_loss {}".format(mse_loss), "red")) 
        
        assert inputs_embeds.shape[1] - 2 == hidden_states.shape[1] 
        mse_lossfunc2 = nn.MSELoss() 
        # print("first 100 elements of input_embeds: {}".format(inputs_embeds[0][0][: 100])) 
        # inputs_embeds = inputs_embeds[:, 1:, :] 
        inputs_embeds = inputs_embeds[:, 2:, :] # NOTE first condensed token is the start of sequence, while the second one is the first token 
        mse_loss_input = mse_lossfunc2(hidden_states, inputs_embeds) 
        l2_distance_input = mse_loss_input.clone().detach() 
        # print(colored("mse_loss_input {}".format(mse_loss_input), "red")) 
        # cossim_input = F.cosine_similarity(hidden_states.reshape(-1, hidden_states.shape[-1]), inputs_embeds.reshape(-1, inputs_embeds.shape[-1]), dim = 1) 
        # print("cossim_input shape {}".format(cossim_input.shape)) 
        # cossim_input = cossim_input.mean(dim = 0) 
        # print("cossim_input {}".format(cossim_input)) 
        
        if self.use_mse_loss: 
            print(colored("mse_loss {}".format(mse_loss), "red")) 
            # still use the small model and get ce 
            hidden_states = hidden_states.detach().clone() 
            # hidden_states = torch.zeros_like(hidden_states).detach() 
            # hidden_states = condensed_embed_labels 
            '''
            return CausalLMOutputWithPastLargeDistance2(
                loss = mse_loss, 
                logits = None, 
                past_key_values = outputs.past_key_values, 
                hidden_states=outputs.hidden_states,
                attentions = outputs.attentions, 
                l2_distance = intermediate_l2_dist, 
                ce_loss = torch.tensor(0), 
            ) 
            ''' 
        # hidden_states has shape (batch_size, seq_length // 7, hidden states) 
        # hidden_states = hidden_states[:, :-1, :] 
        
        # interleave the hidden_states and the input_ids 
        # assert hidden_states.shape[1] == small_input_ids.shape[1] // 7 - 1 
        assert hidden_states.shape[1] == (small_input_ids.shape[1] - self.addonmodel_start) // self.sliding_window_length 
        addonmodeloutput = self.addonsmallmodel( 
            # input_ids = input_ids, 
            input_ids = small_input_ids, 
            attention_mask = original_attention_mask, 
            position_ids = None, 
            past_key_values = None, 
            condensed_embeds = hidden_states, 
            labels = None, 
            use_cache = None, 
            output_attentions = True, 
            output_hidden_states = None, 
            return_dict = True, 
            start_idx = self.addonmodel_start, # NOTE this is very important 
            eval_mode = False, 
            iteration_count = 1, 
            condensed_fashion = "projection_mode", 
            # experiment_setting = "setting3", 
            experiment_setting = self.inference_setting, 
        ) 
        
        logits = addonmodeloutput.logits 
        
        '''
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        ''' 
        
        # seq_length = input_ids.shape[1] + hidden_states.shape[1] 
        seq_length = small_input_ids.shape[1] + hidden_states.shape[1] 
        assert seq_length == logits.shape[1], "seq_length is not compatible to logits" 
        # mask_list_pos = [i * (self.sliding_window_length + 1) for i in range(seq_length // (self.sliding_window_length + 1))] 
        # mask_list_pos = [7 + i * (self.sliding_window_length + 1) for i in range((seq_length - 7) // (self.sliding_window_length + 1))] 
        mask_list_pos = [self.addonmodel_start + i * (self.sliding_window_length + 1) for i in range((seq_length - self.addonmodel_start) // (self.sliding_window_length + 1))] 
        mask_list_pos22 = [x - 1 for x in mask_list_pos] 
        # print(colored("mask_list_pos {}".format(mask_list_pos), "red")) 
        loss = None 
        if labels is not None: 
            # selected_indices = list(range(7)) 
            selected_indices = list(range(self.addonmodel_start - 1)) 
            # for i in range(7, seq_length): 
                # if i not in mask_list_pos: 
                    # selected_indices.append(i) 
            for i in range(self.addonmodel_start - 1, seq_length): 
                if i not in mask_list_pos22: 
                    selected_indices.append(i) 
            # print(colored("selected_indices {}".format(selected_indices), "red")) 
            # select and shift the logits 
            logits = logits[:, selected_indices, :] 
            shift_logits = logits[..., :-1, :].contiguous() 
            shift_labels = labels[..., 1:].contiguous() # shape (batch_size, seq_length - 1) 
            print("shift_logits shape {}; shift_labels shape {}".format(shift_logits.shape, shift_labels.shape)) 
            # Flatten the tokens 
            loss_fct = CrossEntropyLoss() 
            shift_logits = shift_logits.view(-1, self.config.vocab_size) 
            shift_labels = shift_labels.view(-1) 
            # Enable model parallelism 
            shift_labels = shift_labels.to(shift_logits.device) 
            ce_loss = loss_fct(shift_logits, shift_labels) 
            loss = ce_loss 
            # print(colored("rank {} loss {}".format(self.accelerator.state.process_index, loss), "yellow")) 
        if loss is not None and not self.use_mse_loss: 
            if self.ce_loss_only: 
                print(colored("ce_loss only", "red")) 
                loss = ce_loss 
            else: 
                print(colored("ce_loss + mse_loss", "red")) 
                # loss = self.alpha * loss + (1 - self.alpha) * mse_loss 
                loss = self.alpha * ce_loss + (1 - self.alpha) * mse_loss 
        else: 
            print(colored("mse_loss only", "red")) 
            loss = mse_loss 
        
        first_pos_loss = torch.tensor(0) 
        second_pos_loss = torch.tensor(0) 

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output 

        return CausalLMOutputWithPastLargeDistance2(
            loss=loss,
            logits = logits, 
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions, 
            attentions = addonmodeloutput.attentions, # delibrately using the model's attention mask with modifications 
            l2_distance = intermediate_l2_dist, 
            ce_loss = ce_loss.detach().clone(), 
            l2_distance_input = l2_distance_input, 
            cossim_input = cossim_input, 
            first_pos_loss = first_pos_loss, 
            second_pos_loss = second_pos_loss, 
        ) 
    
    def forward_generate( 
        self, 
        large_input_ids: torch.LongTensor = None, 
        small_input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None, # for generation, we choose to make the attention_mask inside the forward_generation 
        position_ids: Optional[torch.LongTensor] = None, # for generation, we choose to not use the position ids 
        past_key_values: Optional[List[torch.FloatTensor]] = None, 
        input_embeds: Optional[torch.FloatTensor] = None, 
        use_cache: Optional[bool] = None, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None, 
        return_dict: Optional[bool] = None, 
        original_attention_mask = None, 
        condensed_embed_labels = None, 
    ) -> Union[Tuple, CausalLMOutputWithPastLargeDistance2]: 
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # inside this function callm, input is through input_embeds 
        # now, we have a start of sequence token 
        print("large_input_ids sequence first element: {}".format(large_input_ids[:, 0])) 
        start_token = self.model.embed_tokens(large_input_ids[:, 0].unsqueeze(1)) 
        extra_pass_in_embeds = self.naive_grouping(large_input_ids[:, 1: ]) 
        extra_pass_in_embeds = torch.cat((start_token, extra_pass_in_embeds), dim = 1) # concatenate at the sequence length dimension 

        # NOTE we don't use the pass-in attention mask 
        start_of_sequenceidx, attention_mask = self.attention_mask_upper(large_input_ids) 
        for i in range(6): 
            print("start_of_sequenceidx[i]: {}".format(start_of_sequenceidx[i])) 
            print("attention_mask of the {}th sequence: {}".format(i, attention_mask[i])) 
        
        # filled in start of sequence token to every line in the batch 
        bos_token = torch.tensor([self.tokenizer_bos_id for i in range(extra_pass_in_embeds.shape[0])]).to(extra_pass_in_embeds.device) 
        bos_embed_token = self.model.embed_tokens(bos_token.unsqueeze(1)) # shape (batch_size, 1, hidden_size) 
        print("shape of start_of_sequenceidx: {}".format(start_of_sequenceidx.shape)) 
        for i in range(extra_pass_in_embeds.shape[0]): 
            extra_pass_in_embeds[i, start_of_sequenceidx[i]] = bos_embed_token[i, 0] 
            print("shape of bos_embed_token: {}".format(bos_embed_token[i, 0].shape)) 
            print("shape of extra_pass_in_embeds[i, start_of_sequenceidx[i]]: {}".format(extra_pass_in_embeds[i, start_of_sequenceidx[i]].shape)) 
        
        # the attention mask should be compatible to the new input_embeds 
        print("attention_mask shape {} and extra_pass_in_embeds shape {}".format(attention_mask.shape, extra_pass_in_embeds.shape)) 
        assert attention_mask.shape[1] == extra_pass_in_embeds.shape[1], "attention_mask shape is not compatible to the new input_embeds" 
        # assert inputs_embeds is None, "inputs_embeds is not None" 
        inputs_embeds = extra_pass_in_embeds 
        print(colored("inputs_embeds shape {} dtype {}".format(inputs_embeds.shape, inputs_embeds.dtype), "yellow")) 
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn) 
        # TODO delete the following line 
        
        if self.generate_iteration_count % self.sliding_window_length == 0: 
            print(colored("running the large model side", "yellow")) 
            outputs = self.model(
                input_ids=None, 
                attention_mask=attention_mask,
                position_ids=position_ids,
                # past_key_values=past_key_values,
                past_key_values = None, 
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ) 

            hidden_states = outputs[0] # we don't need the lm_head 
            print("hidden_states shape {} dtype {}".format(hidden_states.shape, hidden_states.dtype)) 
            if self.small_model_dtype == torch.float32: 
                hidden_states = hidden_states.to(torch.float32) 
            elif self.small_model_dtype == torch.bfloat16: 
                hidden_states = hidden_states.to(torch.bfloat16) 
            # print(colored("small_model_type: {}".format(self.small_model_dtype), "red")) 
            # intermediate_l2_dist = self.l2distancecompute(inputs_embeds, hidden_states) 
            self.generate_model_hidden_states = hidden_states.clone().detach() 
        self.generate_iteration_count += 1 
        
        practical_mask = attention_mask.unsqueeze(-1).expand_as(inputs_embeds) 
        # NOTE here we don't clip the hidden_states sequence length anymore 
        # we try using all the hidden_states including the padding tokens 
        
        # assert input_embeds.shape[1] == hidden_states.shape[1] 
        assert inputs_embeds.shape[1] == self.generate_model_hidden_states.shape[1] 
        
        # interleave the hidden_states and the input_ids 
        # assert hidden_states.shape[1] == small_input_ids.shape[1] // 7 - 1 
        print(colored("running the small model side", "green")) 
        addonmodeloutput = self.addonsmallmodel( 
            # input_ids = input_ids, 
            input_ids = small_input_ids, 
            attention_mask = original_attention_mask, 
            position_ids = None, 
            past_key_values = None, 
            condensed_embeds = self.generate_model_hidden_states, 
            labels = None, 
            use_cache = None, 
            output_attentions = True, 
            output_hidden_states = None, 
            return_dict = True, 
            start_idx = 1, # NOTE this is very important 
            eval_mode = False, 
            iteration_count = 1, 
            condensed_fashion = "projection_mode", 
            # experiment_setting = "setting3", 
            experiment_setting = self.inference_setting, 
            generate_flag = True, 
        ) 
        
        logits = addonmodeloutput.logits 
        
        '''
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        ''' 
        
        # seq_length = input_ids.shape[1] + hidden_states.shape[1] 
        ce_loss = None 
        # seq_length = small_input_ids.shape[1] + hidden_states.shape[1] 
        seq_length = small_input_ids.shape[1] + self.generate_model_hidden_states.shape[1] 
        assert seq_length == logits.shape[1], "seq_length is not compatible to logits" 
        # mask_list_pos = [i * (self.sliding_window_length + 1) for i in range(seq_length // (self.sliding_window_length + 1))] 
        # mask_list_pos = [7 + i * (self.sliding_window_length + 1) for i in range((seq_length - 7) // (self.sliding_window_length + 1))] 
        # addonmodel_start = 1 
        loss = None 

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output 

        return CausalLMOutputWithPastLargeDistance2(
            loss=loss,
            logits = logits, 
            # past_key_values=outputs.past_key_values,
            past_key_values = past_key_values, 
            # hidden_states=outputs.hidden_states, 
            hidden_states = self.generate_model_hidden_states, 
            # attentions=outputs.attentions, 
            attentions = addonmodeloutput.attentions, # delibrately using the model's attention mask with modifications 
            l2_distance = None, 
            ce_loss = None, 
            l2_distance_input = None, 
            cossim_input = None, 
        ) 
    
    def sample( 
        self, 
        input_ids: torch.LongTensor, 
        logits_processor: Optional[LogitsProcessorList] = None, 
        stopping_criteria: Optional[StoppingCriteriaList] = None, 
        logits_warper: Optional[LogitsProcessorList] = None, 
        max_length: Optional[int] = None, 
        pad_token_id: Optional[int] = None, 
        eos_token_id: Optional[int] = None, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None, 
        output_scores: Optional[bool] = None, 
        return_dict_in_generate: Optional[bool] = None, 
        synced_gpus: bool = False, 
        streamer: Optional["BaseStreamer"] = None, 
        **model_kwargs, 
    ) -> Union[SampleOutput, torch.LongTensor]: 
        
        print("inside generate function, output_hidden_states is {}".format(output_hidden_states)) 
        print(colored("inside the function that is overloaded for the model", "yellow")) 
        
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList() 
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList() 
        if max_length is not None: 
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            ) 
            stopping_criteria = StoppingCriteriaList(stopping_criteria, max_length) 
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList() 
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )
        
        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        
        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )
        
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device) 
        
        this_peer_finished = False # used by synced_gpus only 
        
        self.generate_iteration_count = 0 
        # auto-regressive generation 
        while True: 
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, adjustment_scheme = "case1", **model_kwargs) 

            # forward pass to get next token
            # outputs = self(
                # **model_inputs,
                # return_dict=True,
                # output_attentions=output_attentions,
                # output_hidden_states=output_hidden_states,
            # ) 
            outputs = self.forward_generate(
                **model_inputs, 
                return_dict = True, 
                output_attentions = output_attentions, 
                output_hidden_states = output_hidden_states, 
            ) 

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def prepare_inputs_for_generation2(
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

    def prepare_inputs_for_generation( 
        self, input_ids, past_key_values = None, attention_mask = None, inputs_embeds = None, adjustment_scheme = None, **kwargs): 
        # mainly used to debug preparing inputs for generation and not using past_key_values 
        assert past_key_values is None, "past_key_values is not None" 
        batch_size, seq_length = input_ids.shape 
        print("batch_size {}; seq_length {}".format(batch_size, seq_length)) 
        
        # adjusting the inputs and mask 
        print("input_ids {}".format(input_ids[2])) 
        print("attention_mask {}".format(attention_mask[2])) 
        condition_mask = input_ids == self.tokenizer_bos_id 
        input_sequence_indices = torch.nonzero(condition_mask).to(input_ids.device).to(torch.long) 
        print("input_sequence_indices shape {}".format(input_sequence_indices.shape)) 
        print("input_sequence_indices: {}".format(input_sequence_indices[2])) 
        input_sequence_indices2 = [] 
        modified_input_bos_sequence_indices = [] 
        assert input_sequence_indices.shape[0] == input_ids.shape[0], "every row of sequences need to have an bos" 
        for i in range(input_ids.shape[0]): # iterate through the batch_size 
            # if input_sequence_indices[i] % self.sliding_window_length != 0: # we found a sequence that needs to be adjusted 
            if input_sequence_indices[i][1].data % self.sliding_window_length != 0: # we found a sequence that needs to be adjusted 
                input_sequence_indices2.append(torch.tensor([i, (input_sequence_indices[i][1])]).to(input_ids.device).view(1, -1)) 
                if adjustment_scheme == "case1": 
                    modified_input_bos_sequence_indices.append(torch.tensor([i, (input_sequence_indices[i][1] // self.sliding_window_length) * self.sliding_window_length]).to(input_ids.device).view(1, -1)) 
                elif adjustment_scheme == "case2": 
                    modified_input_bos_sequence_indices.append(torch.tensor([i, (input_sequence_indices[i][1] // self.sliding_window_length + 1) * self.sliding_window_length]).to(input_ids.device).view(1, -1)) 
                else: 
                    raise ValueError("adjustment_scheme is not recognized") 
        if len(input_sequence_indices2) != 0: 
            # adjusting the input_ids 
            input_sequence_indices2 = torch.cat(input_sequence_indices2, dim = 0).to(input_ids.device).to(torch.long) 
            modified_input_bos_sequence_indices = torch.cat(modified_input_bos_sequence_indices, dim = 0).to(input_ids.device).to(torch.long) 
            print("shape of modified_input_bos_sequence_indices {}".format(modified_input_bos_sequence_indices.shape)) 
            print(modified_input_bos_sequence_indices) 
            
            row_indices = input_sequence_indices2[:, 0] 
            col_indices = input_sequence_indices2[:, 1] 
            input_ids.index_put_((row_indices, col_indices), torch.full_like(row_indices, fill_value = self.tokenizer_pad_id, device = input_ids.device, dtype = input_ids.dtype), accumulate = False) 
            
            row_indices = modified_input_bos_sequence_indices[:, 0] 
            col_indices = modified_input_bos_sequence_indices[:, 1] 
            input_ids.index_put_((row_indices, col_indices), torch.full_like(row_indices, fill_value = self.tokenizer_bos_id, device = input_ids.device, dtype = input_ids.dtype), accumulate = False) 
            
            print("input_ids {}".format(input_ids[2])) 
        # just for checking 
        checking_indices = torch.nonzero(input_ids == self.tokenizer_bos_id) 
        print("positions of the start of sequence after modification: {}".format(checking_indices)) 
        for i in range(checking_indices.shape[0]): 
            assert checking_indices[i][1] % self.sliding_window_length == 0, "start of sequence is not at the right position" 
            
        # making attention_mask 
        modified_input_bos_sequence_indices = torch.nonzero(input_ids == self.tokenizer_bos_id).to(input_ids.device).to(torch.long) 
        modified_input_bos_sequence_indices = modified_input_bos_sequence_indices[:, 1].unsqueeze(1).expand(-1, seq_length) 
        print("modified_input_bos_sequence_indices shape {}".format(modified_input_bos_sequence_indices.shape)) 
        col_indices = torch.arange(seq_length).expand(batch_size, -1).to(input_ids.device) 
        attention_mask = col_indices >= modified_input_bos_sequence_indices 
        # attention_mask = input_ids != self.tokenizer_pad_id 
        attention_mask = attention_mask.to(torch.long) 
        print("attention_mask {}".format(attention_mask[2])) 
        # just for checking 
        for i in range(checking_indices.shape[0]): 
            if checking_indices[i][1] != 0: 
                assert torch.unique(attention_mask[i][: checking_indices[i][1]]) == 0, "attention_mask is not correct" 
            assert torch.unique(attention_mask[i][checking_indices[i][1] : ]) == 1, "attention_mask is not correct" 
            print(colored("checking attention_mask passed", "green")) 
                
        # past_key_values is not used and input_ids is not changed 
        '''
        position_ids = kwargs.get("position_ids", None) 
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :] 
        ''' 
        position_ids = None 
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"large_input_ids": input_ids, 
                            "small_input_ids": input_ids,} 

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

class LlamaWeirdLarge2(LlamaPreTrainedModel): 
    """ 
    Inside this class, the lm_head is not used 
    We also have a groupping function call at the beginning of the forward function 
    """ 
    
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, *args, sliding_window_length = 7, addonsmallmodel, use_mse_loss = False, ce_loss_only = False, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.model = LlamaModel(self.config) 
        self.vocab_size = self.config.vocab_size 
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False) 
        
        self.addonsmallmodel = addonsmallmodel 
        self.sliding_window_length = sliding_window_length 
        self.small_model_dtype = self.addonsmallmodel.embed_projection.weight.dtype 
        print(colored("small_model_dtype {}".format(self.small_model_dtype), "red")) 
        self.use_mse_loss = use_mse_loss 
        self.ce_loss_only = ce_loss_only 
        self.alpha = 0.5 

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
    
    def naive_grouping(self, input_ids): 
        embedding_searched = self.model.embed_tokens(input_ids) 
        # print("embedding_searched shape {} {}".format(embedding_searched.shape, embedding_searched.dtype)) 
        seq_length = embedding_searched.shape[1] 
        
        assert seq_length % 7 == 0, "seq_length is not divisible by 7" 
        added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // 7, embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
        for i in range(seq_length // 7): 
            sum = torch.zeros((embedding_searched.shape[0], embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
            for j in range(7): 
                sum += embedding_searched[:, i * 7 + j, :] 
                sum /= 7. 
                # print("sum dtype {}".format(sum.dtype)) 
            added_tensor[:, i, :] = sum 
        # print("added_tensor shape {}".format(added_tensor.shape)) 
        
        return added_tensor 
    
    def set_addonsmallmodel(self, addonsmallmodel): 
        self.addonsmallmodel = addonsmallmodel 
    
    def set_smallmodelfull(self): 
        self.addonsmallmodel = self.addonsmallmodel.to(torch.float32) 
    
    def l2distancecompute(self, inputs, hidden_states): 
        input_used = inputs.clone().detach()[:, 1:, :]
        hidden_states_used = hidden_states.clone().detach()[:, :-1, :] 
        assert input_used.shape == hidden_states_used.shape 
        dmod = input_used.shape[-1] 
        input_used = input_used.reshape(-1, dmod) 
        hidden_states_used = hidden_states_used.reshape(-1, dmod) 
        # compute the difference 
        diff = input_used - hidden_states_used 
        
        # compute the square 
        diff = diff ** 2
        
        # sum up the square 
        diff = torch.sum(diff, dim = 1) 
        
        # take square root 
        diff = torch.sqrt(diff) 
        
        # average the l2 distance 
        diff = torch.mean(diff) 
        
        return diff 

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        # input_ids: torch.LongTensor = None, 
        large_input_ids: torch.LongTensor = None, 
        small_input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        original_attention_mask = None, 
        condensed_embed_labels = None, 
    ) -> Union[Tuple, CausalLMOutputWithPastLargeDistance2]: 
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

        # inside this function callm, input is through input_embeds 
        extra_pass_in_embeds = self.naive_grouping(large_input_ids) 
        # the attention mask should be compatible to the new input_embeds 
        assert attention_mask.shape[1] == extra_pass_in_embeds.shape[1], "attention_mask shape is not compatible to the new input_embeds" 
        assert inputs_embeds is None, "inputs_embeds is not None" 
        inputs_embeds = extra_pass_in_embeds 
        print(colored("inputs_embeds shape {} dtype {}".format(inputs_embeds.shape, inputs_embeds.dtype), "yellow")) 
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn) 
        # TODO delete the following line 
        
        outputs = self.model(
            input_ids=None, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) 

        hidden_states = outputs[0] # we don't need the lm_head 
        print("hidden_states shape {} dtype {}".format(hidden_states.shape, hidden_states.dtype)) 
        if self.small_model_dtype == torch.float32: 
            hidden_states = hidden_states.to(torch.float32) 
        elif self.small_model_dtype == torch.bfloat16: 
            hidden_states = hidden_states.to(torch.bfloat16) 
        # print(colored("small_model_type: {}".format(self.small_model_dtype), "red")) 
        # intermediate_l2_dist = self.l2distancecompute(inputs_embeds, hidden_states) 
        
        practical_mask = attention_mask.unsqueeze(-1).expand_as(inputs_embeds) 
        mselabels = condensed_embed_labels 
        hidden_states[practical_mask == 0] = 0 
        hidden_states = hidden_states[:, :-1, :] # NOTE this is very important 
        # assert labels.shape == hidden_states.shape 
        assert mselabels.shape == hidden_states.shape 
        mse_lossfunc = nn.MSELoss() 
        mse_loss = mse_lossfunc(hidden_states, mselabels) 
        intermediate_l2_dist = mse_loss.clone().detach() 
        
        mse_lossfunc2 = nn.MSELoss() 
        inputs_embeds = inputs_embeds[:, 1:, :] 
        mse_loss_input = mse_lossfunc2(hidden_states, inputs_embeds) 
        l2_distance_input = mse_loss_input.clone().detach() 
        
        if self.use_mse_loss: 
            print(colored("mse_loss {}".format(mse_loss), "red")) 
            return CausalLMOutputWithPastLargeDistance2(
                loss = mse_loss, 
                logits = None, 
                past_key_values = outputs.past_key_values, 
                hidden_states=outputs.hidden_states,
                attentions = outputs.attentions, 
                l2_distance = intermediate_l2_dist, 
                ce_loss = torch.tensor(0), 
            ) 
            
        # hidden_states has shape (batch_size, seq_length // 7, hidden states) 
        # hidden_states = hidden_states[:, :-1, :] 
        
        # interleave the hidden_states and the input_ids 
        assert hidden_states.shape[1] == small_input_ids.shape[1] // 7 - 1 
        addonmodeloutput = self.addonsmallmodel( 
            # input_ids = input_ids, 
            input_ids = small_input_ids, 
            attention_mask = original_attention_mask, 
            position_ids = None, 
            past_key_values = None, 
            condensed_embeds = hidden_states, 
            labels = None, 
            use_cache = None, 
            output_attentions = True, 
            output_hidden_states = None, 
            return_dict = True, 
            start_idx = 7, # NOTE this is very important 
            eval_mode = False, 
            iteration_count = 1, 
            condensed_fashion = "projection_mode", 
            experiment_setting = "setting4", 
        ) 
        
        logits = addonmodeloutput.logits 
        
        '''
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        ''' 
        
        # seq_length = input_ids.shape[1] + hidden_states.shape[1] 
        seq_length = small_input_ids.shape[1] + hidden_states.shape[1] 
        assert seq_length == logits.shape[1], "seq_length is not compatible to logits" 
        # mask_list_pos = [i * (self.sliding_window_length + 1) for i in range(seq_length // (self.sliding_window_length + 1))] 
        mask_list_pos = [7 + i * (self.sliding_window_length + 1) for i in range((seq_length - 7) // (self.sliding_window_length + 1))] 
        # print(colored("mask_list_pos {}".format(mask_list_pos), "red")) 
        loss = None 
        if labels is not None: 
            selected_indices = list(range(7)) 
            for i in range(7, seq_length): 
                if i not in mask_list_pos: 
                    selected_indices.append(i) 
            # print(colored("selected_indices {}".format(selected_indices), "red")) 
            # select and shift the logits 
            logits = logits[:, selected_indices, :] 
            shift_logits = logits[..., :-1, :].contiguous() 
            shift_labels = labels[..., 1:].contiguous() # shape (batch_size, seq_length - 1) 
            print("shift_logits shape {}; shift_labels shape {}".format(shift_logits.shape, shift_labels.shape)) 
            # Flatten the tokens 
            loss_fct = CrossEntropyLoss() 
            shift_logits = shift_logits.view(-1, self.config.vocab_size) 
            shift_labels = shift_labels.view(-1) 
            # Enable model parallelism 
            shift_labels = shift_labels.to(shift_logits.device) 
            ce_loss = loss_fct(shift_logits, shift_labels) 
            loss = ce_loss 
            # print(colored("rank {} loss {}".format(self.accelerator.state.process_index, loss), "yellow")) 
        if loss is not None: 
            if self.ce_loss_only: 
                loss = ce_loss 
            else: 
                # loss = self.alpha * loss + (1 - self.alpha) * mse_loss 
                loss = self.alpha * ce_loss + (1 - self.alpha) * mse_loss 
        else: 
            loss = mse_loss 

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output 

        return CausalLMOutputWithPastLargeDistance2(
            loss=loss,
            logits = logits, 
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions, 
            attentions = addonmodeloutput.attentions, # delibrately using the model's attention mask with modifications 
            l2_distance = intermediate_l2_dist, 
            ce_loss = ce_loss.detach().clone(), 
            l2_distance_input = l2_distance_input, 
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

class LlamaWeirdLarge(LlamaPreTrainedModel): 
    """ 
    We call this autoregressive Medusa model 
    """ 
    # almost identical to LlamaWeirdLarge3, but weird fix for some model 
    _tied_weights_keys = ["lm_head.weight"]
    '''
    def __init__(self, *args, small_config, hostname, large_dim, sliding_window_length = 7, use_mse_loss = False, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.model = LlamaModel(self.config) 
        self.vocab_size = self.config.vocab_size 
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False) 
        
        # self.addonsmallmodel = addonsmallmodel 
        # self.addonsmallmodel = SimpleSmallModel(small_config, sliding_window_length = sliding_window_length, hostname = hostname, target_model_dim = large_dim) 
        self.addonsmallmodel = None 
        self.sliding_window_length = sliding_window_length 
        # self.small_model_dtype = self.addonsmallmodel.embed_projection.weight.dtype 
        self.small_model_dtype = torch.bfloat16 
        print(colored("small_model_dtype {}".format(self.small_model_dtype), "red")) 
        self.use_mse_loss = use_mse_loss 
        self.alpha = 0.5 

        # Initialize weights and apply final processing
        self.post_init()
    ''' 
    def __init__(self, config): 
        super().__init__(config) 
        self.model = LlamaModel(config) 
        self.vocab_size = config.vocab_size 
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False) 
        # self.addonsmallmodel = None 
        small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m") 
        self.sliding_window_length = 7 
        self.addonsmallmodel = SimpleSmallModel(small_config, sliding_window_length = self.sliding_window_length, target_model_dim = self.config.hidden_size) 
        self.small_model_dtype = torch.bfloat16 
        self.use_mse_loss = False 
        self.ce_loss_only = False 
        self.alpha = 0.5 
        self.addonmodel_start = self.sliding_window_length + 1 
        self.inference_setting = "setting0" 
        self.use_cosinesimilarity = False 
        self.generate_iteration_count = 0 
        self.generate_model_hidden_states = torch.tensor(0) # this field saves the intermediate tensors generated by the large model 
        self.tokenizer_bos_id = 1 
        self.tokenizer_pad_id = 2 
        self.compression_scheme = "autoregressive_baseline" 
        
        self.post_init() 

    def get_input_embeddings(self):
        return self.model.embed_tokens 
    
    def set_msece_loss(self, use_mse_loss, ce_loss_only): 
        self.use_mse_loss = use_mse_loss 
        self.ce_loss_only = ce_loss_only 
    
    def set_cosinesimilarity(self, use_cosinesimilarity): 
        if use_cosinesimilarity: 
            self.use_cosinesimilarity = True 
    
    def set_hidden_states_compression_scheme(self, scheme): 
        self.compression_scheme = scheme 
    
    def set_addonsmallmodel_statedict(self, small_state_dict_for_model): 
        new_state_dict = {} 

        for key in small_state_dict_for_model.keys(): 
            new_key = key 
            if 'lm_head' in key: 
                print("got here found the following key {}".format(key)) 
            if 'model.' in key: 
                new_key = key[6 :] 
            print(new_key) 
            new_state_dict[new_key] = small_state_dict_for_model[key] 
        # if args.embedding_pretrained: 
        #     new_state_dict["embed_projection.weight"] = torch.load("linearprojectionweighttesting.pt") 
        try: 
            self.addonsmallmodel.load_state_dict(new_state_dict) 
        except RuntimeError as r: 
            print(colored(r, "yellow")) 

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value 
    
    def set_inference_setting(self, setting = "setting0"): 
        self.inference_setting = setting 
    
    def set_slidingwindowlength(self, sliding_window_length, addonmodel_start = None): 
        self.sliding_window_length = sliding_window_length 
        if addonmodel_start is not None: 
            self.addonmodel_start = addonmodel_start 
        else: 
            self.addonmodel_start = self.sliding_window_length + 1 
    
    def set_tokenizer_bos_id(self, bos_id, pad_id): 
        self.tokenizer_bos_id = bos_id 
        self.tokenizer_pad_id = pad_id 
    
    def set_walpha(self, alpha) : 
        self.alpha = alpha 

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder
        
    def reinitialize_embeddings(self, type = "xavieruniform"): 
        from torch.nn import init 
        embedding = self.model.embed_tokens 
        if type == "xavieruniform": 
            init.xavier_uniform_(embedding.weight) 
        elif type == "xaviernormal": 
            init.xavier_normal_(embedding.weight) 
        elif type == "kaimingnormal": 
            init.kaiming_normal_(embedding.weight) 
        else: 
            raise ValueError("type not recognized") 

    def get_decoder(self):
        return self.model 
    
    def naive_grouping(self, input_ids): 
        embedding_searched = self.model.embed_tokens(input_ids) 
        # print("embedding_searched shape {} {}".format(embedding_searched.shape, embedding_searched.dtype)) 
        seq_length = embedding_searched.shape[1] 
        print("seq_length {}".format(seq_length)) 
        
        # assert seq_length % 7 == 0, "seq_length is not divisible by 7" 
        # assert seq_length % self.sliding_window_length == 0, "seq_length is not divisible by sliding_window_length" 
        # added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // 7, embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
        added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // self.sliding_window_length, embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
        # for i in range(seq_length // 7): 
        for i in range(seq_length // self.sliding_window_length): 
            sum = torch.zeros((embedding_searched.shape[0], embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
            # for j in range(7): 
            for j in range(self.sliding_window_length): 
                # sum += embedding_searched[:, i * 7 + j, :] 
                sum += embedding_searched[:, i * self.sliding_window_length + j, :] 
                # sum /= 7. 
                # print("sum dtype {}".format(sum.dtype)) 
            sum /= float(self.sliding_window_length) 
            added_tensor[:, i, :] = sum 
        # print("added_tensor shape {}".format(added_tensor.shape)) 
        
        return added_tensor 
    
    def attention_mask_upper(self, input_ids): 
        sequence_length = ((input_ids.shape[1] - 1) // self.sliding_window_length) + 1 
        batch_size = input_ids.shape[0] 
        condition_mask = input_ids == self.tokenizer_bos_id # finds the index of the start of sequence token 
        start_of_sequenceidx = torch.nonzero(condition_mask)[:, 1] 
        start_of_sequenceidx //= self.sliding_window_length 
        start_of_sequenceidx = start_of_sequenceidx.to(torch.long) 
        # modified_input_bos_sequence_indices = modified_input_bos_sequence_indices[:, 1].unsqueeze(1).expand(-1, seq_length) 
        start_of_sequenceidx2 = start_of_sequenceidx.unsqueeze(1).expand(-1, sequence_length) 
        print("start_of_sequenceidx shape {}".format(start_of_sequenceidx2.shape)) 
        col_indices = torch.arange(sequence_length).expand(batch_size, -1).to(input_ids.device) 
        attention_mask = col_indices >= start_of_sequenceidx2 
        attention_mask = attention_mask.to(torch.long) 
        return start_of_sequenceidx, attention_mask 
        
    def set_addonsmallmodel(self, addonsmallmodel): 
        self.addonsmallmodel = addonsmallmodel 
    
    def set_smallmodelfull(self): 
        self.addonsmallmodel = self.addonsmallmodel.to(torch.float32) 
    
    def l2distancecompute(self, inputs, hidden_states): 
        input_used = inputs.clone().detach()[:, 1:, :]
        hidden_states_used = hidden_states.clone().detach()[:, :-1, :] 
        assert input_used.shape == hidden_states_used.shape 
        dmod = input_used.shape[-1] 
        input_used = input_used.reshape(-1, dmod) 
        hidden_states_used = hidden_states_used.reshape(-1, dmod) 
        # compute the difference 
        diff = input_used - hidden_states_used 
        
        # compute the square 
        diff = diff ** 2
        
        # sum up the square 
        diff = torch.sum(diff, dim = 1) 
        
        # take square root 
        diff = torch.sqrt(diff) 
        
        # average the l2 distance 
        diff = torch.mean(diff) 
        
        return diff 
    
    def avgpool(self, hidden_states): 
        downsampled_vectors = [] 
        sum = torch.zeros((hidden_states.shape[0], hidden_states.shape[2]), dtype = hidden_states.dtype).to(hidden_states.device) 
        for i in range(hidden_states.shape[1]): 
            if i % self.sliding_window_length == self.sliding_window_length - 1: 
                sum += hidden_states[:, i, :] 
                downsampled_vectors.append(sum / self.sliding_window_length) 
                sum.mul_(0.) 
                assert sum.view(-1).sum() == 0 
            else: 
                sum += hidden_states[:, i, :] 
        # downsampled_vectors = downsampled_vectors[1 :] 
        
        return torch.stack(downsampled_vectors, dim = 1) 

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        # input_ids: torch.LongTensor = None, 
        large_input_ids: torch.LongTensor = None, 
        small_input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        input_embeds: Optional[torch.FloatTensor] = None, 
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        original_attention_mask = None, 
        condensed_embed_labels = None, 
    ) -> Union[Tuple, CausalLMOutputWithPastLargeDistance2]: 
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

        outputs = self.model(
            input_ids = large_input_ids, 
            attention_mask = attention_mask, 
            position_ids = position_ids, 
            past_key_values = past_key_values, 
            inputs_embeds = input_embeds, 
            use_cache = use_cache, 
            output_attentions = output_attentions, 
            output_hidden_states = output_hidden_states, 
            return_dict = return_dict, 
        ) 

        hidden_states = outputs[0] # we don't need the lm_head 
        print("hidden_states shape {} dtype {}".format(hidden_states.shape, hidden_states.dtype)) 
        if self.small_model_dtype == torch.float32: 
            hidden_states = hidden_states.to(torch.float32) 
        elif self.small_model_dtype == torch.bfloat16: 
            hidden_states = hidden_states.to(torch.bfloat16) 
        # print(colored("small_model_type: {}".format(self.small_model_dtype), "red")) 
        # intermediate_l2_dist = self.l2distancecompute(inputs_embeds, hidden_states) 
        seq_len = hidden_states.shape[1] 
        
        if self.compression_scheme == "autoregressive_baseline": 
            selected_seq_indices = [i * self.sliding_window_length for i in range(1, (seq_len - 1) // self.sliding_window_length)] 
            print("selected_seq_indices {} total length {}".format(selected_seq_indices, len(selected_seq_indices))) 
            print("using autoregressive_baseline") 
            hidden_states = hidden_states[:, selected_seq_indices, :] 
        elif self.compression_scheme == "group_compress": 
            print("using group_compress") 
            hidden_states = self.avgpool(hidden_states) 
            hidden_states = hidden_states[:, 1 :, :] 
        else: 
            raise ValueError("compression_scheme not recognized") 
        
        mse_loss = torch.tensor(0) 
        cossim_loss = torch.tensor(0) 
        
        # mse_loss = 0.5 * mse_loss + 0.5 * cossim_loss 
        # intermediate_l2_dist = mse_loss.clone().detach() 
        intermediate_l2_dist = mse_loss.clone().detach() 
        if self.use_cosinesimilarity: 
            mse_loss = cossim_loss 
        cossim_input = cossim_loss.clone().detach() 
        # print(colored("mse_loss {}".format(mse_loss), "red")) 
        
        if self.use_mse_loss: 
            print(colored("mse_loss {}".format(mse_loss), "red")) 
            # still use the small model and get ce 
            hidden_states = hidden_states.detach().clone() 
            # hidden_states = torch.zeros_like(hidden_states).detach() 
            # hidden_states = condensed_embed_labels 
            '''
            return CausalLMOutputWithPastLargeDistance2(
                loss = mse_loss, 
                logits = None, 
                past_key_values = outputs.past_key_values, 
                hidden_states=outputs.hidden_states,
                attentions = outputs.attentions, 
                l2_distance = intermediate_l2_dist, 
                ce_loss = torch.tensor(0), 
            ) 
            ''' 
        # hidden_states has shape (batch_size, seq_length // 7, hidden states) 
        # hidden_states = hidden_states[:, :-1, :] 
        
        # interleave the hidden_states and the input_ids 
        # assert hidden_states.shape[1] == small_input_ids.shape[1] // 7 - 1 
        print("expected {}".format(small_input_ids.shape[1] // self.sliding_window_length - 1)) 
        assert hidden_states.shape[1] == (small_input_ids.shape[1] - self.addonmodel_start) // self.sliding_window_length 
        addonmodeloutput = self.addonsmallmodel( 
            # input_ids = input_ids, 
            input_ids = small_input_ids, 
            attention_mask = original_attention_mask, 
            position_ids = None, 
            past_key_values = None, 
            condensed_embeds = hidden_states, 
            labels = None, 
            use_cache = None, 
            output_attentions = True, 
            output_hidden_states = None, 
            return_dict = True, 
            start_idx = self.addonmodel_start, # NOTE this is very important 
            eval_mode = False, 
            iteration_count = 1, 
            condensed_fashion = "projection_mode", 
            # experiment_setting = "setting3", 
            experiment_setting = self.inference_setting, 
        ) 
        
        logits = addonmodeloutput.logits 
        
        '''
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        ''' 
        
        # seq_length = input_ids.shape[1] + hidden_states.shape[1] 
        seq_length = small_input_ids.shape[1] + hidden_states.shape[1] 
        assert seq_length == logits.shape[1], "seq_length is not compatible to logits" 
        # mask_list_pos = [i * (self.sliding_window_length + 1) for i in range(seq_length // (self.sliding_window_length + 1))] 
        # mask_list_pos = [7 + i * (self.sliding_window_length + 1) for i in range((seq_length - 7) // (self.sliding_window_length + 1))] 
        mask_list_pos = [self.addonmodel_start + i * (self.sliding_window_length + 1) for i in range((seq_length - self.addonmodel_start) // (self.sliding_window_length + 1))] 
        mask_list_pos22 = [x - 1 for x in mask_list_pos] 
        # print(colored("mask_list_pos {}".format(mask_list_pos), "red")) 
        loss = None 
        if labels is not None: 
            # selected_indices = list(range(7)) 
            selected_indices = list(range(self.addonmodel_start - 1)) 
            # for i in range(7, seq_length): 
                # if i not in mask_list_pos: 
                    # selected_indices.append(i) 
            for i in range(self.addonmodel_start - 1, seq_length): 
                if i not in mask_list_pos22: 
                    selected_indices.append(i) 
            # print(colored("selected_indices {}".format(selected_indices), "red")) 
            # select and shift the logits 
            logits = logits[:, selected_indices, :] 
            shift_logits = logits[..., :-1, :].contiguous() 
            shift_labels = labels[..., 1:].contiguous() # shape (batch_size, seq_length - 1) 
            print("shift_logits shape {}; shift_labels shape {}".format(shift_logits.shape, shift_labels.shape)) 
            # Flatten the tokens 
            loss_fct = CrossEntropyLoss() 
            shift_logits = shift_logits.view(-1, self.config.vocab_size) 
            shift_labels = shift_labels.view(-1) 
            # Enable model parallelism 
            shift_labels = shift_labels.to(shift_logits.device) 
            ce_loss = loss_fct(shift_logits, shift_labels) 
            loss = ce_loss 
            # print(colored("rank {} loss {}".format(self.accelerator.state.process_index, loss), "yellow")) 
        if loss is not None and not self.use_mse_loss: 
            if self.ce_loss_only: 
                print(colored("ce_loss only", "red")) 
                loss = ce_loss 
            else: 
                print(colored("ce_loss + mse_loss", "red")) 
                # loss = self.alpha * loss + (1 - self.alpha) * mse_loss 
                loss = self.alpha * ce_loss + (1 - self.alpha) * mse_loss 
        else: 
            print(colored("mse_loss only", "red")) 
            loss = mse_loss 

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output 

        return CausalLMOutputWithPastLargeDistance2(
            loss=loss,
            logits = logits, 
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions, 
            attentions = addonmodeloutput.attentions, # delibrately using the model's attention mask with modifications 
            l2_distance = intermediate_l2_dist, 
            ce_loss = ce_loss.detach().clone(), 
            l2_distance_input = torch.tensor(0), 
            cossim_input = cossim_input, 
        ) 
    
    def prepare_inputs_for_generation( 
        self, input_ids, past_key_values = None, attention_mask = None, inputs_embeds = None, adjustment_scheme = None, **kwargs): 
        # mainly used to debug preparing inputs for generation and not using past_key_values 
        assert past_key_values is None, "past_key_values is not None" 
        batch_size, seq_length = input_ids.shape 
        print("batch_size {}; seq_length {}".format(batch_size, seq_length)) 
        
        # adjusting the inputs and mask 
        print("input_ids {}".format(input_ids[2])) 
        print("attention_mask {}".format(attention_mask[2])) 
        condition_mask = input_ids == self.tokenizer_bos_id 
        input_sequence_indices = torch.nonzero(condition_mask).to(input_ids.device).to(torch.long) 
        print("input_sequence_indices shape {}".format(input_sequence_indices.shape)) 
        print("input_sequence_indices: {}".format(input_sequence_indices[2])) 
        input_sequence_indices2 = [] 
        modified_input_bos_sequence_indices = [] 
        assert input_sequence_indices.shape[0] == input_ids.shape[0], "every row of sequences need to have an bos" 
        for i in range(input_ids.shape[0]): # iterate through the batch_size 
            # if input_sequence_indices[i] % self.sliding_window_length != 0: # we found a sequence that needs to be adjusted 
            if input_sequence_indices[i][1].data % self.sliding_window_length != 0: # we found a sequence that needs to be adjusted 
                input_sequence_indices2.append(torch.tensor([i, (input_sequence_indices[i][1])]).to(input_ids.device).view(1, -1)) 
                if adjustment_scheme == "case1": 
                    modified_input_bos_sequence_indices.append(torch.tensor([i, (input_sequence_indices[i][1] // self.sliding_window_length) * self.sliding_window_length]).to(input_ids.device).view(1, -1)) 
                elif adjustment_scheme == "case2": 
                    modified_input_bos_sequence_indices.append(torch.tensor([i, (input_sequence_indices[i][1] // self.sliding_window_length + 1) * self.sliding_window_length]).to(input_ids.device).view(1, -1)) 
                else: 
                    raise ValueError("adjustment_scheme is not recognized") 
        if len(input_sequence_indices2) != 0: 
            # adjusting the input_ids 
            input_sequence_indices2 = torch.cat(input_sequence_indices2, dim = 0).to(input_ids.device).to(torch.long) 
            modified_input_bos_sequence_indices = torch.cat(modified_input_bos_sequence_indices, dim = 0).to(input_ids.device).to(torch.long) 
            print("shape of modified_input_bos_sequence_indices {}".format(modified_input_bos_sequence_indices.shape)) 
            print(modified_input_bos_sequence_indices) 
            
            row_indices = input_sequence_indices2[:, 0] 
            col_indices = input_sequence_indices2[:, 1] 
            input_ids.index_put_((row_indices, col_indices), torch.full_like(row_indices, fill_value = self.tokenizer_pad_id, device = input_ids.device, dtype = input_ids.dtype), accumulate = False) 
            
            row_indices = modified_input_bos_sequence_indices[:, 0] 
            col_indices = modified_input_bos_sequence_indices[:, 1] 
            input_ids.index_put_((row_indices, col_indices), torch.full_like(row_indices, fill_value = self.tokenizer_bos_id, device = input_ids.device, dtype = input_ids.dtype), accumulate = False) 
            
            print("input_ids {}".format(input_ids[2])) 
        # just for checking 
        checking_indices = torch.nonzero(input_ids == self.tokenizer_bos_id) 
        print("positions of the start of sequence after modification: {}".format(checking_indices)) 
        for i in range(checking_indices.shape[0]): 
            assert checking_indices[i][1] % self.sliding_window_length == 0, "start of sequence is not at the right position" 
            
        # making attention_mask 
        modified_input_bos_sequence_indices = torch.nonzero(input_ids == self.tokenizer_bos_id).to(input_ids.device).to(torch.long) 
        modified_input_bos_sequence_indices = modified_input_bos_sequence_indices[:, 1].unsqueeze(1).expand(-1, seq_length) 
        print("modified_input_bos_sequence_indices shape {}".format(modified_input_bos_sequence_indices.shape)) 
        col_indices = torch.arange(seq_length).expand(batch_size, -1).to(input_ids.device) 
        attention_mask = col_indices >= modified_input_bos_sequence_indices 
        # attention_mask = input_ids != self.tokenizer_pad_id 
        attention_mask = attention_mask.to(torch.long) 
        print("attention_mask {}".format(attention_mask[2])) 
        # just for checking 
        for i in range(checking_indices.shape[0]): 
            if checking_indices[i][1] != 0: 
                assert torch.unique(attention_mask[i][: checking_indices[i][1]]) == 0, "attention_mask is not correct" 
            assert torch.unique(attention_mask[i][checking_indices[i][1] : ]) == 1, "attention_mask is not correct" 
            print(colored("checking attention_mask passed", "green")) 
                
        # past_key_values is not used and input_ids is not changed 
        '''
        position_ids = kwargs.get("position_ids", None) 
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :] 
        ''' 
        position_ids = None 
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"large_input_ids": input_ids, 
                            "small_input_ids": input_ids,} 

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

class LlamaWeirdLargeTest(LlamaPreTrainedModel): 
    """ 
    We call this autoregressive Medusa model 
    """ 
    # almost identical to LlamaWeirdLarge3, but weird fix for some model 
    _tied_weights_keys = ["lm_head.weight"]
    '''
    def __init__(self, *args, small_config, hostname, large_dim, sliding_window_length = 7, use_mse_loss = False, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.model = LlamaModel(self.config) 
        self.vocab_size = self.config.vocab_size 
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False) 
        
        # self.addonsmallmodel = addonsmallmodel 
        # self.addonsmallmodel = SimpleSmallModel(small_config, sliding_window_length = sliding_window_length, hostname = hostname, target_model_dim = large_dim) 
        self.addonsmallmodel = None 
        self.sliding_window_length = sliding_window_length 
        # self.small_model_dtype = self.addonsmallmodel.embed_projection.weight.dtype 
        self.small_model_dtype = torch.bfloat16 
        print(colored("small_model_dtype {}".format(self.small_model_dtype), "red")) 
        
        self.use_mse_loss = use_mse_loss 
        self.alpha = 0.5 

        # Initialize weights and apply final processing
        self.post_init()
    ''' 
    def __init__(self, config): 
        super().__init__(config) 
        self.model = LlamaModel(config) 
        self.vocab_size = config.vocab_size 
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False) 
        # self.addonsmallmodel = None 
        small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m") 
        # self.sliding_window_length = 7 
        # self.sliding_window_length = 2 
        # self.sliding_window_length = 1 
        self.addonsmallmodel = SimpleSmallModel(small_config, target_model_dim = self.config.hidden_size) # sliding_window_length is set elsewhere 
        self.small_model_dtype = torch.bfloat16 
        self.use_mse_loss = False 
        self.ce_loss_only = False 
        self.alpha = 0.5 
        self.inference_setting = "setting0" 
        self.use_cosinesimilarity = False 
        self.generate_iteration_count = 0 
        self.generate_model_hidden_states = torch.tensor(0) # this field saves the intermediate tensors generated by the large model 
        self.tokenizer_bos_id = 1 
        self.tokenizer_pad_id = 2 
        
        self.post_init() 
    
    def set_sliding_window_length(self, sliding_window_length): 
        self.sliding_window_length = sliding_window_length 
        self.addonmodel_start = self.sliding_window_length + 1 
        self.addonsmallmodel.set_sliding_window_length(self.sliding_window_length) 

    def get_input_embeddings(self):
        return self.model.embed_tokens 
    
    def set_msece_loss(self, use_mse_loss, ce_loss_only): 
        self.use_mse_loss = use_mse_loss 
        self.ce_loss_only = ce_loss_only 
    
    def set_cosinesimilarity(self, use_cosinesimilarity): 
        if use_cosinesimilarity: 
            self.use_cosinesimilarity = True 
    
    def resetgenerationcount(self): 
        self.generate_iteration_count = 0 
    
    def set_addonsmallmodel_statedict(self, small_state_dict_for_model): 
        new_state_dict = {} 

        for key in small_state_dict_for_model.keys(): 
            new_key = key 
            if 'lm_head' in key: 
                print("got here found the following key {}".format(key)) 
            if 'model.' in key: 
                new_key = key[6 :] 
            print(new_key) 
            new_state_dict[new_key] = small_state_dict_for_model[key] 
        # if args.embedding_pretrained: 
        #     new_state_dict["embed_projection.weight"] = torch.load("linearprojectionweighttesting.pt") 
        try: 
            self.addonsmallmodel.load_state_dict(new_state_dict) 
        except RuntimeError as r: 
            print(colored(r, "yellow")) 

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value 
    
    def set_inference_setting(self, setting = "setting0"): 
        self.inference_setting = setting 
    
    def set_slidingwindowlength(self, sliding_window_length, addonmodel_start = None): 
        self.sliding_window_length = sliding_window_length 
        if addonmodel_start is not None: 
            self.addonmodel_start = addonmodel_start 
        else: 
            self.addonmodel_start = self.sliding_window_length + 1 
    
    def set_tokenizer_bos_id(self, bos_id, pad_id): 
        self.tokenizer_bos_id = bos_id 
        self.tokenizer_pad_id = pad_id 
    
    def set_walpha(self, alpha) : 
        self.alpha = alpha 

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder
        
    def reinitialize_embeddings(self, type = "xavieruniform"): 
        from torch.nn import init 
        embedding = self.model.embed_tokens 
        if type == "xavieruniform": 
            init.xavier_uniform_(embedding.weight) 
        elif type == "xaviernormal": 
            init.xavier_normal_(embedding.weight) 
        elif type == "kaimingnormal": 
            init.kaiming_normal_(embedding.weight) 
        else: 
            raise ValueError("type not recognized") 

    def get_decoder(self):
        return self.model 
    
    def naive_grouping(self, input_ids): 
        embedding_searched = self.model.embed_tokens(input_ids) 
        # print("embedding_searched shape {} {}".format(embedding_searched.shape, embedding_searched.dtype)) 
        seq_length = embedding_searched.shape[1] 
        print("seq_length {}".format(seq_length)) 
        
        # assert seq_length % 7 == 0, "seq_length is not divisible by 7" 
        # assert seq_length % self.sliding_window_length == 0, "seq_length is not divisible by sliding_window_length" 
        # added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // 7, embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
        added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // self.sliding_window_length, embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
        # for i in range(seq_length // 7): 
        for i in range(seq_length // self.sliding_window_length): 
            sum = torch.zeros((embedding_searched.shape[0], embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
            # for j in range(7): 
            for j in range(self.sliding_window_length): 
                # sum += embedding_searched[:, i * 7 + j, :] 
                sum += embedding_searched[:, i * self.sliding_window_length + j, :] 
                # sum /= 7. 
                # print("sum dtype {}".format(sum.dtype)) 
            sum /= float(self.sliding_window_length) 
            added_tensor[:, i, :] = sum 
        # print("added_tensor shape {}".format(added_tensor.shape)) 
        
        return added_tensor 
    
    def attention_mask_upper(self, input_ids): 
        sequence_length = ((input_ids.shape[1] - 1) // self.sliding_window_length) + 1 
        batch_size = input_ids.shape[0] 
        condition_mask = input_ids == self.tokenizer_bos_id # finds the index of the start of sequence token 
        start_of_sequenceidx = torch.nonzero(condition_mask)[:, 1] 
        start_of_sequenceidx //= self.sliding_window_length 
        start_of_sequenceidx = start_of_sequenceidx.to(torch.long) 
        # modified_input_bos_sequence_indices = modified_input_bos_sequence_indices[:, 1].unsqueeze(1).expand(-1, seq_length) 
        start_of_sequenceidx2 = start_of_sequenceidx.unsqueeze(1).expand(-1, sequence_length) 
        print("start_of_sequenceidx shape {}".format(start_of_sequenceidx2.shape)) 
        col_indices = torch.arange(sequence_length).expand(batch_size, -1).to(input_ids.device) 
        attention_mask = col_indices >= start_of_sequenceidx2 
        attention_mask = attention_mask.to(torch.long) 
        return start_of_sequenceidx, attention_mask 
        
    def set_addonsmallmodel(self, addonsmallmodel): 
        self.addonsmallmodel = addonsmallmodel 
    
    def set_smallmodelfull(self): 
        self.addonsmallmodel = self.addonsmallmodel.to(torch.float32) 
    
    def l2distancecompute(self, inputs, hidden_states): 
        input_used = inputs.clone().detach()[:, 1:, :]
        hidden_states_used = hidden_states.clone().detach()[:, :-1, :] 
        assert input_used.shape == hidden_states_used.shape 
        dmod = input_used.shape[-1] 
        input_used = input_used.reshape(-1, dmod) 
        hidden_states_used = hidden_states_used.reshape(-1, dmod) 
        # compute the difference 
        diff = input_used - hidden_states_used 
        
        # compute the square 
        diff = diff ** 2
        
        # sum up the square 
        diff = torch.sum(diff, dim = 1) 
        
        # take square root 
        diff = torch.sqrt(diff) 
        
        # average the l2 distance 
        diff = torch.mean(diff) 
        
        return diff 
    
    def avgpool2(self, hidden_states): 
        seq_len = hidden_states.shape[1] # 0, 1, 2, 3, 4, 5, 6, 7 
        assert (seq_len - 1) % self.sliding_window_length == 0, "seq_len is not compatible with sliding_window_length" 
        buffer_tensor = torch.zeros((hidden_states.shape[0], seq_len // self.sliding_window_length, hidden_states.shape[2]), dtype = hidden_states.dtype).to(hidden_states.device) 
        for k in range(0, seq_len, self.sliding_window_length): # stride is fixed 
            for i in range(self.sliding_window_length): 
                sum = torch.zeros((hidden_states.shape[0], hidden_states.shape[2]), dtype = hidden_states.dtype).to(hidden_states.device) 
                sum += hidden_states[:, k + i, :] 
            sum /= self.sliding_window_length 
            buffer_tensor[:, k // self.sliding_window_length, :] = sum 
        return buffer_tensor 
    
    def avgpool3(self, hidden_states): 
        assert self.sliding_window_length == 1 # remove this line 
        downsampled_vectors = [] 
        sum = torch.zeros((hidden_states.shape[0], hidden_states.shape[2]), dtype = hidden_states.dtype).to(hidden_states.device) 
        for i in range(hidden_states.shape[1]): 
            if i % self.sliding_window_length == self.sliding_window_length - 1: 
                if i == 0: 
                    sum += hidden_states[:, i, :] 
                else: 
                    sum += hidden_states[:, i, :] 
                    sum += hidden_states[:, i - 1, :] # remove this line 
                    sum /= 2. # remove this line 
                downsampled_vectors.append(sum / self.sliding_window_length) 
                sum.mul_(0.) 
                assert sum.view(-1).sum() == 0 
            else: 
                sum += hidden_states[:, i, :] 
        # downsampled_vectors = downsampled_vectors[1 :] 
        
        return torch.stack(downsampled_vectors, dim = 1) 
    
    def avgpool(self, hidden_states): 
        downsampled_vectors = [] 
        sum = torch.zeros((hidden_states.shape[0], hidden_states.shape[2]), dtype = hidden_states.dtype).to(hidden_states.device) 
        for i in range(hidden_states.shape[1]): 
            if i % self.sliding_window_length == self.sliding_window_length - 1: 
                sum += hidden_states[:, i, :] 
                downsampled_vectors.append(sum / self.sliding_window_length) 
                sum.mul_(0.) 
                assert sum.view(-1).sum() == 0 
            else: 
                sum += hidden_states[:, i, :] 
        # downsampled_vectors = downsampled_vectors[1 :] 
        # downsampled_vectors.append(downsampled_vectors[-1]) 
        
        return torch.stack(downsampled_vectors, dim = 1) 

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        # input_ids: torch.LongTensor = None, 
        large_input_ids: torch.LongTensor = None, 
        small_input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        input_embeds: Optional[torch.FloatTensor] = None, 
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        original_attention_mask = None, 
        condensed_embed_labels = None, 
        autoregressive_first_element = False, 
        label_adjustment = False, 
    ) -> Union[Tuple, CausalLMOutputWithPastLargeDistance2]: 
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

        outputs = self.model(
            input_ids = large_input_ids, 
            attention_mask = attention_mask, 
            position_ids = position_ids, 
            past_key_values = past_key_values, 
            inputs_embeds = input_embeds, 
            use_cache = use_cache, 
            output_attentions = output_attentions, 
            output_hidden_states = output_hidden_states, 
            return_dict = return_dict, 
        ) 

        hidden_states = outputs[0] # we don't need the lm_head 
        # print("hidden_states shape {} dtype {}".format(hidden_states.shape, hidden_states.dtype)) 
        if self.small_model_dtype == torch.float32: 
            hidden_states = hidden_states.to(torch.float32) 
        elif self.small_model_dtype == torch.bfloat16: 
            hidden_states = hidden_states.to(torch.bfloat16) 
        # print(colored("small_model_type: {}".format(self.small_model_dtype), "red")) 
        # intermediate_l2_dist = self.l2distancecompute(inputs_embeds, hidden_states) 
        seq_len = hidden_states.shape[1] 
        
        
        # selected_seq_indices = [i * self.sliding_window_length for i in range(1, (seq_len - 1) // self.sliding_window_length)] 
        # print("selected_seq_indices {} total length {}".format(selected_seq_indices, len(selected_seq_indices))) 
        # hidden_states = self.avgpool(hidden_states) 
        if autoregressive_first_element: 
            # selected_seq_indices = [i * self.sliding_window_length for i in range(0, (seq_len - 1) // self.sliding_window_length)] 
            selected_seq_indices = [i * self.sliding_window_length for i in range(0, seq_len // self.sliding_window_length)] 
            print("selected_seq_indices {} total length {}".format(selected_seq_indices, len(selected_seq_indices))) 
            print("using autoregressive_baseline") 
            hidden_states = hidden_states[:, selected_seq_indices, :] 
            print("hidden_states shape {} dtype {}".format(hidden_states.shape, hidden_states.dtype)) 
            removelast = (seq_len % self.sliding_window_length == 0) 
            if removelast: 
                hidden_states = hidden_states[:, :-1, :] 
        else: 
            removelast = (hidden_states.shape[1] % self.sliding_window_length == 0) 
            hidden_states = self.avgpool(hidden_states) 
            if removelast: 
                hidden_states = hidden_states[:, :-1, :] 
                # hidden_states = hidden_states[:, :-2, :] 
        hidden_states = hidden_states[:, 1 :, :] # works with 0 as the start of the sampling index 
        print("hidden_states shape {}".format(hidden_states.shape)) 
        # hidden_states = hidden_states[:, 2 :, :] # works with 1 as the start of the sampling index 
        # hidden_states = hidden_states[:, 3:, :] 
        # print("some hidden states numbers: ", hidden_states.reshape(-1)[: 100]) 
        # hidden_states = hidden_states[:, -28 :, :] 
        
        mse_loss = torch.tensor(0) 
        cossim_loss = torch.tensor(0) 
        
        # mse_loss = 0.5 * mse_loss + 0.5 * cossim_loss 
        # intermediate_l2_dist = mse_loss.clone().detach() 
        intermediate_l2_dist = mse_loss.clone().detach() 
        if self.use_cosinesimilarity: 
            mse_loss = cossim_loss 
        cossim_input = cossim_loss.clone().detach() 
        # print(colored("mse_loss {}".format(mse_loss), "red")) 
        
        if self.use_mse_loss: 
            print(colored("mse_loss {}".format(mse_loss), "red")) 
            # still use the small model and get ce 
            hidden_states = hidden_states.detach().clone() 
            # hidden_states = torch.zeros_like(hidden_states).detach() 
            # hidden_states = condensed_embed_labels 
            '''
            return CausalLMOutputWithPastLargeDistance2(
                loss = mse_loss, 
                logits = None, 
                past_key_values = outputs.past_key_values, 
                hidden_states=outputs.hidden_states,
                attentions = outputs.attentions, 
                l2_distance = intermediate_l2_dist, 
                ce_loss = torch.tensor(0), 
            ) 
            ''' 
        # hidden_states has shape (batch_size, seq_length // 7, hidden states) 
        # hidden_states = hidden_states[:, :-1, :] 
        
        # interleave the hidden_states and the input_ids 
        # assert hidden_states.shape[1] == small_input_ids.shape[1] // 7 - 1 
        print("expected {}".format(small_input_ids.shape[1] // self.sliding_window_length - 1)) 
        print("small_input_ids: {}".format(small_input_ids[0])) 
        print("self.addonmodel_start {}".format(self.addonmodel_start)) 
        print("sliding_window_length {}".format(self.sliding_window_length)) 
        print("hidden_states.shape[1] {}".format(hidden_states.shape[1])) 
        assert hidden_states.shape[1] == (small_input_ids.shape[1] - self.addonmodel_start) // self.sliding_window_length  # please add back 
        # print("condensed_embed_labels shape {} dtype {}".format(condensed_embed_labels.shape, condensed_embed_labels.dtype) if condensed_embed_labels is not None else "condensed_embed_labels is None") 
        addonmodeloutput = self.addonsmallmodel( 
            # input_ids = input_ids, 
            input_ids = small_input_ids, 
            attention_mask = original_attention_mask, 
            # position_ids = None, 
            past_key_values = None, 
            condensed_embeds = hidden_states, 
            # condensed_embeds = condensed_embed_labels, 
            labels = None, 
            # labels = labels, 
            # use_cache = None, 
            output_attentions = True, 
            output_hidden_states = None, 
            return_dict = True, 
            start_idx = self.addonmodel_start, # NOTE this is very important 
            eval_mode = False, 
            iteration_count = None, 
            # condensed_fashion = "projection_mode", 
            # experiment_setting = "setting3", 
            experiment_setting = self.inference_setting, 
        ) 
        
        logits = addonmodeloutput.logits 
        # loss = addonmodeloutput["loss"] 
        # logits = addonmodeloutput["logits"] 
        # ce_loss = loss 
        
        '''
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        ''' 
        # seq_length = input_ids.shape[1] + hidden_states.shape[1] 
        seq_length = small_input_ids.shape[1] + hidden_states.shape[1] 
        assert seq_length == logits.shape[1], "seq_length is not compatible to logits" 
        # mask_list_pos = [i * (self.sliding_window_length + 1) for i in range(seq_length // (self.sliding_window_length + 1))] 
        # mask_list_pos = [7 + i * (self.sliding_window_length + 1) for i in range((seq_length - 7) // (self.sliding_window_length + 1))] 
        mask_list_pos = [self.addonmodel_start + i * (self.sliding_window_length + 1) for i in range((seq_length - self.addonmodel_start) // (self.sliding_window_length + 1))] 
        mask_list_pos22 = [x - 1 for x in mask_list_pos] 
        print("mask list pos22: {}".format(mask_list_pos22)) 
        print("length of mask list pos22: {}".format(len(mask_list_pos22))) 
        # print(colored("mask_list_pos {}".format(mask_list_pos), "red")) 
        loss = None 
        
        if label_adjustment: # we adjust the labels to be completely information loss free 
            print("got inside") 
            copy_idx = [self.addonmodel_start + (self.sliding_window_length * i) for i in range(hidden_states.shape[1])] 
            labels_addition = labels[:, copy_idx] 
            newlabels = labels[:, : self.addonmodel_start] 
            old_label_count = self.addonmodel_start 
            for i in range(labels_addition.shape[1]): 
                newlabels = torch.cat([newlabels, labels_addition[:, i].unsqueeze(1)], dim = 1) 
                if old_label_count < labels.shape[1]: 
                    newlabels = torch.cat([newlabels, labels[:, old_label_count : min(old_label_count + self.sliding_window_length, labels.shape[1])]], dim = 1) 
                old_label_count += self.sliding_window_length 
            assert newlabels.shape[1] == seq_length 
            
            # some visual check, printing index and values together 
            # for i in range(newlabelsone.shape[0]): 
                # if i < labels.shape[0]: 
                    # print("index {} labels value {} newlabels value {}".format(i, labels[0, i], newlabels[0, i])) 
                # else: 
                    # print("index {} labels values {} new labels value {}".format(i, "None", newlabels[0, i])) 
            labels = newlabels 
        
        if labels is not None: 
            # selected_indices = list(range(7)) 
            # selected_indices = list(range(self.addonmodel_start)) 
            # for i in range(7, seq_length): 
                # if i not in mask_list_pos: 
                    # selected_indices.append(i) 
            # for i in range(self.addonmodel_start, seq_length): 
            #     if i not in mask_list_pos: 
            #         selected_indices.append(i) 
            selected_indices = list(range(self.addonmodel_start - 1)) 
            for i in range(self.addonmodel_start - 1, seq_length): 
                if i not in mask_list_pos22: 
                    selected_indices.append(i) 
            # selected_indices = mask_list_pos22 
            # print(colored("selected_indices {}".format(selected_indices), "red")) 
            # select and shift the logits 
            logits = logits[:, selected_indices, :] 
            shift_logits = logits[..., :-1, :].contiguous() 
            shift_labels = labels[..., 1:].contiguous() # shape (batch_size, seq_length - 1) 
            # shift_labels = labels[..., 1:-1].contiguous() # shape (batch_size, seq_length - 1) 
            print("shift_logits shape {}; shift_labels shape {}".format(shift_logits.shape, shift_labels.shape)) 
            # Flatten the tokens 
            loss_fct = CrossEntropyLoss() 
            # shift_logits2 = shift_logits.clone().detach() # used for position investigation 
            # shift_labels2 = shift_labels.clone().detach() # used for position investigation 
            shift_logits = shift_logits.view(-1, self.config.vocab_size) 
            shift_labels = shift_labels.view(-1) 
            
            # position loss performance investigation below 
            # num_chunks = (shift_logits2.shape[1] - 1) // (self.sliding_window_length + 1) 
            # first_pos_indices = [self.addonmodel_start - 1 + (self.sliding_window_length + 1) * i for i in range(num_chunks)] 
            # first_pos_ce_loss = loss_fct(shift_logits2[:, first_pos_indices, :].view(-1, self.config.vocab_size), shift_labels2[:, first_pos_indices].view(-1)) 
            # second_pos_indices = [self.addonmodel_start + (self.sliding_window_length + 1) * i for i in range(num_chunks)] 
            # second_pos_ce_loss = loss_fct(shift_logits2[:, second_pos_indices, :].view(-1, self.config.vocab_size), shift_labels2[:, second_pos_indices].view(-1)) 
            first_pos_ce_loss = torch.tensor(0) 
            second_pos_ce_loss = torch.tensor(0) 
            
            # Enable model parallelism 
            shift_labels = shift_labels.to(shift_logits.device) 
            ce_loss = loss_fct(shift_logits, shift_labels) 
            loss = ce_loss 
            # print(colored("rank {} loss {}".format(self.accelerator.state.process_index, loss), "yellow")) 
        if loss is not None and not self.use_mse_loss: 
            if self.ce_loss_only: 
                print(colored("ce_loss only", "red")) 
                loss = ce_loss 
            else: 
                print(colored("ce_loss + mse_loss", "red")) 
                # loss = self.alpha * loss + (1 - self.alpha) * mse_loss 
                loss = self.alpha * ce_loss + (1 - self.alpha) * mse_loss 
        else: 
            print(colored("mse_loss only", "red")) 
            loss = mse_loss 

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output 

        return CausalLMOutputWithPastLargeDistance2(
            loss=loss,
            first_pos_loss = first_pos_ce_loss, 
            second_pos_loss = second_pos_ce_loss, 
            logits = logits, 
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions, 
            attentions = addonmodeloutput.attentions, # delibrately using the model's attention mask with modifications 
            l2_distance = intermediate_l2_dist, 
            ce_loss = ce_loss.detach().clone(), 
            l2_distance_input = torch.tensor(0), 
            cossim_input = cossim_input, 
        ) 
    
    def top_k_top_p_filter(self, logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0): 
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
    
    def norm_logits(self, logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor: 
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
        logits = self.top_k_top_p_filter(logits, top_k=top_k, top_p=top_p) 
        probs = F.softmax(logits, dim=1)
        return probs 
    
    def sample2(self, probs : torch.Tensor, num_samples: int = 1, random_seed = None): 
        if random_seed:
            torch.manual_seed(random_seed)
        idx_next = torch.multinomial(probs, num_samples=num_samples)
        # if (idx_next.item() == 0):
            # raise RuntimeError 
        return idx_next 
    
    def forward_generate(
        self,
        # input_ids: torch.LongTensor = None, 
        large_input_ids: torch.LongTensor = None, 
        small_input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        input_embeds: Optional[torch.FloatTensor] = None, 
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        original_attention_mask = None, 
        condensed_embed_labels = None, 
        autoregressive_first_element = False, 
        output_large_model_last_hidden_states = False, 
        inmiddlesample = False, 
        target_lmhead = None, 
        temperature = 0.6, 
        top_k = -1, 
        top_p = 0.9, 
    ) -> Union[Tuple, CausalLMOutputWithPastLargeDistance2]: 
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

        last_hidden_states = None 
        start = None 
        if self.generate_iteration_count % self.sliding_window_length == 0: 
            # NOTE for this case, we use the pass-in attention mask 
            # print(colored("running the large model side", "green")) 
            outputs = self.model(
                input_ids = large_input_ids, 
                attention_mask = attention_mask, 
                position_ids = position_ids, 
                past_key_values = past_key_values, 
                inputs_embeds = input_embeds, 
                use_cache = use_cache, 
                output_attentions = output_attentions, 
                output_hidden_states = output_hidden_states, 
                return_dict = return_dict, 
            ) 
            
            hidden_states = outputs[0] # we don't need the lm_head 
            if output_large_model_last_hidden_states: 
                last_hidden_states = hidden_states.clone().detach() 
            if inmiddlesample: 
                outputdist = target_lmhead(hidden_states) 
                probs = self.norm_logits(outputdist[:, -1, :], temperature = temperature, top_k = top_k, top_p = top_p) 
                start = self.sample2(probs) 
                if len(start.shape) == 1: 
                    start = start.unsqueeze(0) 
                small_input_ids = torch.cat([small_input_ids, start], dim = 1) 
                original_attention_mask = torch.cat([original_attention_mask, torch.ones_like(start)], dim = 1) 
            seq_len = hidden_states.shape[1] 
            
            if self.small_model_dtype == torch.float32: 
                hidden_states = hidden_states.to(torch.float32) 
            elif self.small_model_dtype == torch.bfloat16: 
                hidden_states = hidden_states.to(torch.bfloat16) 
            if self.sliding_window_length != 1: 
                selected_seq_indices = [i * self.sliding_window_length for i in range(0, math.ceil(seq_len / self.sliding_window_length))]  # adding the last one to get future tokens 
            else: 
                selected_seq_indices = [i * self.sliding_window_length for i in range(0, seq_len // self.sliding_window_length)] 
            # print("selected_seq_indices {} total length {}".format(selected_seq_indices, len(selected_seq_indices))) 
            
            hidden_states = hidden_states[:, selected_seq_indices, :] 
            hidden_states = hidden_states[:, 1 :, :] # works with 0 as the start of the sampling index 
            self.generate_model_hidden_states = hidden_states.clone().detach() 
        self.generate_iteration_count += 1 
        
        # print(colored("running the small model side", "green")) 
        addonmodeloutput = self.addonsmallmodel.generate_forward(
            input_ids = small_input_ids, 
            attention_mask = original_attention_mask, 
            position_ids = None, 
            past_key_values = None, 
            condensed_embeds = self.generate_model_hidden_states, 
            labels = None, 
            use_cache = None, 
            output_attentions = True, 
            output_hidden_states = None, 
            return_dict = True, 
            start_idx = self.sliding_window_length + 1, # NOTE this is very important 
            eval_mode = False, 
            experiment_setting = self.inference_setting, 
            generate_flag = True, 
        ) 
        
        logits = addonmodeloutput.logits 
        # logits = torch.zeros((small_input_ids.shape[0], small_input_ids.shape[1], self.config.vocab_size)).to(small_input_ids.device).to(torch.float32) 
        
        loss = None 
        
        if not return_dict: 
            output = (logits,) + outputs[1:] 
            return (loss,) + output if loss is not None else output 
        
        return CausalLMOutputWithPastLargeDistance3(
            loss = loss, 
            logits = logits, 
            past_key_values = past_key_values, 
            hidden_states = None, 
            attentions = None, 
            l2_distance = None, 
            ce_loss = None, 
            l2_distance_input = None, 
            cossim_input = None, 
            last_hidden_states = last_hidden_states, 
            start = start if inmiddlesample else None, 
        ) 
    
    def sample( 
        self, 
        input_ids: torch.LongTensor, 
        logits_processor: Optional[LogitsProcessorList] = None, 
        stopping_criteria: Optional[StoppingCriteriaList] = None, 
        logits_warper: Optional[LogitsProcessorList] = None, 
        max_length: Optional[int] = None, 
        pad_token_id: Optional[int] = None, 
        eos_token_id: Optional[int] = None, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None, 
        output_scores: Optional[bool] = None, 
        return_dict_in_generate: Optional[bool] = None, 
        synced_gpus: bool = False, 
        streamer: Optional["BaseStreamer"] = None, 
        **model_kwargs, 
    ) -> Union[SampleOutput, torch.LongTensor]: 
        
        print("inside generate function, output_hidden_states is {}".format(output_hidden_states)) 
        print(colored("inside the function that is overloaded for the model", "yellow")) 
        
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList() 
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList() 
        if max_length is not None: 
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            ) 
            stopping_criteria = StoppingCriteriaList(stopping_criteria, max_length) 
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList() 
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )
        
        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        
        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )
        
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device) 
        
        this_peer_finished = False # used by synced_gpus only 
        
        self.generate_iteration_count = 0 
        # auto-regressive generation 
        while True: 
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, adjustment_scheme = "case1", **model_kwargs) 

            # forward pass to get next token
            # outputs = self(
                # **model_inputs,
                # return_dict=True,
                # output_attentions=output_attentions,
                # output_hidden_states=output_hidden_states,
            # ) 
            outputs = self.forward_generate(
                **model_inputs, 
                return_dict = True, 
                output_attentions = output_attentions, 
                output_hidden_states = output_hidden_states, 
            ) 

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids 
    
    def prepare_inputs_for_generation2( 
        self, input_ids, past_key_values = None, attention_mask = None, inputs_embeds = None, adjustment_scheme = None, **kwargs): 
        # mainly used to debug preparing inputs for generation and not using past_key_values 
        assert past_key_values is None, "past_key_values is not None" 
        batch_size, seq_length = input_ids.shape 
        print("batch_size {}; seq_length {}".format(batch_size, seq_length)) 
        
        # adjusting the inputs and mask 
        print("input_ids {}".format(input_ids[2])) 
        print("attention_mask {}".format(attention_mask[2])) 
        condition_mask = input_ids == self.tokenizer_bos_id 
        input_sequence_indices = torch.nonzero(condition_mask).to(input_ids.device).to(torch.long) 
        print("input_sequence_indices shape {}".format(input_sequence_indices.shape)) 
        print("input_sequence_indices: {}".format(input_sequence_indices[2])) 
        input_sequence_indices2 = [] 
        modified_input_bos_sequence_indices = [] 
        assert input_sequence_indices.shape[0] == input_ids.shape[0], "every row of sequences need to have an bos" 
        for i in range(input_ids.shape[0]): # iterate through the batch_size 
            # if input_sequence_indices[i] % self.sliding_window_length != 0: # we found a sequence that needs to be adjusted 
            if input_sequence_indices[i][1].data % self.sliding_window_length != 0: # we found a sequence that needs to be adjusted 
                input_sequence_indices2.append(torch.tensor([i, (input_sequence_indices[i][1])]).to(input_ids.device).view(1, -1)) 
                if adjustment_scheme == "case1": 
                    modified_input_bos_sequence_indices.append(torch.tensor([i, (input_sequence_indices[i][1] // self.sliding_window_length) * self.sliding_window_length]).to(input_ids.device).view(1, -1)) 
                elif adjustment_scheme == "case2": 
                    modified_input_bos_sequence_indices.append(torch.tensor([i, (input_sequence_indices[i][1] // self.sliding_window_length + 1) * self.sliding_window_length]).to(input_ids.device).view(1, -1)) 
                else: 
                    raise ValueError("adjustment_scheme is not recognized") 
        if len(input_sequence_indices2) != 0: 
            # adjusting the input_ids 
            input_sequence_indices2 = torch.cat(input_sequence_indices2, dim = 0).to(input_ids.device).to(torch.long) 
            modified_input_bos_sequence_indices = torch.cat(modified_input_bos_sequence_indices, dim = 0).to(input_ids.device).to(torch.long) 
            print("shape of modified_input_bos_sequence_indices {}".format(modified_input_bos_sequence_indices.shape)) 
            print(modified_input_bos_sequence_indices) 
            
            row_indices = input_sequence_indices2[:, 0] 
            col_indices = input_sequence_indices2[:, 1] 
            input_ids.index_put_((row_indices, col_indices), torch.full_like(row_indices, fill_value = self.tokenizer_pad_id, device = input_ids.device, dtype = input_ids.dtype), accumulate = False) 
            
            row_indices = modified_input_bos_sequence_indices[:, 0] 
            col_indices = modified_input_bos_sequence_indices[:, 1] 
            input_ids.index_put_((row_indices, col_indices), torch.full_like(row_indices, fill_value = self.tokenizer_bos_id, device = input_ids.device, dtype = input_ids.dtype), accumulate = False) 
            
            print("input_ids {}".format(input_ids[2])) 
        # just for checking 
        checking_indices = torch.nonzero(input_ids == self.tokenizer_bos_id) 
        print("positions of the start of sequence after modification: {}".format(checking_indices)) 
        for i in range(checking_indices.shape[0]): 
            assert checking_indices[i][1] % self.sliding_window_length == 0, "start of sequence is not at the right position" 
            
        # making attention_mask 
        modified_input_bos_sequence_indices = torch.nonzero(input_ids == self.tokenizer_bos_id).to(input_ids.device).to(torch.long) 
        modified_input_bos_sequence_indices = modified_input_bos_sequence_indices[:, 1].unsqueeze(1).expand(-1, seq_length) 
        print("modified_input_bos_sequence_indices shape {}".format(modified_input_bos_sequence_indices.shape)) 
        col_indices = torch.arange(seq_length).expand(batch_size, -1).to(input_ids.device) 
        attention_mask = col_indices >= modified_input_bos_sequence_indices 
        # attention_mask = input_ids != self.tokenizer_pad_id 
        attention_mask = attention_mask.to(torch.long) 
        print("attention_mask {}".format(attention_mask[2])) 
        # just for checking 
        for i in range(checking_indices.shape[0]): 
            if checking_indices[i][1] != 0: 
                assert torch.unique(attention_mask[i][: checking_indices[i][1]]) == 0, "attention_mask is not correct" 
            assert torch.unique(attention_mask[i][checking_indices[i][1] : ]) == 1, "attention_mask is not correct" 
            print(colored("checking attention_mask passed", "green")) 
                
        # past_key_values is not used and input_ids is not changed 
        '''
        position_ids = kwargs.get("position_ids", None) 
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :] 
        ''' 
        position_ids = None 
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"large_input_ids": input_ids, 
                            "small_input_ids": input_ids,} 

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs 
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values = None, attention_mask = None, inputs_embeds = None, **kwargs
    ): 
        assert past_key_values is None, "past_key_values is not None" 
        
        position_ids = None 
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"large_input_ids": input_ids, 
                            "small_input_ids": input_ids,} 
        
        # print("attention_mask", attention_mask.shape) 
        # print("input_ids", input_ids.shape) 
        if attention_mask.shape[1] == input_ids.shape[1] - 1: 
            torch.cat([attention_mask, torch.ones(attention_mask.shape[0], 1, device = attention_mask.device)], dim = 1) 
        assert attention_mask.shape[1] == input_ids.shape[1], "attention_mask is not compatible with input_ids" 
        original_attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.shape[0], (input_ids.shape[1] - self.addonmodel_start)//self.sliding_window_length + 1, device = attention_mask.device)], dim = 1) 
        
        model_inputs.update( 
            { 
                "position_ids": position_ids, 
                "past_key_values": past_key_values, 
                "use_cache": kwargs.get("use_cache"), 
                "attention_mask": attention_mask, 
                "original_attention_mask": original_attention_mask, 
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

class LargeModelLMHeadModel(nn.Module): 
    def __init__(self, original_layer): 
        super(LargeModelLMHeadModel, self).__init__() 
        self.target_lm_head = original_layer 
    
    def forward(self, x): 
        return self.target_lm_head(x) 

class LlamaWeirdLargeTestmixedb(LlamaPreTrainedModel): 
    """ 
    We call this autoregressive Medusa model 
    """ 
    # almost identical to LlamaWeirdLarge3, but weird fix for some model 
    _tied_weights_keys = ["lm_head.weight"]
    '''
    def __init__(self, *args, small_config, hostname, large_dim, sliding_window_length = 7, use_mse_loss = False, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.model = LlamaModel(self.config) 
        self.vocab_size = self.config.vocab_size 
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False) 
        
        # self.addonsmallmodel = addonsmallmodel 
        # self.addonsmallmodel = SimpleSmallModel(small_config, sliding_window_length = sliding_window_length, hostname = hostname, target_model_dim = large_dim) 
        self.addonsmallmodel = None 
        self.sliding_window_length = sliding_window_length 
        # self.small_model_dtype = self.addonsmallmodel.embed_projection.weight.dtype 
        self.small_model_dtype = torch.bfloat16 
        print(colored("small_model_dtype {}".format(self.small_model_dtype), "red")) 
        
        self.use_mse_loss = use_mse_loss 
        self.alpha = 0.5 

        # Initialize weights and apply final processing
        self.post_init()
    ''' 
    def __init__(self, config): 
        super().__init__(config) 
        self.model = LlamaModel(config) 
        self.vocab_size = config.vocab_size 
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False) 
        # self.addonsmallmodel = None 
        small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m") 
        # self.sliding_window_length = 7 
        # self.sliding_window_length = 2 
        # self.sliding_window_length = 1 
        self.addonsmallmodel = SimpleSmallModelmixedb(small_config, target_model_dim = self.config.hidden_size) # sliding_window_length is set elsewhere 
        self.small_model_dtype = torch.bfloat16 
        self.use_mse_loss = False 
        self.ce_loss_only = False 
        self.alpha = 0.5 
        self.inference_setting = "setting0" 
        self.use_cosinesimilarity = False 
        self.generate_iteration_count = 0 
        self.generate_model_hidden_states = torch.tensor(0) # this field saves the intermediate tensors generated by the large model 
        self.tokenizer_bos_id = 1 
        self.tokenizer_pad_id = 2 
        
        self.post_init() 
    
    def set_sliding_window_length(self, sliding_window_length): 
        self.sliding_window_length = sliding_window_length 
        self.addonmodel_start = self.sliding_window_length + 1 
        self.addonsmallmodel.set_sliding_window_length(self.sliding_window_length) 

    def get_input_embeddings(self):
        return self.model.embed_tokens 
    
    def set_msece_loss(self, use_mse_loss, ce_loss_only): 
        self.use_mse_loss = use_mse_loss 
        self.ce_loss_only = ce_loss_only 
    
    def set_cosinesimilarity(self, use_cosinesimilarity): 
        if use_cosinesimilarity: 
            self.use_cosinesimilarity = True 
    
    def resetgenerationcount(self): 
        self.generate_iteration_count = 0 
    
    def set_addonsmallmodel_statedict(self, small_state_dict_for_model): 
        new_state_dict = {} 

        for key in small_state_dict_for_model.keys(): 
            new_key = key 
            if 'lm_head' in key: 
                print("got here found the following key {}".format(key)) 
            if 'model.' in key: 
                new_key = key[6 :] 
            print(new_key) 
            new_state_dict[new_key] = small_state_dict_for_model[key] 
        # if args.embedding_pretrained: 
        #     new_state_dict["embed_projection.weight"] = torch.load("linearprojectionweighttesting.pt") 
        try: 
            self.addonsmallmodel.load_state_dict(new_state_dict) 
        except RuntimeError as r: 
            print(colored(r, "yellow")) 

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value 
    
    def set_inference_setting(self, setting = "setting0"): 
        self.inference_setting = setting 
    
    def set_slidingwindowlength(self, sliding_window_length, addonmodel_start = None): 
        self.sliding_window_length = sliding_window_length 
        if addonmodel_start is not None: 
            self.addonmodel_start = addonmodel_start 
        else: 
            self.addonmodel_start = self.sliding_window_length + 1 
    
    def set_tokenizer_bos_id(self, bos_id, pad_id): 
        self.tokenizer_bos_id = bos_id 
        self.tokenizer_pad_id = pad_id 
    
    def set_walpha(self, alpha) : 
        self.alpha = alpha 

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder
        
    def reinitialize_embeddings(self, type = "xavieruniform"): 
        from torch.nn import init 
        embedding = self.model.embed_tokens 
        if type == "xavieruniform": 
            init.xavier_uniform_(embedding.weight) 
        elif type == "xaviernormal": 
            init.xavier_normal_(embedding.weight) 
        elif type == "kaimingnormal": 
            init.kaiming_normal_(embedding.weight) 
        else: 
            raise ValueError("type not recognized") 

    def get_decoder(self):
        return self.model 
    
    def naive_grouping(self, input_ids): 
        embedding_searched = self.model.embed_tokens(input_ids) 
        # print("embedding_searched shape {} {}".format(embedding_searched.shape, embedding_searched.dtype)) 
        seq_length = embedding_searched.shape[1] 
        print("seq_length {}".format(seq_length)) 
        
        # assert seq_length % 7 == 0, "seq_length is not divisible by 7" 
        # assert seq_length % self.sliding_window_length == 0, "seq_length is not divisible by sliding_window_length" 
        # added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // 7, embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
        added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // self.sliding_window_length, embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
        # for i in range(seq_length // 7): 
        for i in range(seq_length // self.sliding_window_length): 
            sum = torch.zeros((embedding_searched.shape[0], embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
            # for j in range(7): 
            for j in range(self.sliding_window_length): 
                # sum += embedding_searched[:, i * 7 + j, :] 
                sum += embedding_searched[:, i * self.sliding_window_length + j, :] 
                # sum /= 7. 
                # print("sum dtype {}".format(sum.dtype)) 
            sum /= float(self.sliding_window_length) 
            added_tensor[:, i, :] = sum 
        # print("added_tensor shape {}".format(added_tensor.shape)) 
        
        return added_tensor 
    
    def attention_mask_upper(self, input_ids): 
        sequence_length = ((input_ids.shape[1] - 1) // self.sliding_window_length) + 1 
        batch_size = input_ids.shape[0] 
        condition_mask = input_ids == self.tokenizer_bos_id # finds the index of the start of sequence token 
        start_of_sequenceidx = torch.nonzero(condition_mask)[:, 1] 
        start_of_sequenceidx //= self.sliding_window_length 
        start_of_sequenceidx = start_of_sequenceidx.to(torch.long) 
        # modified_input_bos_sequence_indices = modified_input_bos_sequence_indices[:, 1].unsqueeze(1).expand(-1, seq_length) 
        start_of_sequenceidx2 = start_of_sequenceidx.unsqueeze(1).expand(-1, sequence_length) 
        print("start_of_sequenceidx shape {}".format(start_of_sequenceidx2.shape)) 
        col_indices = torch.arange(sequence_length).expand(batch_size, -1).to(input_ids.device) 
        attention_mask = col_indices >= start_of_sequenceidx2 
        attention_mask = attention_mask.to(torch.long) 
        return start_of_sequenceidx, attention_mask 
        
    def set_addonsmallmodel(self, addonsmallmodel): 
        self.addonsmallmodel = addonsmallmodel 
    
    def set_smallmodelfull(self): 
        self.addonsmallmodel = self.addonsmallmodel.to(torch.float32) 
    
    def l2distancecompute(self, inputs, hidden_states): 
        input_used = inputs.clone().detach()[:, 1:, :]
        hidden_states_used = hidden_states.clone().detach()[:, :-1, :] 
        assert input_used.shape == hidden_states_used.shape 
        dmod = input_used.shape[-1] 
        input_used = input_used.reshape(-1, dmod) 
        hidden_states_used = hidden_states_used.reshape(-1, dmod) 
        # compute the difference 
        diff = input_used - hidden_states_used 
        
        # compute the square 
        diff = diff ** 2
        
        # sum up the square 
        diff = torch.sum(diff, dim = 1) 
        
        # take square root 
        diff = torch.sqrt(diff) 
        
        # average the l2 distance 
        diff = torch.mean(diff) 
        
        return diff 
    
    def avgpool2(self, hidden_states): 
        seq_len = hidden_states.shape[1] # 0, 1, 2, 3, 4, 5, 6, 7 
        assert (seq_len - 1) % self.sliding_window_length == 0, "seq_len is not compatible with sliding_window_length" 
        buffer_tensor = torch.zeros((hidden_states.shape[0], seq_len // self.sliding_window_length, hidden_states.shape[2]), dtype = hidden_states.dtype).to(hidden_states.device) 
        for k in range(0, seq_len, self.sliding_window_length): # stride is fixed 
            for i in range(self.sliding_window_length): 
                sum = torch.zeros((hidden_states.shape[0], hidden_states.shape[2]), dtype = hidden_states.dtype).to(hidden_states.device) 
                sum += hidden_states[:, k + i, :] 
            sum /= self.sliding_window_length 
            buffer_tensor[:, k // self.sliding_window_length, :] = sum 
        return buffer_tensor 
    
    def avgpool3(self, hidden_states): 
        assert self.sliding_window_length == 1 # remove this line 
        downsampled_vectors = [] 
        sum = torch.zeros((hidden_states.shape[0], hidden_states.shape[2]), dtype = hidden_states.dtype).to(hidden_states.device) 
        for i in range(hidden_states.shape[1]): 
            if i % self.sliding_window_length == self.sliding_window_length - 1: 
                if i == 0: 
                    sum += hidden_states[:, i, :] 
                else: 
                    sum += hidden_states[:, i, :] 
                    sum += hidden_states[:, i - 1, :] # remove this line 
                    sum /= 2. # remove this line 
                downsampled_vectors.append(sum / self.sliding_window_length) 
                sum.mul_(0.) 
                assert sum.view(-1).sum() == 0 
            else: 
                sum += hidden_states[:, i, :] 
        # downsampled_vectors = downsampled_vectors[1 :] 
        
        return torch.stack(downsampled_vectors, dim = 1) 
    
    def avgpool(self, hidden_states): 
        downsampled_vectors = [] 
        sum = torch.zeros((hidden_states.shape[0], hidden_states.shape[2]), dtype = hidden_states.dtype).to(hidden_states.device) 
        for i in range(hidden_states.shape[1]): 
            if i % self.sliding_window_length == self.sliding_window_length - 1: 
                sum += hidden_states[:, i, :] 
                downsampled_vectors.append(sum / self.sliding_window_length) 
                sum.mul_(0.) 
                assert sum.view(-1).sum() == 0 
            else: 
                sum += hidden_states[:, i, :] 
        # downsampled_vectors = downsampled_vectors[1 :] 
        # downsampled_vectors.append(downsampled_vectors[-1]) 
        
        return torch.stack(downsampled_vectors, dim = 1) 

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        # input_ids: torch.LongTensor = None, 
        large_input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        input_embeds: Optional[torch.FloatTensor] = None, 
        # labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        original_attention_mask = None, 
        condensed_embed_labels = None, 
        autoregressive_first_element = False, 
        label_adjustment = False, 
        first_n_rows = None, 
    ) -> Union[Tuple, CausalLMOutputWithPastLargeDistance2]: 
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

        outputs = self.model(
            input_ids = large_input_ids, 
            attention_mask = attention_mask, 
            position_ids = position_ids, 
            past_key_values = past_key_values, 
            inputs_embeds = input_embeds, 
            use_cache = use_cache, 
            output_attentions = output_attentions, 
            output_hidden_states = output_hidden_states, 
            return_dict = return_dict, 
        ) 

        hidden_states = outputs[0] # we don't need the lm_head 
        # print("hidden_states shape {} dtype {}".format(hidden_states.shape, hidden_states.dtype)) 
        if self.small_model_dtype == torch.float32: 
            hidden_states = hidden_states.to(torch.float32) 
        elif self.small_model_dtype == torch.bfloat16: 
            hidden_states = hidden_states.to(torch.bfloat16) 
        seq_len = hidden_states.shape[1] 
        # catenhidden = torch.zeros((hidden_states.shape[0] * self.sliding_window_length, hidden_states.shape[1], hidden_states.shape[2])).to(hidden_states.device).to(hidden_states.dtype) 
        catenhidden = None 
        
        for j in range(self.sliding_window_length if first_n_rows is None else first_n_rows):
        # for j in range(6): 
            # selected_seq_indices = [i * self.sliding_window_length for i in range(0, seq_len // self.sliding_window_length)] 
            selected_seq_indices = [i * self.sliding_window_length + j for i in range(0, seq_len // self.sliding_window_length)] 
            print("selected_seq_indices {} total length {}".format(selected_seq_indices, len(selected_seq_indices))) 
            selected_hidden = hidden_states[:, selected_seq_indices, :][:, 1 :, :].clone() 
            # print("hidden_states shape {} dtype {}".format(hidden_states.shape, hidden_states.dtype)) 
            print("hidden_states shape {} dtype {}".format(selected_hidden.shape, selected_hidden.dtype)) 
            if catenhidden is None: 
                catenhidden = selected_hidden 
            else: 
                catenhidden = torch.cat([catenhidden, selected_hidden], dim = 0) # we modify the batch_size dimension 
        
        # hidden_states = hidden_states[:, 1 :, :] # works with 0 as the start of the sampling index 
        # print("hidden_states shape {}".format(hidden_states.shape)) 
        print("catenhidden shape {}".format(catenhidden.shape)) 
        
        small_input_ids = None 
        practical_attention_mask = None 
        # making small_input_ids 
        # together with the small attention mask 
        for j in range(self.sliding_window_length if first_n_rows is None else first_n_rows): 
        # for j in range(6): # debugging using the setting we are familiar with 
            nummagic = self.sliding_window_length - 1 - j # from 6 to 0 inclusive 
            if nummagic != 0: 
                stageinputids = torch.cat([torch.full((large_input_ids.shape[0], nummagic), self.tokenizer_pad_id, dtype = large_input_ids.dtype).to(large_input_ids.device), large_input_ids[:, : -nummagic].clone()], dim = 1) # sequence length dimension and unchanged 
                stageattentionmask = torch.cat([torch.zeros((large_input_ids.shape[0], nummagic), dtype = attention_mask.dtype).to(attention_mask.device), original_attention_mask[:, : -nummagic].clone()], dim = 1) 
            else: 
                stageinputids = large_input_ids.clone() 
                stageattentionmask = original_attention_mask.clone() 
            if small_input_ids is None: 
                small_input_ids = stageinputids 
                practical_attention_mask = stageattentionmask 
            else: 
                small_input_ids = torch.cat([small_input_ids, stageinputids], dim = 0) 
                practical_attention_mask = torch.cat([practical_attention_mask, stageattentionmask], dim = 0) 
                
        # making label 
        labels = small_input_ids.clone() 
        labels[labels == self.tokenizer_pad_id] = -100 
        
        mse_loss = torch.tensor(0) 
        cossim_loss = torch.tensor(0) 
        
        intermediate_l2_dist = mse_loss.clone().detach() 
        if self.use_cosinesimilarity: 
            mse_loss = cossim_loss 
        cossim_input = cossim_loss.clone().detach() 
        # print(colored("mse_loss {}".format(mse_loss), "red")) 
        
        print("expected {}".format(small_input_ids.shape[1] // self.sliding_window_length - 1)) 
        print("small_input_ids[0] {}".format(small_input_ids[0])) 
        # print("small_input_ids[large_input_ids.shape[0]] {}".format(small_input_ids[large_input_ids.shape[0]])) 
        # print("small_input_ids[2 * large_input_ids.shape[0]] {}".format(small_input_ids[2 * large_input_ids.shape[0]])) 
        # print("practical_attention_mask[0] {}".format(practical_attention_mask[0])) 
        # print("practical_attention_mask[large_input_ids.shape[0]] {}".format(practical_attention_mask[large_input_ids.shape[0]])) 
        # print("practical_attention_mask[2 * large_input_ids.shape[0]] {}".format(practical_attention_mask[2 * large_input_ids.shape[0]])) 
        # print("labels[0] {}".format(labels[0])) 
        # print("labels[large_input_ids.shape[0]] {}".format(labels[large_input_ids.shape[0]]) if large_input_ids.shape[0] < labels.shape[0] else "labels[large_input_ids.shape[0]] is None") 
        # print("labels[2 * large_input_ids.shape[0]] {}".format(labels[2 * large_input_ids.shape[0]]) if 2 * large_input_ids.shape[0] < labels.shape[0] else "labels[2 * large_input_ids.shape[0]] is None") 
        print("self.addonmodel_start {}".format(self.addonmodel_start)) 
        print("sliding_window_length {}".format(self.sliding_window_length)) 
        # print("hidden_states.shape[1] {}".format(hidden_states.shape[1])) 
        # assert hidden_states.shape[1] == (small_input_ids.shape[1] - self.addonmodel_start) // self.sliding_window_length  # please add back 
        print("catenhidden.shape[1] {}".format(catenhidden.shape[1])) 
        assert catenhidden.shape[1] == (small_input_ids.shape[1] - self.addonmodel_start) // self.sliding_window_length 
        # print("condensed_embed_labels shape {} dtype {}".format(condensed_embed_labels.shape, condensed_embed_labels.dtype) if condensed_embed_labels is not None else "condensed_embed_labels is None") 
        addonmodeloutput = self.addonsmallmodel( 
            # input_ids = input_ids, 
            input_ids = small_input_ids, 
            # attention_mask = original_attention_mask, 
            attention_mask = practical_attention_mask, 
            # position_ids = None, 
            past_key_values = None, 
            # condensed_embeds = hidden_states, 
            condensed_embeds = catenhidden, 
            # condensed_embeds = condensed_embed_labels, 
            labels = None, 
            # labels = labels, 
            # use_cache = None, 
            output_attentions = True, 
            output_hidden_states = None, 
            return_dict = True, 
            start_idx = self.addonmodel_start, # NOTE this is very important 
            eval_mode = False, 
            iteration_count = None, 
            # condensed_fashion = "projection_mode", 
            # experiment_setting = "setting3", 
            experiment_setting = self.inference_setting, 
        ) 
        
        logits = addonmodeloutput.logits 
        # loss = addonmodeloutput["loss"] 
        # logits = addonmodeloutput["logits"] 
        # ce_loss = loss 
        
        '''
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        ''' 
        # seq_length = input_ids.shape[1] + hidden_states.shape[1] 
        # seq_length = small_input_ids.shape[1] + hidden_states.shape[1] 
        seq_length = small_input_ids.shape[1] + catenhidden.shape[1] 
        assert seq_length == logits.shape[1], "seq_length is not compatible to logits" 
        # mask_list_pos = [self.addonmodel_start + i * (self.sliding_window_length + 1) for i in range((seq_length - self.addonmodel_start) // (self.sliding_window_length + 1))] 
        mask_list_pos = [self.addonmodel_start + self.sliding_window_length - 1 + i * (self.sliding_window_length + 1) for i in range((seq_length - self.addonmodel_start) // (self.sliding_window_length + 1))] 
        mask_list_pos22 = [x - 1 for x in mask_list_pos] 
        print("mask list pos22: {}".format(mask_list_pos22)) 
        print("length of mask list pos22: {}".format(len(mask_list_pos22))) 
        loss = None 
        
        if labels is not None: 
            # selected_indices = list(range(7)) 
            # selected_indices = list(range(self.addonmodel_start)) 
            # for i in range(7, seq_length): 
                # if i not in mask_list_pos: 
                    # selected_indices.append(i) 
            # for i in range(self.addonmodel_start, seq_length): 
            #     if i not in mask_list_pos: 
            #         selected_indices.append(i) 
            selected_indices = list(range(self.addonmodel_start - 1)) 
            for i in range(self.addonmodel_start - 1, seq_length): 
                if i not in mask_list_pos22: 
                    selected_indices.append(i) 
            # selected_indices = mask_list_pos22 
            # print(colored("selected_indices {}".format(selected_indices), "red")) 
            # select and shift the logits 
            logits = logits[:, selected_indices, :] 
            shift_logits = logits[..., :-1, :].contiguous() 
            shift_labels = labels[..., 1:].contiguous() # shape (batch_size, seq_length - 1) 
            # shift_labels = labels[..., 1:-1].contiguous() # shape (batch_size, seq_length - 1) 
            print("shift_logits shape {}; shift_labels shape {}".format(shift_logits.shape, shift_labels.shape)) 
            # Flatten the tokens 
            loss_fct = CrossEntropyLoss() 
            # shift_logits2 = shift_logits.clone().detach() # used for position investigation 
            # shift_labels2 = shift_labels.clone().detach() # used for position investigation 
            shift_logits = shift_logits.view(-1, self.config.vocab_size) 
            shift_labels = shift_labels.view(-1) 
            
            # position loss performance investigation below 
            # num_chunks = (shift_logits2.shape[1] - 1) // (self.sliding_window_length + 1) 
            # first_pos_indices = [self.addonmodel_start - 1 + (self.sliding_window_length + 1) * i for i in range(num_chunks)] 
            # first_pos_ce_loss = loss_fct(shift_logits2[:, first_pos_indices, :].view(-1, self.config.vocab_size), shift_labels2[:, first_pos_indices].view(-1)) 
            # second_pos_indices = [self.addonmodel_start + (self.sliding_window_length + 1) * i for i in range(num_chunks)] 
            # second_pos_ce_loss = loss_fct(shift_logits2[:, second_pos_indices, :].view(-1, self.config.vocab_size), shift_labels2[:, second_pos_indices].view(-1)) 
            first_pos_ce_loss = torch.tensor(0) 
            second_pos_ce_loss = torch.tensor(0) 
            
            # Enable model parallelism 
            shift_labels = shift_labels.to(shift_logits.device) 
            ce_loss = loss_fct(shift_logits, shift_labels) 
            loss = ce_loss 
            # print(colored("rank {} loss {}".format(self.accelerator.state.process_index, loss), "yellow")) 
        if loss is not None and not self.use_mse_loss: 
            if self.ce_loss_only: 
                print(colored("ce_loss only", "red")) 
                loss = ce_loss 
            else: 
                print(colored("ce_loss + mse_loss", "red")) 
                # loss = self.alpha * loss + (1 - self.alpha) * mse_loss 
                loss = self.alpha * ce_loss + (1 - self.alpha) * mse_loss 
        else: 
            print(colored("mse_loss only", "red")) 
            loss = mse_loss 

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output 

        return CausalLMOutputWithPastLargeDistance2(
            loss=loss,
            first_pos_loss = first_pos_ce_loss, 
            second_pos_loss = second_pos_ce_loss, 
            logits = logits, 
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions, 
            attentions = addonmodeloutput.attentions, # delibrately using the model's attention mask with modifications 
            l2_distance = intermediate_l2_dist, 
            ce_loss = ce_loss.detach().clone(), 
            l2_distance_input = torch.tensor(0), 
            cossim_input = cossim_input, 
        ) 
    
    def top_k_top_p_filter(self, logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0): 
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
    
    def norm_logits(self, logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor: 
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
        logits = self.top_k_top_p_filter(logits, top_k=top_k, top_p=top_p) 
        probs = F.softmax(logits, dim=1)
        return probs 
    
    def sample2(self, probs : torch.Tensor, num_samples: int = 1, random_seed = None): 
        if random_seed:
            torch.manual_seed(random_seed)
        idx_next = torch.multinomial(probs, num_samples=num_samples)
        # if (idx_next.item() == 0):
            # raise RuntimeError 
        return idx_next 
    
    def forward_generate(
        self,
        # input_ids: torch.LongTensor = None, 
        large_input_ids: torch.LongTensor = None, 
        small_input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        input_embeds: Optional[torch.FloatTensor] = None, 
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        original_attention_mask = None, 
        condensed_embed_labels = None, 
        autoregressive_first_element = False, 
        output_large_model_last_hidden_states = False, 
        inmiddlesample = False, 
        target_lmhead = None, 
        temperature = 0.6, 
        top_k = -1, 
        top_p = 0.9, 
    ) -> Union[Tuple, CausalLMOutputWithPastLargeDistance2]: 
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

        last_hidden_states = None 
        start = None 
        if self.generate_iteration_count % self.sliding_window_length == 0: 
            # NOTE for this case, we use the pass-in attention mask 
            # print(colored("running the large model side", "green")) 
            outputs = self.model(
                input_ids = large_input_ids, 
                attention_mask = attention_mask, 
                position_ids = position_ids, 
                past_key_values = past_key_values, 
                inputs_embeds = input_embeds, 
                use_cache = use_cache, 
                output_attentions = output_attentions, 
                output_hidden_states = output_hidden_states, 
                return_dict = return_dict, 
            ) 
            
            hidden_states = outputs[0] # we don't need the lm_head 
            if output_large_model_last_hidden_states: 
                last_hidden_states = hidden_states.clone().detach() 
            if inmiddlesample: 
                outputdist = target_lmhead(hidden_states) 
                probs = self.norm_logits(outputdist[:, -1, :], temperature = temperature, top_k = top_k, top_p = top_p) 
                start = self.sample2(probs) 
                if len(start.shape) == 1: 
                    start = start.unsqueeze(0) 
                small_input_ids = torch.cat([small_input_ids, start], dim = 1) 
                original_attention_mask = torch.cat([original_attention_mask, torch.ones_like(start)], dim = 1) 
            seq_len = hidden_states.shape[1] 
            
            if self.small_model_dtype == torch.float32: 
                hidden_states = hidden_states.to(torch.float32) 
            elif self.small_model_dtype == torch.bfloat16: 
                hidden_states = hidden_states.to(torch.bfloat16) 
            if self.sliding_window_length != 1: 
                selected_seq_indices = [i * self.sliding_window_length for i in range(0, math.ceil(seq_len / self.sliding_window_length))]  # adding the last one to get future tokens 
            else: 
                selected_seq_indices = [i * self.sliding_window_length for i in range(0, seq_len // self.sliding_window_length)] 
            # print("selected_seq_indices {} total length {}".format(selected_seq_indices, len(selected_seq_indices))) 
            
            hidden_states = hidden_states[:, selected_seq_indices, :] 
            hidden_states = hidden_states[:, 1 :, :] # works with 0 as the start of the sampling index 
            self.generate_model_hidden_states = hidden_states.clone().detach() 
        self.generate_iteration_count += 1 
        
        # print(colored("running the small model side", "green")) 
        addonmodeloutput = self.addonsmallmodel.generate_forward(
            input_ids = small_input_ids, 
            attention_mask = original_attention_mask, 
            position_ids = None, 
            past_key_values = None, 
            condensed_embeds = self.generate_model_hidden_states, 
            labels = None, 
            use_cache = None, 
            output_attentions = True, 
            output_hidden_states = None, 
            return_dict = True, 
            start_idx = self.sliding_window_length + 1, # NOTE this is very important 
            eval_mode = False, 
            experiment_setting = self.inference_setting, 
            generate_flag = True, 
        ) 
        
        logits = addonmodeloutput.logits 
        # logits = torch.zeros((small_input_ids.shape[0], small_input_ids.shape[1], self.config.vocab_size)).to(small_input_ids.device).to(torch.float32) 
        
        loss = None 
        
        if not return_dict: 
            output = (logits,) + outputs[1:] 
            return (loss,) + output if loss is not None else output 
        
        return CausalLMOutputWithPastLargeDistance3(
            loss = loss, 
            logits = logits, 
            past_key_values = past_key_values, 
            hidden_states = None, 
            attentions = None, 
            l2_distance = None, 
            ce_loss = None, 
            l2_distance_input = None, 
            cossim_input = None, 
            last_hidden_states = last_hidden_states, 
            start = start if inmiddlesample else None, 
        ) 
    
    def sample( 
        self, 
        input_ids: torch.LongTensor, 
        logits_processor: Optional[LogitsProcessorList] = None, 
        stopping_criteria: Optional[StoppingCriteriaList] = None, 
        logits_warper: Optional[LogitsProcessorList] = None, 
        max_length: Optional[int] = None, 
        pad_token_id: Optional[int] = None, 
        eos_token_id: Optional[int] = None, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None, 
        output_scores: Optional[bool] = None, 
        return_dict_in_generate: Optional[bool] = None, 
        synced_gpus: bool = False, 
        streamer: Optional["BaseStreamer"] = None, 
        **model_kwargs, 
    ) -> Union[SampleOutput, torch.LongTensor]: 
        
        print("inside generate function, output_hidden_states is {}".format(output_hidden_states)) 
        print(colored("inside the function that is overloaded for the model", "yellow")) 
        
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList() 
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList() 
        if max_length is not None: 
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            ) 
            stopping_criteria = StoppingCriteriaList(stopping_criteria, max_length) 
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList() 
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )
        
        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        
        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )
        
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device) 
        
        this_peer_finished = False # used by synced_gpus only 
        
        self.generate_iteration_count = 0 
        # auto-regressive generation 
        while True: 
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, adjustment_scheme = "case1", **model_kwargs) 

            # forward pass to get next token
            # outputs = self(
                # **model_inputs,
                # return_dict=True,
                # output_attentions=output_attentions,
                # output_hidden_states=output_hidden_states,
            # ) 
            outputs = self.forward_generate(
                **model_inputs, 
                return_dict = True, 
                output_attentions = output_attentions, 
                output_hidden_states = output_hidden_states, 
            ) 

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids 
    
    def prepare_inputs_for_generation2( 
        self, input_ids, past_key_values = None, attention_mask = None, inputs_embeds = None, adjustment_scheme = None, **kwargs): 
        # mainly used to debug preparing inputs for generation and not using past_key_values 
        assert past_key_values is None, "past_key_values is not None" 
        batch_size, seq_length = input_ids.shape 
        print("batch_size {}; seq_length {}".format(batch_size, seq_length)) 
        
        # adjusting the inputs and mask 
        print("input_ids {}".format(input_ids[2])) 
        print("attention_mask {}".format(attention_mask[2])) 
        condition_mask = input_ids == self.tokenizer_bos_id 
        input_sequence_indices = torch.nonzero(condition_mask).to(input_ids.device).to(torch.long) 
        print("input_sequence_indices shape {}".format(input_sequence_indices.shape)) 
        print("input_sequence_indices: {}".format(input_sequence_indices[2])) 
        input_sequence_indices2 = [] 
        modified_input_bos_sequence_indices = [] 
        assert input_sequence_indices.shape[0] == input_ids.shape[0], "every row of sequences need to have an bos" 
        for i in range(input_ids.shape[0]): # iterate through the batch_size 
            # if input_sequence_indices[i] % self.sliding_window_length != 0: # we found a sequence that needs to be adjusted 
            if input_sequence_indices[i][1].data % self.sliding_window_length != 0: # we found a sequence that needs to be adjusted 
                input_sequence_indices2.append(torch.tensor([i, (input_sequence_indices[i][1])]).to(input_ids.device).view(1, -1)) 
                if adjustment_scheme == "case1": 
                    modified_input_bos_sequence_indices.append(torch.tensor([i, (input_sequence_indices[i][1] // self.sliding_window_length) * self.sliding_window_length]).to(input_ids.device).view(1, -1)) 
                elif adjustment_scheme == "case2": 
                    modified_input_bos_sequence_indices.append(torch.tensor([i, (input_sequence_indices[i][1] // self.sliding_window_length + 1) * self.sliding_window_length]).to(input_ids.device).view(1, -1)) 
                else: 
                    raise ValueError("adjustment_scheme is not recognized") 
        if len(input_sequence_indices2) != 0: 
            # adjusting the input_ids 
            input_sequence_indices2 = torch.cat(input_sequence_indices2, dim = 0).to(input_ids.device).to(torch.long) 
            modified_input_bos_sequence_indices = torch.cat(modified_input_bos_sequence_indices, dim = 0).to(input_ids.device).to(torch.long) 
            print("shape of modified_input_bos_sequence_indices {}".format(modified_input_bos_sequence_indices.shape)) 
            print(modified_input_bos_sequence_indices) 
            
            row_indices = input_sequence_indices2[:, 0] 
            col_indices = input_sequence_indices2[:, 1] 
            input_ids.index_put_((row_indices, col_indices), torch.full_like(row_indices, fill_value = self.tokenizer_pad_id, device = input_ids.device, dtype = input_ids.dtype), accumulate = False) 
            
            row_indices = modified_input_bos_sequence_indices[:, 0] 
            col_indices = modified_input_bos_sequence_indices[:, 1] 
            input_ids.index_put_((row_indices, col_indices), torch.full_like(row_indices, fill_value = self.tokenizer_bos_id, device = input_ids.device, dtype = input_ids.dtype), accumulate = False) 
            
            print("input_ids {}".format(input_ids[2])) 
        # just for checking 
        checking_indices = torch.nonzero(input_ids == self.tokenizer_bos_id) 
        print("positions of the start of sequence after modification: {}".format(checking_indices)) 
        for i in range(checking_indices.shape[0]): 
            assert checking_indices[i][1] % self.sliding_window_length == 0, "start of sequence is not at the right position" 
            
        # making attention_mask 
        modified_input_bos_sequence_indices = torch.nonzero(input_ids == self.tokenizer_bos_id).to(input_ids.device).to(torch.long) 
        modified_input_bos_sequence_indices = modified_input_bos_sequence_indices[:, 1].unsqueeze(1).expand(-1, seq_length) 
        print("modified_input_bos_sequence_indices shape {}".format(modified_input_bos_sequence_indices.shape)) 
        col_indices = torch.arange(seq_length).expand(batch_size, -1).to(input_ids.device) 
        attention_mask = col_indices >= modified_input_bos_sequence_indices 
        # attention_mask = input_ids != self.tokenizer_pad_id 
        attention_mask = attention_mask.to(torch.long) 
        print("attention_mask {}".format(attention_mask[2])) 
        # just for checking 
        for i in range(checking_indices.shape[0]): 
            if checking_indices[i][1] != 0: 
                assert torch.unique(attention_mask[i][: checking_indices[i][1]]) == 0, "attention_mask is not correct" 
            assert torch.unique(attention_mask[i][checking_indices[i][1] : ]) == 1, "attention_mask is not correct" 
            print(colored("checking attention_mask passed", "green")) 
                
        # past_key_values is not used and input_ids is not changed 
        '''
        position_ids = kwargs.get("position_ids", None) 
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :] 
        ''' 
        position_ids = None 
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"large_input_ids": input_ids, 
                            "small_input_ids": input_ids,} 

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs 
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values = None, attention_mask = None, inputs_embeds = None, **kwargs
    ): 
        assert past_key_values is None, "past_key_values is not None" 
        
        position_ids = None 
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"large_input_ids": input_ids, 
                            "small_input_ids": input_ids,} 
        
        # print("attention_mask", attention_mask.shape) 
        # print("input_ids", input_ids.shape) 
        if attention_mask.shape[1] == input_ids.shape[1] - 1: 
            torch.cat([attention_mask, torch.ones(attention_mask.shape[0], 1, device = attention_mask.device)], dim = 1) 
        assert attention_mask.shape[1] == input_ids.shape[1], "attention_mask is not compatible with input_ids" 
        original_attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.shape[0], (input_ids.shape[1] - self.addonmodel_start)//self.sliding_window_length + 1, device = attention_mask.device)], dim = 1) 
        
        model_inputs.update( 
            { 
                "position_ids": position_ids, 
                "past_key_values": past_key_values, 
                "use_cache": kwargs.get("use_cache"), 
                "attention_mask": attention_mask, 
                "original_attention_mask": original_attention_mask, 
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
    
    def _modify_decoder_attention_mask_for_hardest_neo(self, combined_attention_mask, dtype, mask_list_pos, start_idx = None, kernel_size = None): 
        mask_shape = combined_attention_mask.shape # (batch_size, 1, tgt_seq_len, src_seq_len) 
        seq_len = mask_shape[-1] 
        start_idx = start_idx if start_idx is not None else self.start_idx 
        kernel_size = kernel_size if kernel_size is not None else self.sliding_window_length 
        
        # row dimensional masking 
        # row_idx_masked_out = [start_idx + i * (kernel_size + 1) for i in range((seq_len - start_idx) / (kernel_size + 1))] 
        row_mask = torch.zeros(mask_shape[-2], mask_shape[-1], device = combined_attention_mask.device) # NOTE currently, this line only works for training 
        # row_mask[row_idx_masked_out] = 1 
        # row_mask[mask_list_pos] = 1 

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
        assert (seq_length - 8) % 7 == 0, "seq_length is not compatible to the sliding window length" 
        self._modify_decoder_attention_mask_for_hardest_neo(attention_mask, inputs_embeds.dtype, mask_list_pos = [8 + i * 7 for i in range((seq_length - 8) // 7)], start_idx = 8, kernel_size = 7) 

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

class LlamaModelWeirdAttentionMap2(LlamaPreTrainedModel): 
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
        # self._modify_attention_mask_in_a_weird_way(attention_mask, inputs_embeds.dtype, start_idx = 64, kernel_size = 4) 

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

    def __init__(self, *args, lookaheadcount = 3, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.model = LlamaModelWeirdAttentionMap2(*args, **kwargs) 
        self.lookaheadcount = lookaheadcount 
        self.output_n_projection = nn.Linear(self.config.hidden_size, self.config.hidden_size * self.lookaheadcount, bias = False) 
        self.vocab_size = self.config.vocab_size 
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False) 
        self.act = nn.SiLU() 

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
    
    def set_lookaheadcount(self, lookaheadcount): 
        self.lookaheadcount = lookaheadcount 
    
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
        hot_n_grams = None, 
        use_filtered_hot_labels = False, 
        compute_original_output = False, 
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
        residual_values = hidden_states # size (batch_size, seq_length, hidden_size) 
        residual_list = [residual_values for _ in range(self.lookaheadcount)] 
        residual_values = torch.stack(residual_list, dim = 2) # size (batch_size, seq_length, hidden_size * lookaheadcount) 
        if compute_original_output: 
            # original_prob_output = self.lm_head(hidden_states) 
            original_prob_output = None 
        # hidden_states should have dimension (batch_size, seq_length, hidden_size) 
        # after the output_tripple_projection, we expect it should be (batch_size, seq_length, hidden_size * n) 
        hidden_states = hidden_states.to(torch.float32) 
        hidden_states = self.output_n_projection(hidden_states) 
        hidden_states = hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], self.lookaheadcount, -1) 
        hidden_states = hidden_states.to(torch.bfloat16) 
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else: 
            # residual_values = residual_values.reshape(residual_values.shape[0], residual_values.shape[1], self.lookaheadcount, -1) 
            hidden_states = residual_values + self.act(hidden_states) 
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # shift_logits = logits[..., :-1, :].contiguous() 
            # shift_labels = labels[..., 1:].contiguous() 
            shift_logits = logits[:, :-(self.lookaheadcount), :, :].contiguous() 
            # only changing logits sequenc length dimension 
            if labels.shape[-1] != self.lookaheadcount: 
                shift_labels = [] # hold all the shifted versions together 
                originalseqlength = labels.shape[1] 
                # print(labels[0]) 
                label_actual_mask = (labels[:, 1 : 1 + (originalseqlength - self.lookaheadcount)] == -100).to(torch.bool) # True to write -100, False to not write 
                for i in range(1, self.lookaheadcount + 1): 
                    shift_labels.append(labels[:, i : i + (originalseqlength - self.lookaheadcount)].contiguous()) 
                shift_labels = torch.stack(shift_labels, dim = 2) # I think dimension should be at 2 
                label_actual_mask = label_actual_mask.unsqueeze(-1).expand(-1, -1, self.lookaheadcount) 
                shift_labels[label_actual_mask] = -100 
                # print(shift_labels[0]) 

                if use_filtered_hot_labels: 
                    # interact with only the hot 1000 tokens 
                    # code generated by GPT4 
                    shift_labels_expand = shift_labels.long().unsqueeze(2) # turn into shape of (batch_size, sequence length, 1, lookaheadcount) 
                    hot_n_grams_expand = hot_n_grams.unsqueeze(0).unsqueeze(0).to(shift_labels_expand.device) # turn into shape of (1, 1, 1000, 3) 
                    matches = torch.all(shift_labels_expand == hot_n_grams_expand, dim = -1).to(torch.bool) # matches have dimension of (batch_size, seq_length, 1000) 
                    mask = ~torch.any(matches, dim = -1) # mask has dimension of (batch_size, seq_length) 
                    # print("first five of mask {}".format(mask[: 5, :])) 
                    # print("first five of shift labels {}".format(shift_labels[: 5, :, 0])) 
                    mask = mask.unsqueeze(-1).expand(-1, -1, self.lookaheadcount) # mask has dimension of (batch_size, seq_length, 3) 
                    shift_labels[mask] = -100 
                    # print("first five of shift labels {}".format(shift_labels[: 5, :, 0])) 
            else: 
                shift_labels = labels 
            output_actual_labelmasks = (shift_labels != -100)[:, :, 0] # shape would be of dimension (batch_size, seq_length) 

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

        if not compute_original_output: 
            print(colored("we are not outputing the original output", "red")) 
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ) 
        else: 
            print(colored("we output the original output", "yellow")) 
            return CausalLMOutputWithPastWithOriginalOutput(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions, 
                original_model_output = original_prob_output, 
                labels_actual_mask = output_actual_labelmasks if labels is not None else None, 
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
            input_ids = input_ids, 
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
    
    def __init__(self, *args, sliding_window_length = 4, hostname = None, target_model_dim = 4096, **kwargs): 
        super().__init__(*args, **kwargs) 
        # copied from LlamaModel 
        config = self.config 
        self.padding_idx = config.pad_token_id 
        self.vocab_size = config.vocab_size 
        
        # cross model projection of the hidden states dimension 
        self.target_model_dim = target_model_dim 
        self.embed_projection = nn.Linear(self.target_model_dim, config.hidden_size, bias = False) 
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx) 
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) 
        
        self.gradient_checkpointing = False 
        
        # copied from LlamaForCausalLM 
        self.vocab_size = config.vocab_size 
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 

        # needed to be used for the interleaving embeddings 
        # self.sliding_window_length = sliding_window_length 
        
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
            else: 
                self.criticalpath = "/fsx-storygen/beidic/yang/" 

        # if self.criticalpath is None or hostname is None: 
            # raise ValueError("critical path is not set") 
    
    def set_sliding_window_length(self, sliding_window_length): 
        self.sliding_window_length = sliding_window_length 
    
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
    
    def set_criticalpath(self, hostname): 
        if hostname is not None: 
            if "lovelace" in hostname: 
                self.criticalpath = "/home/yangzho6/" 
            elif "ada" in hostname: 
                self.criticalpath = "/home/beidic/yangzho6/" 
            else: 
                self.criticalpath = "/fsx-storygen/beidic/yang/" 

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
        
        if attention_mask is not None: 

            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len = input_shape[-1]).to( #008000 
                inputs_embeds.device 
            ) 
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
    
    def _modify_decoder_attention_mask_neo(self, combined_attention_mask, dtype, mask_list_pos, start_idx = None, kernel_size = None): 
        mask_shape = combined_attention_mask.shape # (batch_size, 1, tgt_seq_len, src_seq_len) 
        seq_len = mask_shape[-1] 
        start_idx = start_idx if start_idx is not None else self.start_idx 
        kernel_size = kernel_size if kernel_size is not None else self.sliding_window_length 
        
        # row dimensional masking 
        # row_idx_masked_out = [start_idx + i * (kernel_size + 1) for i in range((seq_len - start_idx) / (kernel_size + 1))] 
        row_mask = torch.zeros(mask_shape[-2], mask_shape[-1], device = combined_attention_mask.device) # NOTE currently, this line only works for training 
        # row_mask[row_idx_masked_out] = 1 
        # row_mask[mask_list_pos] = 1 
        # row_mask[mask_list_pos, :] = 1 

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
    
    def _modify_decoder_attention_mask_for_large_model_addon(self, combined_attention_mask, dtype, mask_list_pos, kernel_size = None): 
        # in this setting, we assume the starting idx to be 0 
        mask_shape = combined_attention_mask.shape # (batch_size, 1, tgt_seq_len, src_seq_len) 
        seq_len = mask_shape[-1] 
        start_idx = 0 
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
        print("condensed token idx list {}".format(condensed_token_idx_list)) 
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
    
    def _modify_decoder_attention_mask_for_hardest_neo(self, combined_attention_mask, dtype, mask_list_pos, start_idx = None, kernel_size = None): 
        mask_shape = combined_attention_mask.shape # (batch_size, 1, tgt_seq_len, src_seq_len) 
        seq_len = mask_shape[-1] 
        start_idx = start_idx if start_idx is not None else self.start_idx 
        kernel_size = kernel_size if kernel_size is not None else self.sliding_window_length 
        
        # row dimensional masking 
        # row_idx_masked_out = [start_idx + i * (kernel_size + 1) for i in range((seq_len - start_idx) / (kernel_size + 1))] 
        row_mask = torch.zeros(mask_shape[-2], mask_shape[-1], device = combined_attention_mask.device) # NOTE currently, this line only works for training 
        # row_mask[row_idx_masked_out] = 1 
        # row_mask[mask_list_pos] = 1 

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
    '''
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
    ''' 
    
    def interleaving_embeddings_inputs2(self, input_embeds, condensed_embeds, kernel_size = 4, start_idx = 64, generate_flag = False): 
        # print("start_idx is {}".format(start_idx)) # debug this is 2 
        if not generate_flag: 
            assert (input_embeds.shape[1] - start_idx)/kernel_size == condensed_embeds.shape[1] 
            # assert (input_embeds.shape[1] - start_idx)//kernel_size == condensed_embeds.shape[1] 
            # combined_embeds = input_embeds[:, : start_idx, :] 
            combined_embeds = input_embeds[:, : start_idx - 1, :] 
            # input_embeds_count = start_idx 
            input_embeds_count = start_idx - 1 
        else: 
            assert (input_embeds.shape[1] - start_idx)//kernel_size + 1 == condensed_embeds.shape[1] 
            # combined_embeds = input_embeds[:, : start_idx, :] 
            combined_embeds = input_embeds[:, : start_idx - 1, :] 
            # input_embeds_count = start_idx 
            input_embeds_count = start_idx - 1 
        for i in range(condensed_embeds.shape[1]): 
            # print("i is {} length of combined_embeds is {}".format(i, combined_embeds.shape[1])) 
            combined_embeds = torch.cat([combined_embeds, condensed_embeds[:, i, :].unsqueeze(1)], dim = 1) 
            if (input_embeds_count < input_embeds.shape[1]): 
                combined_embeds = torch.cat([combined_embeds, input_embeds[:, input_embeds_count : min(input_embeds_count + kernel_size, input_embeds.shape[1]), :]], dim = 1) 
            input_embeds_count += kernel_size 
        if input_embeds_count < input_embeds.shape[1]: 
            combined_embeds = torch.cat([combined_embeds, input_embeds[:, input_embeds_count :, :]], dim = 1) 
        
        return combined_embeds 
    
    def interleaving_embeddings_inputs(self, input_embeds, condensed_embeds, kernel_size = 4, start_idx = 64, generate_flag = False): 
        if not generate_flag: 
            # assert (input_embeds.shape[1] - start_idx)/kernel_size == condensed_embeds.shape[1] 
            assert (input_embeds.shape[1] - start_idx)//kernel_size == condensed_embeds.shape[1] 
            combined_embeds = input_embeds[:, : start_idx, :] 
            input_embeds_count = start_idx 
        else: 
            assert (input_embeds.shape[1] - start_idx)//kernel_size + 1 == condensed_embeds.shape[1] 
            combined_embeds = input_embeds[:, : start_idx, :] 
            input_embeds_count = start_idx 
        for i in range(condensed_embeds.shape[1]): 
            # print("i is {} length of combined_embeds is {}".format(i, combined_embeds.shape[1])) 
            combined_embeds = torch.cat([combined_embeds, condensed_embeds[:, i, :].unsqueeze(1)], dim = 1) 
            if (input_embeds_count < input_embeds.shape[1]): 
                combined_embeds = torch.cat([combined_embeds, input_embeds[:, input_embeds_count : min(input_embeds_count + kernel_size, input_embeds.shape[1]), :]], dim = 1) 
            input_embeds_count += kernel_size 
        
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
        tensordtype = tensor.dtype 
        if tensordtype == torch.bfloat16: 
            tensor_np = tensor.cpu().clone().to(torch.float32).numpy() 
        else: 
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
    def plot_attention_map(attention_maps, layer_num, head_num, seq_length, filename, batch_idx = 0): 
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
        # attention_map = attention_maps[layer_num][0][head_num].to(torch.float32).cpu().detach().numpy() 
        attention_map = attention_maps[layer_num][batch_idx][head_num].to(torch.float32).cpu().detach().numpy() 
        
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
        fig, ax = plt.subplots(figsize=(100, 60)) 
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
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, ha = "right") 

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
        generate_flag = False, 
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
            print("input_ids shape {}".format(input_ids.shape)) 
            print("condensed_embeds shape {}".format(condensed_embeds.shape)) 
            assert input_ids.shape[0] == condensed_embeds.shape[0] # batch size has to match 
            print("input_ids shape {} condensed_embeds shape {}".format(input_ids.shape, condensed_embeds.shape)) 
            print("sliding window length {}".format(self.sliding_window_length)) 
            if not generate_flag: 
                print("input_ids.shape[1]: {}".format(input_ids.shape[1])) 
                print("start_idx: {}".format(start_idx)) 
                print("self.sliding_window_length: {}".format(self.sliding_window_length)) 
                print("condensed_embeds.shape[1]: {}".format(condensed_embeds.shape[1])) 
                assert (input_ids.shape[1] - start_idx)//self.sliding_window_length == condensed_embeds.shape[1] # number of condensed tokens should have desired mapping with sequence length 
            else: 
                print("start_idx: {}".format(start_idx)) 
                # print(math.ceil((input_ids.shape[1] - start_idx)/self.sliding_window_length)) 
                print(math.ceil((input_ids.shape[1] - (start_idx - 1))/self.sliding_window_length)) 
                print(condensed_embeds.shape[1]) 
                # assert ((input_ids.shape[1] - start_idx)//self.sliding_window_length) + 1 == condensed_embeds.shape[1] 
                assert math.ceil((input_ids.shape[1] - (start_idx - 1))/self.sliding_window_length) == condensed_embeds.shape[1] 
                # assert math.ceil((input_ids.shape[1] - start_idx)/self.sliding_window_length) == condensed_embeds.shape[1] 

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
        
        seq_length += condensed_embeds.shape[1] 
        seq_length_with_past = seq_length 
        past_key_values_length = 0 
        
        if past_key_values is not None: 
            past_key_values_length = past_key_values[0][0].shape[2] 
            seq_length_with_past = seq_length_with_past + past_key_values_length 
        
        # self.mask_list_pos = [self.start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - self.start_idx) // (self.sliding_window_length + 1))] 
        # mask_list_pos = [self.start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - self.start_idx) // (self.sliding_window_length + 1))] 
        # mask_list_pos = [start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - start_idx) // (self.sliding_window_length + 1))] 
        if not generate_flag: 
            mask_list_pos = [start_idx - 1 + i * (self.sliding_window_length + 1) for i in range((seq_length - start_idx) // (self.sliding_window_length + 1))] 
        else: 
            print((seq_length - start_idx) / (self.sliding_window_length + 1)) 
            mask_list_pos = [start_idx - 1 + i * (self.sliding_window_length + 1) for i in range(int(math.ceil((seq_length - start_idx) / (self.sliding_window_length + 1))))] 
        if position_ids is None: 
            device = input_ids.device 
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
            
            print("position ids found is {}".format(position_ids.shape)) 
            print("position ids found is {}".format(position_ids)) 
        
        # the important part 
        # input_embeds should not be None 
        torch.set_printoptions(threshold = 500) 
        input_embeds = None 
        if condensed_embeds is not None: 
            print(colored("condensed_embeds dtype: {} input_ids dtype: {}".format(condensed_embeds.dtype, input_ids.dtype), "yellow")) 
            print("embed_projection dtype: {}".format(self.embed_projection.weight.dtype)) 
            if self.condensed_fashion == "projection_mode": 
                print(colored("condensed_embeds dtype: {}".format(condensed_embeds.dtype), "red")) 
                condensed_embeds = self.embed_projection(condensed_embeds) 
            input_embeds = self.embed_tokens(input_ids) 
            input_embeds = self.interleaving_embeddings_inputs2(input_embeds, condensed_embeds, kernel_size = self.sliding_window_length, start_idx = start_idx, generate_flag = generate_flag) 
        else: 
            raise ValueError("We cannot have an inference or any forward propagation without the inputs_embeds") 
        
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
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), input_embeds, past_key_values_length 
        ) 
        
        if self.eval_mode: 
            # the attention_mask ignores the condensed tokens 
            self._convert_to_normal_attention_mask(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
        else: 
            if self.experiment_setting == "setting0": 
                self._modify_decoder_attention_mask_neo(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting1": 
                self._modify_decoder_attention_mask_for_harder(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting2": 
                self._modify_decoder_attention_mask_for_harder2(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting3": 
                self._modify_decoder_attention_mask_for_hardest_neo(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting4": 
                self._modify_decoder_attention_mask_for_hardest_neo(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting5": 
                # we make no change to the original attention mask 
                pass 
            else: 
                raise ValueError("We do not have the experiment setting you are looking for") 
            
        if iteration_count is not None and iteration_count == 1: 
            working_dir = self.criticalpath 
            self.visualize_attention_mask(seq_length, attention_mask[0][0], working_dir + "attention_mask_after_modification.jpg") 
        
        hidden_states = input_embeds 
        print("hidden_states shape {}".format(hidden_states.shape)) 
        
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
                # horizontal_bar_enabled = False 
                horizontal_bar_enabled = True 
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask, 
                    mask_list_pos = mask_list_pos, 
                    horizontal_bar_enabled = horizontal_bar_enabled, 
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
        
        if self.config.pretraining_tp > 1: 
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1) 
        else: 
            logits = self.lm_head(hidden_states) 
        logits = logits.float() 

        mask_list_pos22 = [x - 1 for x in mask_list_pos] # just trying 
        loss = None 
        if labels is not None: 
            # Shift so that tokens < n predict n 
            # selected_indices = list(range(start_idx)) 
            selected_indices = list(range(start_idx - 1)) 
            # for i in range(start_idx, seq_length): 
            for i in range(start_idx - 1, seq_length): 
                # if i not in mask_list_pos: 
                if i not in mask_list_pos22: 
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
    
    def generate_forward( 
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
        experiment_setting = "setting0", 
        generate_flag = None, 
    ) -> Union[Tuple, CausalLMOutputWithPast]: 

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
        
        # dimension matching 
        assert input_ids.shape[0] == condensed_embeds.shape[0] # batch size has to match 
        
        # assert ((input_ids.shape[1] - start_idx)//self.sliding_window_length) + 1 == condensed_embeds.shape[1] 
        assert math.ceil((input_ids.shape[1] - (start_idx - 1))/self.sliding_window_length) == condensed_embeds.shape[1] 
        # assert math.ceil((input_ids.shape[1] - start_idx)/self.sliding_window_length) == condensed_embeds.shape[1] 
                
        
        seq_length += condensed_embeds.shape[1] 
        seq_length_with_past = seq_length 
        past_key_values_length = 0 
        
        if past_key_values is not None: 
            past_key_values_length = past_key_values[0][0].shape[2] 
            seq_length_with_past = seq_length_with_past + past_key_values_length 
        
        # self.mask_list_pos = [self.start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - self.start_idx) // (self.sliding_window_length + 1))] 
        # mask_list_pos = [self.start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - self.start_idx) // (self.sliding_window_length + 1))] 
        # mask_list_pos = [start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - start_idx) // (self.sliding_window_length + 1))] 
        
        mask_list_pos = [start_idx - 1 + i * (self.sliding_window_length + 1) for i in range(int(math.ceil((seq_length - start_idx) / (self.sliding_window_length + 1))))] 
        if position_ids is None: # TODO: currently the position_ids isn't adaptive to attention_mask, which needs to be improved 
            device = input_ids.device 
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
            
            # print("position ids found is {}".format(position_ids.shape)) 
            # print("position ids found is {}".format(position_ids)) 
        
        # the important part 
        # input_embeds should not be None 
        input_embeds = None 
        condensed_embeds = self.embed_projection(condensed_embeds) 
        input_embeds = self.embed_tokens(input_ids) 
        input_embeds = self.interleaving_embeddings_inputs2(input_embeds, condensed_embeds, kernel_size = self.sliding_window_length, start_idx = start_idx, generate_flag = generate_flag) 
        
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
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), input_embeds, past_key_values_length 
        ) 
        
        if self.eval_mode: 
            # the attention_mask ignores the condensed tokens 
            self._convert_to_normal_attention_mask(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
        else: 
            if self.experiment_setting == "setting0": 
                self._modify_decoder_attention_mask_neo(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting1": 
                self._modify_decoder_attention_mask_for_harder(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting2": 
                self._modify_decoder_attention_mask_for_harder2(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting3": 
                self._modify_decoder_attention_mask_for_hardest_neo(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting4": 
                self._modify_decoder_attention_mask_for_hardest_neo(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting5": 
                # we make no change to the original attention mask 
                pass 
            else: 
                raise ValueError("We do not have the experiment setting you are looking for") 
        
        hidden_states = input_embeds 
        # print("hidden_states shape {}".format(hidden_states.shape)) 
        
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
                # horizontal_bar_enabled = False 
                horizontal_bar_enabled = True 
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask, 
                    mask_list_pos = mask_list_pos, 
                    horizontal_bar_enabled = horizontal_bar_enabled, 
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
        
        if self.config.pretraining_tp > 1: 
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1) 
        else: 
            logits = self.lm_head(hidden_states) 
        logits = logits.float() 

        mask_list_pos22 = [x - 1 for x in mask_list_pos] # just trying 
        loss = None 
        if labels is not None: 
            # Shift so that tokens < n predict n 
            # selected_indices = list(range(start_idx)) 
            selected_indices = list(range(start_idx - 1)) 
            # for i in range(start_idx, seq_length): 
            for i in range(start_idx - 1, seq_length): 
                # if i not in mask_list_pos: 
                if i not in mask_list_pos22: 
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

class SimpleSmallModelmixedb(LlamaPreTrainedModel): 
    _tied_weights_keys = ["lm_head.weight"] 
    
    def __init__(self, *args, sliding_window_length = 4, hostname = None, target_model_dim = 4096, **kwargs): 
        super().__init__(*args, **kwargs) 
        # copied from LlamaModel 
        config = self.config 
        self.padding_idx = config.pad_token_id 
        self.vocab_size = config.vocab_size 
        
        # cross model projection of the hidden states dimension 
        self.target_model_dim = target_model_dim 
        self.embed_projection = nn.Linear(self.target_model_dim, config.hidden_size, bias = False) 
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx) 
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) 
        
        self.gradient_checkpointing = False 
        
        # copied from LlamaForCausalLM 
        self.vocab_size = config.vocab_size 
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 

        # needed to be used for the interleaving embeddings 
        # self.sliding_window_length = sliding_window_length 
        
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
            else: 
                self.criticalpath = "/fsx-storygen/beidic/yang/" 

        # if self.criticalpath is None or hostname is None: 
            # raise ValueError("critical path is not set") 
    
    def set_sliding_window_length(self, sliding_window_length): 
        self.sliding_window_length = sliding_window_length 
    
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
    
    def set_criticalpath(self, hostname): 
        if hostname is not None: 
            if "lovelace" in hostname: 
                self.criticalpath = "/home/yangzho6/" 
            elif "ada" in hostname: 
                self.criticalpath = "/home/beidic/yangzho6/" 
            else: 
                self.criticalpath = "/fsx-storygen/beidic/yang/" 

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
        
        if attention_mask is not None: 

            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len = input_shape[-1]).to( #008000 
                inputs_embeds.device 
            ) 
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
    
    def _modify_decoder_attention_mask_neo(self, combined_attention_mask, dtype, mask_list_pos, start_idx = None, kernel_size = None): 
        mask_shape = combined_attention_mask.shape # (batch_size, 1, tgt_seq_len, src_seq_len) 
        seq_len = mask_shape[-1] 
        start_idx = start_idx if start_idx is not None else self.start_idx 
        kernel_size = kernel_size if kernel_size is not None else self.sliding_window_length 
        
        # row dimensional masking 
        # row_idx_masked_out = [start_idx + i * (kernel_size + 1) for i in range((seq_len - start_idx) / (kernel_size + 1))] 
        row_mask = torch.zeros(mask_shape[-2], mask_shape[-1], device = combined_attention_mask.device) # NOTE currently, this line only works for training 
        # row_mask[row_idx_masked_out] = 1 
        # row_mask[mask_list_pos] = 1 
        # row_mask[mask_list_pos, :] = 1 

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
    
    def _modify_decoder_attention_mask_for_large_model_addon(self, combined_attention_mask, dtype, mask_list_pos, kernel_size = None): 
        # in this setting, we assume the starting idx to be 0 
        mask_shape = combined_attention_mask.shape # (batch_size, 1, tgt_seq_len, src_seq_len) 
        seq_len = mask_shape[-1] 
        start_idx = 0 
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
        print("condensed token idx list {}".format(condensed_token_idx_list)) 
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
    
    def _modify_decoder_attention_mask_for_hardest_neo(self, combined_attention_mask, dtype, mask_list_pos, start_idx = None, kernel_size = None): 
        mask_shape = combined_attention_mask.shape # (batch_size, 1, tgt_seq_len, src_seq_len) 
        seq_len = mask_shape[-1] 
        start_idx = start_idx if start_idx is not None else self.start_idx 
        kernel_size = kernel_size if kernel_size is not None else self.sliding_window_length 
        
        # row dimensional masking 
        # row_idx_masked_out = [start_idx + i * (kernel_size + 1) for i in range((seq_len - start_idx) / (kernel_size + 1))] 
        row_mask = torch.zeros(mask_shape[-2], mask_shape[-1], device = combined_attention_mask.device) # NOTE currently, this line only works for training 
        # row_mask[row_idx_masked_out] = 1 
        # row_mask[mask_list_pos] = 1 

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
    '''
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
    ''' 
    
    def interleaving_embeddings_inputs2(self, input_embeds, condensed_embeds, kernel_size = 4, start_idx = 64, generate_flag = False): 
        # print("start_idx is {}".format(start_idx)) # debug this is 2 
        if not generate_flag: 
            assert (input_embeds.shape[1] - start_idx)/kernel_size == condensed_embeds.shape[1] 
            # combined_embeds = input_embeds[:, : start_idx - 1, :] 
            combined_embeds = input_embeds[:, : start_idx - 1 + self.sliding_window_length - 1, :] 
            # input_embeds_count = start_idx - 1 
            input_embeds_count = start_idx - 1 + self.sliding_window_length - 1 
        else: 
            assert (input_embeds.shape[1] - start_idx)//kernel_size + 1 == condensed_embeds.shape[1] 
            # combined_embeds = input_embeds[:, : start_idx, :] 
            combined_embeds = input_embeds[:, : start_idx - 1, :] 
            # input_embeds_count = start_idx 
            input_embeds_count = start_idx - 1 
        for i in range(condensed_embeds.shape[1]): 
            # print("i is {} length of combined_embeds is {}".format(i, combined_embeds.shape[1])) 
            combined_embeds = torch.cat([combined_embeds, condensed_embeds[:, i, :].unsqueeze(1)], dim = 1) 
            if (input_embeds_count < input_embeds.shape[1]): 
                combined_embeds = torch.cat([combined_embeds, input_embeds[:, input_embeds_count : min(input_embeds_count + kernel_size, input_embeds.shape[1]), :]], dim = 1) 
            input_embeds_count += kernel_size 
        if input_embeds_count < input_embeds.shape[1]: 
            print(colored("inside input_interleaving encunter leftover wordtokens", "green")) 
            combined_embeds = torch.cat([combined_embeds, input_embeds[:, input_embeds_count :, :]], dim = 1) 
        
        return combined_embeds 
    
    def interleaving_embeddings_inputs(self, input_embeds, condensed_embeds, kernel_size = 4, start_idx = 64, generate_flag = False): 
        if not generate_flag: 
            # assert (input_embeds.shape[1] - start_idx)/kernel_size == condensed_embeds.shape[1] 
            assert (input_embeds.shape[1] - start_idx)//kernel_size == condensed_embeds.shape[1] 
            combined_embeds = input_embeds[:, : start_idx, :] 
            input_embeds_count = start_idx 
        else: 
            assert (input_embeds.shape[1] - start_idx)//kernel_size + 1 == condensed_embeds.shape[1] 
            combined_embeds = input_embeds[:, : start_idx, :] 
            input_embeds_count = start_idx 
        for i in range(condensed_embeds.shape[1]): 
            # print("i is {} length of combined_embeds is {}".format(i, combined_embeds.shape[1])) 
            combined_embeds = torch.cat([combined_embeds, condensed_embeds[:, i, :].unsqueeze(1)], dim = 1) 
            if (input_embeds_count < input_embeds.shape[1]): 
                combined_embeds = torch.cat([combined_embeds, input_embeds[:, input_embeds_count : min(input_embeds_count + kernel_size, input_embeds.shape[1]), :]], dim = 1) 
            input_embeds_count += kernel_size 
        
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
        tensordtype = tensor.dtype 
        if tensordtype == torch.bfloat16: 
            tensor_np = tensor.cpu().clone().to(torch.float32).numpy() 
        else: 
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
        attention_map = attention_maps[layer_num][0][head_num].to(torch.float32).cpu().detach().numpy() 
        
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
        fig, ax = plt.subplots(figsize=(100, 60)) 
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
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, ha = "right") 

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
        generate_flag = False, 
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
            print("input_ids shape {}".format(input_ids.shape)) 
            print("condensed_embeds shape {}".format(condensed_embeds.shape)) 
            assert input_ids.shape[0] == condensed_embeds.shape[0] # batch size has to match 
            print("input_ids shape {} condensed_embeds shape {}".format(input_ids.shape, condensed_embeds.shape)) 
            print("sliding window length {}".format(self.sliding_window_length)) 
            if not generate_flag: 
                print("input_ids.shape[1]: {}".format(input_ids.shape[1])) 
                print("start_idx: {}".format(start_idx)) 
                print("self.sliding_window_length: {}".format(self.sliding_window_length)) 
                print("condensed_embeds.shape[1]: {}".format(condensed_embeds.shape[1])) 
                assert (input_ids.shape[1] - start_idx)//self.sliding_window_length == condensed_embeds.shape[1] # number of condensed tokens should have desired mapping with sequence length 
            else: 
                print("start_idx: {}".format(start_idx)) 
                # print(math.ceil((input_ids.shape[1] - start_idx)/self.sliding_window_length)) 
                print(math.ceil((input_ids.shape[1] - (start_idx - 1))/self.sliding_window_length)) 
                print(condensed_embeds.shape[1]) 
                # assert ((input_ids.shape[1] - start_idx)//self.sliding_window_length) + 1 == condensed_embeds.shape[1] 
                assert math.ceil((input_ids.shape[1] - (start_idx - 1))/self.sliding_window_length) == condensed_embeds.shape[1] 
                # assert math.ceil((input_ids.shape[1] - start_idx)/self.sliding_window_length) == condensed_embeds.shape[1] 

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
        
        seq_length += condensed_embeds.shape[1] 
        seq_length_with_past = seq_length 
        past_key_values_length = 0 
        
        if past_key_values is not None: 
            past_key_values_length = past_key_values[0][0].shape[2] 
            seq_length_with_past = seq_length_with_past + past_key_values_length 
        
        # self.mask_list_pos = [self.start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - self.start_idx) // (self.sliding_window_length + 1))] 
        # mask_list_pos = [self.start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - self.start_idx) // (self.sliding_window_length + 1))] 
        # mask_list_pos = [start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - start_idx) // (self.sliding_window_length + 1))] 
        if not generate_flag: 
            # mask_list_pos = [start_idx - 1 + i * (self.sliding_window_length + 1) for i in range((seq_length - start_idx) // (self.sliding_window_length + 1))] 
            mask_list_pos = [start_idx - 1 + self.sliding_window_length - 1 + i * (self.sliding_window_length + 1) for i in range((seq_length - start_idx) // (self.sliding_window_length + 1))] 
        else: 
            print((seq_length - start_idx) / (self.sliding_window_length + 1)) 
            mask_list_pos = [start_idx - 1 + i * (self.sliding_window_length + 1) for i in range(int(math.ceil((seq_length - start_idx) / (self.sliding_window_length + 1))))] 
        if position_ids is None: 
            device = input_ids.device 
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
            
            print("position ids found is {}".format(position_ids.shape)) 
            print("position ids found is {}".format(position_ids)) 
        
        # the important part 
        # input_embeds should not be None 
        torch.set_printoptions(threshold = 500) 
        input_embeds = None 
        if condensed_embeds is not None: 
            print(colored("condensed_embeds dtype: {} input_ids dtype: {}".format(condensed_embeds.dtype, input_ids.dtype), "yellow")) 
            print("embed_projection dtype: {}".format(self.embed_projection.weight.dtype)) 
            if self.condensed_fashion == "projection_mode": 
                print(colored("condensed_embeds dtype: {}".format(condensed_embeds.dtype), "red")) 
                condensed_embeds = self.embed_projection(condensed_embeds) 
            input_embeds = self.embed_tokens(input_ids) 
            input_embeds = self.interleaving_embeddings_inputs2(input_embeds, condensed_embeds, kernel_size = self.sliding_window_length, start_idx = start_idx, generate_flag = generate_flag) 
        else: 
            raise ValueError("We cannot have an inference or any forward propagation without the inputs_embeds") 
        
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
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), input_embeds, past_key_values_length 
        ) 
        
        if self.eval_mode: 
            # the attention_mask ignores the condensed tokens 
            self._convert_to_normal_attention_mask(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
        else: 
            if self.experiment_setting == "setting0": 
                self._modify_decoder_attention_mask_neo(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting1": 
                self._modify_decoder_attention_mask_for_harder(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting2": 
                self._modify_decoder_attention_mask_for_harder2(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting3": 
                self._modify_decoder_attention_mask_for_hardest_neo(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting4": 
                self._modify_decoder_attention_mask_for_hardest_neo(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting5": 
                # we make no change to the original attention mask 
                pass 
            else: 
                raise ValueError("We do not have the experiment setting you are looking for") 
            
        if iteration_count is not None and iteration_count == 1: 
            working_dir = self.criticalpath 
            self.visualize_attention_mask(seq_length, attention_mask[0][0], working_dir + "attention_mask_after_modification.jpg") 
        
        hidden_states = input_embeds 
        print("hidden_states shape {}".format(hidden_states.shape)) 
        
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
                # horizontal_bar_enabled = False 
                horizontal_bar_enabled = True 
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask, 
                    mask_list_pos = mask_list_pos, 
                    horizontal_bar_enabled = horizontal_bar_enabled, 
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
        
        if self.config.pretraining_tp > 1: 
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1) 
        else: 
            logits = self.lm_head(hidden_states) 
        logits = logits.float() 

        mask_list_pos22 = [x - 1 for x in mask_list_pos] # just trying 
        loss = None 
        if labels is not None: 
            # Shift so that tokens < n predict n 
            # selected_indices = list(range(start_idx)) 
            selected_indices = list(range(start_idx - 1)) 
            # for i in range(start_idx, seq_length): 
            for i in range(start_idx - 1, seq_length): 
                # if i not in mask_list_pos: 
                if i not in mask_list_pos22: 
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
    
    def generate_forward( 
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
        experiment_setting = "setting0", 
        generate_flag = None, 
    ) -> Union[Tuple, CausalLMOutputWithPast]: 

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
        
        # dimension matching 
        assert input_ids.shape[0] == condensed_embeds.shape[0] # batch size has to match 
        
        # assert ((input_ids.shape[1] - start_idx)//self.sliding_window_length) + 1 == condensed_embeds.shape[1] 
        assert math.ceil((input_ids.shape[1] - (start_idx - 1))/self.sliding_window_length) == condensed_embeds.shape[1] 
        # assert math.ceil((input_ids.shape[1] - start_idx)/self.sliding_window_length) == condensed_embeds.shape[1] 
                
        
        seq_length += condensed_embeds.shape[1] 
        seq_length_with_past = seq_length 
        past_key_values_length = 0 
        
        if past_key_values is not None: 
            past_key_values_length = past_key_values[0][0].shape[2] 
            seq_length_with_past = seq_length_with_past + past_key_values_length 
        
        # self.mask_list_pos = [self.start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - self.start_idx) // (self.sliding_window_length + 1))] 
        # mask_list_pos = [self.start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - self.start_idx) // (self.sliding_window_length + 1))] 
        # mask_list_pos = [start_idx + i * (self.sliding_window_length + 1) for i in range((seq_length - start_idx) // (self.sliding_window_length + 1))] 
        
        mask_list_pos = [start_idx - 1 + i * (self.sliding_window_length + 1) for i in range(int(math.ceil((seq_length - start_idx) / (self.sliding_window_length + 1))))] 
        if position_ids is None: # TODO: currently the position_ids isn't adaptive to attention_mask, which needs to be improved 
            device = input_ids.device 
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
            
            # print("position ids found is {}".format(position_ids.shape)) 
            # print("position ids found is {}".format(position_ids)) 
        
        # the important part 
        # input_embeds should not be None 
        input_embeds = None 
        condensed_embeds = self.embed_projection(condensed_embeds) 
        input_embeds = self.embed_tokens(input_ids) 
        input_embeds = self.interleaving_embeddings_inputs2(input_embeds, condensed_embeds, kernel_size = self.sliding_window_length, start_idx = start_idx, generate_flag = generate_flag) 
        
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
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), input_embeds, past_key_values_length 
        ) 
        
        if self.eval_mode: 
            # the attention_mask ignores the condensed tokens 
            self._convert_to_normal_attention_mask(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
        else: 
            if self.experiment_setting == "setting0": 
                self._modify_decoder_attention_mask_neo(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting1": 
                self._modify_decoder_attention_mask_for_harder(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting2": 
                self._modify_decoder_attention_mask_for_harder2(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting3": 
                self._modify_decoder_attention_mask_for_hardest_neo(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting4": 
                self._modify_decoder_attention_mask_for_hardest_neo(attention_mask, dtype = input_embeds.dtype, mask_list_pos = mask_list_pos, start_idx = start_idx, kernel_size = self.sliding_window_length) 
            elif self.experiment_setting == "setting5": 
                # we make no change to the original attention mask 
                pass 
            else: 
                raise ValueError("We do not have the experiment setting you are looking for") 
        
        hidden_states = input_embeds 
        # print("hidden_states shape {}".format(hidden_states.shape)) 
        
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
                # horizontal_bar_enabled = False 
                horizontal_bar_enabled = True 
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask, 
                    mask_list_pos = mask_list_pos, 
                    horizontal_bar_enabled = horizontal_bar_enabled, 
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
        
        if self.config.pretraining_tp > 1: 
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1) 
        else: 
            logits = self.lm_head(hidden_states) 
        logits = logits.float() 

        mask_list_pos22 = [x - 1 for x in mask_list_pos] # just trying 
        loss = None 
        if labels is not None: 
            # Shift so that tokens < n predict n 
            # selected_indices = list(range(start_idx)) 
            selected_indices = list(range(start_idx - 1)) 
            # for i in range(start_idx, seq_length): 
            for i in range(start_idx - 1, seq_length): 
                # if i not in mask_list_pos: 
                if i not in mask_list_pos22: 
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

class SimpleSmallModel2(LlamaPreTrainedModel): 
    _tied_weights_keys = ["lm_head.weight"] 
    
    def __init__(self, *args, sliding_window_length = 4, hostname = None, target_model_dim = 4096, generate_flag = False, **kwargs): 
        super().__init__(*args, **kwargs) 
        # copied from LlamaModel 
        config = self.config 
        self.padding_idx = config.pad_token_id 
        self.vocab_size = config.vocab_size 
        
        # cross model projection of the hidden states dimension 
        self.target_model_dim = target_model_dim 
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
            else: 
                self.criticalpath = "/fsx-storygen/beidic/yang/" 

        if self.criticalpath is None or hostname is None: 
            raise ValueError("critical path is not set") 
        self.generate_flag = generate_flag 
    
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
    
    def interleaving_embeddings_inputs(self, input_embeds, condensed_embeds, kernel_size = 4, start_idx = 64): 
        if not self.generate_flag: 
            assert (input_embeds.shape[1] - start_idx - kernel_size)/kernel_size == condensed_embeds.shape[1] 
            combined_embeds = input_embeds[:, : start_idx + kernel_size, :] 
            input_embeds_count = start_idx + kernel_size 
        else: 
            assert (input_embeds.shape[1] - start_idx)//kernel_size == condensed_embeds.shape[1] 
            combined_embeds = input_embeds[:, : start_idx, :] 
            input_embeds_count = start_idx 
        for i in range(condensed_embeds.shape[1]): 
            combined_embeds = torch.cat([combined_embeds, condensed_embeds[:, i, :].unsqueeze(1)], dim = 1) 
            combined_embeds = torch.cat([combined_embeds, input_embeds[:, input_embeds_count : input_embeds_count + kernel_size, :]], dim = 1) 
            input_embeds_count += kernel_size 
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
        tensordtype = tensor.dtype 
        if tensordtype == torch.bfloat16: 
            tensor_np = tensor.cpu().clone().to(torch.float32).numpy() 
        else: 
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
        attention_map = attention_maps[layer_num][0][head_num].to(torch.float32).cpu().detach().numpy() 
        
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
        fig, ax = plt.subplots(figsize=(100, 60)) 
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
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, ha = "right") 

        plt.colorbar(cbar, orientation = "vertical") 

        # Save to file
        plt.savefig(filename, bbox_inches='tight')
        plt.close() 
    
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
        
        batch_size = input_ids.shape[0] 
        seq_length = input_ids.shape[1] 
        if not self.eval_mode: 
            # dimension matching 
            print("input_ids shape {}".format(input_ids.shape)) 
            print("condensed_embeds shape {}".format(condensed_embeds.shape)) 
            assert input_ids.shape[0] == condensed_embeds.shape[0] # batch size has to match 
            print("input_ids shape {} condensed_embeds shape {}".format(input_ids.shape, condensed_embeds.shape)) 
            print("sliding window length {}".format(self.sliding_window_length)) 
            if self.training: 
                assert (input_ids.shape[1] - start_idx - self.sliding_window_length)/self.sliding_window_length == condensed_embeds.shape[1] # number of condensed tokens should have desired mapping with sequence length 
            
        else: 
            # for the eval mode we simply ignore the condensed_embeds 
            condensed_length = int((input_ids.shape[1] - start_idx)/self.sliding_window_length) 
            condensed_embeds = torch.zeros((batch_size, condensed_length, self.target_model_dim)).to(input_ids.device) 
        seq_length += condensed_embeds.shape[1] # we make sure the before entering the forward function, the condensed token has dropped the last one 
        seq_length_with_past = seq_length 
        past_key_values_length = 0 
        
        if past_key_values is not None: 
            past_key_values_length = past_key_values[0][0].shape[2] 
            seq_length_with_past = seq_length_with_past + past_key_values_length 
            
        mask_list_pos = [start_idx + self.sliding_window_length + i * (self.sliding_window_length + 1) for i in range((seq_length - start_idx - self.sliding_window_length) // (self.sliding_window_length + 1))] 
        if position_ids is None: # this should work for this case
            device = input_ids.device 
            position_list = [] 
            pos_count = past_key_values_length 
            # following_flag = False 
            for i in range(seq_length): 
                if i in mask_list_pos: 
                    position_list.append(pos_count) 
                else: 
                    pos_count += 1 
                    position_list.append(pos_count) 
            position_ids = torch.tensor(position_list, dtype = torch.long, device = device) 
            position_ids = position_ids.unsqueeze(0) 
        
        torch.set_printoptions(threshold = 500) 
        input_embeds = None 
        if condensed_embeds is not None: 
            print(colored("condensed_embeds dtype: {}".format(condensed_embeds.dtype), "red")) 
            print("embed_projection dtype: {}".format(self.embed_projection.weight.dtype)) 
            if self.condensed_fashion == "projection_mode": 
                print(colored("condensed_embeds dtype: {}".format(condensed_embeds.dtype), "red")) 
                condensed_embeds = self.embed_projection(condensed_embeds) 
            input_embeds = self.embed_tokens(input_ids) 
            input_embeds = self.interleaving_embeddings_inputs(input_embeds, condensed_embeds, kernel_size = self.sliding_window_length, start_idx = start_idx) 
        else: 
            raise ValueError("We cannot have an inference or any forward propagation without the inputs_embeds") 

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
        
        if iteration_count is not None and iteration_count == 1: 
            working_dir = self.criticalpath 
            self.visualize_attention_mask(seq_length, attention_mask[0][0], working_dir + "attention_mask_after_modification.jpg") 
        
        hidden_states = input_embeds 
        
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False 
        
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
                horizontal_bar_enabled = False 
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask, 
                    mask_list_pos = mask_list_pos, 
                    horizontal_bar_enabled = horizontal_bar_enabled, 
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
        
        mask_list_pos22 = [x - 1 for x in mask_list_pos] 
        loss = None 
        if labels is not None: 
            selected_indices = list(range(start_idx + self.sliding_window_length - 1)) 
            for i in range(start_idx + self.sliding_window_length - 1, seq_length): 
                if i not in mask_list_pos22: 
                    selected_indices.append(i) 
            logits = logits[:, selected_indices, :] 
            shift_logits = logits[..., :-1, :].contiguous() 
            shift_labels = labels[..., 1:].contiguous() 
            loss_fct = CrossEntropyLoss() 
            shift_logits = shift_logits.view(-1, self.config.vocab_size) 
            shift_labels = shift_labels.view(-1) 
            shift_labels = shift_labels.to(shift_logits.device) 
            loss = loss_fct(shift_logits, shift_labels) 
        
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
        