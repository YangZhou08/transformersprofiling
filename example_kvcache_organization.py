import math 
import time 
import torch 
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union 
import torch.functional as F 

def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True) 
    return x_max / x_max_sum 

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
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p) 
    probs = F.softmax(logits, dim=1)
    return probs 

def sample(probs : torch.Tensor, num_samples: int = 1, random_seed = None):
    if random_seed:
        torch.manual_seed(random_seed)
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    if (idx_next.item() == 0):
        raise RuntimeError
    return idx_next 

@torch.inference_mode()
def Vanilla_Spec_cache(tokenizer, target, target_cache, draft, draft_cache, input_ids, gamma=4, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None):
    # reset cache
    target_cache.reset()
    draft_cache.reset()
    
    ############ Iterative Pre-fill ############
    iter_prefill = math.ceil(input_ids.shape[1] / 100)
    for i in (range(iter_prefill)):
        outputs = target(
            input_ids=input_ids[:, i*100:(i+1)*100],
            past_key_values=target_cache,
            use_cache=True,
        )

        outputs_draft = draft(
            input_ids=input_ids[:, i*100:(i+1)*100],
            past_key_values=draft_cache,
            use_cache=True,
        )

    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0

    next_token = sample(norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    
    if verbose:
        spec_stream(next_token[0], tokenizer, 'cyan')

    n = 0
    time1 = time.time()
    
    ############ Spec Decoding ############
    while n < max_len:
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)
        
        pred_token_idx = next_token

        speculation_probs = []
        generated_ids = []

        for _ in range(gamma):
            outputs = draft(
                input_ids=pred_token_idx,
                past_key_values=draft_cache,
                use_cache=True,
            )

            probs = norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
            pred_token_idx = sample(probs)
            speculation_probs.append(probs[0])
            
            generated_ids.append(pred_token_idx.item())
            draft_count += 1

        # verification
        verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(draft.device)], dim=1)

        with torch.no_grad():
            outputs = target(
                input_ids=verify_tokens,
                past_key_values=target_cache,
                use_cache=True,
            )

        count = 0
        verify_probs = []
    
        for i in range(gamma + 1):
            assert outputs.logits.shape[1] == gamma + 1
            verify_probs.append(norm_logits(outputs.logits[:, i, :], temperature=temperature ,top_k=top_k, top_p=top_p)[0])

        for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs[:-1]):
            r = torch.rand(1, device = draft.device)

            if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                count += 1
                accepted_count += 1
                n += 1
                pred_token_idx = torch.tensor([[i]]).to(draft.device)
                if verbose:
                    spec_stream(i, tokenizer, 'green')

                # if eos
                if tokenizer.eos_token_id == i:
                    draft_count -= gamma - count
                    break

            else:
                resample_count += 1
                n += 1
                pred_token_idx = sample(max_fn(verify_prob-speculation_prob))
                if verbose:
                    spec_stream(pred_token_idx, tokenizer, 'red')
                break

        # if eos
        if tokenizer.eos_token_id == pred_token_idx:
            break

        if count == len(generated_ids):
            target_sample_count += 1
            n += 1
            pred_token_idx = sample(verify_probs[-1])
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'blue')

        next_token = pred_token_idx
        
        if gamma - count > 0:
            draft_cache.seq_len -= (gamma - count) - 1
        else:
            # gamma == count, we need to update the cache for draft
            with torch.no_grad():
                outputs = draft(
                    input_ids=torch.tensor([[generated_ids[-1]]]).to(draft.device),
                    past_key_values=draft_cache,
                    use_cache=True,
                )

        target_cache.seq_len -= (gamma - count)
        assert target_cache.seq_len == draft_cache.seq_len, f"{target_cache.seq_len} != {draft_cache.seq_len}"


    time2 = time.time()
    acceptance_rate = accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma
    if verbose:
        print(f"Use {time2 - time1} sec to generate {n} tokens (now {target_cache.seq_len} tokens), Tokens/s: {n / (time2 - time1)}", flush=True)
        print(f"accepted rate {acceptance_rate}, avg generated tokens {avg_tokens}")

    return acceptance_rate 

class Cache: 
    pass 

class SimpleCache(Cache):
    def __init__(self, model, max_budget=1024) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seq_len = 0
        self.max_budget = max_budget

        self.hidden_size = model.config.hidden_size
        if hasattr(model.config, 'num_key_value_heads'):
            self.num_heads = model.config.num_key_value_heads # for multi-query heads 
        else:
            self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        for i in range(self.layers):
            if hasattr(model, 'gpt_neox'):
                device = model.gpt_neox.layers[i].attention.query_key_value.weight.device
                dtype = model.gpt_neox.layers[i].attention.query_key_value.weight.dtype
            else:
                # device = model.device
                # dtype = torch.float16
                device = model.model.layers[i].self_attn.q_proj.weight.device
                dtype = model.model.layers[i].self_attn.q_proj.weight.dtype
            self.key_cache.append(torch.zeros([1, self.num_heads, self.max_budget, self.head_dim], dtype=dtype).to(device))
            self.value_cache.append(torch.zeros([1, self.num_heads, self.max_budget, self.head_dim], dtype=dtype).to(device))

    def print_status(self):
        print("Cached Size:", self.seq_len, "| Max Budget:", self.max_budget)
    
    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # print(self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + key_states.shape[-2]].shape, self.seq_len)
        self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + key_states.shape[-2]] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + value_states.shape[-2]] = value_states

        key = self.key_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]]
        value = self.value_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-2]

        return key, value
