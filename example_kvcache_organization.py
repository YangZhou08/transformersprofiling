import math 
import time 
import torch 
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union 
import torch.nn.functional as F 
from termcolor import colored 
from datasets import load_dataset 

from transformers import LlamaTokenizer 
from transformers import LlamaConfig, LlamaPreTrainedModel 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 

from transformers.models.llama.modeling_llama import LlamaForCausalLM 
from transformers.models.llama.modeling_llama import LlamaWeirdLargeTest 
from transformers.models.llama.modeling_llama import LargeModelLMHeadModel 

import socket 
from tqdm import tqdm 
import argparse 
import gc 

torch.set_printoptions(threshold=50000) 

hostname = socket.gethostname() 
print("Hostname:", hostname) 

if "lovelace" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/yangzho6/model_checkpoints/" 
    dir_c4llmsynthesized = "/home/yangzho6/c4llm_synthesized/" 
    # dir_c4llmsynthesized = "/home/beidic/yangzho6/c4llm_synthesized/" 
    dir_c4 = "/home/yangzho6/c4_parts/downloads/" 
    # dir_sdata = "/home/yangzho6/slimpajama/SlimPajama-627B/test/chunk1/" 
elif "ada" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/beidic/yangzho6/model_checkpoints/" 
    dir_c4llmsynthesized = "/home/beidic/yangzho6/c4llm_synthesized/" 
else: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    # dir_models = "/home/yangzho6/model_checkpoints/" 
    dir_models = "/fsx-storygen/beidic/yang/model_checkpoints/" 
    # dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
    # dir_sdata = "/fsx-storygen/beidic/yang/c4llm_synthesized/" 
    dir_c4llmsynthesized = "/fsx-storygen/beidic/yang/c4llm_synthesized/" 
    dir_c4 = "/fsx-storygen/beidic/yang/c4_parts/downloads/" 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 

def spec_stream(pred_token_idx, tokenizer, color='blue'): 
    # print("pred_token_idx: ", pred_token_idx) 
    pred_token_idx = pred_token_idx.squeeze(0) 
    decoded_token = tokenizer.decode(
            pred_token_idx,
            # skip_special_tokens=True,
            skip_special_tokens = False, 
            clean_up_tokenization_spaces=True,
            # spaces_between_special_tokens=False,
        ) 

    decoded_token = decoded_token.replace("<0x0A>", "\n")

    print(colored(decoded_token, color), flush=True, end=" ")

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
    # if (idx_next.item() == 0):
        # raise RuntimeError 
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

@torch.inference_mode()
def Vanilla_specu_dectesting2(tokenizer, train_dataloader, target_lmhead, model, gamma=4, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None): 
    previous_addone = None 
    
    for batch in train_dataloader: 
        input_ids = batch["input_ids"].to(model.device) 
        attention_mask = batch["attention_mask"].to(model.device) 
        input_ids = torch.cat([torch.tensor([[tokenizer.eos_token_id]]).to(model.device), input_ids], dim = 1) # pad one more extra 
        attention_mask = torch.cat([torch.zeros([1, 1]).to(model.device), attention_mask], dim = 1) # pad one more extra 
        
        # now the input_ids is (batch_size, seq_len + 1) and attention_mask is (batch_size, seq_len + 1) 
        # reset cache 
        
        resample_count = 0
        accepted_count = 0
        target_sample_count = 0
        draft_count = 0

        n = 0
        time1 = time.time()
        
        ############ Spec Decoding ############ 

        speculation_probs = []
        generated_ids = [] 
        original_large_model_hidden_states = None 

        for _ in range(gamma): 
            model_inputs = model.prepare_inputs_for_generation(input_ids, past_key_values = None, input_embeds = None, attention_mask = attention_mask) 
            print("model_inputs[large_input_ids]: {}".format(model_inputs["large_input_ids"].shape)) 
            print("model_inputs[attention_mask]: {}".format(model_inputs["attention_mask"].shape)) 
            outputs = model.forward_generate(
                **model_inputs, 
                return_dict = True, 
                output_attentions = False, 
                output_hidden_states = False, 
                output_large_model_last_hidden_states = True, 
            ) 
            
            # print("outputs.logits.shape: {}".format(outputs.logits.shape)) 

            probs = norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
            pred_token_idx = sample(probs)
            speculation_probs.append(probs[0]) 
            original_large_model_hidden_states = outputs.last_hidden_states 
            assert original_large_model_hidden_states is not None 
            
            generated_ids.append(pred_token_idx.item())
            draft_count += 1 

        # verification
        verify_tokens = torch.cat([input_ids, torch.LongTensor([generated_ids]).to(model.device)], dim = 1) 
        large_model_start_verifying_index = input_ids.shape[1] - 1 

        with torch.no_grad():
            # outputs = target(
            #     input_ids=verify_tokens,
            #     past_key_values = None, 
            #     use_cache = False, 
            # ) 
            target_output_logits = target_lmhead(original_large_model_hidden_states) 

        count = 0
        verify_probs = []

        for i in range(gamma + 1):
            # assert outputs.logits.shape[1] == gamma + 1 
            idx = i + large_model_start_verifying_index 
            verify_probs.append(norm_logits(outputs.logits[:, idx, :], temperature=temperature ,top_k=top_k, top_p=top_p)[0]) 

        for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs[:-1]):
            r = torch.rand(1, device = model.device) 

            if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                count += 1
                accepted_count += 1
                n += 1
                pred_token_idx = torch.tensor([[i]]).to(model.device) 
                if verbose:
                    # spec_stream(i, tokenizer, 'green') 
                    spec_stream(pred_token_idx, tokenizer, "green") 

                # if eos
                if tokenizer.eos_token_id == i:
                    draft_count -= gamma - count
                    break

            else:
                resample_count += 1
                n += 1 
                verify_prob = verify_prob.unsqueeze(0) 
                speculation_prob = speculation_prob.unsqueeze(0) 
                pred_token_idx = sample(max_fn(verify_prob-speculation_prob))
                if verbose:
                    spec_stream(pred_token_idx, tokenizer, 'red')
                break 
            
            # print("speculation_probs: {}, verify_probs: {}".format(speculation_prob.shape, verify_prob.shape)) 

        if count == len(generated_ids):
            target_sample_count += 1
            n += 1
            pred_token_idx = sample(verify_probs[-1])
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'blue') 


        time2 = time.time()
        acceptance_rate = accepted_count / draft_count
        avg_tokens = accepted_count / draft_count * gamma
        if verbose:
            # print(f"Use {time2 - time1} sec to generate {n} tokens (now {target_cache.seq_len} tokens), Tokens/s: {n / (time2 - time1)}", flush=True) 
            print(f"accepted rate {acceptance_rate}, avg generated tokens {avg_tokens}")

    return acceptance_rate, draft_count 

@torch.inference_mode()
def Vanilla_specu_dectesting3(tokenizer, 
                              target_lmhead, 
                              model, 
                              input_ids, 
                              attention_mask, 
                              gamma=4, 
                              max_len=256, 
                              top_k=-1, 
                              top_p=0.9, 
                              temperature=0.6, 
                              verbose=False, 
                              file_path=None, 
                              target_model = None, 
): 
    # reset cache 
    '''
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
    ''' 
    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0

    n = 0
    time1 = time.time()
    
    ############ Spec Decoding ############ 

    speculation_probs = []
    generated_ids = []

    target_model_last_hidden_states = None 
    for _ in range(gamma): 
        model_inputs = model.prepare_inputs_for_generation(input_ids, past_key_values = None, input_embeds = None, attention_mask = attention_mask) 
        print("model_inputs[large_input_ids]: {}".format(model_inputs["large_input_ids"].shape)) 
        print("model_inputs[attention_mask]: {}".format(model_inputs["attention_mask"].shape)) 
        outputs = model.forward_generate(
            **model_inputs, 
            return_dict = True, 
            output_attentions = False, 
            output_hidden_states = False, 
            output_large_model_last_hidden_states = True, 
        ) 
        # print("outputs.logits.shape: {}".format(outputs.logits.shape)) 

        probs = norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
        pred_token_idx = sample(probs)
        speculation_probs.append(probs[0]) 
        target_model_last_hidden_states = outputs.last_hidden_states # target_model_last_hidden_states is in shape (batch_size, 1, hidden_size) 
        assert target_model_last_hidden_states is not None 
        
        generated_ids.append(pred_token_idx.item())
        draft_count += 1 

    with torch.no_grad(): 
        target_model_logits = target_lmhead(target_model_last_hidden_states) # shape (batch_size, 1, vocab_size) 
        target_model_logits = target_model_logits[:, -1, :] # shape (batch_size, vocab_size) 
        
        outputs2 = target_model( 
            input_ids = input_ids, 
            past_key_values = None, 
            use_cache = False, 
            output_hidden_states = True
        ) 
        
        # assert torch.allclose(target_model_logits, outputs2.logits[:, -1, :].to(torch.bfloat16))  # check if the two logits are the same 

    count = 0
    verify_probs = []

    verify_probs.append(norm_logits(target_model_logits, temperature = temperature, top_k = top_k, top_p = top_p)[0]) 
    # assert torch.allclose(verify_probs[0], norm_logits(outputs2.logits[:, -1, :].to(torch.bfloat16), temperature = temperature, top_k = top_k, top_p = top_p)[0]) 

    for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs): 
        r = torch.rand(1, device = model.device) 

        if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
            count += 1
            accepted_count += 1
            n += 1
            pred_token_idx = torch.tensor([[i]]).to(model.device) 
            if verbose:
                # spec_stream(i, tokenizer, 'green') 
                spec_stream(pred_token_idx, tokenizer, "green") 

            # if eos
            if tokenizer.eos_token_id == i:
                draft_count -= gamma - count
                break

        else:
            resample_count += 1
            n += 1 
            verify_prob = verify_prob.unsqueeze(0) 
            speculation_prob = speculation_prob.unsqueeze(0) 
            pred_token_idx = sample(max_fn(verify_prob-speculation_prob))
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'red')
            break 
        
        # print("speculation_probs: {}, verify_probs: {}".format(speculation_prob.shape, verify_prob.shape)) 

    time2 = time.time()
    acceptance_rate = accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma 
    
    if verbose:
        # print(f"Use {time2 - time1} sec to generate {n} tokens (now {target_cache.seq_len} tokens), Tokens/s: {n / (time2 - time1)}", flush=True) 
        print(f"accepted rate {acceptance_rate}, avg generated tokens {avg_tokens}")

    return acceptance_rate, draft_count 

@torch.inference_mode()
def Vanilla_spec_dectesting(tokenizer, target, model, input_ids, attention_mask, gamma=4, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None): 
    # reset cache 
    '''
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
    ''' 
    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0

    n = 0
    time1 = time.time()
    
    ############ Spec Decoding ############ 

    speculation_probs = []
    generated_ids = []

    for _ in range(gamma): 
        model_inputs = model.prepare_inputs_for_generation(input_ids, past_key_values = None, input_embeds = None, attention_mask = attention_mask) 
        print("model_inputs[large_input_ids]: {}".format(model_inputs["large_input_ids"].shape)) 
        print("model_inputs[attention_mask]: {}".format(model_inputs["attention_mask"].shape)) 
        outputs = model.forward_generate(
            **model_inputs, 
            return_dict = True, 
            output_attentions = False, 
            output_hidden_states = False, 
        ) 
        # print("outputs.logits.shape: {}".format(outputs.logits.shape)) 

        probs = norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
        pred_token_idx = sample(probs)
        speculation_probs.append(probs[0])
        
        generated_ids.append(pred_token_idx.item())
        draft_count += 1 

    # verification
    verify_tokens = torch.cat([input_ids, torch.LongTensor([generated_ids]).to(model.device)], dim = 1) 
    large_model_start_verifying_index = input_ids.shape[1] - 1 

    with torch.no_grad():
        outputs = target(
            input_ids=verify_tokens,
            past_key_values = None, 
            use_cache = False, 
        ) 

    count = 0
    verify_probs = []

    for i in range(gamma + 1):
        # assert outputs.logits.shape[1] == gamma + 1 
        idx = i + large_model_start_verifying_index 
        verify_probs.append(norm_logits(outputs.logits[:, idx, :], temperature=temperature ,top_k=top_k, top_p=top_p)[0]) 

    for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs[:-1]):
        r = torch.rand(1, device = model.device) 

        if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
            count += 1
            accepted_count += 1
            n += 1
            pred_token_idx = torch.tensor([[i]]).to(model.device) 
            if verbose:
                # spec_stream(i, tokenizer, 'green') 
                spec_stream(pred_token_idx, tokenizer, "green") 

            # if eos
            if tokenizer.eos_token_id == i:
                draft_count -= gamma - count
                break

        else:
            resample_count += 1
            n += 1 
            verify_prob = verify_prob.unsqueeze(0) 
            speculation_prob = speculation_prob.unsqueeze(0) 
            pred_token_idx = sample(max_fn(verify_prob-speculation_prob))
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'red')
            break 
        
        # print("speculation_probs: {}, verify_probs: {}".format(speculation_prob.shape, verify_prob.shape)) 

    if count == len(generated_ids):
        target_sample_count += 1
        n += 1
        pred_token_idx = sample(verify_probs[-1])
        if verbose:
            spec_stream(pred_token_idx, tokenizer, 'blue') 


    time2 = time.time()
    acceptance_rate = accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma
    if verbose:
        # print(f"Use {time2 - time1} sec to generate {n} tokens (now {target_cache.seq_len} tokens), Tokens/s: {n / (time2 - time1)}", flush=True) 
        print(f"accepted rate {acceptance_rate}, avg generated tokens {avg_tokens}")

    return acceptance_rate, draft_count 

@torch.inference_mode()
def Vanilla_spec_decnokv22(tokenizer, target, draft, input_ids, gamma=4, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None, attention_mask = None): 
    '''
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
    ''' 
    
    print("input_ids shape", input_ids.shape) 
    spec_stream(input_ids, tokenizer, 'black') 
    for k in range(max_len): 
        outputs = target(
            input_ids = input_ids, 
            past_key_values = None, 
            use_cache = False, 
            attention_mask = attention_mask, 
        ) 

        resample_count = 0
        accepted_count = 0
        target_sample_count = 0
        draft_count = 0 

        next_token = sample(norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)) # predicting for the next token 
        # next_token = torch.argmax(outputs.logits[:, -1, :], dim = -1).unsqueeze(0) 
        # print(next_token) 
        # print("next_token shape: ", next_token.shape) 
        spec_stream(next_token[0], tokenizer, 'cyan') 
        input_ids = torch.cat([input_ids, next_token], dim = 1) 
        attention_mask = torch.cat([attention_mask, torch.ones([1, 1]).to(draft.device)], dim = 1) 
    
    if verbose: 
        print("\n") 
        spec_stream(input_ids, tokenizer, "black") 
        spec_stream(next_token[0], tokenizer, 'cyan') 
    # print("input_ids shape: {}".format(input_ids.shape)) 

    n = 0
    time1 = time.time() 
    small_model_input_full_context = input_ids 
    
    ############ Spec Decoding ############
    while n < max_len:
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)
        
        small_model_input_full_context = torch.cat([small_model_input_full_context, next_token], dim = 1).to(draft.device) 
        # print("pred_token_idx: {}".format(pred_token_idx.shape)) 

        speculation_probs = []
        generated_ids = []

        for _ in range(gamma): 
            if len(generated_ids) > 0: 
                outputs = draft(
                    # input_ids = small_model_input_full_context, 
                    input_ids = torch.cat([small_model_input_full_context, torch.LongTensor([generated_ids]).to(draft.device)], dim = 1), 
                    # past_key_values=draft_cache, 
                    # use_cache=True, 
                    past_key_values = None, 
                    use_cache = False, 
                    # attention_mask = torch.cat([attention_mask, torch.ones([1, len(generated_ids)]).to(draft.device)], dim = 1), 
                ) 
            else: 
                outputs = draft(
                    input_ids = small_model_input_full_context, 
                    past_key_values = None, 
                    use_cache = False, 
                ) 
            
            probs = norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
            pred_token_idx = sample(probs)
            speculation_probs.append(probs[0]) 
            # print("pred_token_idx: {} speculation_probs: {}".format(pred_token_idx, speculation_probs[0].shape)) 
            
            generated_ids.append(pred_token_idx.item())
            draft_count += 1 

        # verification
        # verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(draft.device)], dim=1) 
        # verify_tokens = torch.cat([pred_token_idx, torch.LongTensor([generated_ids]).to(draft.device)], dim = 1) 
        verify_tokens = torch.cat([small_model_input_full_context, torch.LongTensor([generated_ids]).to(draft.device)], dim = 1) 
        large_model_start_verifying_index = small_model_input_full_context.shape[1] - 1 
        # print("verify_tokens: {}".format(verify_tokens.shape[1])) 
        # print("large_model_start_verifying_index: {}".format(large_model_start_verifying_index)) 

        with torch.no_grad():
            outputs = target(
                input_ids=verify_tokens,
                past_key_values = None, 
                use_cache = False, 
            ) 

        count = 0
        verify_probs = []
    
        for i in range(gamma + 1): 
            # assert outputs.logits.shape[1] == gamma + 1 
            idx = i + large_model_start_verifying_index 
            verify_probs.append(norm_logits(outputs.logits[:, idx, :], temperature=temperature ,top_k=top_k, top_p=top_p)[0]) 
        # verify_probs.append(norm_logits(outputs.logits[:, -2, :], temperature = temperature, top_k = top_k, top_p = top_p)[0]) 
        # print("length of speculation_probs: {} length of verify_probs: {}".format(len(speculation_probs), len(verify_probs))) 
        
        # print("verify_probs: {}".format(verify_probs[0].shape)) 
        for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs[:-1]):
            r = torch.rand(1, device = draft.device)

            if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                count += 1
                accepted_count += 1
                n += 1
                pred_token_idx = torch.tensor([[i]]).to(draft.device)
                if verbose:
                    spec_stream(pred_token_idx, tokenizer, 'green') 

                # if eos
                if tokenizer.eos_token_id == i:
                    draft_count -= gamma - count
                    break

            else:
                resample_count += 1
                n += 1 
                # print("verify_prob: {}".format(verify_prob.shape)) 
                # print("speculative_prob: {}".format(speculation_prob.shape)) 
                verify_prob = verify_prob.unsqueeze(0) 
                speculation_prob = speculation_prob.unsqueeze(0) 
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

    time2 = time.time()
    acceptance_rate = accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma
    if verbose:
        # print(f"Use {time2 - time1} sec to generate {n} tokens (now {target_cache.seq_len} tokens), Tokens/s: {n / (time2 - time1)}", flush=True) 
        print(f"accepted rate {acceptance_rate}, avg generated tokens {avg_tokens}") 

    return acceptance_rate, draft_count 

@torch.inference_mode()
def Vanilla_Spec_nokvcache(tokenizer, target, draft, input_ids, gamma=4, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None, attention_mask = None): 
    '''
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
    ''' 
    outputs = target(
        input_ids = input_ids, 
        past_key_values = None, 
        use_cache = False, 
        attention_mask = attention_mask, 
    ) 
    attention_mask = torch.cat([attention_mask, torch.ones([1, 1]).to(draft.device)], dim = 1) 

    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0 

    next_token = sample(norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)) # predicting for the next token 
    
    if verbose: 
        print("\n") 
        spec_stream(input_ids, tokenizer, "black") 
        spec_stream(next_token[0], tokenizer, 'cyan') 
    # print("input_ids shape: {}".format(input_ids.shape)) 

    n = 0
    time1 = time.time() 
    
    ############ Spec Decoding ############
    while n < max_len:
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)
        
        pred_token_idx = next_token 
        small_model_input_full_context = torch.cat([input_ids, next_token], dim = 1).to(draft.device) 
        # print("pred_token_idx: {}".format(pred_token_idx.shape)) 

        speculation_probs = []
        generated_ids = []

        for _ in range(gamma):
            outputs = draft(
                input_ids = small_model_input_full_context, 
                # past_key_values=draft_cache, 
                # use_cache=True, 
                past_key_values = None, 
                use_cache = False, 
                attention_mask = attention_mask, 
            ) 

            probs = norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
            pred_token_idx = sample(probs)
            speculation_probs.append(probs[0]) 
            # print("pred_token_idx: {} speculation_probs: {}".format(pred_token_idx, speculation_probs[0].shape)) 
            
            generated_ids.append(pred_token_idx.item())
            draft_count += 1 

        # verification
        # verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(draft.device)], dim=1) 
        # verify_tokens = torch.cat([pred_token_idx, torch.LongTensor([generated_ids]).to(draft.device)], dim = 1) 
        verify_tokens = torch.cat([small_model_input_full_context, torch.LongTensor([generated_ids]).to(draft.device)], dim = 1) 
        large_model_start_verifying_index = small_model_input_full_context.shape[1] - 1 
        # print("verify_tokens: {}".format(verify_tokens.shape[1])) 
        # print("large_model_start_verifying_index: {}".format(large_model_start_verifying_index)) 

        with torch.no_grad():
            outputs = target(
                input_ids=verify_tokens,
                past_key_values = None, 
                use_cache = False, 
                attention_mask = torch.cat([attention_mask, torch.ones([1, len(generated_ids)]).to(draft.device)], dim = 1), 
            ) 

        count = 0
        verify_probs = []
    
        for i in range(gamma + 1): 
            # assert outputs.logits.shape[1] == gamma + 1 
            idx = i + large_model_start_verifying_index 
            verify_probs.append(norm_logits(outputs.logits[:, idx, :], temperature=temperature ,top_k=top_k, top_p=top_p)[0]) 
        # verify_probs.append(norm_logits(outputs.logits[:, -2, :], temperature = temperature, top_k = top_k, top_p = top_p)[0]) 
        # print("length of speculation_probs: {} length of verify_probs: {}".format(len(speculation_probs), len(verify_probs))) 
        
        # print("verify_probs: {}".format(verify_probs[0].shape)) 
        for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs[:-1]):
            r = torch.rand(1, device = draft.device)

            if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                count += 1
                accepted_count += 1
                n += 1
                pred_token_idx = torch.tensor([[i]]).to(draft.device)
                if verbose:
                    spec_stream(pred_token_idx, tokenizer, 'green') 

                # if eos
                if tokenizer.eos_token_id == i:
                    draft_count -= gamma - count
                    break

            else:
                resample_count += 1
                n += 1 
                # print("verify_prob: {}".format(verify_prob.shape)) 
                # print("speculative_prob: {}".format(speculation_prob.shape)) 
                verify_prob = verify_prob.unsqueeze(0) 
                speculation_prob = speculation_prob.unsqueeze(0) 
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

    time2 = time.time()
    acceptance_rate = accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma
    if verbose:
        # print(f"Use {time2 - time1} sec to generate {n} tokens (now {target_cache.seq_len} tokens), Tokens/s: {n / (time2 - time1)}", flush=True) 
        print(f"accepted rate {acceptance_rate}, avg generated tokens {avg_tokens}") 

    return acceptance_rate, draft_count 

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

@torch.inference_mode()
def Vanilla_Spec_cache(tokenizer, target, target_cache, draft, draft_cache, input_ids, gamma=4, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None):
    # reset cache
    target_cache.reset()
    draft_cache.reset()
    '''
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
    ''' 
    outputs = target( 
        input_ids = input_ids, 
        past_key_values = target_cache, 
        use_cache = True, 
    ) 
    
    outputs_draft = draft(
        input_ids = input_ids, 
        past_key_values = draft_cache, 
        use_cache = True, 
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
    
def get_dataset(datasetname = None, tokenizer = None, max_length = None, limit = None): 
    
    def encode_with_truncation(examples): 
        # tokdictionary = tokenizer(examples['text'][100000 : 100000 + 3000], padding = "max_length", max_length = 260, 
        #                  return_attention_mask = True, return_tensors = "pt", truncation = True, 
        #                  add_special_tokens = True) 
        tokdictionary = tokenizer(examples['text'], padding = "max_length", max_length = max_length, 
                                return_attention_mask = True, return_tensors = "pt", truncation = True, 
                                add_special_tokens = True) 
        newdictionary = {} 
        newdictionary['input_ids'] = tokdictionary['input_ids'].squeeze(0) 
        newdictionary['attention_mask'] = tokdictionary['attention_mask'].squeeze(0) 
        return newdictionary 
    
    def encode_with_truncationspecialized(examples): 
        # tokdictionary = tokenizer(examples['text'][100000 : 100000 + 3000], padding = "max_length", max_length = max_length, 
        #                           eturn_attention_mask = True, return_tensors = "pt", truncation = True, 
        #                           add_special_tokens = True) 
        tokdictionary = tokenizer(examples['text'][50000 : 50000 + 3000], padding = "max_length", max_length = max_length, 
                                  return_attention_mask = True, return_tensors = "pt", truncation = True, 
                                  add_special_tokens = True) 
        # tokdictionary = tokenizer(examples['text'], padding = "max_length", max_length = 260, 
        #                          return_attention_mask = True, return_tensors = "pt", truncation = True, 
        #                          add_special_tokens = True) 
        newdictionary = {} 
        newdictionary['input_ids'] = tokdictionary['input_ids'].squeeze(0) 
        newdictionary['attention_mask'] = tokdictionary['attention_mask'].squeeze(0) 
        return newdictionary 
    
    if datasetname == "c4": 
        dfiles = [] 
        filename = "c4_file1.json" 
        # filename = "c4_file15.json" 
        dfiles.append(dir_c4 + filename) 
        datasetnew = load_dataset("json", data_files = dfiles, split = "train[:{}]".format(limit)) 
        # datasetnew = load_dataset("json", data_files = dfiles, split = "train[:10000]") 
        
        datasetnew = datasetnew.map(encode_with_truncation, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask", "text"]) 
    elif datasetname == "pg19": 
        # TODO: loading another dataset 
        # datasetnew = load_dataset('emozilla/pg19', split = "test") 
        datasetnew = load_dataset('emozilla/pg19', split = "train[:{}]".format(limit)) 
        # datasetnew = load_dataset('emozilla/pg19', split = "train[:1000]") 
        
        datasetnew = datasetnew.map(encode_with_truncationspecialized, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask", "text"]) 
    elif datasetname == "openwebtext": 
        # datasetnew = load_dataset("Skylion007/openwebtext", split = "train[:10000]") 
        datasetnew = load_dataset("Skylion007/openwebtext", split = "train[:{}]".format(limit)) 
        
        datasetnew = datasetnew.map(encode_with_truncation, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask", "text"]) 
    
    return datasetnew 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description = "Speculative Acceptance Rate") 
    parser.add_argument("--loading_from_checkpoint", type = str, default = None) 
    parser.add_argument("--experiment_setting", type = str, default = None) 
    parser.add_argument("--kernel_size", type = int, default = None) 
    parser.add_argument("--dataset_name", type = str, default = None) 
    parser.add_argument("--use_small_draft", action = "store_true") 
    parser.add_argument("--double_decking", action = "store_true") 
    
    args = parser.parse_args() 
    
    max_length_table = {1 : 64, 2 : 63, 3 : 64, 4 : 65, 5 : 66, 6: 67, 7 : 64, 10 : 71} # the pattern is as follows: the first position is corresponding to the sequence_length that mod kernel_size by 1, for example, 63 % 2 = 1, etc. 
    
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
    if tokenizer.pad_token is not None: 
        print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
    else: 
        tokenizer.pad_token = tokenizer.eos_token 
        print("We now use eos_token as pad token") 
    tokenizer.padding_side = "left" 
    
    if not args.double_decking: 
        # large model 
        target_largemodel = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
        # target_largemodel = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
    else: 
        target_largemodel = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
        target_largemodellmhead = LargeModelLMHeadModel(target_largemodel.lm_head).to(torch.bfloat16).to(torch_device) 
        # del target_largemodel 
        # gc.collect() 
    # target_largemodel = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
    if not args.use_small_draft: 
        large_model = LlamaWeirdLargeTest.from_pretrained(args.loading_from_checkpoint).to(torch.bfloat16).to(torch_device) 
        large_model.set_sliding_window_length(args.kernel_size) 
        large_model.addonsmallmodel.set_criticalpath(hostname = hostname) 
        large_model.set_msece_loss(use_mse_loss = False, ce_loss_only = True) 
        large_model.to(torch.bfloat16).to(torch_device) 
        large_model.set_inference_setting(args.experiment_setting) 
        large_model.set_walpha(0.5) 
        large_model.set_slidingwindowlength(sliding_window_length = args.kernel_size, addonmodel_start = args.kernel_size + 1) 
        large_model.set_tokenizer_bos_id(bos_id = tokenizer.bos_token_id, pad_id = tokenizer.pad_token_id) 
        large_model.set_cosinesimilarity(False) 
        
        large_model.config.pad_token_id = tokenizer.pad_token_id 
        large_model.addonsmallmodel.config.pad_token_id = tokenizer.pad_token_id 
        # small_model.config.pad_token_id = tokenizer.pad_token_id 
        large_model.model.eval() 
        large_model.addonsmallmodel.eval() 
    
    small_model = None 
    # small model 
    if args.use_small_draft and args.loading_from_checkpoint is not None: 
        small_model = LlamaForCausalLM.from_pretrained(args.loading_from_checkpoint).to(torch.bfloat16).to(torch_device) 
    elif args.use_small_draft: 
        # small_model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
        small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-68m", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
    
    acceptanceratelist = {"c4": [], "pg19": [], "openwebtext": []} 
    
    iterationscounts = args.kernel_size if not args.use_small_draft else 1 
    
    # datasetlist = ["c4", "pg19", "openwebtext"] 
    # datasetlist = ["pg19", "openwebtext"] 
    datasetlist = ["c4"] 
    for datasetname in datasetlist: 
        for i in range(iterationscounts): # we need a forloop 
            if args.use_small_draft: 
                datasetnew = get_dataset(datasetname = datasetname, tokenizer = tokenizer, max_length = 64, limit = 10000) 
            else: 
                datasetnew = get_dataset(datasetname = datasetname, tokenizer = tokenizer, max_length = max_length_table[args.kernel_size] + i, limit = 10000) # i 0 means the first position, i 1 means the second position, etc. 
            
            # dataloader = torch.utils.data.DataLoader(datasetnew, batch_size = 32, shuffle = False) 
            dataloader = torch.utils.data.DataLoader(datasetnew, batch_size = 1, shuffle = False) 
            
            globalacceptancerate = 0 
            globaldraftcount = 0 
            
            for batch in tqdm(dataloader): 
                input_ids = batch["input_ids"].to(torch_device) 
                attention_mask = batch["attention_mask"].to(torch_device) 
                if args.use_small_draft: 
                    acceptancer, draftcount = Vanilla_Spec_nokvcache(tokenizer, 
                                        target_largemodel, 
                                        small_model, 
                                        input_ids, 
                                        gamma = 1, 
                                        max_len = 1, 
                                        verbose = True, 
                                        attention_mask = attention_mask, 
                    ) 
                    '''
                    acceptancer, draftcount = Vanilla_spec_decnokv22(tokenizer, 
                                                target_largemodel, 
                                                small_model, 
                                                input_ids, 
                                                gamma = 1, 
                                                max_len = 128, 
                                                verbose = True, 
                                                temperature = 0.6, 
                                                attention_mask = attention_mask, 
                                                )  
                    print("input_ids: ", input_ids) 
                    ''' 
                else: 
                    if not args.double_decking: 
                        large_model.resetgenerationcount() 
                        acceptancer, draftcount = Vanilla_spec_dectesting(tokenizer, 
                                            target_largemodel, 
                                            large_model, 
                                            input_ids, 
                                            attention_mask, 
                                            gamma = 1, 
                                            max_len = 1, 
                                            verbose = True, 
                                            ) 
                    else: 
                        # print("using specudectesting version3") 
                        large_model.resetgenerationcount() 
                        acceptancer, draftcount = Vanilla_specu_dectesting3(tokenizer, 
                                            target_largemodellmhead, 
                                            large_model, 
                                            input_ids, 
                                            attention_mask, 
                                            gamma = 1, 
                                            max_len = 1, 
                                            verbose = True, 
                                            target_model = target_largemodel, 
                                            ) 
                                    
                globalacceptancerate += (acceptancer * draftcount) 
                globaldraftcount += draftcount 
            
            # print("global acceptance rate: ", globalacceptancerate / globaldraftcount) 
            print("position {} acceptancerate: {}".format(i + 1, globalacceptancerate / globaldraftcount)) 
            acceptanceratelist[datasetname].append(globalacceptancerate / globaldraftcount) 
        
# for datasetname in acceptanceratelist.keys(): 
for datasetname in datasetlist: 
    print("datasetname: {}".format(datasetname)) 
    for i in range(iterationscounts): 
        print("position {} acceptance rate: {}".format(i + 1, acceptanceratelist[datasetname][i])) 
    print() 
