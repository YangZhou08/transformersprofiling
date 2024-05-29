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

# from transformers.griffin.llama9 import LlamaForCausalLM 
from transformers import LlamaForCausalLM 
from transformers.griffin.llama9 import get_llama_griffin 

import socket 
from tqdm import tqdm 
import argparse 
import gc 
from time import sleep 

torch.set_printoptions(threshold=50000) 

hostname = socket.gethostname() 
print("Hostname:", hostname) 

from scipy.optimize import fsolve 

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
    # pred_token_idx = pred_token_idx.squeeze(0) 
    if isinstance(pred_token_idx, torch.Tensor): 
        if len(pred_token_idx.shape) > 1: 
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
    # x_max_sum = torch.sum(x_max, dim=1, keepdim=True) 
    x_max_sum = torch.sum(x_max, dim = -1, keepdim = True) 
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

def set_inference_mode(model, mode): 
    for layer in model.model.layers: 
        layer.mlp.set_inference_mode(mode) 

@torch.inference_mode() 
def Vanilla_Spec_cache(tokenizer, model, cache, input_ids, gamma = 4, max_len = 256, top_k = -1, top_p = 0.9, temperature = 0.6, verbose = False, file_path = None, attention_mask = None): 
    # reset cache 
    cache = None 
    
    set_inference_mode(model, "full") 
    # newinputids = input_ids 
    n = 0 
    '''
    while n < max_len: 
        outputs = model(
            input_ids = newinputids, 
            # attention_mask = attention_mask, 
            past_key_values = cache, # using large model's cache 
            use_cache = True, 
        ) 
        cache = outputs.past_key_values 
        newinputids = sample(norm_logits(outputs.logits[:, -1, :], temperature = temperature, top_k = top_k, top_p = top_p)) 
        print(tokenizer.decode(newinputids[0]), end = " ") 
        cache = outputs.past_key_values 
        n += 1 
    exit(0) 
    ''' 
    outputs = model(
        input_ids = input_ids, 
        attention_mask = attention_mask, 
        past_key_values = cache, # using large model's cache 
        use_cache = True, 
    ) 
    cache = outputs.past_key_values 
    # attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(attention_mask.device)], dim = 1) 
    
    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0
    
    next_token = sample(norm_logits(outputs.logits[:, -1, :], temperature = temperature, top_k = top_k, top_p = top_p)) 
    
    if verbose: 
        spec_stream(next_token[0], tokenizer, 'cyan') 
    
    n = 0 
    while n < max_len:
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)
        
        pred_token_idx = next_token

        speculation_probs = []
        generated_ids = []
        
        # disposableattentionmask = attention_mask.clone() 

        set_inference_mode(model, "partial") 
        for _ in range(gamma):
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=cache, 
                use_cache=True,
                # attention_mask = disposableattentionmask, 
            ) 
            cache = outputs.past_key_values 
            # disposableattentionmask = torch.cat([disposableattentionmask, torch.ones(1, 1).to(disposableattentionmask.device)], dim = 1) 

            probs = norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
            pred_token_idx = sample(probs)
            speculation_probs.append(probs[0])
            
            generated_ids.append(pred_token_idx.item())
            draft_count += 1
        
        new_cache = [] 
        for layer in cache: 
            new_layer = [] 
            # print("len(layer), ", len(layer)) 
            # print("kv shape, ", layer[0].shape) 
            for kv in layer: 
                new_layer.append(kv[:, :, :-gamma, :].contiguous()) 
                # new_layer.append(v[:, :-gamma, :].contiguous()) 
            new_layer = tuple(new_layer) 
            new_cache.append(new_layer) 
        new_cache = tuple(new_cache) 
        cache = new_cache 
        
        # verification
        verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(model.device)], dim = 1) 
        # print("verify_tokens shape: ", verify_tokens.shape) 
        
        set_inference_mode(model, "full") 
        with torch.no_grad():
            outputs = model(
                input_ids=verify_tokens,
                past_key_values=cache, 
                use_cache=True,
                # attention_mask = disposableattentionmask, 
            ) 
        cache = outputs.past_key_values 
        
        count = 0
        verify_probs = []
        
        for i in range(gamma + 1):
            assert outputs.logits.shape[1] == gamma + 1
            verify_probs.append(norm_logits(outputs.logits[:, i, :], temperature=temperature ,top_k=top_k, top_p=top_p)[0]) 
            # print(tokenizer.decode(sample(verify_probs[-1])), end = " ") 
        
        for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs[:-1]):
            r = torch.rand(1, device = model.device) 

            if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                count += 1
                accepted_count += 1
                n += 1
                pred_token_idx = torch.tensor([[i]]).to(model.device) 
                if verbose:
                    spec_stream(i, tokenizer, 'green')

                # if eos
                if tokenizer.eos_token_id == i:
                    draft_count -= gamma - count
                    break

            else:
                resample_count += 1
                n += 1
                # print("shape of verify_prob {} shape of speculation_prob {}".format(verify_prob.shape, speculation_prob.shape)) 
                # verify_prob = verify_prob.unsqueeze(0) 
                # speculation_prob = speculation_prob.unsqueeze(0) 
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
            # roll back the large model's cache 
            new_cache = [] 
            for layer in cache: 
                new_layer = [] 
                # print("len(layer), ", len(layer)) 
                for kv in layer: 
                    new_layer.append(kv[:, :, :-(gamma - count), :].contiguous()) 
                    # new_layer.append(v[:, :-gamma+count-1, :].contiguous()) 
                # print("length of kv cache {} count {} expected {}".format(new_layer[0].shape[2], count, input_ids.shape[1] + n)) 
                assert new_layer[0].shape[2] == input_ids.shape[1] + n 
                new_layer = tuple(new_layer) 
                new_cache.append(new_layer) 
            new_cache = tuple(new_cache) 
            cache = new_cache 
        else: 
            # print("length of kv cache: ", new_layer[0].shape[2]) 
            # assert new_layer[0].shape[2] == input_ids.shape[1] + n 
            assert cache[0][0].shape[2] == input_ids.shape[1] + n 
        
    acceptance_rate = accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma
    if verbose:
        print(f"accepted rate {acceptance_rate}, avg generated tokens {avg_tokens}")

    # return acceptance_rate 
    return acceptance_rate, draft_count 

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
        # filename = "c4_file1.json" 
        filename = "c4_file15.json" 
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
    elif datasetname == "gsm8k": 
        datasetnew = load_dataset("gsm8k", "main", split = "train[:{}]".format(limit)) 
    
    return datasetnew 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description = "Speculative Acceptance Rate") 
    parser.add_argument("--usegriffin", action = "store_true") 
    parser.add_argument("--datasetname", choices = ["c4", "gsm8k"], default = "gsm8k") 
    
    args = parser.parse_args() 
    print(args) 
    
    # tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code = True) 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code = True) 
    if tokenizer.pad_token is not None: 
        print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
    else: 
        tokenizer.pad_token = tokenizer.eos_token 
        print("We now use eos_token as pad token") 
    tokenizer.padding_side = "left" 
    # model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models, device_map = torch_device, torch_dtype = torch.bfloat16) 
    model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir = dir_models, device_map = torch_device, torch_dtype = torch.bfloat16) 
    
    schedule_k = [0.5 for _ in range(model.config.num_hidden_layers)] 
    model.config.mode = "gen"
    model.config.selection_method = "topk" 
    
    model = get_llama_griffin(model, schedule_k, notcats = args.usegriffin) 
    model.config.pad_token_id = tokenizer.pad_token_id 
    model.eval() 
    
    if args.datasetname == "c4": 
        datasetnew = get_dataset("c4", tokenizer, 128, 1000) 
    else: 
        datasetnew = get_dataset("gsm8k", tokenizer, 128, 500) 
        
    if args.datasetname == "c4": 
        datasetnew = torch.utils.data.DataLoader(datasetnew, batch_size = 1, shuffle = False) 
    
    globalacceptancerate = 0 
    globaldraftcount = 0 
    globalacceptedtokenscount = 0 
    globalnumverifications = 0 
    
    totalinstances = 0 
    prefixi = None 
    for i, batch in enumerate(tqdm(datasetnew)): 
        if args.datasetname == "gsm8k": 
            if i < 10: 
                continue 
            if prefixi == None: 
                for j in range(5): 
                    # input_ids = datasetone[j]["question"] 
                    print("Question: " + datasetnew[j]["question"] + "\n" + "Answer: " + datasetnew[j]["answer"]) 
                    tokenizedinput = tokenizer.encode("Question: " + datasetnew[j]["question"] + "\n" + "Answer: " + datasetnew[j]["answer"] + "\n\n", return_tensors = "pt", add_special_tokens = False) 
                    if prefixi == None: 
                        prefixi = tokenizedinput 
                    else: 
                        # input_ids = torch.cat((input_ids, tokenizedinput), dim = -1) 
                        prefixi = torch.cat((prefixi, tokenizedinput), dim = -1) 
                    # tokenizedinput = tokenizer.encode(datasetone[j]["answer"], return_tensors = "pt", add_special_tokens = False) 
                    # input_ids = torch.cat((input_ids, tokenizedinput), dim = -1) 
                    print(prefixi) 
                    print(prefixi.shape) 
            question_input = tokenizer.encode("Question: " + datasetnew[i]["question"] + "\n" + "Answer: ", return_tensors = "pt", add_special_tokens = False) 
            print(question_input) 
            input_ids = torch.cat((prefixi, question_input), dim = -1) 
            print(tokenizer.decode(input_ids[0])) 
            attention_mask = None 
        else: 
            input_ids = batch["input_ids"].to(torch_device) 
            attention_mask = batch["attention_mask"].to(torch_device) 
        if attention_mask[0][0] == 0: 
            continue 
        
        totalinstances += 1 
        if totalinstances > 100: 
            break 
        
        print(tokenizer.decode(input_ids[0]), end = " ") 
        
        acceptancer, draftcount = Vanilla_Spec_cache(tokenizer, 
                                                     model, 
                                                     None, 
                                                     input_ids, 
                                                     gamma = 1, 
                                                     max_len = 128, 
                                                     top_k = -1, 
                                                     top_p = 0.9, 
                                                     temperature = 0.6, 
                                                     verbose = True, 
                                                     attention_mask = attention_mask, 
        ) 
        break 
    
        globalacceptancerate += (acceptancer * draftcount) 
        globaldraftcount += draftcount 
    print("globalacceptancerate: ", globalacceptancerate / globaldraftcount) 
