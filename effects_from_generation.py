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
from time import sleep 
import numpy as np 

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

def compute_entropy(prob_dist):
    # Ensure the probability distribution is normalized and contains no zero values
    prob_dist = np.clip(prob_dist, 1e-10, 1)  # Avoid log(0) by replacing 0 with a small number
    entropy = -np.sum(prob_dist * np.log2(prob_dist))
    return entropy 

def plain_single_model(tokenizer, model, input_ids, attention_mask, max_len = 256, top_k = -1, top_p = 0.9, temperature = 0.6, verbose = False): 
    n = 0 
    generated_ids = None 
    collected_probs = [] 
    if verbose: 
        spec_stream(input_ids, tokenizer) 
    
    while n < max_len: 
        outputs = model(
            input_ids = input_ids, 
            past_key_values = None, 
            use_cache = False, 
            attention_mask = attention_mask, 
        ) 
        next_token_logits = outputs.logits[:, -1, :] 
        probs = norm_logits(next_token_logits, temperature = temperature, top_k = top_k, top_p = top_p) 
        collected_probs.append(probs) 
        next_token = sample(probs) 
        generated_ids = next_token 
        input_ids = torch.cat((input_ids, generated_ids), dim = 1) 
        attention_mask = torch.cat((attention_mask, torch.ones((1, 1)).to(attention_mask.device)), dim = 1) 
        n += 1 
        if verbose: 
            spec_stream(next_token, tokenizer) 
    return input_ids, collected_probs 

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
    
    return datasetnew 

if __name__ == "__main__": 
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
    if tokenizer.pad_token is not None: 
        print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
    else: 
        tokenizer.pad_token = tokenizer.eos_token 
        print("We now use eos_token as pad token") 
    tokenizer.padding_side = "left" 
    
    small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-68m", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
    
    datasetnew = get_dataset("c4", tokenizer, 64, 10) 
    dataloader = torch.utils.data.DataLoader(datasetnew, batch_size = 1, shuffle = False) 
    
    for batch in dataloader: 
        input_ids = batch["input_ids"].to(torch_device) 
        attention_mask = batch["attention_mask"].to(torch_device) 
        
        outputsequence, collected_probs = plain_single_model(tokenizer, small_model, input_ids, attention_mask, max_len = 256, top_k = -1, top_p = 0.9, temperature = 0.6, verbose = True) 
        print("outputsequence: {}".format(outputsequence)) 
        print("outputsequence shape: {}".format(outputsequence.shape)) 
        print("len(collected_probs) {} collected_probs[0].shape {}".format(len(collected_probs), collected_probs[0].shape)) 
        
        list_collected_entropies = [] 
        for i in range(len(collected_probs)): 
            probs = collected_probs[i].squeeze(0) 
            probs = probs.detach().cpu().numpy() 
            ent = compute_entropy(probs) 
            list_collected_entropies.append(ent) 
        print("list_collected_entropies: {}".format(list_collected_entropies)) 
        
        exit(0) 
