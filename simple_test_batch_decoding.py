import torch 
import argparse 
# import contexttimer 

from src.transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from src.transformers import GPTNeoXForCausalLM 
from src.transformers import LlamaForCausalLM 

from tqdm import tqdm
# from sampling.utils import norm_logits, sample 

import torch.nn.functional as F 

from src.transformers.generation.logits_process import LogitsProcessorList 
import time 
import numpy as np 

# set_logger("/rscratch/zhendong/yang_tasc/transformersprofiling/simple_tb3b_log.txt") 
# cache_dir = "/home/bc20/yang/transformersprofiling" 
# cache_dir = "/home/yangzho6/model_checkpoints" 
cache_dir = "/home/yangzho6/model_checkpoints" 

def run(): 
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    # torch_device = 'cpu' 
    
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b-deduped", cache_dir = cache_dir) 
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir) 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = cache_dir) 
    
    # small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", revision = "step3000", cache_dir = "/rscratch/zhendong/yang_tasc").to(torch_device) 
    # small_model = GPTNeoXForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = cache_dir).to(torch_device) 
    # small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir).to(torch_device) 
    small_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = cache_dir).to(torch_device) 
    # small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = cache_dir).to(torch_device) 
    # print(type(small_model)) 
    small_model.eval() 
    weightmodelfirst = next(small_model.parameters()) 
    print(weightmodelfirst.dtype) 

    for name, tensor in small_model.named_parameters(): 
        print(name, tensor.shape) 
    
    pad_token_id = tokenizer.pad_token_id 
    eos_token_id = tokenizer.eos_token_id 
    # decoder_input_ids = torch.full((input_ids.shape[0], 1), pad_token_id, dtype=torch.long).to(input_ids.device) 
    
    n = 0 
    top_k = 10
    top_p = 0.9 
    
    temperature = 1 
    past_key_values = None 
    
    measure_time_list = [] 
    
    generated_sequence = input_ids 
    past_output = None 

    input_ids = torch.randint(0, 32000, (5, 100)).to(torch_device) 
    print("Warming up ...") 
    for i in range(10): 
        print("Warm up iteration {}".format(i)) 
        outputs = small_model.generate(input_ids = input_ids, max_length = 500, do_sample = False) 
    
    print("warm up done") 
    for i in range(1, 11): 
        print("using batch size of {}".format(i)) 
        input_ids = torch.randint(0, 32000, (i, 100)).to(torch_device) 
        print("input_ids shape is {}".format(input_ids.shape)) 
        start_time = time.time() 
        outputs = small_model.generate(input_ids = input_ids, max_length = 500, do_sample = False) 
        torch.cuda.synchronize() 
        end_time = time.time() 
        print("time for batch size of {} is {}".format(i, end_time - start_time)) 

if __name__ == "__main__": 
    run() 
