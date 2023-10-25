import torch 
import argparse 
# import contexttimer 

import datasets 
from datasets import load_dataset 

from src.transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from src.transformers import GPTNeoXForCausalLM 

from tqdm import tqdm
# from sampling.utils import norm_logits, sample 

import torch.nn.functional as F 

from src.transformers.generation.logits_process import LogitsProcessorList 
import time 
import numpy as np 

from termcolor import colored 

cache_dir = "/home/bc20/yang/transformersprofiling" 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
onedataset = load_dataset('json', data_files = "/home/bc20/yang/transformersprofiling/downloads/c4_subset.json", split = "train") 
# onedataset = load_dataset("c4", "en", split = "train", cache_dir = cache_dir) 

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir) 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = cache_dir) 
    
# small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", revision = "step3000", cache_dir = "/rscratch/zhendong/yang_tasc").to(torch_device) 
# small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", revision = "step3000", cache_dir = cache_dir).to(torch_device) 
# small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b", revision = "step3000", cache_dir = cache_dir).to(torch_device) 
# small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir).to(torch_device) 
small_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = cache_dir).to(torch_device) 
small_model.eval() 

weightmodelfirst = next(small_model.parameters()) 
print(weightmodelfirst.dtype) 

# train_dataset = onedataset["train"] 
# validation_dataset = onedataset["validation"] 

# for i in range(10): 
#     print(onedataset[i]) 

print("*** Below is the selected line to test ***") 
for i in range(10): 
    print(colored("On the {}th line we print out the sequence".format(i), "green")) 
    word_seq = onedataset[i]["text"] 
    # print(word_seq) 

    input_ids = tokenizer.encode(word_seq, return_tensors = 'pt').to(torch_device) 

    print("the input ids is {}".format(input_ids.shape)) 
    # print(input_ids) 
    print() 
    
    print("the original input first 100 tokens should be: \n{}".format(tokenizer.decode(input_ids[0][:100]))) 

    # halfindex = int(input_ids.shape[-1]/2) 
    # input_first_part = input_ids[:, :halfindex] 
    input_first_part = input_ids[:, :50] 

    # n = 0 
    top_k = 10
    top_p = 0.9 

    temperature = 1 

    outputs = small_model.generate(input_ids = input_first_part, max_length = 200, do_sample = True, top_k = top_k, top_p = top_p, temperature = temperature) 
    print(outputs.shape) 

    output_t = tokenizer.decode(outputs[0]) 
    print(output_t) 
    print() 
