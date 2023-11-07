import torch 
import argparse 
# import contexttimer 

import datasets 
from datasets import load_dataset 

from src.transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from src.transformers.models.llama.modeling_llama import LlamaForCausalLM 
from src.transformers import GPTNeoXForCausalLM 

from tqdm import tqdm
# from sampling.utils import norm_logits, sample 

import torch.nn.functional as F 

from src.transformers.generation.logits_process import LogitsProcessorList 
import time 
import numpy as np 

from termcolor import colored 

from src.transformers import BitsAndBytesConfig 
from src.transformers import Trainer, TrainingArguments 
from src.transformers import DataCollatorForLanguageModeling 
from src.transformers.generation.utils import GenerationConfig 

import os 
import json 

# cache_dir = "/home/bc20/yang/transformersprofiling" 
dir_dataset = "/home/yangzho6/c4_parts" 
dir_models = "/home/yangzho6/model_checkpoints" 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir) 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
# tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
# print("the tokenizer pad token id is {}".format(tokenizer.pad_token_id)) 
# tokenizer.pad_token = "[PAD]" 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

large_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
large_model.eval() 

# batch_size = 50 
batch_size = 150 
# word_input = "His signature speed, explosiveness and quickness for transition, attacking downhill, finishing and defense were evident during two games versus the Perth Wildcats and three during the Intercontinental Cup in Singapore. But the flashes of self-creation and shot-making appeared earlier than expected. And though he’s likely to go through stretches of cold shooting or inefficient one-on-one execution, Holland — who’ll still be 18 years old by the 2024 draft — has looked competent enough with his off-the-dribble footwork, pull-up, floater and rhythm threes for scouts to buy gradual improvement moving forward." 

generated_input_data = torch.randint(low = 0, high = tokenizer.vocab_size, size = (batch_size, 60), dtype = torch.long).to(torch_device) 
# generated_input_data = torch.cat((generated_input_data, torch.full((2, 4), tokenizer.pad_token_id, dtype = torch.long).to(torch_device)), dim = 1) 
generated_input_data = torch.cat((torch.full((batch_size, 4), tokenizer.pad_token_id).to(torch_device), generated_input_data), dim = 1) 

# attention_mask = torch.cat((torch.ones((2, 60), dtype = torch.long).to(torch_device), torch.zeros((2, 4), dtype = torch.long).to(torch_device)), dim = 1) 
attention_mask = torch.cat((torch.zeros((batch_size, 4), dtype = torch.long).to(torch_device), torch.ones((batch_size, 60), dtype = torch.long).to(torch_device)), dim = 1) 
n = 0 
top_k = 10
top_p = 0.9 

temperature = 1 
past_key_values = None 
print("warming up ...") 
for i in range(5): 
    output_seqences = large_model.generate(input_ids = generated_input_data, attention_mask = attention_mask, do_sample = True, top_k = top_k, top_p = top_p, temperature = temperature, max_length = 128) 
    outputcheck = tokenizer.decode(output_seqences[0]) 
    print(outputcheck) 

print("start measuring time ...") 
latency_timelist = [] 
for i in range(100): 
    start_time = time.time() 
    output_seqences = large_model.generate(input_ids = generated_input_data, attention_mask = attention_mask, do_sample = True, top_k = top_k, top_p = top_p, temperature = temperature, max_length = 128) 
    torch.cuda.synchronize() 
    end_time = time.time() 
    print("time measurementL: {}".format(end_time - start_time)) 
    latency_timelist.append(end_time - start_time) 
print("average time measurement is {}".format(np.mean(latency_timelist))) 