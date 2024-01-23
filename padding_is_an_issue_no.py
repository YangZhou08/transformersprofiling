import torch 
import argparse 
# import contexttimer 

import datasets 
from datasets import load_dataset 

from src.transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from src.transformers import GPTNeoXForCausalLM 
from src.transformers import LlamaConfig, LlamaPreTrainedModel 

from tqdm import tqdm
# from sampling.utils import norm_logits, sample 

import torch.nn.functional as F 

from src.transformers.generation.logits_process import LogitsProcessorList 
import time 
import numpy as np 

from termcolor import colored 
from src.transformers import Trainer, TrainingArguments 
from torch import nn 
from src.transformers import DataCollatorForLanguageModeling 
from src.transformers.generation.utils import GenerationConfig 
from src.transformers.models.llama.modeling_llama import LlamaForCausalLM, SimpleSmallModel 
import time 

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False 

has_wandb = False # disable for debugging 

from src.transformers.utils import ( 
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
    strtobool,
) 

if is_apex_available():
    from apex import amp 

from src.transformers import BitsAndBytesConfig 

# cache_dir = "/home/bc20/yang/transformersprofiling" 
dir_dataset = "/home/yangzho6/c4_parts" 
dir_models = "/home/yangzho6/model_checkpoints" 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 

print() 

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir) 
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
# tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
# print("the tokenizer pad token id is {}".format(tokenizer.pad_token_id)) 
# tokenizer.pad_token = "[PAD]" 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

# large_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
# large_model.eval() 

# small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = cache_dir).to(torch_device) 
# small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = cache_dir).to(torch_device) 
small_config = LlamaConfig.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 

small_state_dict_for_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = dir_models).state_dict() 
small_model = SimpleSmallModel(small_config) 

new_state_dict = {} 

for key in small_state_dict_for_model.keys(): 
    new_key = key 
    if 'lm_head' in key: 
        print("got here found the following key {}".format(key)) 
    if 'model.' in key: 
        new_key = key[6 :] 
    print(new_key) 
    new_state_dict[new_key] = small_state_dict_for_model[key] 

try: 
    small_model.load_state_dict(new_state_dict) 
except RuntimeError as r: 
    print(colored(r, "yellow")) 
small_model = small_model.to(torch_device) 
small_model.train() 

# batch_size = 50 
# batch_size = 150 
batch_size = 10 
# word_input = "His signature speed, explosiveness and quickness for transition, attacking downhill, finishing and defense were evident during two games versus the Perth Wildcats and three during the Intercontinental Cup in Singapore. But the flashes of self-creation and shot-making appeared earlier than expected. And though he’s likely to go through stretches of cold shooting or inefficient one-on-one execution, Holland — who’ll still be 18 years old by the 2024 draft — has looked competent enough with his off-the-dribble footwork, pull-up, floater and rhythm threes for scouts to buy gradual improvement moving forward." 

generated_input_data = torch.randint(low = 0, high = tokenizer.vocab_size, size = (batch_size, 128), dtype = torch.long).to(torch_device) 
# generated_input_data = torch.cat((generated_input_data, torch.full((2, 4), tokenizer.pad_token_id, dtype = torch.long).to(torch_device)), dim = 1) 
# generated_input_data = torch.cat((torch.full((batch_size, 4), tokenizer.pad_token_id).to(torch_device), generated_input_data), dim = 1) 
generated_condensed_embeds = torch.randn((batch_size, 16, 4096)).to(torch_device) 
# attention_mask = torch.cat((torch.ones((2, 60), dtype = torch.long).to(torch_device), torch.zeros((2, 4), dtype = torch.long).to(torch_device)), dim = 1) 
# attention_mask = torch.cat((torch.zeros((batch_size, 4), dtype = torch.long).to(torch_device), torch.ones((batch_size, 60), dtype = torch.long).to(torch_device)), dim = 1) 
attention_mask = torch.ones((batch_size, 144), dtype = torch.long).to(torch_device) 
n = 0 
top_k = 10
top_p = 0.9 
'''
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
''' 
for i in range(5): 
    outputs = small_model( 
        input_ids = generated_input_data, 
        attention_mask = attention_mask, 
        labels = generated_input_data, 
        condensed_embeds = generated_condensed_embeds, 
        output_hidden_states = True, 
        output_attentions = True, 
        return_dict = True, 
    ) 
