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

from torch.utils.data import DataLoader 

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

torch_device = "cuda:0" 

class CustomDataset: 
    def __init__(self, data_dir, tokenizer = None, max_length = 128): 
        # self.synthesize_dir = "/home/yangzho6/c4llm_synthesized/" 
        self.synthesize_dir = data_dir 
        self.dataset = load_dataset('json', data_files = self.synthesize_dir + "c4synthesized_file1.json") 
        self.dataset = self.dataset["train"] 

        self.tokenizer = tokenizer 
        self.max_length = max_length 
    
    def __len__(self): 
        return len(self.dataset) 
    
    def __getitem__(self, idx): 
        item = self.dataset[idx] 
        tensor = torch.load(item["condensed_token_path"]) 

        if self.tokenizer is not None: 
            # the following line is under investigation 
            encoded_text = self.tokenizer( 
                item["text"], 
                add_special_tokens = False, 
                padding = "max_length", 
                max_length = 128, 
                return_attention_mask = True, 
                return_tensors = "pt" 
            ) 
            item['input_ids'] = encoded_text['input_ids'].squeeze(0)  # remove the batch dimension
            item['attention_mask'] = encoded_text['attention_mask'].squeeze(0)  # remove the batch dimension 
        
        item["condensed_embeds"] = tensor 

        return item 

def compute_perplexity(model, tokenizer, text): 
    encodings = tokenizer(text, return_tensors = "pt") 

    max_length = model.config.n_positions 
    stride = 512 # should not be used 

    nlls = [] 
    for i in range(0, encodings.input_ids.size(1), stride): 
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        nlls.append(log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item() 

dir_dataset = "/home/yangzho6/c4_parts" 
dir_models = "/home/yangzho6/model_checkpoints" 
dir_sdata = "/home/yangzho6/c4llm_synthesized/" 

onedataset = load_dataset('json', data_files = '/home/yangzho6/c4_parts/downloads/c4_file1.json', split = "train[:1000]") 

tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 

datasetnew = CustomDataset(data_dir = dir_sdata, tokenizer = tokenizer) 

# small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = cache_dir).to(torch_device) 
small_config = LlamaConfig.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
'''
print("print out configurations") 
for k, v in small_config.__dict__.items(): 
    print(k, v) 
''' 
small_state_dict_for_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = dir_models).state_dict() 
small_model = SimpleSmallModel(small_config) 
'''
print("we expect the following keys") 
print(len(small_model.state_dict().keys())) 
for key, _ in small_model.named_parameters(): 
    print(key) 

print() 
print("from the pretrained model, we found the following keys") 
print(type(small_state_dict_for_model)) 
print(len(small_state_dict_for_model.keys())) 
''' 
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
small_model.eval_mode = True 
# small_model.train() 

dataloader = DataLoader(datasetnew, batch_size = 8) 

# generated using GPT-4 
# Compute perplexity over the dataset
total_perplexity = 0
num_batches = 0 

for batch in dataloader: 
    input_ids = batch["input_ids"].to(torch_device) 
    attention_mask = batch["attention_mask"].to(torch_device) 
    labels = input_ids.clone() 
    labels[labels == tokenizer.pad_token_id] = -100 
    outputs = small_model(input_ids = input_ids, attention_mask = attention_mask, labels = labels) 
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
    perplexity = torch.exp(loss).mean().item() 
    total_perplexity += perplexity 
    num_batches += 1 

average_perplexity = total_perplexity / num_batches 
