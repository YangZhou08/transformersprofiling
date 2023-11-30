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

import socket

hostname = socket.gethostname()
print("Hostname:", hostname)

if "lovelace" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    datasetsrc = "/home/yangzho6/c4_parts/downloads/c4_file2.json" 
    dir_models = "/home/yangzho6/model_checkpoints" 
    synthesized_dir_path = "/home/yangzho6/c4llm_synthesized/" 
    synthesized_data_path = "/home/yangzho6/c4llm_synthesized/tensor_dir/" 
elif "ada" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    datasetsrc = "/home/beidic/yangzho6/c4_parts/downloads/c4_file2.json" 
    dir_models = "/home/beidic/yangzho6/model_checkpoints" 
    synthesized_dir_path = "/home/beidic/yangzho6/c4llm_synthesized/" 
    # synthesized_data_path = "/home/beidic/yangzho6/c4llm_synthesized/tensor_dir/" 
    synthesized_data_path = "/home/beidic/yangzho6/c4llm_synthesized/tensor_dir2/" 
else: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/yangzho6/model_checkpoints" 
    synthesized_dir_path = "/home/yangzho6/c4llm_synthesized/" 
    synthesized_data_path = "/home/yangzho6/c4llm_synthesized/tensor_dir/" 

from termcolor import colored 

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
                return_tensors = "pt", 
                truncation = True, 
            ) 
            
            item['input_ids'] = encoded_text['input_ids'].squeeze(0)  # remove the batch dimension 
            if item["input_ids"].shape[0] > 128: 
                print("shape is {}".format(item["input_ids"].shape)) 
                print("this example is {}".format(item["text"])) 
                print("the tokenizer after output is {}".format(item["input_ids"])) 
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

# onedataset = load_dataset('json', data_files = '/home/yangzho6/c4_parts/downloads/c4_file1.json', split = "train[:1000]") 

# tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", cache_dir = dir_models) 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 

if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
tokenizer.padding_side = "left" 

# datasetnew = CustomDataset(data_dir = dir_sdata, tokenizer = tokenizer) 
datasetnew = load_dataset('json', data_files = datasetsrc, split = "train[:1000]") 

def encode_with_truncation(examples): 
    return tokenizer(examples["text"], truncation = True, max_length = 256, padding = "max_length", 
                     return_special_tokens_mask = True) 

train_dataset = datasetnew.map(encode_with_truncation, batched = True, num_proc = 4) 
'''
# small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = cache_dir).to(torch_device) 
small_config = LlamaConfig.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 

# small_state_dict_for_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = dir_models).state_dict() 
small_state_dict_for_model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).state_dict() 
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
small_model.eval_mode = True 
# small_model.train() 
''' 
'''
model = = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch_device) 
''' 
config = LlamaConfig.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models) 

if config.rope_scaling is not None: 
    print("model.config.rope_scaling[type] is {}".format(config.rope_scaling["type"])) 
    print("model.config.rope_scaling[factor] is {}".format(config.rope_scaling["factor"])) 
else: 
    config.rope_scaling = {} 

config.rope_scaling["type"] = "sep_q_k" 
config.rope_scaling["factor"] = 2.0 

state_dict_for_model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).state_dict() 
model = LlamaForCausalLM(config) 
model.load_state_dict(state_dict_for_model) 

model.eval() 

'''
small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = dir_models).to(torch_device) 

# small_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", cache_dir = dir_models).to(torch_device) 
# large_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
# large_model.eval() 

# small_model = large_model 
''' 
data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

batch_size = 50 
# dataloader = DataLoader(train_dataset, batch_size = batch_size) 
model_path = dir_models 
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=1,            # number of training epochs, feel free to tweak
    per_device_train_batch_size= 100, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
) 

trainer = Trainer( 
    model = model, 
    args = training_args, 
    train_dataset = train_dataset, 
    # eval_dataset = test_dataset, 
    data_collator = data_collator, 
) 

train_dataloader = trainer.get_train_dataloader() 

# generated using GPT-4 
# Compute perplexity over the dataset
total_perplexity = 0 
# total_loss = torch.zeros(1).to(torch_device) 
total_loss = 0 
num_batches = 0 
count = 0 

with torch.no_grad(): 
    for batch in train_dataloader: 
        input_ids = batch["input_ids"].to(torch_device) 
        attention_mask = batch["attention_mask"].to(torch_device) 
        labels = input_ids.clone() 
        labels[labels == tokenizer.pad_token_id] = -100 
        
        if isinstance(model, SimpleSmallModel): 
            condensed_embeds = batch["condensed_embeds"].to(torch_device) 
            batch_size, seq_len = attention_mask.shape 
            addedon_length = condensed_embeds.shape[1] 
            # print("get the input sentence: {}".format(tokenizer.decode(input_ids[0]))) 
            attention_mask = torch.cat((attention_mask, torch.ones((batch_size, addedon_length), dtype = torch.long).to(input_ids.device)), dim = 1) 
            
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, eval_mode = True, iteration_count = count) 
        else: 
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels) 
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
        print("size of loss is {}".format(loss)) 
        total_loss += loss.item() 
        perplexity = torch.exp(loss).mean().item() 
        print(colored("perplexity is {}".format(perplexity), "yellow")) 
        print() 
        total_perplexity += perplexity 
        num_batches += 1 
        count += 1 

average_perplexity = total_perplexity / num_batches 
reference_perplexity = np.exp(total_loss / num_batches) 
print(colored("reference perplexity is {}".format(reference_perplexity), "yellow")) 
print("average perplexity is {}".format(average_perplexity)) 
