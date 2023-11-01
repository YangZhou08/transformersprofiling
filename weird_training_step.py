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
from src.transformers import Trainer, TrainingArguments 
from torch import nn 
from src.transformers import DataCollatorForLanguageModeling 
from src.transformers.generation.utils import GenerationConfig 
from src.transformers.models.llama.modeling_llama import LlamaForCausalLM, SimpleSmallModel 
import time 
from termcolor import colored 

class CustomTrainer(Trainer): 
    def __init__(self, large_model = None, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.large_model = large_model 
        self.generation_config = GenerationConfig(return_dict_in_generate = True) 
        # self.time_checkpoint = time.time() 
        self.time_checkpoint = 0 
    
    def downsample_vectors(self, listoflasthiddenstates, kernel_size = 4): 
        downsampled_vectors = [] 
        shape = listoflasthiddenstates[0].shape 
        device = listoflasthiddenstates[0].device 
        for i in range(len(listoflasthiddenstates)): 
            sum = torch.zeros(shape, device = device) 
            if i % kernel_size == kernel_size - 1: 
                sum += listoflasthiddenstates[i] 
                downsampled_vectors.append(sum/kernel_size) 
                sum.mul_(0.) 
            else: 
                sum += listoflasthiddenstates[i] 
        return downsampled_vectors 

    def compute_loss(self, model, inputs, return_outputs = False): 
        torch.cuda.synchronize() 
        print(colored("time elasped in the last iteration is {}".format(time.time() - self.time_checkpoint)), "red") 
        self.time_checkpoint = time.time() 
        labels = None 
        for k, v in inputs.items(): 
            if isinstance(v, tuple): 
                print(k, len(v)) 
            elif isinstance(v, torch.Tensor): 
                print(k, v.shape) 
            else: 
                print(k, v) 
        
        print("attention_mask: {}".format(inputs["attention_mask"])) 
        input_ids = inputs["input_ids"] 
        attention_mask = inputs["attention_mask"] 
        labels = inputs["labels"] 
        top_k = 10
        top_p = 0.9 

        temperature = 1 

        large_outputs = self.large_model.generate(input_ids = input_ids, max_length = 128, do_sample = True, top_k = top_k, top_p = top_p, temperature = temperature, output_hidden_states = True, return_dict_in_generate = True) 
        list_of_last_hidden_states = [token_hidden_states[-1][:, -1, :] for token_hidden_states in large_outputs.hidden_states] 
        downsampled_vectors = self.downsample_vectors(list_of_last_hidden_states) 
        assert len(downsampled_vectors) == 64/4 
        print("shape of the downsampled vectors is {} hidden states dim {}".format(len(downsampled_vectors), downsampled_vectors[0].shape)) 
        
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels) 
        
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            ) 
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
        print("the loss is {}".format(loss)) 

        return (loss, outputs) if return_outputs else loss 

from src.transformers import BitsAndBytesConfig 

cache_dir = "/home/bc20/yang/transformersprofiling" 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
onedataset = load_dataset('json', data_files = "/home/bc20/yang/transformersprofiling/downloads/c4_subset.json", split = "train[:1000]")  
# onedataset = load_dataset("c4", "en", split = "train", cache_dir = cache_dir) 

d = onedataset.train_test_split(test_size = 0.1) 
print(d["train"], d["test"]) 

print() 

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir) 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = cache_dir) 
# tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
# print("the tokenizer pad token id is {}".format(tokenizer.pad_token_id)) 
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "left" 

'''
quant_config = BitsAndBytesConfig(
    load_in_4bit = True, 
    llm_int4_has_fp16_weight = True, 
    bnb_4bit_quant_type = "nf4", 
    bnb_4bit_compute_dtype = "float16", 
    bnb_4bit_use_double_quant = False 
) 
''' 
small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = cache_dir).to(torch_device) 
# small_model = SimpleSmallModel.from_pretrained("JackFram/llama-160m", cache_dir = cache_dir).to(torch_device) 
large_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = cache_dir).to(torch_device).half() 
large_model.eval() 

small_model.config.pad_token_id = tokenizer.pad_token_id 
small_model.train() 

# max_length = small_model.config.max_position_embeddings 
max_length = 64 
# def encode_with_truncation(examples): 
    # return tokenizer(examples["text"], truncation=True, padding="max_length",
                #    max_length=max_length, return_special_tokens_mask=True) 
def encode_with_truncation(examples): 
    return tokenizer(examples["text"], truncation = True, padding = "max_length", 
                     max_length = max_length, return_special_tokens_mask = True) 

train_dataset = d['train'].map(encode_with_truncation, batched = True, num_proc = 4) 
test_dataset = d['test'].map(encode_with_truncation, batched = True, num_proc = 4) 

print("The model max length is {}".format(small_model.config.max_position_embeddings)) 

train_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 
test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

model_path = "/home/bc20/yang" 
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=10,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=10, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
) 

weightmodelfirst = next(small_model.parameters()) 
print(weightmodelfirst.dtype) 

trainer = CustomTrainer( 
    large_model = large_model, 
    model = small_model, 
    args = training_args, 
    train_dataset = train_dataset, 
    eval_dataset = test_dataset, 
    data_collator = data_collator, 
) 

trainer.train() 

'''
print("*** Below is the selected line to test ***") 
print(colored("On the {}th line we print out the sequence".format(i), "green")) 
word_seq = onedataset[i]["text"] 
# print(word_seq) 

input_ids = tokenizer.encode(word_seq, return_tensors = 'pt').to(torch_device) 

print("the input ids is {}".format(input_ids.shape)) 
# print(input_ids) 
print() 

print("the original input first 100 tokens should be: ") 
print(colored(tokenizer.decode(input_ids[0][:64]), "yellow"), end = '') 
print(tokenizer.decode(input_ids[0][64:])) 

# halfindex = int(input_ids.shape[-1]/2) 
# input_first_part = input_ids[:, :halfindex] 
input_first_part = input_ids[:, :64] 

# n = 0 
top_k = 10
top_p = 0.9 

temperature = 1 

outputs = small_model.generate(input_ids = input_first_part, max_length = 128, do_sample = True, top_k = top_k, top_p = top_p, temperature = temperature) 
print(outputs.shape) 

output_t = tokenizer.decode(outputs[0][:64]) 
print(output_t, end = '') 
output_t = tokenizer.decode(outputs[0][64:]) 
print(colored(output_t, "green")) 
print() 
''' 
