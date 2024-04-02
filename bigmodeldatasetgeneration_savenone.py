import torch 
import argparse 
# import contexttimer 

import datasets 
from datasets import load_dataset 

from src.transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from src.transformers.models.llama.modeling_llama import LlamaForCausalLM 
from src.transformers import LlamaTokenizer 
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
'''
# cache_dir = "/home/bc20/yang/transformersprofiling" 
dir_dataset = "/home/yangzho6/c4_parts" 
dir_models = "/home/yangzho6/model_checkpoints" 
''' 
import socket

hostname = socket.gethostname()
print("Hostname:", hostname) 
start_time = time.time() 

parser = argparse.ArgumentParser() 
parser.add_argument("--advanced_data_layout", type = bool, default = False) 
parser.add_argument("--path_d", type = int, default = 0) 
parser.add_argument("--model_name", type = str, default = "openllama3b") 
parser.add_argument("--topk", type = int, default = None) 
parser.add_argument("--batch_size", type = int, default = 64) 
parser.add_argument("--debug", action = "store_true") 
# parser.add_argument("--datasetsubname", type = str, default = None) 
parser.add_argument("--task_id", type = int, default = 0) 
parser.add_argument("--num_workers", type = int, default = 8) 
parser.add_argument("--saving_condensed", action = "store_true") 

args = parser.parse_args() 
print("the args are {}".format(args)) 
# if args.datasetsubname is None: 
    # raise ValueError("datasetsubname should be specified") 

# model_name = "openllama3b" 
# model_name = "shearedllama2_7b" 
model_name = args.model_name 

if "lovelace" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    datasetsrc = "/home/yangzho6/c4_parts/downloads/c4_file2.json" 
    datasetparent = "/home/yangzho6/c4_parts/downloads/" 
    dir_models = "/home/yangzho6/model_checkpoints" 
    synthesized_dir_path = "/home/yangzho6/c4llm_synthesized/{}_topk{}/".format(model_name, args.topk if args.topk is not None else "na") 
    synthesized_data_path = "/home/yangzho6/c4llm_synthesized/{}_topk{}/tensor_dir/".format(model_name, args.topk if args.topk is not None else "na") 
elif "ada" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    datasetsrc = "/home/beidic/yangzho6/c4_parts/downloads/c4_file2.json" 
    datasetparent = "/home/beidic/yangzho6/c4_parts/downloads/" 
    dir_models = "/home/beidic/yangzho6/model_checkpoints" 
    synthesized_dir_path = "/home/beidic/yangzho6/c4llm_synthesized/{}_topk{}/".format(model_name, args.topk if args.topk is not None else "na") 
    # synthesized_data_path = "/home/beidic/yangzho6/c4llm_synthesized/tensor_dir/" 
    synthesized_data_path = "/home/beidic/yangzho6/c4llm_synthesized/{}_topk{}/tensor_dir2/".format(model_name, args.topk if args.topk is not None else "na") 
else: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    datasetsrc = "/fsx-storygen/beidic/yang/c4/en/c4_file2.json" 
    datasetparent = "/fsx-storygen/beidic/yang/c4_parts/downloads/" 
    dir_models = "/fsx-storygen/beidic/yang/model_checkpoints" 
    synthesized_dir_path = "/fsx-storygen/beidic/yang/c4llm_synthesized/{}_topk{}/".format(model_name, args.topk if args.topk is not None else "na") 
    synthesized_data_path = "/fsx-storygen/beidic/yang/c4llm_synthesized/{}_topk{}/tensor_dir/".format(model_name, args.topk if args.topk is not None else "na") 

from termcolor import colored 

import argparse 
import subprocess 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
# onedataset = load_dataset('json', data_files = '/home/yangzho6/c4_parts/downloads/c4_file1.json', split = "train") 
# onedataset = load_dataset('json', data_files = ['/home/beidic/yangzho6/c4_parts/downloads/c4_file1.json', '/home/beidic/yangzho6/c4_parts/downloads/c4_file2.json'], split = "train") 
# onedataset = load_dataset("json", data_files = '/home/beidic/yangzho6/c4_parts/downloads/c4_file1.json', split = "train") 
# say I want to use 96 GPUs 

#step1 I need to know how large the number of lines in the datasetfile is 
d_files = [datasetparent + "c4_file{}.json".format(args.task_id)] # number of files is 3 
print(colored("path_d: {}, the dataset file is {}".format(args.path_d, d_files), "yellow")) 
line_count = 0 
for file in d_files: 
    result = subprocess.run(["wc", "-l", file], capture_output = True, text = True) 
    if result.returncode == 0: 
        line_count += int(result.stdout.split()[0]) 
        # print("path_d: {}, the line count is {}".format(args.path_d, line_count)) 
    else: 
        raise Exception(f"Error counting lines: {result.stderr}") 
print("path_d: {}, the line count is {}".format(args.path_d, line_count)) 
# task_id is which task is in, while path_d is which GPU is on 

#step2 I need to know how many lines each GPU should process 
# we hardcode the following, the total number of tasks is 12, each task uses 1 node with 8 GPUs (no longer used) 
# args.path_d = args.task_id * 8 + args.path_d 
# each_gpu_line_count_ref = (line_count + 6) // 7 
# each_gpu_line_count_ref = (line_count + 7) // 8 
each_gpu_line_count_ref = (line_count + args.num_workers - 1) // args.num_workers 
if args.path_d < args.num_workers - 1: 
    each_gpu_line_count = each_gpu_line_count_ref 
else: # numworkers - 1 
    # each_gpu_line_count = line_count - (6 * each_gpu_line_count_ref) 
    each_gpu_line_count = line_count - ((args.num_workers - 1) * each_gpu_line_count_ref) 
print(colored("the global proc id is {} start_idx {} end_idx {}".format(args.path_d, args.path_d * each_gpu_line_count_ref, args.path_d * each_gpu_line_count_ref + each_gpu_line_count), "blue")) 

# print(colored("path_d: {}, the processing files are {}".format(args.path_d, d_files), "yellow")) 
print(colored("path_d: {}, Using model name {} for synthesized data".format(args.path_d, model_name), "yellow")) 
print(colored("path_d: {}, Using topk {} and debug is {}".format(args.path_d, args.topk, args.debug), "yellow")) 

if not args.debug: 
    onedataset = load_dataset("json", data_files = d_files, split = "train[{}:{}]".format(args.path_d * each_gpu_line_count_ref, args.path_d * each_gpu_line_count_ref + each_gpu_line_count)) 
    # onedataset = load_dataset("json", data_files = d_files, split = "train") 
else: 
    onedataset = load_dataset("json", data_files = d_files, split = "train[:2000]") 

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
        sum = torch.zeros(shape, device = device) 
        for i in range(len(listoflasthiddenstates)): 
            if i % kernel_size == kernel_size - 1: 
                sum += listoflasthiddenstates[i] 
                downsampled_vectors.append(sum/kernel_size) 
                sum.mul_(0.) 
                assert sum.view(-1).sum() == 0 
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
        with torch.no_grad(): 
            input_ids = inputs["input_ids"] 
            attention_mask = inputs["attention_mask"] 
            labels = inputs["labels"] 
            top_k = 10
            top_p = 0.9 

            temperature = 1 

            large_outputs = self.large_model.generate(input_ids = input_ids, max_length = 128, do_sample = True, top_k = top_k, top_p = top_p, temperature = temperature, output_hidden_states = True, return_dict_in_generate = True) 
            # print("the shape of the sequence is {}".format(large_outputs.sequences.shape)) 
            # print("output last hidden states list has length {}".format(len(large_outputs.hidden_states))) 
            # print("output last hidden states list first element has shape {}".format([len(large_outputs.hidden_states[i]) for i in range(len(large_outputs.hidden_states))])) 
            # print("each token output hiddens states has shape {}".format(large_outputs.hidden_states[-1][-1].shape)) 
            list_of_last_hidden_states = [token_hidden_states[-1][:, -1, :] for token_hidden_states in large_outputs.hidden_states] 
            print(colored("sequences of the large model output sequence has shape {}".format(large_outputs.sequences.shape), "yellow")) 
            downsampled_vectors = self.downsample_vectors(list_of_last_hidden_states) 
            assert len(downsampled_vectors) == 64/4 
            # print("each dim of downsampled_vectors is {}".format(downsampled_vectors[0].shape)) 
            downsampled_vectors = torch.stack(downsampled_vectors, dim = 1) 
            print("downsampled vector dimension is {}".format(downsampled_vectors.shape)) 
            attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.shape[0], 80), device = attention_mask.device)), dim = 1) #TODO make it more general 
            # print("shape of the downsampled vectors is {} hidden states dim {}".format(len(downsampled_vectors), downsampled_vectors[0].shape)) 
        
        outputs = model(input_ids = large_outputs.sequences, attention_mask = attention_mask, labels = large_outputs.sequences, condensed_embeds = downsampled_vectors) 
        
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            ) 
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
        print("the loss is {}".format(loss)) 

        return (loss, outputs) if return_outputs else loss 

small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = dir_models).to(torch_device) 
small_model.eval() 

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir) 
if model_name == "shearedllama2_7b": 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
elif model_name == "openllama3b": 
    tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models) 
elif model_name == "tinyllama": 
    tokenizer = LlamaTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models) 
    # tokenizer2 = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
elif model_name == "phi-2": 
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", cache_dir = dir_models) 
elif model_name == "llama2_7b": 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
else: 
    raise ValueError("model name should be one of shearedllama2_7b, openllama3b") 
# tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
# print("the tokenizer pad token id is {}".format(tokenizer.pad_token_id)) 
# tokenizer.pad_token = "[PAD]" 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 
# tokenizer2.pad_token = tokenizer2.eos_token 
# tokenizer2.padding_side = "left" 

if model_name == "openllama3b": 
    # large_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) # pad_id = 2 
    large_model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
elif model_name == "tinyllama": 
    large_model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
elif model_name == "shearedllama2_7b": 
    large_model = LlamaForCausalLM.from_pretrained("princeton-nlp/Sheared-LLaMA-2.7B", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) # pad_id = 2 
elif model_name == "phi-2": 
    large_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
elif model_name == "llama2_7b": 
    large_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
else: 
    raise ValueError("model name should be one of shearedllama2_7b, openllama3b") 
large_model.eval() 

max_length = 64 

def encode_with_truncation(examples): 
    return tokenizer(examples["text"], truncation = True, padding = "max_length", 
                     max_length = max_length, return_special_tokens_mask = True) 

# train_dataset = onedataset["train"].map(encode_with_truncation, batched = True, num_proc = 4) 

train_dataset = onedataset.map(encode_with_truncation, batched = True, num_proc = 16) 
# train_dataset = d['train'].map(encode_with_truncation, batched = True, num_proc = 4) 
# test_dataset = d['test'].map(encode_with_truncation, batched = True, num_proc = 4) 

train_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 
# test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

# model_path = "/home/bc20/yang" 
# model_path = "/home/yangzho6/model_checkpoints" 
model_path = dir_models 
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=1,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=8, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=1,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=8,  # evaluation batch size
    # logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
    # save_steps=1000,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
) 

trainer = CustomTrainer( 
    large_model = large_model, 
    model = small_model, 
    args = training_args, 
    train_dataset = train_dataset, 
    # eval_dataset = test_dataset, 
    data_collator = data_collator, 
) 

# synthesized_data_path = synthesized_data_path[: -1] + "_kernel_{}_{}_{}/".format(args.kernel_size, args.task_id, args.path_d) 
synthesized_data_path = synthesized_data_path[: -1] + "_{}_{}/".format(args.task_id, args.path_d) 
# json_file_name = "c4synthesized_file1_kernel{}_{}_{}.json".format(args.kernel_size, args.task_id, args.path_d) 
json_file_name = "c4synthesized_file1_{}_{}.json".format(args.task_id, args.path_d) 

os.makedirs(synthesized_data_path, exist_ok = True) 

json_file1 = open(synthesized_dir_path + json_file_name, "a") 

train_dataloader = trainer.get_train_dataloader() 
print("path_d: {}, the length of the train dataloader is {}".format(args.path_d, len(train_dataloader))) 

for step, inputs in enumerate(train_dataloader): 
    inputs = trainer._prepare_inputs(inputs) 
    input_ids = inputs["input_ids"] 
    attention_mask = inputs["attention_mask"] 
    if step % 1000 == 0: 
        print("input_ids.shape[0] {}".format(input_ids.shape[0])) 
    labels = inputs["labels"] 
    if args.topk is not None: 
        top_k = args.topk 
    top_p = 0.9 

    temperature = 1 

    if args.topk is not None: 
        large_outputs = large_model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            max_length = 260, 
            do_sample = True, 
            top_k = top_k, 
            output_hidden_states = False, 
            return_dict_in_generate = True 
        ) 
    else: 
        large_outputs = large_model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            max_length = 260, 
            do_sample = True, 
            # do_sample = False, 
            output_hidden_states = False, 
            return_dict_in_generate = True 
        ) 
    
    if args.debug: 
        for i in range(input_ids.shape[0]): 
            example = large_outputs.sequences[i] 
            print(tokenizer.decode(example[: max_length])) 
            print(colored(tokenizer.decode(example[max_length : ]), "blue")) 
            print("attention_mask is {}".format(attention_mask[i])) 
            print() 
        exit(0) 
    # if step > 1: 
    
    textsynthesized = tokenizer.batch_decode(large_outputs.sequences) 
    if step % 100 == 0: 
        print("path_d: {}, step is {} and the text first synthesized is {}".format(args.path_d, step, textsynthesized[0])) 
    
    for i in range(large_outputs.sequences.shape[0]): 
        example_synthesized = textsynthesized[i] # I don't think this line is used 
        
        outputs = large_outputs.sequences[i] 
        
        seq_len = len(outputs) 
        for j in range(seq_len): 
            if outputs[j] == 1: 
                outputs = outputs[(j + 1) :] 
                break 
        
        new_output = tokenizer.decode(outputs) 
        
        example_data = {
            "text": new_output, 
        } 
        json_file1.write(json.dumps(example_data) + "\n") 
    
json_file1.close() 
torch.cuda.synchronize() 
end_time = time.time() 
print("path_d: {}, the time elasped is {}".format(args.path_d, end_time - start_time)) 
