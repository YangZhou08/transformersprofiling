import torch 
import argparse 
# import contexttimer 

import datasets 
from datasets import load_dataset 
import sys 
import os 

current_dir = os.path.dirname(__file__) 
parent_dir = os.path.dirname(current_dir) 
src_folder = os.path.join(parent_dir, "src") 
sys.path.append(src_folder) 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from transformers.models.llama.modeling_llama import LlamaForCausalLM 
from transformers.models.llama.modeling_llama import LlamaWeirdLarge3 
from transformers.models.llama.modeling_llama import SimpleSmallModel 
from transformers.models.llama.modeling_llama import LlamaWeirdLargeTest 
from transformers.models.llama.configuration_llama import LlamaConfig 
from transformers import LlamaTokenizer 
from transformers import GPTNeoXForCausalLM 

from tqdm import tqdm
# from sampling.utils import norm_logits, sample 

import torch.nn.functional as F 

from transformers.generation.logits_process import LogitsProcessorList 
import time 
import numpy as np 

from termcolor import colored 

from transformers import BitsAndBytesConfig 
from transformers import Trainer, TrainingArguments 
from transformers import DataCollatorForLanguageModeling 
from transformers.generation.utils import GenerationConfig 

import json 
'''
# cache_dir = "/home/bc20/yang/transformersprofiling" 
dir_dataset = "/home/yangzho6/c4_parts" 
dir_models = "/home/yangzho6/model_checkpoints" 
''' 
import socket

hostname = socket.gethostname()
print("Hostname:", hostname) 

parser = argparse.ArgumentParser() 
parser.add_argument("--kernel_size", type = int, default = 7) 
parser.add_argument("--advanced_data_layout", type = bool, default = False) 
parser.add_argument("--path_d", type = int, default = 0) 
parser.add_argument("--model_name", type = str, default = "debugging") 
parser.add_argument("--topk", type = int, default = None) 
parser.add_argument("--batch_size", type = int, default = 64) 
parser.add_argument("--debug", action = "store_true") 
parser.add_argument("--small_model_checkpoint", type = str, default = None) 
parser.add_argument("--use_large_model", action = "store_true") 

args = parser.parse_args() 

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
    datasetsrc = "/data/home/beidic/yang/c4/en/c4_file2.json" 
    datasetparent = "/data/home/beidic/yang/c4_parts/downloads/" 
    dir_models = "/data/home/beidic/yang/model_checkpoints" 
    synthesized_dir_path = "/data/home/beidic/yang/c4llm_synthesized/{}_topk{}/".format(model_name, args.topk if args.topk is not None else "na") 
    synthesized_data_path = "/data/home/beidic/yang/c4llm_synthesized/{}_topk{}/tensor_dir/".format(model_name, args.topk if args.topk is not None else "na") 

from termcolor import colored 

import argparse 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
# onedataset = load_dataset('json', data_files = '/home/yangzho6/c4_parts/downloads/c4_file1.json', split = "train") 
# onedataset = load_dataset('json', data_files = ['/home/beidic/yangzho6/c4_parts/downloads/c4_file1.json', '/home/beidic/yangzho6/c4_parts/downloads/c4_file2.json'], split = "train") 
# onedataset = load_dataset("json", data_files = '/home/beidic/yangzho6/c4_parts/downloads/c4_file1.json', split = "train") 
# interest_idx_file = [1, 2, 3] if args.path_d == 0 else [4, 5] 
# interest_idx_file = [1, 2, 3, 4, 5] 
interest_idx_file = [2 * args.path_d + i for i in range(2)] 
# d_files = ["c4_file{}.json".format(i) for i in interest_idx_file] 
d_files = "c4synthesized_file1_kernel7_0.json" 
dfiles = ["/home/yangzho6/c4llm_synthesized/tinyllama/" + d_files] 
print(colored("path_d: {}, the processing files are {}".format(args.path_d, d_files), "yellow")) 
print(colored("path_d: {}, Using model name {} for synthesized data".format(args.path_d, model_name), "yellow")) 
print(colored("path_d: {}, Using topk {} and debug is {}".format(args.path_d, args.topk, args.debug), "yellow")) 
if not args.debug: 
    onedataset = load_dataset("json", data_files = dfiles, split = "train[:5000]") 
else: 
    onedataset = load_dataset("json", data_files = dfiles, split = "train[:2000]") 

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

# train_dataset = onedataset["train"] 
# validation_dataset = onedataset["validation"] 

# for i in range(10): 
#     print(onedataset[i]) 

# d = onedataset.train_test_split(test_size = 0.1) 
# print(d["train"], d["test"]) 
# print(d["train"]) 

print() 

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir) 
if model_name == "shearedllama2_7b": 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
elif model_name == "openllama3b": 
    tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models) 
elif model_name == "tinyllama": 
    # tokenizer = LlamaTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models) 
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
elif model_name == "llama2_7b": 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
elif model_name == "phi-2": 
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", cache_dir = dir_models) 
elif model_name == "debugging": 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
elif model_name == "debugging2": 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
else: 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
# tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
# print("the tokenizer pad token id is {}".format(tokenizer.pad_token_id)) 
# tokenizer.pad_token = "[PAD]" 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

if model_name == "openllama3b": 
    # large_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) # pad_id = 2 
    large_model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
elif model_name == "tinyllama": 
    large_model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
elif model_name == "llama2_7b": 
    large_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
elif model_name == "shearedllama2_7b": 
    large_model = LlamaForCausalLM.from_pretrained("princeton-nlp/Sheared-LLaMA-2.7B", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) # pad_id = 2 
elif model_name == "phi-2": 
    large_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
elif model_name == "debugging": 
    # large_model = LlamaWeirdLarge3.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
    large_model = LlamaWeirdLarge3.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
    
    small_state_dict_for_model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).state_dict() 
    small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models) 
    small_model = SimpleSmallModel(small_config, hostname = hostname, sliding_window_length = args.kernel_size, target_model_dim = 2048) 

    new_state_dict = {} 

    for key in small_state_dict_for_model.keys(): 
        new_key = key 
        if 'lm_head' in key: 
            print("got here found the following key {}".format(key)) 
        if 'model.' in key: 
            new_key = key[6 :] 
        print(new_key) 
        new_state_dict[new_key] = small_state_dict_for_model[key] 
    # if args.embedding_pretrained: 
    #     new_state_dict["embed_projection.weight"] = torch.load("linearprojectionweighttesting.pt") 

    try: 
        small_model.load_state_dict(new_state_dict) 
    except RuntimeError as r: 
        print(colored(r, "yellow")) 

    small_model = small_model.to(torch.bfloat16).to(torch_device) 
    large_model.set_msece_loss(False, False) 
    large_model.set_addonsmallmodel(small_model) 
    large_model.set_inference_setting("setting3") 
    large_model.set_walpha(0.5) 
    large_model.set_slidingwindowlength(sliding_window_length = 7, addonmodel_start = 1) 
    large_model.set_tokenizer_bos_id(bos_id = tokenizer.bos_token_id, pad_id = tokenizer.pad_token_id) 
    large_model.set_cosinesimilarity(False) 
    
    large_model.config.pad_token_id = tokenizer.pad_token_id 
    small_model.config.pad_token_id = tokenizer.pad_token_id 
    
    large_model.model.eval() 
    large_model.addonsmallmodel.eval() 
elif model_name == "debugging2": 
    large_model = LlamaWeirdLargeTest.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
    
    small_state_dict_for_model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).state_dict() 
    small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models) 
    small_model = SimpleSmallModel(small_config, hostname = hostname, sliding_window_length = args.kernel_size, target_model_dim = 2048) 

    new_state_dict = {} 

    for key in small_state_dict_for_model.keys(): 
        new_key = key 
        if 'lm_head' in key: 
            print("got here found the following key {}".format(key)) 
        if 'model.' in key: 
            new_key = key[6 :] 
        print(new_key) 
        new_state_dict[new_key] = small_state_dict_for_model[key] 
    # if args.embedding_pretrained: 
    #     new_state_dict["embed_projection.weight"] = torch.load("linearprojectionweighttesting.pt") 

    try: 
        small_model.load_state_dict(new_state_dict) 
    except RuntimeError as r: 
        print(colored(r, "yellow")) 

    small_model = small_model.to(torch.bfloat16).to(torch_device) 
    large_model.set_msece_loss(False, False) 
    # large_model.set_addonsmallmodel(small_model) 
    large_model.set_addonsmallmodel_statedict(new_state_dict) 
    large_model.addonsmallmodel.set_criticalpath(hostname = hostname) 
    large_model.set_inference_setting("setting3") 
    large_model.set_walpha(0.5) 
    # large_model.set_slidingwindowlength(sliding_window_length = 7, addonmodel_start = 1) 
    large_model.set_slidingwindowlength(sliding_window_length = 7) 
    large_model.set_tokenizer_bos_id(bos_id = tokenizer.bos_token_id, pad_id = tokenizer.pad_token_id) 
    large_model.set_cosinesimilarity(False) 
    
    large_model.config.pad_token_id = tokenizer.pad_token_id 
    small_model.config.pad_token_id = tokenizer.pad_token_id 
    
    large_model.model.eval() 
    large_model.addonsmallmodel.eval() 
elif model_name == "debugging3": 
    print(colored("found the checkpoint of the model", "yellow")) 
    large_model = LlamaWeirdLargeTest.from_pretrained("/home/yangzho6/model_checkpoints/kernelsize4checkpoint-3200").to(torch.bfloat16).to(torch_device) 
    large_model.set_msece_loss(False, False) 
    # large_model.set_addonsmallmodel(small_model) 
    # large_model.set_addonsmallmodel_statedict(new_state_dict) 
    large_model.addonsmallmodel.set_criticalpath(hostname = hostname) 
    large_model.set_inference_setting("setting5") 
    large_model.set_walpha(0.5) 
    # large_model.set_slidingwindowlength(sliding_window_length = 7, addonmodel_start = 1) 
    large_model.set_slidingwindowlength(sliding_window_length = 7) 
    large_model.set_tokenizer_bos_id(bos_id = tokenizer.bos_token_id, pad_id = tokenizer.pad_token_id) 
    large_model.set_cosinesimilarity(False) 
    
    large_model.config.pad_token_id = tokenizer.pad_token_id 
    small_model.config.pad_token_id = tokenizer.pad_token_id 
    
    large_model.model.eval() 
    large_model.addonsmallmodel.eval() 
elif model_name == "plainsmall": 
    small_model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
    small_model.eval() 
    large_model = None 
elif model_name == "finetunedsmall": 
    small_model = LlamaForCausalLM.from_pretrained(args.small_model_checkpoint).to(torch.bfloat16).to(torch_device) 
    small_model.eval() 
    large_model = None 
else: 
    raise ValueError("model name should be one of shearedllama2_7b, openllama3b") 
# large_model.eval() 

# max_length = small_model.config.max_position_embeddings 
max_length = 71 
# def encode_with_truncation(examples): 
    # return tokenizer(examples["text"], truncation=True, padding="max_length",
                #    max_length=max_length, return_special_tokens_mask=True) 
def encode_with_truncation(examples): 
    return tokenizer(examples["text"], truncation = True, padding = "max_length", 
                     max_length = max_length, return_special_tokens_mask = True) 

# train_dataset = onedataset["train"].map(encode_with_truncation, batched = True, num_proc = 4) 
train_dataset = onedataset.map(encode_with_truncation, batched = True, num_proc = 16) 
# train_dataset = d['train'].map(encode_with_truncation, batched = True, num_proc = 4) 
# test_dataset = d['test'].map(encode_with_truncation, batched = True, num_proc = 4) 

print("path_d: {}, The model max length is {}".format(args.path_d, small_model.config.max_position_embeddings)) 

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
    per_device_train_batch_size=args.batch_size, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
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

synthesized_data_path = synthesized_data_path[: -1] + "_kernel_{}_{}/".format(args.kernel_size, args.path_d) 
json_file_name = "c4synthesized_file1_kernel{}_{}.json".format(args.kernel_size, args.path_d) 

os.makedirs(synthesized_data_path, exist_ok = True) 
# json_file_name = "c4synthesized_file1.json" 
# json_file_name = "c4synthesized_file2.json" 
json_file1 = open(synthesized_dir_path + json_file_name, "a") 

train_dataloader = trainer.get_train_dataloader() 
print("path_d: {}, the length of the train dataloader is {}".format(args.path_d, len(train_dataloader))) 
# dict_kernel_maxlength = {3 : 63, 4 : 64, 5 : 65, 6 : 66, 7 : 70} 
dict_kernel_maxlength = {2 : 64, 3 : 63, 4 : 64, 5 : 65, 6 : 66, 7 : 70, 10 : 70} 
# kernel_size = 4 
if args.kernel_size not in dict_kernel_maxlength: 
    raise ValueError("kernel size should be one of 3, 4, 5, 6, 7") 
else: 
    kernel_size = int(args.kernel_size) 

def visualizing_generated_sequence(original_sequence, large_outputs, index_to_peek): 
    print("the sequence of inputs: {}".format(large_outputs.sequences[index_to_peek])) 
    print("the sequence in words: {}".format(tokenizer.decode(large_outputs.sequences[index_to_peek][0 : 64])), end = " ") 
    print(colored(tokenizer.decode(large_outputs.sequences[index_to_peek][64: ]), "blue")) 
    print("expected sequence: {}".format(tokenizer.decode(original_sequence[index_to_peek][: 64])), end = " ") 
    print(colored(tokenizer.decode(original_sequence[index_to_peek][64 :]), "green")) 

def compute_hit_counts(original_sequence, large_outputs): 
    agreement_matrix = original_sequence[:, 64 : ] == large_outputs.sequences[:, 64 : ] # (batch_size, 7) 
    agreement_matrix = agreement_matrix.to(torch.long) 
    summation_stats = torch.sum(agreement_matrix, dim = 0).reshape((1, 7))  # summing along the column dimension 
    return summation_stats[0][0], summation_stats[0][1], summation_stats[0][2], summation_stats[0][3], summation_stats[0][4], summation_stats[0][5], summation_stats[0][6] 
    
first_pos_hit = 0 
second_pos_hit = 0 
third_pos_hit = 0 
fourth_pos_hit = 0 
fifth_pos_hit = 0 
sixth_pos_hit = 0 
seventh_pos_hit = 0 
total_examples = 0 

for step, inputs in enumerate(train_dataloader): 
    inputs = trainer._prepare_inputs(inputs) 
    original_sequence = inputs["input_ids"] 
    input_ids = inputs["input_ids"] 
    input_ids = input_ids[:, : 64] 
    attention_mask = inputs["attention_mask"] 
    labels = inputs["labels"] 
    if args.topk is not None: 
        top_k = args.topk 
    top_p = 0.9 

    temperature = 1 

    # large_outputs = large_model.generate(input_ids = input_ids, max_length = 128, do_sample = False, output_hidden_states = True, return_dict_in_generate = True) 
    # large_outputs = large_model.generate(input_ids = input_ids, max_length = max_length + dict_kernel_maxlength[kernel_size], do_sample = False, output_hidden_states = True, return_dict_in_generate = True) 
    if args.topk is not None: 
        large_outputs = large_model.generate(input_ids = input_ids, max_length = 260, do_sample = True, top_k = top_k, output_hidden_states = True, return_dict_in_generate = True) 
    elif args.use_large_model: 
        index_to_peek = 2 
        # print("input in words is {}".format(tokenizer.decode(input_ids[0]))) 
        # print("input ids is {}".format(input_ids[index_to_peek])) 
        large_outputs = large_model.generate(input_ids = input_ids, max_length = 71, do_sample = True, output_hidden_states = True, return_dict_in_generate = True) 
        visualizing_generated_sequence(original_sequence, large_outputs, index_to_peek) 
        first_pos, second_pos, third_pos, fourth_pos, fifth_pos, sixth_pos, seventh_pos = compute_hit_counts(original_sequence, large_outputs) 
    else: 
        index_to_peek = 2 
        large_outputs = small_model.generate(input_ids = input_ids, max_length = 71, do_sample = True, output_hidden_states = True, return_dict_in_generate = True) 
        visualizing_generated_sequence(original_sequence, large_outputs, index_to_peek) 
        first_pos, second_pos, third_pos, fourth_pos, fifth_pos, sixth_pos, seventh_pos = compute_hit_counts(original_sequence, large_outputs) 
    first_pos_hit += first_pos 
    second_pos_hit += second_pos 
    third_pos_hit += third_pos 
    fourth_pos_hit += fourth_pos 
    fifth_pos_hit += fifth_pos 
    sixth_pos_hit += sixth_pos 
    seventh_pos_hit += seventh_pos 
    total_examples += input_ids.shape[0] 
    print("first_pos hit rate {} second_pos hit rate {} third_pos hit rate {} fourth_pos hit rate {} fifth_pos hit rate {} sixth_pos hit rate {} seventh_pos hit rate {}".format(first_pos_hit/total_examples, second_pos_hit/total_examples, third_pos_hit/total_examples, fourth_pos_hit/total_examples, fifth_pos_hit/total_examples, sixth_pos_hit/total_examples, seventh_pos_hit/total_examples)) 
print("first_pos hit rate {} second_pos hit rate {} third_pos hit rate {} fourth_pos hit rate {} fifth_pos hit rate {} sixth_pos hit rate {} seventh_pos hit rate {}".format(first_pos_hit/total_examples, second_pos_hit/total_examples, third_pos_hit/total_examples, fourth_pos_hit/total_examples, fifth_pos_hit/total_examples, sixth_pos_hit/total_examples, seventh_pos_hit/total_examples)) 
