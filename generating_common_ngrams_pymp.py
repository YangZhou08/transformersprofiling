from collections import Counter
import re 

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
import json 
import multiprocessing as mp 

torch_device = "cuda:0" 

parser = argparse.ArgumentParser() 
parser.add_argument("--num_ngrams", type = int, default = 1000) 
parser.add_argument("--length_of_ngram", type = int, default = 3) 
parser.add_argument("--num_workers", type = int, default = 8) 
parser.add_argument("--num_pass_iteration", type = int, default = 1) 

args = parser.parse_args() 

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 

dataset = load_dataset('json', data_files = datasetsrc, split = "train[:4000]") 

length_of_dataset = len(dataset) 
num_workers = args.num_workers 
subdatasets = [] 
# subset_in_use = [dataset[i : min(length_of_dataset, (i + 1) * ((length_of_dataset + num_workers - 1) // num_workers))] for i in range(num_workers)] # evenly partitioned the dataset into num_workers splits 

def generate_ngrams(tokens, n=3):
    return zip(*[tokens[i:] for i in range(n)]) 

def worker(num, iteration_count): 
    idx_start, idx_end = subdatasets[num] 
    # subdatasplit = dataset[idx_start : idx_end] 
    print("worker {} started idx_start {} idx_end {}".format(num, idx_start, idx_end)) 
    batch_counter = Counter() 
    if num == 0: 
        for i in tqdm(range(idx_start, idx_end)): 
            # text = subdatasplit[i]["text"] 
            text = dataset[i]["text"] 
            tokens = tokenizer.tokenize(text) 
            three_ngrams = generate_ngrams(tokens, args.length_of_ngram) 
            three_ngrams = list(three_ngrams) 
            print("worker {} length of three_ngrams {}".format(num, len(three_ngrams))) 
            batch_counter.update(three_ngrams) 
    else: 
        for i in range(idx_start, idx_end): 
            # text = subdatasplit[i]["text"] 
            text = dataset[i]["text"] 
            tokens = tokenizer.tokenize(text) 
            three_ngrams = generate_ngrams(tokens, args.length_of_ngram) 
            three_ngrams = list(three_ngrams) 
            print("worker {} length of three_ngrams {}".format(num, len(three_ngrams))) 
            batch_counter.update(three_ngrams) 
    return 
    most_common_3grams = batch_counter.most_common(args.num_ngrams) 
    with open(synthesized_dir_path + "mostcommon1000003gramsworker{}_iterationcount{}.json".format(num, iteration_count), "w") as f: 
        json.dump(most_common_3grams, f) 
    print("worker {} write file to {}".format(num, synthesized_dir_path + "mostcommon1000003gramsworker{}_iterationcount{}.json".format(num, iteration_count))) 


processes = [] 
global_datasetidx = 0 
num_iterations = args.num_pass_iteration 

for j in range(num_iterations): 
    print("iteration {}".format(j)) 
    # set_in_used = dataset[global_datasetidx : min(length_of_dataset, global_datasetidx + (length_of_dataset + num_iterations - 1) // num_iterations)] 
    # global_datasetidx += (length_of_dataset + 5 - 1) // 5 
    global_datasetidx += (length_of_dataset + num_iterations - 1) // num_iterations 
    # length_of_subset = len(set_in_used) 
    length_of_subset = (len(dataset) + num_iterations - 1) // num_iterations 
    subdivision_length = (length_of_subset + num_workers - 1)//num_workers 
    # subdatasets = [set_in_used[i : min(length_of_subset, (i + 1) * ((length_of_subset + num_workers - 1) // num_workers))] for i in range(num_workers)] # evenly partitioned the dataset into num_workers splits 
    # subdatasets = [set_in_used[k * subdivision_length : min(length_of_subset, (k + 1) * subdivision_length)] for k in range(num_workers)] 
    subdatasets = [(k * subdivision_length, min(length_of_subset, (k + 1) * subdivision_length)) for k in range(num_workers)] 
    for i in range(num_workers): 
        p = mp.Process(target = worker, args = (i, j)) 
        processes.append(p) 
        p.start() 

    for p in processes: 
        p.join() 
    print("finish iteration {}".format(j)) 
    exit(0) 

collection = Counter() 
for i in range(num_iterations): 
    for j in range(num_workers): 
        with open(synthesized_dir_path + "mostcommon1000003gramsworker{}_iterationcount{}.json".format(j, i), "r") as f: 
            data = json.load(f) 
            print(len(data)) 
            for d in data: 
                # print(d) 
                # print(type(d)) 
                # print(d[0]) 
                collection.update([tuple(d[0]), d[1]]) 
exit(0) 

globalhottestngram = collection.most_common(args.num_ngrams) 
print(type(globalhottestngram), len(globalhottestngram)) 
exit(0) 
with open(synthesized_dir_path + "mostcommon1000003grams.json", "w") as f: 
    json.dump(globalhottestngram, f) 

greedy_finding = set() 
for i in range(args.num_ngrams): 
    greedy_finding.add(globalhottestngram[i][0]) 

print("checking with the sequential implementation") 
sequential_counts = Counter() 
for text in tqdm(dataset["text"]): 
    tokens = tokenizer.tokenize(text) 
    three_ngrams = zip(*[tokens[i:] for i in range(3)]) 
    three_ngrams = list(three_ngrams) 
    sequential_counts.update(three_ngrams) 

sequential_n = sequential_counts.most_common(args.num_ngrams) 
sequential_finding = set() 
for i in range(args.ngrams): 
    sequential_finding.add(sequential_n[i][0]) 

hottestsequentialintersection = greedy_finding & sequential_finding 
print(len(hottestsequentialintersection)/len(sequential_finding)) 
