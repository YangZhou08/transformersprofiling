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

torch_device = "cuda:0" 

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 

dataset = load_dataset('json', data_files = datasetsrc, split = "train[:1000]") 

def generate_ngrams(tokens, n=3):
    return zip(*[tokens[i:] for i in range(n)])

def count_batch(batch): 
    batch_counter = Counter() 
    # print(len(batch["text"])) 
    for text in batch["text"]: 
        tokens = tokenizer.tokenize(text) 
        three_ngrams = zip(*[tokens[i:] for i in range(3)]) 
        three_ngrams = list(three_ngrams) 
        batch_counter.update(three_ngrams) 

    ngrams_counts = [(ngram, count) for ngram, count in batch_counter.items()] 
    # print(len(ngrams_counts)) 
    # ngram, count = ngrams_counts[0] 
    ngrams, counts = zip(*ngrams_counts) 
    ngrams = list(ngrams) 
    counts = list(counts) 
    print(len(ngrams), len(counts)) 
    # ngrams, counts = zip(*ngrams_counts) if ngrams_counts else ([], []) 
    # ngrams = list(ngrams) 
    # counts = list(counts) 

    # return {"ngrams_counts": [ngrams_counts] * 100} 
    ngram_list = [ngrams] * len(batch["text"]) 
    print(len(ngram_list)) 
    print(len(ngram_list[0])) 
    counts_list = [counts] * len(batch["text"]) 
    print(len(counts_list[0])) 
    return {"ngrams": ngram_list, "counts": counts_list}  

num_pros = 4 
batch_size = 100 

batched_counts = dataset.map( 
    count_batch, 
    batched = True, 
    num_proc = num_pros, 
    batch_size = batch_size, 
) 

total_counts = Counter()
total_length = len(batched_counts) 
num_batch = total_length // (num_pros * batch_size) 
remaining_length = total_length % (num_pros * batch_size) 
size_stride = [100] * (num_batch * num_pros) 
for i in range(4): 
    size_stride.append(remaining_length//4) 
print(size_stride) 
index = 0 
for idx in tqdm((size_stride[:-1])): 
    index += idx 
    ngrams = batched_counts[index]["ngrams"]  # Access the first (and only) element in the list
    counts = batched_counts[index]["counts"]  # Access the first (and only) element in the list 
    print(len(ngrams), len(counts)) 

    # Update the total counts
    for j in range(len(ngrams)): 
        ngram = ngrams[j] 
        # print(len(ngram)) 
        ngram = tuple(ngram) 
        count = counts[j] 
        # print(len(ngram)) 
        # for ngram, count in zip(ngrams, counts):
        total_counts[ngram] += count 

# Now total_counts has the aggregated count of all ngrams 

print(total_counts.most_common(200)) 

print("checking with the sequential implementation") 
sequential_counts = Counter() 
for text in tqdm(dataset["text"]): 
    tokens = tokenizer.tokenize(text) 
    three_ngrams = zip(*[tokens[i:] for i in range(3)]) 
    three_ngrams = list(three_ngrams) 
    sequential_counts.update(three_ngrams) 

print(sequential_counts.most_common(200)) 
