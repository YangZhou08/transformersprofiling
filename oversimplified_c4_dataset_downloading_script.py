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

import torch 
import argparse 
# import contexttimer 

import datasets 
from datasets import load_dataset 
import socket

hostname = socket.gethostname()
print("Hostname:", hostname)

if "lovelace" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_dataset = "/home/yangzho6/c4_parts" 
    dir_models = "/home/yangzho6/model_checkpoints" 
elif "ada" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_dataset = "/home/beidic/yangzho6/c4_parts" 
    dir_models = "/home/beidic/yangzho6/model_checkpoints" 
else: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    datasetsrc = "/fsx-storygen/beidic/yang/c4/en/c4_file2.json" 
    datasetparent = "/fsx-storygen/beidic/yang/c4/en/" 
    dir_dataset = "/fsx-storygen/beidic/yang/c4_parts" 
    dir_models = "/fsx-storygen/beidic/yang/model_checkpoints" 
    # synthesized_dir_path = "/data/home/beidic/yang/c4llm_synthesized/{}_topk{}/".format(model_name, args.topk if args.topk is not None else "na") 
    # synthesized_data_path = "/data/home/beidic/yang/c4llm_synthesized/{}_topk{}/tensor_dir/".format(model_name, args.topk if args.topk is not None else "na") 

onedataset = load_dataset("c4", "en", split = "train", cache_dir = dir_dataset) 
# onedataset = load_dataset("c4", "en", split = "train", streaming = True) 
# train_dataloader = torch.utils.data.DataLoader(onedataset, batch_size = 20, shuffle = False) 
# print("length of the dataset is {}".format(len(onedataset))) 
'''
for i, example in enumerate(onedataset): 
    print(example["text"]) 
    print() 
    if i > 19: 
        break 
''' 
