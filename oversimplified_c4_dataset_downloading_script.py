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
    dir_dataset = "/home/yangzho6/c4_parts" 
    dir_models = "/home/yangzho6/model_checkpoints" 

# onedataset = load_dataset("c4", "en", split = "train", cache_dir = dir_dataset) 
onedataset = load_dataset("c4", "en", split = "train", streaming = True) 
print("length of the dataset is {}".format(len(onedataset))) 
for i in range(len(onedataset)): 
    print(onedataset[i]) 
