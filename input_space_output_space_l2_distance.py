import torch 
import argparse 
# import contexttimer 

import datasets 
from datasets import load_dataset 

from src.transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from src.transformers import GPTNeoXForCausalLM 
from src.transformers import LlamaConfig, LlamaPreTrainedModel 
from src.transformers import LlamaTokenizer 

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
from src.transformers.models.llama.modeling_llama import LlamaCausalLMWeirdTwo 
# from src.transformers.models.llama.modeling_llama import LlamaWeirdLarge 
from src.transformers.models.llama.modeling_llama import LlamaWeirdLarge2 
from src.transformers.models.llama.modeling_llama import LlamaWeirdLarge3 
import time 
from torch.utils.data import random_split 
from src.transformers import BitsAndBytesConfig 
from packaging import version 
from src.transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model 
from src.transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES 

import datetime 
import os 
import inspect 

# # cache_dir = "/home/bc20/yang/" 
# dir_dataset = "/home/yangzho6/c4_parts" 
# dir_models = "/home/yangzho6/model_checkpoints2" 
# dir_sdata = "/home/yangzho6/c4llm_synthesized/" 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False 

# has_wandb = False # disable for debugging 

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

from src.transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_dataloader_sampler,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
) 
from src.transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union 
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler 
from src.transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available 

if is_apex_available():
    from apex import amp 

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met 

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin

    if version.parse(accelerate_version) > version.parse("0.20.3"):
        from accelerate.utils import (
            load_fsdp_model,
            load_fsdp_optimizer,
            save_fsdp_model,
            save_fsdp_optimizer,
        )
    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper 

if is_peft_available():
    from peft import PeftModel 

import subprocess 

def get_git_commit_hash():
    try:
        # Run the git command to get the current commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        # Decode from bytes to string
        return commit_hash.decode('utf-8')
    except subprocess.CalledProcessError:
        # Handle cases where the git command fails (e.g., not a git repository)
        return None

commit_hash = get_git_commit_hash()[: 7] # only 7 digits 
print("the commit hash is {}".format(commit_hash)) 

import datetime 
hash_of_time = str(datetime.datetime.now()).split('.')[-1] 
print("the hash of time is {}".format(hash_of_time)) 

import socket

hostname = socket.gethostname()
print("Hostname:", hostname)

if "lovelace" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/yangzho6/model_checkpoints/" 
    dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
    dir_unprocessed_dataset = "/home/yangzho6/c4_parts/downloads/" 
elif "ada" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/beidic/yangzho6/model_checkpoints/" 
    dir_sdata = "/home/beidic/yangzho6/c4llm_synthesized/" 
    dir_unprocessed_dataset = "/home/beidic/yangzho6/c4_parts/downloads/" 
else: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/yangzho6/model_checkpoints/" 
    dir_sdata = "/home/yangzho6/c4llm_synthesized/" 

logger = logging.get_logger(__name__) 

model_name = "openllama3b" 

class CustomDataset: 
    def __init__(self, data_dir, tokenizer = None, max_length = 256, kernel_size = 7): 
        # self.synthesize_dir = "/home/yangzho6/c4llm_synthesized/" 
        self.synthesize_dir = data_dir 
        # self.dataset = load_dataset('json', data_files = self.synthesize_dir + "c4synthesized_file1.json", split = "train") 
        # self.dataset = load_dataset('json', data_files = [self.synthesize_dir + 'c4synthesized_file1.json', self.synthesize_dir + 'c4synthesized_file2.json'], split="train") 
        dfiles = [] 
        if kernel_size != 4: 
            if hostname == "ada": 
                for i in range(0, 2): 
                    # filename = "c4synthesized_file1_kernel{}_{}.json".format(kernel_size, i) 
                    filename = "c4synthesized_file1_kernel{}_{}.json".format(kernel_size, i) 
                    dfiles.append(self.synthesize_dir + "{}/".format(model_name) + filename) 
            else: 
                filename = "c4synthesized_file1_kernel{}_{}.json".format(kernel_size, 0) 
                dfiles.append(self.synthesize_dir + "{}/".format(model_name) + filename) 
        else: 
            filename = "c4synthesized_file1.json" 
        self.dataset = load_dataset('json', data_files = dfiles, split = "train") 
        # self.dataset = load_dataset('json', data_files = dfiles, split = "train[:2000]") 
        # self.dataset = load_dataset('json', data_files = dfiles, split = "train[:2000]") 
        self.dict_kernel_maxlength = {2 : 64, 3 : 63, 4 : 64, 5 : 65, 6 : 66, 7 : 70, 10 : 70} 
        self.kernel_size = kernel_size 
        # self.dataset = self.dataset["train"][0: 5120] 

        self.tokenizer = tokenizer 
        self.max_length = max_length 
    
    def __len__(self): 
        return len(self.dataset) 
    
    def preprocess_dataset(self): 
        def encode_with_truncation(examples): 
            # return tokenizer(examples["text"], truncation = True, padding = "max_length", 
                            #  max_length = max_length, return_special_tokens_mask = True) 
            return tokenizer(examples["text"], padding = "max_length", max_length = self.max_length, 
                            return_attention_mask = True, return_tensors = "pt", truncation = True, 
                            add_special_tokens = True) 
        
        def loading_condensed_embeds(examples): 
            # not used because it consumes too much memory 
            return {"condensed_embeds": torch.load(examples["condensed_token_path"])} 
        
        self.dataset = self.dataset.map(encode_with_truncation, batched = True, num_proc = 4) 
        # self.dataset = self.dataset.map(loading_condensed_embeds, batched = True, num_proc = 4) 
        # self.dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 
    
    def __getitem__(self, idx): 
        item = self.dataset[idx] 
        try: 
            tensor = torch.load(item["condensed_token_path"]) 
        except IOError as e: 
            print(colored("///IOError occured replacing with an empty tensor///", "red")) 
            if model_name == "shearedllama2_7b": 
                dmodel = 2560 
            elif model_name == "openllama3b": 
                dmodel = 3200 
            elif model_name == "tinyllama": 
                dmodel = 2048 
            tensor = torch.zeros((28, dmodel), dtype = torch.float32) 
        
        if self.tokenizer is not None: 
            encoded_text = self.tokenizer( 
                item["text"][58 :], # 6 word-level tokens + BOS to be the first chunk 
                # add_special_tokens = False, 
                add_special_tokens = True, 
                padding = "max_length", 
                # max_length = 64 + self.dict_kernel_maxlength[self.kernel_size], 
                max_length = self.max_length, 
                return_attention_mask = True, 
                return_tensors = "pt", 
                truncation = True, 
            ) 
            
            # print("text is {}".format(item["text"][58 :])) 
            item['input_ids'] = encoded_text['input_ids'].squeeze(0)  # remove the batch dimension 
            # print("input_ids is {}, the length is {}".format(item["input_ids"], item["input_ids"].shape[0])) 
            item['attention_mask'] = encoded_text['attention_mask'].squeeze(0)  # remove the batch dimension 
        
        item["condensed_embeds"] = tensor 
        # print(colored("the shape of condensed_embeds is {}".format(tensor.shape), "yellow")) 
        # item["input_ids"] = torch.tensor(item["input_ids"]) 
        # item["attention_mask"] = torch.tensor(item["attention_mask"]) 

        return item 

    def split(self, train_size): 
        if isinstance(train_size, float): 
            train_size = int(train_size * len(self)) 
        eval_size = len(self) - train_size 
        return random_split(self, [train_size, eval_size]) 

tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models) 
if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
    tokenizer.padding_side = "left" 

def naive_grouping(input_ids, model): 
    embedding_searched = model.embed_tokens(input_ids) 
    # print("embedding_searched shape {} {}".format(embedding_searched.shape, embedding_searched.dtype)) 
    seq_length = embedding_searched.shape[1] 
    
    assert seq_length % 7 == 0, "seq_length is not divisible by 7" 
    added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // 7, embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
    for i in range(seq_length // 7): 
        sum = torch.zeros((embedding_searched.shape[0], embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
        for j in range(7): 
            sum += embedding_searched[:, i * 7 + j, :] 
            sum /= 7. 
            # print("sum dtype {}".format(sum.dtype)) 
        added_tensor[:, i, :] = sum 
    # print("added_tensor shape {}".format(added_tensor.shape)) 
    
    return added_tensor 

# model = LlamaWeirdLarge2.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models, sliding_window_length = 7, addonsmallmodel = 
model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 

datasetnew = CustomDataset(max_length = 203, data_dir = dir_sdata, tokenizer = tokenizer, kernel_size = 7) 
mse_loss = nn.MSELoss() 
accumulated_loss = 0. 

for i in range(len(datasetnew)): 
    data1 = datasetnew[i] 
    input_ids = data1["input_ids"].reshape(1, -1).to(torch_device) 
    added_tensor = naive_grouping(input_ids, model) 
    print("added_tensor shape {}".format(added_tensor.shape)) 
    added_tensor = added_tensor[:, : -1, :] 
    print("added_tensor shape {}".format(added_tensor.shape)) 
    loss = mse_loss(added_tensor, data1["condensed_embeds"].to(torch_device)) 
    print(colored("loss is {}".format(loss), "yellow")) 
    accumulated_loss += loss.item() 

print(colored("averaged loss is {}".format(accumulated_loss / len(datasetnew)), "yellow")) 
