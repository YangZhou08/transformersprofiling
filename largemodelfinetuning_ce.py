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
from src.transformers.models.llama.modeling_llama import LlamaCausalLMWeirdTwo 
import time 
from torch.utils.data import random_split 
from src.transformers import BitsAndBytesConfig 
from packaging import version 
from src.transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model 
from src.transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES 

import datetime 
import os 

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
    dir_models = "/home/yangzho6/model_checkpoints" 
    dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
    dir_unprocessed_dataset = "/home/yangzho6/c4_parts/downloads/" 
elif "ada" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/beidic/yangzho6/model_checkpoints" 
    dir_sdata = "/home/beidic/yangzho6/c4llm_synthesized/" 
    dir_unprocessed_dataset = "/home/beidic/yangzho6/c4_parts/downloads/" 
else: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/yangzho6/model_checkpoints" 
    dir_sdata = "/home/yangzho6/c4llm_synthesized/" 

logger = logging.get_logger(__name__) 

class CustomTrainer(Trainer): 
    def __init__(self, n = 7, tokenizer = None, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.n = n 
        self.tokenizer = tokenizer 
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None 
        
        print("printing out input keys and values shape, there are {} in total".format(len(inputs.keys()))) 
        for key, value in inputs.items(): 
            print("key {}, value shape {}".format(key, value.shape)) 
        exit(0) 
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss 

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
# tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models) 
# tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
tokenizer.padding_side = "left" 

list_of_datasets = ["c4_file{}.json".format(i) for i in range(1, 3)] 
list_of_datasets = [dir_unprocessed_dataset + path for path in list_of_datasets] 
onedataset = load_dataset("json", data_files = list_of_datasets, split = "train[:1000]") 
d = onedataset.train_test_split(test_size = 0.005) # 0.995 for training, 0.005 for testing 

def encode_with_truncation(examples): 
    # return tokenizer(examples["text"], truncation = True, padding = "max_length", 
                    #  max_length = max_length, return_special_tokens_mask = True) 
    return tokenizer(examples["text"], padding = "max_length", max_length = 259, 
                     return_attention_mask = True, return_tensors = "pt", truncation = True) 

train_dataset = d["train"].map(encode_with_truncation, batched = True, num_proc = 4) 
test_dataset = d["test"].map(encode_with_truncation, batched = True, num_proc = 4) 

large_model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
# large_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
# large_model = LlamaForCausalLM.from_pretrained("princeton-nlp/Sheared-LLaMA-2.7B", cache_dir = dir_models) 
large_model.train() 

# TODO change the following code to use the checkpoint of the best trained window 7 model 
small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models) 

small_state_dict_for_model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).state_dict() 
small_model = SimpleSmallModel(small_config, hostname = hostname, sliding_window_length = 7) 

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

small_model = small_model.to(torch_device) 
small_model.eval() # at start we avoid training the small model 

large_model.config.pad_token_id = tokenizer.pad_token_id 
small_model.config.pad_token_id = tokenizer.pad_token_id 

def naive_grouping(examples): 
    # I found that using the multiprocessing in the dataset cannot process neural networks 
    # this function is not used switching everything inside the neural network forward function 
    input_ids = examples["input_ids"] 
    input_ids = torch.tensor(input_ids).to(torch_device) 
    print("got here inside the naive_grouping function") 
    embedding_searched = large_model.get_input_embeddings()(input_ids) # shape (batch_size, seq_len, hidden_size) 
    print("embedding_searched shape {}".format(embedding_searched.shape)) 
    # operate on the seq_len dimension 
    # because of truncation and padding, the seq_len dimension is guaranteed to be multiples of 7 
    seq_length = embedding_searched.shape[1] 
    added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // 7, embedding_searched.shape[2])).to(torch_device) 
    practice_attention_mask = torch.ones_like(added_tensor).to(torch_device) 
    for i in range(seq_length // 7): 
        sum = torch.zeros((embedding_searched.shape[0], 1, embedding_searched.shape[2])) 
        all_pad = True 
        for j in range(7): 
            sum += embedding_searched[:, i * 7 + j, :] 
            sum /= 7. 
            if (input_ids[:, i * 7 + j] != tokenizer.pad_token_id): 
                all_pad = False 
        added_tensor[:, i, :] = sum 
        if all_pad: 
            practice_attention_mask[:, i, :] = 0 
    print("added_tensor shape {}".format(added_tensor.shape)) 
    
    return {"input_ids_chunk": added_tensor, "attention_mask_chunk": practice_attention_mask} 

def group_attention_map_chunked_generation(examples): 
    # this function is for generating the chunked attention mask 
    
    input_ids = examples["input_ids"] 
    input_ids = torch.tensor(input_ids) 
    print("input_ids shape {}".format(input_ids.shape)) 
    if len(input_ids.shape) == 1: 
        input_ids = input_ids.unsqueeze(0) 
    
    seq_length = input_ids.shape[1] 
    attention_mask_chunk = torch.ones((input_ids.shape[0], seq_length // 7)) 
    assert input_ids.shape[1] % 7 == 0 
    
    for i in range(input_ids.shape[0]): 
        for j in range(seq_length // 7): 
            all_pad = True 
            for k in range(7): 
                if input_ids[i, j * 7 + k] != tokenizer.pad_token_id: 
                    all_pad = False 
            
            if all_pad: 
                attention_mask_chunk[i, j] = 0 
    
    return {"attention_mask_chunk": attention_mask_chunk} 

train_dataset = train_dataset.map(group_attention_map_chunked_generation, batch_size = 10, num_proc = 4) 
exit(0) 
test_dataset = test_dataset.map(naive_grouping, batched = True, num_proc = 8) 

# large_model = large_model.to(torch_device) 

train_dataset.set_format(type = "torch", columns = ["input_ids_chunk", "attention_mask_chunk", "input_ids", "attention_mask"]) 
test_dataset.set_format(type = "torch", columns = ["input_ids_chunk", "attention_mask_chunk", "input_ids", "attention_mask"]) 

param_group = [] 
for param in large_model.parameters(): 
    param.requires_grad = True 
    param_group.append(param) 
print("length of param_group {}".format(len(param_group))) 

custom_optimizer = torch.optim.AdamW(param_group, lr = 5e-5) 
# custom_optimizer = torch.optim.AdamW(param_group, lr = 1e-4) 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

# model_path = "/home/bc20/yang" 
# model_path = "/home/yangzho6/model_checkpoints" 
model_path = dir_models 
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    # evaluation_strategy="steps",    # evaluate each `logging_steps` steps 
    overwrite_output_dir=True,      
    num_train_epochs=5,            # number of training epochs, feel free to tweak
    per_device_train_batch_size = 2, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=4,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size= 2,  # evaluation batch size
    # logging_steps=1, 
    logging_steps = 300,          # evaluate, log and save model checkpoints every 1000 step
    # save_steps=1000, 
    # save_steps = 2000, 
    save_steps = 300, 
    # learning_rate=5e-7, 
    # learning_rate=5e-5, 
    # learning_rate=2e-4, 
    learning_rate = 2e-4, 
    # learning_rate = 1e-4, 
    # learning_rate = 5e-6, 
    # learning_rate = 0, 
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training 
    save_total_limit=5,            # whether you don't have much space so you let only 3 model weights saved in the disk 
    # lr_scheduler_type = "cosine", 
    warmup_steps = 25, 
    label_names = ["labels"], 
    remove_unused_columns = True, 
    save_strategy = "steps", 
    evaluation_strategy = "steps", 
) 

if has_wandb: 
    today = datetime.date.today() 
    wandblogconfigs = training_args.to_dict() 
    wandblogconfigs["git_commit"] = commit_hash 
    wandblogconfigs["time_hash"] = hash_of_time 
    wandb.init(project = "chunkedlargefinetuning", config = wandblogconfigs, name = "large_small_ce{}_{}".format(today, "unmasked")) 

trainer = CustomTrainer(
    model = small_model, 
    args = training_args, 
    train_dataset = train_dataset, 
    eval_dataset = test_dataset, 
    data_collator = data_collator, 
    optimizers = (custom_optimizer, None), 
    tokenizer = tokenizer, 
) 

torch.autograd.set_detect_anomaly(True) 

trainer.train() 

wandb.finish() 
