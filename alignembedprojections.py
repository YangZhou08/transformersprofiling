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
from torch.utils.data import random_split 
from src.transformers import BitsAndBytesConfig 
from packaging import version 

import datetime 

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

commit_hash = get_git_commit_hash() 
print("the commit hash is {}".format(commit_hash)) 

import socket

hostname = socket.gethostname()
print("Hostname:", hostname)

if "lovelace" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_dataset = "/home/yangzho6/c4_parts" 
    dir_models = "/home/yangzho6/model_checkpoints" 
    synthesized_dir_path = "/home/yangzho6/c4llm_synthesized/" 
    synthesized_data_path = "/home/yangzho6/c4llm_synthesized/tensor_dir/" 
    dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
elif "ada" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_dataset = "/home/beidic/yangzho6/c4_parts" 
    dir_models = "/home/beidic/yangzho6/model_checkpoints" 
    synthesized_dir_path = "/home/beidic/yangzho6/c4llm_synthesized/" 
    synthesized_data_path = "/home/beidic/yangzho6/c4llm_synthesized/tensor_dir/" 
    dir_sdata = "/home/beidic/yangzho6/c4llm_synthesized/" 
else: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_dataset = "/home/yangzho6/c4_parts" 
    dir_models = "/home/yangzho6/model_checkpoints" 
    synthesized_dir_path = "/home/yangzho6/c4llm_synthesized/" 
    synthesized_data_path = "/home/yangzho6/c4llm_synthesized/tensor_dir/" 
    dir_sdata = "/home/yangzho6/c4llm_synthesized/" 

logger = logging.get_logger(__name__) 

large_model_state_dict = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).state_dict() 
small_model_state_dict = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).state_dict() 

large_model_embeddings = None 
for key in large_model_state_dict.keys(): 
    if key == "model.embed_tokens.weight": 
        print("got here found the following key {}".format(key)) 
        large_model_embeddings = large_model_state_dict[key] 
        break 
    else: 
        del large_model_state_dict[key] 

small_model_embeddings = None 
for key in small_model_state_dict.keys(): 
    if key == "model.embed_tokens.weight": 
        small_model_embeddings = small_model_state_dict[key] 
        break 
    else: 
        del small_model_state_dict[key] 

class SingleLayerProjection(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.linear = torch.nn.Linear(4096, 768) 
    
    def forward(self, x): 
        return self.linear(x) 


loss_fn = torch.nn.MSELoss() 

layerone = SingleLayerProjection() 
layerone.to("cuda") 
optimizer = torch.optim.Adam(layerone.parameters(), lr = 1e-4) 
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.9) 

large_model_embeddings = large_model_embeddings.to("cuda") 
small_model_embeddings = small_model_embeddings.to("cuda") 

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False 

if has_wandb: 
    wandb.init(project = "llm160m", name = "linearmodeltrainingalignment4") 

for i in range(10000): # 100 epochs 
    print("iteration {} out of {}".format(i, 10000)) 
    optimizer.zero_grad() 
    loss = loss_fn(layerone(large_model_embeddings), small_model_embeddings) 
    loss.backward() 
    wandb.log({"global iteration count": i, "loss": loss.item()}) 
    optimizer.step() 

wandb.finish() 

torch.save(layerone.linear.weight, "linearprojectionweighttesting.pt") 

layerone_checking = SingleLayerProjection() 
layerone_checking.linear.weight = torch.nn.Parameter(torch.load("linearprojectionweighttesting.pt")) 
layerone_checking.to("cuda") 

print("loading model to cuda to print loss ", loss_fn(layerone_checking(large_model_embeddings), small_model_embeddings).item()) 
