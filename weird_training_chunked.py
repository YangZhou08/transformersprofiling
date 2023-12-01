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
    datasetsrc = "/home/yangzho6/c4_parts/downloads/c4_file2.json" 
    dir_models = "/home/yangzho6/model_checkpoints" 
    dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
elif "ada" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    datasetsrc = "/home/beidic/yangzho6/c4_parts/downloads/c4_file2.json" 
    dir_models = "/home/beidic/yangzho6/model_checkpoints" 
    dir_sdata = "/home/beidic/yangzho6/c4llm_synthesized/" 
else: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    datasetsrc = "/home/yangzho6/c4_parts/downloads/c4_file2.json" 
    dir_models = "/home/yangzho6/model_checkpoints" 
    dir_sdata = "/home/yangzho6/c4llm_synthesized/" 

logger = logging.get_logger(__name__) 

class CustomDataset: 
    def __init__(self, data_dir, tokenizer = None, max_length = 128, kernel_size = 4): 
        # self.synthesize_dir = "/home/yangzho6/c4llm_synthesized/" 
        self.synthesize_dir = data_dir 
        # self.dataset = load_dataset('json', data_files = self.synthesize_dir + "c4synthesized_file1.json", split = "train") 
        # self.dataset = load_dataset('json', data_files = [self.synthesize_dir + 'c4synthesized_file1.json', self.synthesize_dir + 'c4synthesized_file2.json'], split="train") 
        if kernel_size != 4: 
            filename = "c4synthesized_file1_kernel{}.json".format(kernel_size) 
        else: 
            filename = "c4synthesized_file1.json" 
        self.dataset = load_dataset('json', data_files = self.synthesize_dir + filename, split = "train") 
        self.dict_kernel_maxlength = {2 : 64, 3 : 63, 4 : 64, 5 : 65, 6 : 66, 7 : 70} 
        self.kernel_size = kernel_size 
        # self.dataset = self.dataset["train"][0: 5120] 

        self.tokenizer = tokenizer 
        self.max_length = max_length 
    
    def __len__(self): 
        return len(self.dataset) 
    
    def __getitem__(self, idx): 
        item = self.dataset[idx] 
        tensor = torch.load(item["condensed_token_path"]) 

        if self.tokenizer is not None: 
            encoded_text = self.tokenizer( 
                item["text"], 
                # add_special_tokens = False, 
                add_special_tokens = True, 
                padding = "max_length", 
                max_length = 64 + self.dict_kernel_maxlength[self.kernel_size], 
                return_attention_mask = True, 
                return_tensors = "pt", 
                truncation = True, 
            ) 
            
            item['input_ids'] = encoded_text['input_ids'].squeeze(0)  # remove the batch dimension
            item['attention_mask'] = encoded_text['attention_mask'].squeeze(0)  # remove the batch dimension 
        
        item["condensed_embeds"] = tensor 

        return item 

    def split(self, train_size): 
        if isinstance(train_size, float): 
            train_size = int(train_size * len(self)) 
        eval_size = len(self) - train_size 
        return random_split(self, [train_size, eval_size]) 

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help') 

parser.add_argument("--group1lr", type = float, default = 2e-4) 
parser.add_argument("--group2lr", type = float, default = 2e-3) 
parser.add_argument("--experiment_setting", type = str, default = "setting0") 
parser.add_argument("--eval_mode", action="store_true", default = False) 
parser.add_argument("--embedding_pretrained", action = "store_true", default = False) 
parser.add_argument("--kernel_size", type = int, default = 4) 
parser.add_argument("--use_plain_model", action = "store_true", default = False) 

args = parser.parse_args() 
if args.embedding_pretrained: 
    args.group2lr = None # we enforce it 
print(args) 

# defining tokenizer 
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir) 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
# tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", cache_dir = dir_models) 
# tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
# print("the tokenizer pad token id is {}".format(tokenizer.pad_token_id)) 
# tokenizer.pad_token = "[PAD]" 
if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
tokenizer.padding_side = "left" 

# backup dataset 
# onedataset = load_dataset('json', data_files = '/home/yangzho6/c4llm_synthesized/c4synthesized_file1.json', split = "train") 
onedataset = load_dataset('json', data_files = datasetsrc, split = "train") 
# onedataset = load_dataset("c4", "en", split = "train", cache_dir = dir_dataset) 
d = onedataset.train_test_split(test_size = 0.1) 
# print(d["train"], d["test"]) 
# max_length = small_model.config.max_position_embeddings 
# def encode_with_truncation(examples): 
    # return tokenizer(examples["text"], truncation=True, padding="max_length",
                #    max_length=max_length, return_special_tokens_mask=True) 
def encode_with_truncation(examples): 
    # return tokenizer(examples["text"], truncation = True, padding = "max_length", 
                    #  max_length = max_length, return_special_tokens_mask = True) 
    return tokenizer(examples["text"], padding = "max_length", max_length = 128, 
                     return_attention_mask = True, return_tensors = "pt", truncation = True) 
train_dataset = d["train"].map(encode_with_truncation, batched = True, num_proc = 4) 
test_dataset = d['test'].map(encode_with_truncation, batched = True, num_proc = 4) 
# print("The model max length is {}".format(small_model.config.max_position_embeddings)) 
train_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 
test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 
'''
# custom dataset 
# defining custom dataset 
kernel_size = args.kernel_size 

datasetnew = CustomDataset(data_dir = dir_sdata, tokenizer = tokenizer, kernel_size = kernel_size) 
train_set, test_set = datasetnew.split(0.98)     # 712k * 0.95 = 676k 712k * 0.05 = 36k 
                                                 # 356k * 0.99 = 352k 356k * 0.01 = 3.6k 
''' 

model = LlamaCausalLMWeirdTwo.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
for name, param in model.named_parameters(): 
    print(name) 
exit(0) 


# for llama model we need to add the padding token 
small_model.config.pad_token_id = tokenizer.pad_token_id 
# print(small_model.embed_projection.weight.dtype) 

pretraining_weights_group = []
newly_initialized_group = [] 
for k, v in small_model.named_parameters(): 
    if "embed_projection" in k: 
        print(k) 
        newly_initialized_group.append(v) 
    else: 
        pretraining_weights_group.append(v) 
print(len(pretraining_weights_group), len(newly_initialized_group)) 

if not args.embedding_pretrained: 
    print("*** we are not using pretrained embeddings ***") 
    custom_optimizer = torch.optim.AdamW([
        # {"params": pretraining_weights_group, "lr": 2e-4}, 
        # {"params": newly_initialized_group, "lr": 2e-3}, 
        {"params": pretraining_weights_group, "lr": float(args.group1lr)}, 
        {"params": newly_initialized_group, "lr": float(args.group2lr)}, 
    ]) 
else: 
    print("*** we are using pretrained embeddings ***") 
    if not os.path.exists("linearprojectionweighttesting.pt"): 
        raise ValueError("please run analyzing_initial_perfromance.py before runnint this setting") 
    for param in newly_initialized_group: 
        pretraining_weights_group.append(param) 
    custom_optimizer = torch.optim.AdamW([
        # {"params": pretraining_weights_group, "lr": 2e-4}, 
        {"params": pretraining_weights_group, "lr": float(args.group1lr)}, 
    ]) 

def _lr_scheduler_rewriting(current_step, *, num_warmup_steps: int, num_training_steps: int): 
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

# model_path = "/home/bc20/yang" 
# model_path = "/home/yangzho6/model_checkpoints" 
model_path = dir_models 
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=5,            # number of training epochs, feel free to tweak
    per_device_train_batch_size = 128, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=4,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=256,  # evaluation batch size
    # logging_steps=1, 
    logging_steps = 250,           # evaluate, log and save model checkpoints every 1000 step
    # save_steps=1000, 
    # save_steps = 2000, 
    save_steps = 250, 
    # learning_rate=5e-7, 
    # learning_rate=5e-5, 
    learning_rate=2e-4, 
    # learning_rate = 1e-4, 
    # learning_rate = 5e-6, 
    # learning_rate = 0, 
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=5,            # whether you don't have much space so you let only 3 model weights saved in the disk 
    # lr_scheduler_type = "cosine", 
    warmup_steps = 100, 
) 

max_length = 128 
if has_wandb: 
    project_setting = args.experiment_setting if args.eval_mode is False else "finetuning" 
    today = datetime.date.today() 
    wandblogconfigs = {**(training_args.to_dict()), **(args.__dict__)} 
    wandblogconfigs["git_commit"] = commit_hash 
    wandblogconfigs["time_hash"] = hash_of_time 
    # wandb.init(project = "llm160m", config = training_args, name="{}_{}".format(today, project_setting)) 
    wandb.init(project = "llm160m", config = wandblogconfigs, name = "{}_{}_{}".format(today, project_setting, "custom" if args.use_plain_model is False else "plain")) 

weightmodelfirst = next(small_model.parameters()) 
# print(weightmodelfirst.dtype) 
print(colored(weightmodelfirst.dtype, "red")) 

def compute_metrics(p): 
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits = p.predictions 
    labels = p.label_ids 
    # logits = logits[: -1] 
    print("logits have shape {}".format(len(logits))) 
    # for i in range(len(logits)): 
    #     print("logits[{}] has shape {}".format(i, logits[i].shape)) 
    logits = logits[0][:, : -1, :] 
    logits = torch.tensor(logits) 
    logits = logits.view(-1, logits.shape[-1]) 
    # logits = logits[:, : -1, :] 
    labels = labels[:, 1:] 
    labels = torch.tensor(labels) 
    labels = labels.view(-1) 
    print("logits have shape {}".format(logits.shape)) 
    probs = torch.softmax(torch.tensor(logits), dim = -1) 
    
    loss = nn.CrossEntropyLoss()(torch.tensor(logits), torch.tensor(labels)).item() 
    perplexity = torch.exp(torch.tensor(loss)).item() 

    pred = torch.argmax(probs, dim = -1) 
    '''
    wandb.login() 

    wandb.log({"evaluation_acc": accuracy_score(p.labels_ids, pred), 
                "evaluation_f1": precision_recall_fscore_support(p.label_ids, pred, average = 'weighted'), 
                "evaluation_perplexity": perplexity, 
    }) 
    ''' 
    output = {
        'accuracy': accuracy_score(labels, pred), 
        # 'f1': precision_recall_fscore_support(p.label_ids, pred, average = 'weighted'), 
        'perplexity': perplexity,
    } 
    print(colored(output, "red")) 
    return output 

# print(trainer.lr_scheduler.state_dict()) 
# exit(0) 

'''
trainer = Trainer(
    model = small_model, 
    args = training_args, 
    train_dataset = train_dataset, 
    data_collator = data_collator, 
) 
''' 

torch.autograd.set_detect_anomaly(True) 

trainer.train() 

wandb.finish() 

