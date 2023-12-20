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
from src.transformers.data.data_collator import DataCollatorForLanguageModeling2 
from src.transformers.generation.utils import GenerationConfig 
from src.transformers.models.llama.modeling_llama import LlamaForCausalLM, SimpleSmallModel 
from src.transformers.models.llama.modeling_llama import LlamaCausalLMWeirdTwo 
from torch.utils.data import random_split 
from src.transformers import BitsAndBytesConfig 
from packaging import version 
from collections.abc import Mapping 
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

has_wandb = False # disable for debugging 

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
import json 

hostname = socket.gethostname()
print("Hostname:", hostname)

if "lovelace" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    # datasetsrc = "/home/yangzho6/c4_parts/downloads/c4_file2.json" 
    datasetsrc = "/home/yangzho6/c4llm_synthesized/c4synthesized_file1_kernel5.json" 
    synthesized_dir_path = "/home/yangzho6/c4llm_synthesized/" 
    dir_models = "/home/yangzho6/model_checkpoints" 
    dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
elif "ada" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    # datasetsrc = "/home/beidic/yangzho6/c4_parts/downloads/c4_file2.json" 
    datasetsrc = "/home/beidic/yangzho6/c4llm_synthesized/c4synthesized_file1.json" 
    synthesized_dir_path = "/home/beidic/yangzho6/c4llm_synthesized/" 
    dir_models = "/home/beidic/yangzho6/model_checkpoints" 
    dir_sdata = "/home/beidic/yangzho6/c4llm_synthesized/" 
else: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    datasetsrc = "/home/yangzho6/c4_parts/downloads/c4_file2.json" 
    synthesized_dir_path = "/home/yangzho6/c4llm_synthesized/" 
    dir_models = "/home/yangzho6/model_checkpoints" 
    dir_sdata = "/home/yangzho6/c4llm_synthesized/" 

logger = logging.get_logger(__name__) 

parser = argparse.ArgumentParser( 
                    prog = "ProgramName", 
                    description = "What the program does", 
                    epilog = "Text at the bottom of help") 

parser.add_argument("--n", type = int, default = 3) 

args = parser.parse_args() 
print(args) 

def log_dict_converterc(filename, preproc, tokenizer): 
    import ast 

    with open(filename, "r") as f: 
        data = json.load(f) 
        
        print(len(data)) 
        
        data = {tuple(d[0]): d[1] for d in data} 
        if not preproc: 
            return data 
        else: 
            # first take all the keys out 
            keys = list(data.keys()) 

            # then we tokenize them 
            assert tokenizer is not None 
            output_keys = [] 
            for idx, key in tqdm(enumerate(keys)): 
                # print(key) 
                if not isinstance(key, list): 
                    key = list(key) 
                trial_key = [] 
                for i, seg in enumerate(key): 
                    if seg == "<0x0A>": 
                        trial_key.append("\n") 
                    else: 
                        trial_key.append(seg) 
                keybinding = "".join(trial_key) 
                encodedtensor = tokenizer(keybinding, add_special_tokens = False, return_attention_mask = False, return_tensors = "pt")["input_ids"].squeeze(0) 
                if encodedtensor[0].item() == 29871: 
                    encodedtensor = encodedtensor[1: ] 
                # print(encodedtensor) 
                tokencat = encodedtensor 
                if encodedtensor.shape[0] != args.n: 
                    local_tensor = [] 
                    # print(colored("encodedtensor shape not equalling what we want", "red")) 
                    if encodedtensor.shape[0] < args.n: 
                        for seg in key: 
                            if seg == "<0x0A>": 
                                seg = "\n" 
                            output_tokenized_keys = tokenizer(seg, add_special_tokens = False, return_attention_mask = False, return_tensors = "pt") 
                            # local_tensor.append(output_tokenized_keys["input_ids"].squeeze(0)) 
                            tensorofinterest = output_tokenized_keys["input_ids"].squeeze(0) 
                            # print(tensorofinterest) 
                            # if local_tensor.shape[0] == 1: 
                            if tensorofinterest.shape[0] != 1: 
                                # assert local_tensor.shape[0] == 2 
                                if tensorofinterest[0] == 29871: 
                                    # print(seg, tensorofinterest) 
                                    tensorofinterest = tensorofinterest[1:] 
                            local_tensor.append(tensorofinterest) 
                        tokencat = torch.cat(local_tensor, dim = 0) 
                        # print(tokencat) 
                
                # tokencat = torch.cat(local_tensor, dim = 0) 
                if tokencat.shape[0] != args.n: 
                    for i in range(tokencat.shape[0] - (args.n - 1)): 
                        cat1 = tokencat[i : i + args.n] 
                        output_keys.append(cat1) 
                        print(colored("adding tokens tensor {}".format(cat1), "yellow")) 
                else: 
                    output_keys.append(tokencat) 
                    print(colored("adding tokens tensor {}".format(tokencat), "yellow")) 
                '''
                print(local_tensor) 
                for seg in local_tensor: 
                    for i in range(seg.shape[0]): 
                        print(tokenizer.decode(seg[i])) 
                ''' 
                # output_keys.append(output_tokenized_keys["input_ids"].squeeze(1)) 
            output_keys = torch.stack(output_keys, dim = 0) 
            print("output_keys shape is {}".format(output_keys.shape)) 
            return output_keys 

class CustomTrainer(Trainer): 
    def __init__(self, common_n_gram_list, n = 3, use_filtered_hot_labels = False, generated_token_start_idx = 64, tokenizer = None, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.n = n 
        self.common_n_gram_list = common_n_gram_list 
        self.use_filtered_hot_labels = use_filtered_hot_labels 
        self.training_mode = True 
        self.generated_token_start_idx = generated_token_start_idx 
        self.iteration_count = 0 
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
        # outputs = model(**inputs) 
        input_ids = inputs["input_ids"] 
        attention_mask = inputs["attention_mask"] 
        label2 = inputs["labels"] 

        # hot_n_grams = self.common_n_gram_dict.keys() 

        # further data collator steps 
        print(type(model)) 
        assert isinstance(model, LlamaCausalLMWeirdTwo) 
        outputs = model(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            labels = label2, 
            output_hidden_states = True, 
            output_attentions = True, 
            return_dict = True, 
            hot_n_grams = self.common_n_gram_list, 
            use_filtered_hot_labels = self.use_filtered_hot_labels, 
            compute_original_output = not self.training_mode, 
        ) 
        
        print("outputs have shape {}".format(len(outputs))) 
        print(colored("model running loss: {}".format(outputs[0].item()), "yellow")) 
        if self.training_mode and has_wandb: 
            wandb.log({"training_loss": outputs[0].item()}) 

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

    def not_used_acceptance_length_compute(self, logits, labels, loss, input_attention_mask, output_step): 
        # computing the average acceptance length 
        # first, folding the original_model_logits 
        label_actual_mask = torch.cat((label_actual_mask, torch.ones((label_actual_mask.shape[0], 1)).to(torch_device)), dim = 1) # dimension (batch_size, seq_len - n + 1) 
        list_folding_logits = [] 
        for i in range(self.n): 
            list_folding_logits.append(original_model_logits[:, i : original_seq_len - self.n + i + 1, :]) 
        original_model_logits = torch.stack(list_folding_logits, dim = 2) # dimension (batch_size, seq_len - n + 1, n, vocab_size) 
        # print("the shape of original_model_logits is {} expected (batch_size, seq_len - n + 1, n, vocab_size)".format(original_model_logits.shape)) 
        model_output_logits = model_output_logits[:, : -self.n + 1, ...] # dimension (batch_size, seq_len - n + 1, n, vocab_size) 
        # print("the shape of model_output_logits is {} expected (batch_size, seq_len - n + 1, n, vocab_size)".format(model_output_logits.shape)) 
        q = F.softmax(model_output_logits, dim = -1) 
        # print("the shape of q is {} expected (batch_size, seq_len - n + 1, n, vocab_size)".format(q.shape)) 
        p = F.softmax(original_model_logits, dim = -1) 
        
        outputsq = torch.max(q, dim = -1) 
        q = outputsq.values 
        idx_q = outputsq.indices 
        # print("the shape of index_q is {}".format(idx_q.shape)) 
        # print("the shape of p is {}".format(p.shape)) 
        
        p = torch.gather(p, -1, idx_q.unsqueeze(-1)).squeeze(-1) # The reason why regular direct index won't work is because the index is not of the same dimension as p 
        p = p.to(torch_device) 
        # print("the shape of p is {}".format(p.shape)) 
        
        r = torch.rand_like(q).to(q.device) # dimension (batch_size, seq_len - n + 1, n) 
        # print("printing out r {}".format(r[0].shape)) 
        mask = r > (p/q) # 1 is reject, 0 is accept, dimension is (batch_size, seq_len - n + 1, n) 
        # print("printing out mask shape {}".format(mask.shape)) 
        assert mask.shape[-1] == self.n 
        '''
        # a small checking piece 
        print("batch 0, first 50 elements in p are {}".format(p[0, : 20, :])) 
        print("batch 0, first 50 elements in q are {}".format(q[0, : 20, :])) 
        print("batch 0, first 20 elements in (p/q) are {}".format((p/q)[0, : 20, :])) 
        print("batch 0, first 20 elements in r are {}".format(r[0, : 20, :])) 
        print("batch 0, first 20 elements in mask are {}".format(mask[0, : 20, :])) 
        ''' 
        mask = mask.reshape(-1, self.n) # dimension (batch_size * (seq_len - n + 1), n) 
        total_acceptance_length = 0 
        row_indices, col_indices = torch.nonzero(mask, as_tuple = True) 
        # print("shape of row_indices is {} shape of col_indices is {}".format(row_indices.shape, col_indices.shape)) 
        idx_row_col_traversal = 0 
        total_counted_pos = 0 
        # print("the shape of input_attention_mask is {}".format(input_attention_mask.shape)) 
        # print("the shape of label_actual_mask is {}".format(label_actual_mask.shape)) 
        for i in range(q.shape[0]): 
            # boundary check 
            if idx_row_col_traversal >= row_indices.shape[0]: 
                break 
            for j in range(q.shape[1]): 
                # row_i = i * mask.shape[1] + j 
                row_i = i * q.shape[1] + j 
                print(i, j) 
                # if input_attention_mask[i, j] == 0: 
                if label_actual_mask[i, j] == 0: 
                    # we skip this token 
                    print("we skip at batch size {} position {} row_i {} row_indices is at {}.format(i, j, row_i, row_indices[idx_row_col_traversal])") 
                    while idx_row_col_traversal < row_indices.shape[0] and row_indices[idx_row_col_traversal] == row_i: 
                    # while row_indices[idx_row_col_traversal] <= row_i: # should essentailly be ==, since previously we guarantee that row_indices is right at the new pos 
                        idx_row_col_traversal += 1 
                    print("idx_row_col_traversal now at {}".format(idx_row_col_traversal)) 
                    continue 
                # boundary check 
                if idx_row_col_traversal >= row_indices.shape[0]: 
                    break 
                total_counted_pos += 1 
                assert row_i <= row_indices[idx_row_col_traversal] 
                if row_i < row_indices[idx_row_col_traversal]: 
                    print("we accept all n tokens at {} since row index is at {}".format(row_i, row_indices[idx_row_col_traversal])) 
                    # we accept all n tokens at row_i position 
                    total_acceptance_length += self.n + 1 
                elif row_i == row_indices[idx_row_col_traversal]: 
                    print("we accept some tokens {}".format(row_i)) 
                    # we accept some tokens
                    total_acceptance_length += col_indices[idx_row_col_traversal] + 1 
                    idx_row_col_traversal += 1 
                    # boundary check 
                    if idx_row_col_traversal >= row_indices.shape[0]: 
                        print("we break at {}".format(idx_row_col_traversal)) 
                        break 
                    print("index_row_col_traversal now at {} and row_indices has length {}".format(idx_row_col_traversal, row_indices.shape[0])) 
                    while idx_row_col_traversal < row_indices.shape[0] and row_indices[idx_row_col_traversal] == row_i: 
                        idx_row_col_traversal += 1 
                else: 
                    raise ValueError("We cannot have this scenario") 
                
                print("inspect where is idx_row_col_traversal at {}".format(idx_row_col_traversal)) 
                print("total acceptance length is {}".format(total_acceptance_length)) 
                print("total counted pos is {}".format(total_counted_pos)) 
        
        print("total acceptance length is {}".format(total_acceptance_length)) 
        print("total counted pos is {}".format(total_counted_pos)) 
        print("average acceptance length is {}".format(total_acceptance_length / total_counted_pos)) 
    
    def local_compute_metrics(
        self, 
        logits, 
        labels, 
        loss, 
        input_attention_mask, 
        outside_step, 
        old_label_mask, 
    ): 
        with torch.no_grad(): 
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support

            print("length of logits is {}".format(len(logits))) 
            for index in range(len(logits)): 
                print("logits[{}] is {}".format(index, logits[index].shape)) 
            # print(colored("printing out the type of logits {}".format(type(logits)), "red")) 
            
            # original_model_logits = logits[1] # dimension (batch_size, seq_len, vocab_size) 
            model_output_logits = logits[0] # dimension (batch_size, seq_len, n, vocab_size) 
            label_actual_mask = logits[1] # dimension (batch_size, seq_len - n) 
            '''
            # besides, we also want to know how many of the tokens here actually has their ngram found out 
            assert old_label_mask.shape[1] == label_actual_mask.shape[1] and old_label_mask.shape[0] == label_actual_mask.shape[0] 
            old_label_mask = old_label_mask.to(torch.long) 
            total_unfiltered_tokens = torch.sum(old_label_mask.view(-1), dim = 0).item() 
            ''' 
            # print("as a sanity check, we see the datatype of label_actual_mask is {}".format(label_actual_mask.dtype)) 
            # input_attention_mask = input_attention_mask[:, :-1] 
            input_attention_mask = input_attention_mask[:, 1:] 
            # labels = labels[:, 1:] 
            # preds = torch.argmax(logits, dim = -1) # dimension (batch_size, seq_len - 1) 
            if outside_step == 0: 
                # print("*** evaluating at step {} ***".format(self.iteration_count)) 
                # visualize the actual output prediction during the evaluation 
                if self.use_filtered_hot_labels: 
                    pass 
                else: 
                    pass 
            
            indices_to_keep = input_attention_mask == 1 
            total_valid_tokens = torch.sum(indices_to_keep.view(-1), dim = 0).item() 
            
            # computing the total accuracy of prediction 
            '''
            shift_labels = [] 
            # original_seq_len = original_model_logits.shape[1] 
            original_seq_len = model_output_logits.shape[1] 
            print(colored("original seq len is {}".format(original_seq_len), "yellow")) 
            for i in range(1, self.n + 1): 
                shift_labels.append(labels[:, i : i + original_seq_len - self.n].contiguous()) 
            shift_labels = torch.stack(shift_labels, dim = 2) # dimension (batch_size, seq_len - n, n) 
            ''' 
            
            print("shift_labels has shape {}".format(shift_labels.shape)) 
            # shift_labels[shift_labels.unsqueeze(-1).expand(-1, -1, 3)] = -100 
            # total_acc_poscount = (~(label_actual_mask.unsqueeze(-1).expand(-1, -1, self.n).to(torch.bool))).to(torch.long).view(-1).sum(dim = 0).item() 
            total_acc_poscount = (label_actual_mask.unsqueeze(-1).expand(-1, -1, self.n).to(torch.bool)).to(torch.long).view(-1).sum(dim = 0).item() 
            model_output_logits2 = model_output_logits[:, :-(self.n), :, :].contiguous() 
            pred = torch.argmax(model_output_logits2, dim = -1) 
            assert pred.shape == shift_labels.shape 
            correctness_matrix = (pred == shift_labels).to(torch.long) # 1 for for matching while 0 is for not matching 
            # filter the matrix with the original filter 
            correctness_matrix = correctness_matrix * (label_actual_mask.unsqueeze(-1).expand(-1, -1, self.n)) 
            correct_words = torch.sum(correctness_matrix.view(-1), dim = 0) 
            print(colored("total counted words is {} correct words is {}".format(total_acc_poscount, correct_words), "yellow")) 
            
            # nothing fancy now, just greedy speculative sampling 
            # starting from 64th token, the rest 64th token should be used to compute the acceptance length 
            # pred has shape (batch_size, seq_len - n, n) 
            # pred = pred[:, self.generated_token_start_idx :, :] 
            total_unfiltered_tokens = shift_labels.shape[1] - (self.generated_token_start_idx - 1) 
            pred = pred[:, self.generated_token_start_idx - 1 :, :] 
            shift_labels = shift_labels[:, self.generated_token_start_idx - 1 :, :] 
            label_accept = label_actual_mask.unsqueeze(-1).expand(-1, -1, self.n)[:, self.generated_token_start_idx - 1 :] 
            acceptance_intermediate = (pred == shift_labels).to(torch.long) # here one is for correct, zero is for incorrect 
            acceptance_intermediate = acceptance_intermediate * label_accept # after filtering one is for keep and correct, zero is for discard 
            dim0  = acceptance_intermediate.shape[0] 
            dim1 = acceptance_intermediate.shape[1] 
            # print("dim0 is {} dim1 is {}".format(dim0, dim1)) # we have to make sure dim0 and dim1 are assigned before we reshape acceptance_intermediate 
            # print("pred, atch size {}, first 20 elements on dim 0 are {}".format(0, pred[0, : 20, 0])) 
            # print("pred, batch size {}, first 20 elements on dim 1 are {}".format(0, pred[0, :20, 1])) 
            # print("labels, batch size {}, first 20 elements on dim 0 are {}".format(0, shift_labels[0, : 20, 0])) 
            # print("labels, batch size {}, first 20 elements on dim 1 are {}".format(0, shift_labels[0, :20, 1])) 
            # print("acceptance_intermediate, batch size {}, first 20 elements are {}".format(0, acceptance_intermediate[0, : 20, 0])) 
            # print("acceptance_intermediate, batch size {}, first 20 elements are {}".format(0, acceptance_intermediate[0, : 20, 1])) 
            holding_diff_dimensionacc = {} 
            for i in range(0, self.n): 
                print("dimension {} has prediction accuracy: {}".format(i, torch.sum(acceptance_intermediate[:, :, i].view(-1), dim = 0).item() / (dim0 * dim1))) 
                holding_diff_dimensionacc["dimension acc {}".format(i)] = torch.sum(acceptance_intermediate[:, :, i].view(-1), dim = 0).item() / (dim0 * dim1) 
            acceptance_intermediate = acceptance_intermediate.reshape(-1, self.n) 
            
            row_indices, col_indices = torch.nonzero(~(acceptance_intermediate.to(torch.bool)), as_tuple = True) # this is very important, now one is for wrong or discard, zero is for correct and keep 
            idx_row_col_traversal = 0 
            total_counted_pos = 0 
            total_acceptance_length = 0 
            for i in range(dim0): 
                # boundary check 
                if idx_row_col_traversal >= row_indices.shape[0]: 
                    break 
                for j in range(dim1): 
                    # row_i = i * mask.shape[1] + j 
                    row_i = i * dim1 + j 
                    # print(i, j) 
                    # if input_attention_mask[i, j] == 0: 
                    if label_accept[i, j, 0] == 0: # label_accept has dimension (batch_size, some length, n) 
                        # we have this filtering such that after the if, we have 1 to signify keep and wrong, 0 is to signify keep and correct 
                        # we skip this token 
                        # print("we skip at batch size {} position {} row_i {} row_indices is at {}.format(i, j, row_i, row_indices[idx_row_col_traversal])") 
                        while idx_row_col_traversal < row_indices.shape[0] and row_indices[idx_row_col_traversal] == row_i: 
                        # while row_indices[idx_row_col_traversal] <= row_i: # should essentailly be ==, since previously we guarantee that row_indices is right at the new pos 
                            idx_row_col_traversal += 1 
                        # print("idx_row_col_traversal now at {}".format(idx_row_col_traversal)) 
                        continue 
                    # boundary check 
                    if idx_row_col_traversal >= row_indices.shape[0]: 
                        break 
                    total_counted_pos += 1 
                    assert row_i <= row_indices[idx_row_col_traversal] 
                    if row_i < row_indices[idx_row_col_traversal]: 
                        # print("we accept all n tokens at {} since row index is at {}".format(row_i, row_indices[idx_row_col_traversal])) 
                        # we accept all n tokens at row_i position 
                        total_acceptance_length += self.n 
                    elif row_i == row_indices[idx_row_col_traversal]: 
                        # print("we accept some tokens {}".format(row_i)) 
                        # we accept some tokens
                        total_acceptance_length += col_indices[idx_row_col_traversal] 
                        # print("col_indices[idx_row_col_traversal] is {}".format(col_indices[idx_row_col_traversal])) 
                        idx_row_col_traversal += 1 
                        # boundary check 
                        if idx_row_col_traversal >= row_indices.shape[0]: 
                            # print("we break at {}".format(idx_row_col_traversal)) 
                            break 
                        # print("index_row_col_traversal now at {} and row_indices has length {}".format(idx_row_col_traversal, row_indices.shape[0])) 
                        while idx_row_col_traversal < row_indices.shape[0] and row_indices[idx_row_col_traversal] == row_i: 
                            idx_row_col_traversal += 1 
                    else: 
                        raise ValueError("We cannot have this scenario") 

                    # print("inspect where is idx_row_col_traversal at {}".format(idx_row_col_traversal)) 
                    # print("total acceptance length is {}".format(total_acceptance_length)) 
                    # print("total counted pos is {}".format(total_counted_pos)) 
        
        print("total unfiltered tokens is {}".format(total_unfiltered_tokens)) 
        print("total acceptance length is {}".format(total_acceptance_length)) 
        print("total counted pos is {}".format(total_counted_pos)) 
        print("average acceptance length is {}".format(total_acceptance_length / total_counted_pos)) 
            
        # use preds to compute f1 score 
        # f1 = precision_recall_fscore_support(labels, preds, average = "weighted") 
        # return {"perplexity": perplexity, "correct_words": correct_words, "total_words": total_valid_tokens, "interest_correct_words": interest_correct_count, "interest_total_words": interest_token_count} 
        # return {"correct_words": correct_words, "total_words": total_acc_poscount, "total_counted_pos": total_counted_pos, "total_acceptance_length": total_acceptance_length, **holding_diff_dimensionacc} 
        return {"correct_words": correct_words, "total_words": total_acc_poscount, "total_counted_pos": total_counted_pos, "total_acceptance_length": total_acceptance_length, "total_unfiltered_token": total_unfiltered_tokens, **holding_diff_dimensionacc} 
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        print("got inside the subclass") 
        self.training_mode = False 
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        # NOTE: note that this modification is only helpful for single GPU training 
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size
        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        total_correct_words = 0 
        total_words = 0 
        # sum_of_perplexity = 0 # used to compute the average perplexity 
        total_loss = 0 # used to compute the correct perplexity 
        total_length_token = 0 
        total_acceptance_length = 0 
        total_unfiltered_length = 0 

        observed_num_examples = 0 
        total_num_steps = len(dataloader) 
        # Main evaluation loop
        holding_diff_dimensionacc = {} 
        for i in range(0, self.n): 
            holding_diff_dimensionacc["dimension acc {}".format(i)] = [] 
        for step, inputs in enumerate(tqdm(dataloader, desc = "description")): 
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step 
            ignore_keys = ["hidden_states", "attentions", "past_key_values"] 
            loss, logits, labels = self.prediction_step(model, inputs, False, ignore_keys=ignore_keys) 
            # print(ignore_keys) 
            # print(colored("the loss is {}".format(loss), "yellow")) 
            # print(colored("the shape of logits is {} {}".format(logits.shape, "yellow"))) 
            # print(colored("the shape of logits if {} {}".format(len(logits), logits[0].shape), "yellow")) 
            # print(colored("the shape of logits is {}".format(logits.shape), "yellow")) 
            # print(colored("the shape of labels is {}".format(labels.shape), "yellow")) 
            total_loss += loss.item() 
            old_labels = inputs["input_ids"].clone() 
            old_labels[old_labels == self.tokenizer.pad_token_id] = -100 
            old_labels = old_labels[:, 1:] 
            old_labels = old_labels[:, : -(self.n + 1)] 
            old_labels = (old_labels != -100).to(torch.bool) 
            local_metrics = self.local_compute_metrics(logits, labels, loss, inputs["attention_mask"], step, old_labels) 
            total_correct_words += local_metrics["correct_words"] 
            total_words += local_metrics["total_words"] 
            total_length_token += local_metrics["total_counted_pos"] 
            total_acceptance_length += local_metrics["total_acceptance_length"] 
            total_unfiltered_length += local_metrics["total_unfiltered_token"] 
            for i in range(0, self.n): 
                # holding_diff_dimensionacc["dimension acc {}".format(i)] = local_metrics["dimension acc {}".format(i)] 
                holding_diff_dimensionacc["dimension acc {}".format(i)].append(local_metrics["dimension acc {}".format(i)]) 

            if is_torch_tpu_available():
                xm.mark_step()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples
        
        global_accuracy = total_correct_words / total_words 
        all_losses = total_loss / total_num_steps 
        holding_dimsionacc = {} 
        for i in range(self.n): 
            holding_dimsionacc["dimension acc {}".format(i)] = sum(holding_diff_dimensionacc["dimension acc {}".format(i)])/len(holding_diff_dimensionacc["dimension acc {}".format(i)]) 
            print("dimension {} has accuracy {}".format(i, holding_dimsionacc["dimension acc {}".format(i)])) 
        
        # have an extra reference only when hotness measure is activated 
        length_diff = total_unfiltered_length - total_length_token # this is the number of tokens that are not filtered by the hotness measure 
        extra_ref_acceptance_length = length_diff + total_acceptance_length # for all the tokens that doesn't have their corresponding ngram in the list, we use one entire forward pass to process them 
        print(colored("average_accpetance_length is {} total_acceptance_length is {} total_length_counted_token is {}".format(total_acceptance_length / total_length_token, total_acceptance_length, total_length_token), "cyan")) 
        print(colored("extra_ref_acceptance_length is {} extra_ref_total_acceptance_length {} extra_ref_total_unfiltered_length {}".format(extra_ref_acceptance_length / total_unfiltered_length, extra_ref_acceptance_length, total_unfiltered_length), "yellow")) 
        exit(0) 

        metrics = {"accuracy": global_accuracy, "average_acceptance_length": total_acceptance_length / total_length_token} 
        print(colored(metrics, "magenta")) 
        if has_wandb: 
            wandb.log({"global_eval_accuracy": global_accuracy, 
                       "average_acceptance_length": total_acceptance_length / total_length_token, 
                       "total_acceptance_length": total_acceptance_length, 
                       "total_length_counted_tokens": total_length_token, 
                       "extra_ref_acceptance_length": extra_ref_acceptance_length / total_unfiltered_length, 
                       "extra_ref_total_acceptance_length": extra_ref_acceptance_length, 
                       "extra_ref_total_unfiltered_length": total_unfiltered_length, 
                       **holding_dimsionacc
            }) 
        # # Metrics!
        # if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
        #     if args.include_inputs_for_metrics:
        #         metrics = self.compute_metrics(
        #             EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
        #         )
        #     else:
        #         metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        # else:
        #     metrics = {}
        # # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            # metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item() 
            metrics[f"{metric_key_prefix}_loss"] = all_losses 
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key) 
        
        self.training_mode = True 

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples) 

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
'''
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
''' 

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
onedataset = load_dataset('json', data_files = datasetsrc, split = "train[:3000]") 
# onedataset = load_dataset('json', data_files = datasetsrc, split = "train") 
# onedataset = load_dataset("c4", "en", split = "train", cache_dir = dir_dataset) 
d = onedataset.train_test_split(test_size = 0.05) 
# print(d["train"], d["test"]) 
# max_length = small_model.config.max_position_embeddings 
# def encode_with_truncation(examples): 
    # return tokenizer(examples["text"], truncation=True, padding="max_length",
                #    max_length=max_length, return_special_tokens_mask=True) 
hot_1000_3_grams = log_dict_converterc(synthesized_dir_path + "mostcommon100000{}grams.json".format(args.n), preproc = True, tokenizer = tokenizer) 

def encode_with_truncation(examples): 
    # return tokenizer(examples["text"], truncation = True, padding = "max_length", 
                    #  max_length = max_length, return_special_tokens_mask = True) 
    return tokenizer(examples["text"], padding = "max_length", max_length = 128, 
                     return_attention_mask = True, return_tensors = "pt", truncation = True) 

def encode_with_truncation2(examples): 
    ''' 
    examples need to have the "input_ids" fiels in it already 
    ''' 
    # first make the labels 
    labels = torch.tensor(examples["input_ids"]).clone() if isinstance(examples["input_ids"], list) else examples["input_ids"].clone() 
    if tokenizer.pad_token_id is not None: 
        labels[labels == tokenizer.pad_token_id] = -100 
    
    # second shift labels in a way to do ngram loss 
    shift_labels = [] 
    originalseqlength = labels.shape[1] 
    label_actual_mask = (labels[:, 1 : 1 + (originalseqlength - args.n)] == -100).to(torch.bool) 
    for i in range(1, args.n + 1): 
        shift_labels.append(labels[:, i : i + (originalseqlength - args.n)].contiguous()) 
    shift_labels = torch.stack(shift_labels, dim = 2) 
    label_actual_mask = label_actual_mask.unsqueeze(-1).expand(-1, -1, args.n) 
    shift_labels[label_actual_mask] = -100 
    print("shift labels shape {}".format(shift_labels.shape)) 
    
    shift_labels_expand = shift_labels.long().unsqueeze(2) # shape of (batch_size, seq_len - n, 1, n) 
    hot_n_grams_expand = hot_1000_3_grams.unsqueeze(0).unsqueeze(0).to(shift_labels_expand.device) # shape (1, 1, hottestcount, n) 
    print("hot n grams expand shape {}".format(hot_n_grams_expand.shape)) 
    matches = torch.all(shift_labels_expand == hot_n_grams_expand, dim = -1).to(torch.bool) # matches have dimension of (batch_size, seq_len - n, hottestcount) 
    mask = ~torch.any(matches, dim = -1) # mask has dimension of (batch_size, seq_len - n) 
    
    # do a bit stats 
    total_pos = torch.tensor(mask.shape[1]).expand(mask.shape[0], 1).to(mask.device) # expecting to be tenosr (batch_size, seq_len - n) 
    total_found_num = torch.sum(~mask, dim = 1).to(mask.device) # expecting to be tensor of first dimension to be batch_size 
    # print("total pos shape {} total_found_num shape {}".format(total_pos.shape, total_found_num.shape)) 
    
    mask = mask.unsqueeze(-1).expand(-1, -1, args.n) # mask has dimension of (batch_size, seq_len - n, n) 
    shift_labels[mask] = -100 
    
    outputform = {"labels": shift_labels, "total_pos": total_pos, "total_found_num": total_found_num} 
    return outputform 
    
train_dataset = d["train"].map(encode_with_truncation, batched = True, num_proc = 4) 
test_dataset = d['test'].map(encode_with_truncation, batched = True, num_proc = 4) 
'''
for i in range(len(train_dataset)): 
    print(type(train_dataset[i])) 
    for k, v in train_dataset[i].items(): 
        print("type of k is {} and k is {}".format(type(k), k)) 
        print("type of v is {} and v is {}".format(type(v), v)) 
''' 
train_dataset = train_dataset.map(encode_with_truncation2, batched = True, num_proc = 4) 
test_dataset = test_dataset.map(encode_with_truncation2, batched = True, num_proc = 4) 

collection_verify = [] 
for i in range(10): 
    print(type(train_dataset[i])) 
    for k, v in train_dataset[i].items(): 
        if k != "labels": 
            print(k) 
            print(v) 
        else: 
            print(k) 
            for i in range(len(v)): 
                if v[i][0] == -100: 
                    print(v[i]) 
                else: 
                    print(colored(v[i], "yellow")) 
                    collection_verify.append(v[i]) 

for it in collection_verify: 
    print(colored(tokenizer.decode(torch.tensor(it)), "blue"), end = ", ") # collection verified are the ngrams that appeared inthat 10 examples, not representative to the entire 10^5 ngrams collected 

# print("The model max length is {}".format(small_model.config.max_position_embeddings)) 
train_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels', 'total_pos', 'total_found_num']) 
test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels', 'total_pos', 'total_found_num']) 
'''
# custom dataset 
# defining custom dataset 
kernel_size = args.kernel_size 

datasetnew = CustomDataset(data_dir = dir_sdata, tokenizer = tokenizer, kernel_size = kernel_size) 
train_set, test_set = datasetnew.split(0.98)     # 712k * 0.95 = 676k 712k * 0.05 = 36k 
                                                 # 356k * 0.99 = 352k 356k * 0.01 = 3.6k 
''' 

total_seq_count = 0 
total_found_seg_collector = 0 
for example in train_dataset: 
    total_seq_count += example["total_pos"].reshape(-1).sum(dim = 0).item() 
    total_found_seg_collector += example["total_found_num"].sum(dim = 0).item() 

print("percentage of found segments is {} total seq found is {} total word in the train dataset is {}".format(total_found_seg_collector / total_seq_count, total_found_seg_collector, total_seq_count)) 
if has_wandb: 
    wandb.log({"percentage of found segments": total_found_seg_collector / total_seq_count, "total seq found": total_found_seg_collector, "total word in the train dataset": total_seq_count}) 

param_group = [] 
module_projection_name = "output_n_projection.weight" 
model = LlamaCausalLMWeirdTwo.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
model.set_lookaheadcount(args.n) 
for name, param in model.named_parameters(): 
    if name == module_projection_name: 
        print("we got inside this if statement") 
        param.requires_grad = True 
        param.data = param.data.to(torch.float32) 
        param_group.append(param) 
    else: 
        param.requires_grad = False 
model.train() 
print("length of param_group is {}".format(len(param_group))) 

custom_optimizer = torch.optim.AdamW(param_group, lr = 1e-3) 


# for llama model we need to add the padding token 
model.config.pad_token_id = tokenizer.pad_token_id 
# print(small_model.embed_projection.weight.dtype) 

# data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 
data_collator = DataCollatorForLanguageModeling2(tokenizer = tokenizer, mlm = False) 

# model_path = "/home/bc20/yang" 
# model_path = "/home/yangzho6/model_checkpoints" 
model_path = dir_models 
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    # evaluation_strategy="steps",    # evaluate each `logging_steps` steps 
    overwrite_output_dir=True,      
    num_train_epochs=5,            # number of training epochs, feel free to tweak
    per_device_train_batch_size = 30, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=4,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size= 25,  # evaluation batch size
    # logging_steps=1, 
    logging_steps = 1,             # evaluate, log and save model checkpoints every 1000 step
    # save_steps=1000, 
    # save_steps = 2000, 
    save_steps = 1, 
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

max_length = 128 

if has_wandb: 
    # project_setting = args.experiment_setting if args.eval_mode is False else "finetuning" 
    today = datetime.date.today() 
    wandblogconfigs = training_args.to_dict() 
    wandblogconfigs["git_commit"] = commit_hash 
    wandblogconfigs["time_hash"] = hash_of_time 
    # wandb.init(project = "llm160m", config = training_args, name="{}_{}".format(today, project_setting)) 
    # wandb.init(project = "llm160m", config = wandblogconfigs, name = "{}_{}_{}".format(today, project_setting, "custom" if args.use_plain_model is False else "plain")) 
    wandb.init(project = "chunklargefinetuning", config = wandblogconfigs, name = "{}_{}".format(today, "unmasked")) 

weightmodelfirst = next(model.parameters()) 
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

trainer = CustomTrainer(
    model = model, 
    args = training_args, 
    train_dataset = train_dataset, 
    eval_dataset = test_dataset, 
    data_collator = data_collator, 
    optimizers = (custom_optimizer, None), 
    common_n_gram_list = hot_1000_3_grams, 
    use_filtered_hot_labels = False, 
    n = args.n, 
    tokenizer = tokenizer, 
) 

torch.autograd.set_detect_anomaly(True) 

trainer.train() 

wandb.finish() 

