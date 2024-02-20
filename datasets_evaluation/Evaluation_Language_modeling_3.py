# this script is mainly for evaluating different checkpoints (large + small or small)
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
from transformers import GPTNeoXForCausalLM 
from transformers import LlamaConfig, LlamaPreTrainedModel 
from transformers import LlamaTokenizer 

from tqdm import tqdm
# from sampling.utils import norm_logits, sample 

import torch.nn.functional as F 

from transformers.generation.logits_process import LogitsProcessorList 
import time 
import numpy as np 
import inspect 

from termcolor import colored 
from transformers import Trainer, TrainingArguments 
from torch import nn 
from transformers import DataCollatorForLanguageModeling 
from transformers.generation.utils import GenerationConfig 
from transformers.models.llama.modeling_llama import LlamaForCausalLM, SimpleSmallModel 
from transformers.models.llama.modeling_llama import LlamaCausalLMWeirdTwo 
from transformers.models.llama.modeling_llama import LlamaWeirdLarge3 
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model 
import time 
from torch.utils.data import random_split 
from transformers import BitsAndBytesConfig 
from packaging import version 
# import torch.nn.parallel.distributed.DistributedDataParallel as DDP 

import datetime 

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union 
from itertools import chain 

if TYPE_CHECKING: 
    import optuna 

# # cache_dir = "/home/bc20/yang/" 
# dir_dataset = "/home/yangzho6/c4_parts" 
# dir_models = "/home/yangzho6/model_checkpoints2" 
# dir_sdata = "/home/yangzho6/c4llm_synthesized/" 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# Set a global seed for reproducibility
seed_value = 42
from transformers import set_seed 
set_seed(seed_value) 

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False 

from transformers.utils import ( 
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
    is_torch_npu_available, 
) 

from transformers.trainer_pt_utils import (
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
from transformers.trainer_utils import (
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
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available 

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

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt" 

import warnings 
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 

import socket 

hostname = socket.gethostname()
print("Hostname:", hostname) 

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help') 

parser.add_argument("--model_name", type = str, default = "openllama3b") 
parser.add_argument("--loading_from_checkpoint", type = str, default = None) 
parser.add_argument("--kernel_size", type = int, default = 7) 
parser.add_argument("--experiment_setting", type = str, default = "setting0") 
parser.add_argument("--condensed_token_random", action = "store_true") 
parser.add_argument("--task_id", type = int, default = 0) 

args = parser.parse_args() 

# model_name = "openllama3b" 
model_name = args.model_name 

if "lovelace" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/yangzho6/model_checkpoints/" 
    # dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
    dir_sdata = "/home/yangzho6/slimpajama/SlimPajama-627B/test/chunk1/" 
elif "ada" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/beidic/yangzho6/model_checkpoints/" 
    dir_sdata = "/home/beidic/yangzho6/c4llm_synthesized/" 
else: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    # dir_models = "/home/yangzho6/model_checkpoints/" 
    dir_models = "/fsx-storygen/beidic/yang/model_checkpoints/" 
    # dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
    dir_sdata = "/fsx-storygen/beidic/yang/c4llm_synthesized/" 

class CustomTrainer(Trainer): 
    def __init__(self, 
                 n = 7, 
                 tokenizer = None, 
                 commit_hash = None, 
                 time_hash = None, 
                 model_name = None, 
                 text_eval = None, 
                 *args, 
                 **kwargs, 
    ): 
        super().__init__(*args, **kwargs) 
        self.n = n 
        self.tokenizer = tokenizer 
        # self.start_idx = start_idx 
        self.iteration_count = 0 
        self.commit_hash = commit_hash 
        self.time_hash = time_hash 
        self.model_name = model_name 
        self.text_eval = text_eval 
        
        if self.args.resume_from_checkpoint is not None: 
            self.time_checkpoint = int(self.args.resume_from_checkpoint.split("-")[-1]) 
            print(colored("resuming from checkpoint {}".format(self.time_checkpoint), "yellow")) 
            print(colored("the learning rate is {}".format(self.optimizer.param_groups[0]["lr"]), "yellow")) 
            print(colored("the step count is {}".format(self.state.global_step), "yellow")) 
            if self.iteration_count == 0: 
                self.iteration_count = 4 * self.state.global_step 
    
    def _set_signature_columns_if_needed(self): 
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names)) 
        self._signature_columns += ["attention_mask_chunk"] 
        self._signature_columns += ["condensed_embeds"] 
        self._signature_columns += ["large_input_ids"] 
        # self._signature_columns += ["small_input_ids"] 
        self._signature_columns += ["input_ids"] 
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        print("self.optimizer has {} parameter groups, we have {} parameters, and the learning rate is {}".format(len(self.optimizer.param_groups), len(self.optimizer.param_groups[0]["params"]), self.optimizer.param_groups[0]["lr"])) 
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None 
        
        for key, values in inputs.items(): 
            print("key: {} and values have shape {}".format(key, values.shape)) 
        
        print(colored("iteration_count {}".format(self.iteration_count), "yellow")) 
        
        # input_ids = inputs["input_ids"] # (batch_size, 203) 
        large_input_ids = inputs["large_input_ids"] # (batch_size, 203) 
        small_input_ids = inputs["input_ids"] # (batch_size, 203) 
        # attention_mask = inputs["attention_mask_chunk"] 
        condensed_embeds_labels = inputs["condensed_embeds"] # (batch_size, 28, 3200) 
        condensed_embeds_labels = condensed_embeds_labels.to(self.model.small_model_dtype) 
        original_attention_mask = inputs["attention_mask"] # (batch_size, 203) 
        label2 = inputs["labels"] # (batch_size, 203) 
        print("shape of large_input_ids {} shape of small_input_ids {}".format(large_input_ids.shape, small_input_ids.shape)) 
        # attention_mask = torch.ones((large_input_ids.shape[0], condensed_embeds_labels.shape[1] + 1), dtype = torch.long).to(large_input_ids.device) 
        # attention_mask = torch.ones((large_input_ids.shape[0], condensed_embeds_labels.shape[1] + 2), dtype = torch.long).to(large_input_ids.device) # sequence length is 204, one bos, 29 more tokens, so 30 in total, we have 28 condensed tokens 
        attention_mask = torch.ones((large_input_ids.shape[0], (large_input_ids.shape[1] - 1) // self.n + 1), dtype = torch.long).to(large_input_ids.device) 
        
        batch_size, seq_len = original_attention_mask.shape 
        # addedon_length = (seq_len - 8) // self.n 
        addedon_length = (seq_len - self.n - 1) // self.n 
        original_attention_mask = torch.cat((original_attention_mask, torch.ones((batch_size, addedon_length), dtype = torch.long).to(small_input_ids.device)), dim = 1) 
        
        outputs = model(
            # input_ids = input_ids, 
            large_input_ids = large_input_ids, 
            small_input_ids = small_input_ids, 
            attention_mask = attention_mask, 
            output_hidden_states = True, 
            output_attentions = True, 
            return_dict = True, 
            condensed_embed_labels = condensed_embeds_labels, 
            original_attention_mask = original_attention_mask, 
            labels = label2, 
        ) 
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
            ce_loss = outputs["ce_loss"] if isinstance(outputs, dict) else outputs[-2] 
            l2_distance = outputs["l2_distance"] if isinstance(outputs, dict) else outputs[-3] 
            l2_distance_input = outputs["l2_distance_input"] if isinstance(outputs, dict) else outputs[-1] 
            cossim_input = outputs["cossim_input"] if isinstance(outputs, dict) else outputs[-1] 
        
        print(colored("rank {} loss {}".format(self.accelerator.state.process_index, loss), "yellow")) 
        print(colored("rank {} ce_loss {}".format(self.accelerator.state.process_index, ce_loss), "yellow")) 
        print(colored("rank {} l2_distance {}".format(self.accelerator.state.process_index, l2_distance), "yellow")) 
        print(colored("rank {} l2_distance_input {}".format(self.accelerator.state.process_index, l2_distance_input), "yellow")) 
        print(colored("rank {} cossim_input {}".format(self.accelerator.state.process_index, cossim_input), "yellow")) 
        if self.accelerator.is_main_process and has_wandb and self.iteration_count % 20 == 0: 
            if len(self.optimizer.param_groups) > 1: 
                wandb.log({"loss": loss, 
                        "group1.lr": self.optimizer.param_groups[0]["lr"], 
                        "group2.lr": self.optimizer.param_groups[1]["lr"], 
                        # "iteration_count": self.iteration_count * 50 
                        "iteration_count": self.iteration_count, 
                        "ce_loss": ce_loss, 
                        "l2_distance": l2_distance, 
                        "l2_distance_input": l2_distance_input, 
                        "cosin_similarity_input": cossim_input, 
                }) 
                
            else: 
                wandb.log({"loss": loss, 
                        "group1.lr": self.optimizer.param_groups[0]["lr"], 
                        "iteration_count": self.iteration_count, 
                        "ce_loss": ce_loss, 
                        "l2_distance": l2_distance, 
                        "l2_distance_input": l2_distance_input, 
                        "cosin_similarity_input": cossim_input, 
                }) 
                
        # if self.accelerator.is_main_process and self.iteration_count % 1000 == 0 and has_wandb: 
        if self.accelerator.is_main_process and has_wandb and self.iteration_count % 500 == 0: 
            print(colored("generating images ... at iteration {}".format(self.iteration_count), "yellow")) 
            for layer in [0, 6, 11]: 
                for head in [0, 6, 11]: 
                    # SimpleSmallModel.plot_attention_map(outputs.attentions, 0, 0, 144, "testing_attention_map.jpg") 
                    plot_name = "testing_attention_map_{}_{}.jpg".format(self.commit_hash, self.time_hash) 
                    SimpleSmallModel.plot_attention_map(outputs.attentions, layer, head, small_input_ids.shape[1] + addedon_length, plot_name) 
                    # print(outputs.attentions[0][0][0][64]) 
                    # time.sleep(0.1) # ensure the file is written to disk 
                    field_name = "layer{}_head{}".format(layer, head) 

                    try: 
                        wandb.log({field_name: wandb.Image(plot_name)}) 
                    except Exception as e: 
                        print(f"An error has occured during logging attention map: {e}") 
                        # try again 
                        # time.sleep(1) 
                        # if try_count < 2: 
                        #     wandb.log({field_name: wandb.Image("testing_attention_map.jpg")}) 
                        #     try_count += 1 
        self.iteration_count += 1 
        return (loss, outputs) if return_outputs else loss 

    def local_compute_metrics(
        self, 
        logits, 
        labels, 
        loss, 
        input_attention_mask, 
        outside_step, 
    ): 
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support 
        print("length of logits {}".format(len(logits))) 
        print("logits[0].shape {}".format(logits[0].shape)) 
        print("logits[1].shape {}".format(logits[1].shape)) 
        print("logits[2].shape {}".format(logits[2].shape)) 
        print("logits[3].shape {}".format(logits[3].shape)) 
        print("logits[4].shape {}".format(logits[4].shape)) 
        # assert len(logits) == 4 
        l2dist = logits[1].reshape(-1) 
        ce_loss = logits[2].reshape(-1) 
        l2dist_input = logits[3].reshape(-1) 
        cos_sim_input = logits[4].reshape(-1) 
        logits = logits[0] 
        # print(l2dist) 
        logits = logits[:, :-1, :] 
        # input_attention_mask = input_attention_mask[:, :-1] 
        input_attention_mask = input_attention_mask[:, 1:] 
        labels = labels[:, 1:] 
        preds = torch.argmax(logits, dim = -1) 
        write_out_text = [] 
        if self.accelerator.is_main_process and outside_step == 0: 
            # print("*** evaluating at step {} ***".format(self.iteration_count)) 
            mask_correctness = (preds == labels).to(torch.bool) 
            pred_outputs = preds[: 20] 
            for i in range(len(pred_outputs)): 
                prediction_text = "the prediction is: " 
                for j in range(mask_correctness.shape[1]): 
                    if mask_correctness[i][j]: 
                        prediction_text += colored(self.tokenizer.decode(pred_outputs[i][j]), "green") + " " 
                    else: 
                        prediction_text += colored(self.tokenizer.decode(pred_outputs[i][j]), "red") + " " 
                print(prediction_text) 
                print() 
                
                mask_filtered = labels[i][input_attention_mask[i] == 1] 
                mask_filtered[mask_filtered == -100] = 0 
                labels_output = self.tokenizer.decode(mask_filtered) 
                write_out_text.append(prediction_text + "\n" + labels_output + "\n") 
                print(colored(labels_output, "cyan")) 
                print() 
                print() 
            
            # with open("{}evaluation_printout_{}_{}_{}_{}_{}.txt".format(dir_models, self.commit_hash, self.time_hash, self.state.global_step, self.n, self.model_name), "a") as f: 
            with open(self.text_eval, "a") as f: 
                f.write("*** at step {} {}".format(self.iteration_count, self.state.global_step)) 
                f.write("\n") 
                for i, text in enumerate(write_out_text): 
                    f.write("example {}/{}\n".format(i, len(write_out_text))) 
                    f.write(text) 
                    f.write("\n") 
                f.write("\n") 
        
        if self.accelerator.state.num_processes > 1: 
            self.accelerator.wait_for_everyone() 
            
        perplexity = torch.exp(loss).mean().item() 
        indices_to_keep = input_attention_mask == 1 # not sure whether we need this 
        total_valid_tokens = torch.sum(indices_to_keep.view(-1), dim = 0).item() 
        correct_words = torch.sum((preds[indices_to_keep] == labels[indices_to_keep]).view(-1), dim = 0).item() 
        print("correct words: {} and total words: {}".format(correct_words, total_valid_tokens)) 
        return {"perplexity": perplexity, "correct_words": correct_words, "total_words": total_valid_tokens, "l2_distance": l2dist.item(), "ce_loss": ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss, "l2_distance_input": l2dist_input.item(), "cosin_similarity": cos_sim_input.item()} 
                
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput: 
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
        
        all_losses = None 
        all_preds = None 
        all_labels = None 
        
        total_correct_words = 0 
        total_words = 0 
        sum_of_perplexity = 0 # used to compute the average perplexity 
        total_loss = 0 # used to compute the correct perplexity 
        l2_distance = 0 
        l2_distance_input = 0 
        cosine_similarity_input = 0 
        ce_loss = 0 
        
        observed_num_examples = 0 
        total_num_steps = len(dataloader) 
        local_device = None 
        # Main evaluation loop
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
            if local_device == None: 
                local_device = loss.device 
            
            # print(ignore_keys) 
            # print(colored("the loss is {}".format(loss), "yellow")) 
            # print(colored("the shape of logits is {} {}".format(logits.shape, "yellow"))) 
            # print(colored("the shape of logits if {} {}".format(len(logits), logits[0].shape), "yellow")) 
            # print(colored("the shape of logits is {}".format(logits.shape), "yellow")) 
            # print(colored("the shape of labels is {}".format(labels.shape), "yellow")) 
            total_loss += loss.item() 
            local_metrics = self.local_compute_metrics(logits, labels, loss, inputs["attention_mask"], step) 
            total_correct_words += local_metrics["correct_words"] 
            total_words += local_metrics["total_words"] 
            sum_of_perplexity += local_metrics["perplexity"] 
            l2_distance += local_metrics["l2_distance"] 
            l2_distance_input += local_metrics["l2_distance_input"] 
            cosine_similarity_input += local_metrics["cosin_similarity"] 
            ce_loss += local_metrics["ce_loss"] 

            if is_torch_tpu_available(): 
                xm.mark_step() 
            
        if self.accelerator.is_main_process: 
            print("rank {} total_loss before aggregation is {}".format(self.accelerator.state.process_index, total_loss)) 
        aggregated_loss = self.gather_function(torch.tensor(total_loss).reshape(1, -1).to(local_device)) 
        if self.accelerator.is_main_process: 
            print("rank {} total_loss after aggregation is {}".format(self.accelerator.state.process_index, aggregated_loss)) 
        total_loss = self.gather_function(torch.tensor(total_loss).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).div(self.accelerator.state.num_processes).item() 
        total_correct_words = self.gather_function(torch.tensor(total_correct_words).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).item() 
        total_words = self.gather_function(torch.tensor(total_words).reshape(-1, 1).to(local_device)).view(-1).sum(dim = -1).item() 
        sum_of_perplexity = self.gather_function(torch.tensor(sum_of_perplexity).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).item() 
        l2_distance = self.gather_function(torch.tensor(l2_distance).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).div(self.accelerator.state.num_processes).item() 
        l2_distance_input = self.gather_function(torch.tensor(l2_distance_input).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).div(self.accelerator.state.num_processes).item() 
        cosine_similarity_input = self.gather_function(torch.tensor(cosine_similarity_input).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).div(self.accelerator.state.num_processes).item() 
        ce_loss = self.gather_function(torch.tensor(ce_loss).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).div(self.accelerator.state.num_processes).item() 
        
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
        
        global_perplexity = np.exp(total_loss / total_num_steps) 
        global_accuracy = total_correct_words / total_words 
        all_losses = total_loss / total_num_steps 
        l2_distance = l2_distance / total_num_steps 
        l2_distance_input = l2_distance_input / total_num_steps 
        cosine_similarity_input = cosine_similarity_input / total_num_steps 
        ce_loss = ce_loss / total_num_steps 

        metrics = {"perplexity": global_perplexity, "accuracy": global_accuracy, "l2_distance": l2_distance, "ce_loss": ce_loss, "l2_distance_input": l2_distance_input, "cosine_similarity_input": cosine_similarity_input} 
        if self.accelerator.is_main_process: 
            print(colored(metrics, "magenta")) 
            wandb.log({"global_eval_perplexity": global_perplexity, "global_eval_accuracy": global_accuracy, "l2_distance": l2_distance, "ce_loss": ce_loss, "eval_loss_upd": all_losses, "l2_distance_input": l2_distance_input, "cosine_similarity_input": cosine_similarity_input}) 
        
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
        
        # print(metrics) 
        # exit(0) 
        
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples) 

model_type = "use_large_model" 
# model_type = "use_small_model" 
if not model_type == "use_small_model" and model_name == "openllama3b": 
    tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models) 
elif not model_type == "use_small_model": 
    # tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models) 
    # tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models) 
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
elif model_type == "use_small_model": 
    tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
else: 
    raise ValueError("model_type is not recognized") 
if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
tokenizer.padding_side = "left" 

kernel_size = 7 # this is definitely subject to change 
# datasetnew = CustomDataset(max_length = 260, data_dir = dir_sdata, tokenizer = tokenizer, kernel_size = kernel_size) 
# datasetnew = CustomDataset(max_length = 260, data_dir = dir_sdata, tokenizer = tokenizer, kernel_size = kernel_size) 
# dfiles = ["example_holdout_{}.jsonl".format(i) for i in range(6282)] 
dfiles = [dir_sdata + "example_holdout_{}combined.jsonl".format(0)] 
# datasetnew = load_dataset('json', data_files = dfiles, split = "train[:10000]") 
datasetnew = load_dataset('emozilla/pg19', split = "train") 

# train_set, test_set = datasetnew.split(0.99) 
print(tokenizer(datasetnew[0]['text'][100000 : 100000 + 3000], padding = "max_length", max_length = 256, 
                return_attention_mask = True, return_tensors = "pt", truncation = True, 
                add_special_tokens = True)) 

def encode_with_truncation(examples): 
    tokdictionary = tokenizer(examples['text'][100000 : 100000 + 3000], padding = "max_length", max_length = 260, 
                     return_attention_mask = True, return_tensors = "pt", truncation = True, 
                     add_special_tokens = True) 
    newdictionary = {} 
    newdictionary['input_ids'] = tokdictionary['input_ids'].squeeze(0) 
    newdictionary['attention_mask'] = tokdictionary['attention_mask'].squeeze(0) 
    return newdictionary 

def unflatten_list_func(examples): 
    examples['input_ids'] = examples['input_ids'].squeeze(0) 
    examples['attention_mask'] = examples['attention_mask'].squeeze(0) 

# datasetnew = datasetnew.map(encode_with_truncation, batched = True, num_proc = 8) 
datasetnew = datasetnew.map(encode_with_truncation, num_proc = 8) 
# datasetnew = datasetnew.map(unflatten_list_func, num_proc = 8) 

datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask", "text"]) 
# datasetnew = datasetnew.map(unflatten_list_func, num_proc = 8) 

for i in range(0, 10): 
    print(datasetnew[i]['text'][100000 : 100000 + 3000]) 
    print(datasetnew[i]['input_ids']) 
    print("length of every example: {}".format(datasetnew[i]['input_ids'].shape)) 
    print() 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

if model_type == "use_small_model": 
    if args.loading_from_checkpoint is not None: 
        small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models) 
        # target_model_dim = 3200 if model_name == "openllama3b" else 2560 
        if model_name == "openllama3b": 
            target_model_dim = 3200 
        elif model_name == "shearedllama2_7b": 
            target_model_dim = 2560 
        elif model_name == "tinyllama": 
            target_model_dim = 2048 
        else: 
            target_model_dim = 2048 
        model = SimpleSmallModel.from_pretrained(args.loading_from_checkpoint, sliding_window_length = args.kernel_size, hostname = hostname, target_model_dim = target_model_dim) 
        model.config.pad_token_id = tokenizer.pad_token_id 
        # model = model.to(torch_device).to(torch.bfloat16) 
        model = model.to(torch_device) 
        model.eval() 
    else: 
        model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models) 
        model = model.to(torch_device) 
        model.eval() 
else: 
    if model_name == "openllama3b": 
        model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
    elif model_name == "shearedllama2_7b": 
        model = LlamaForCausalLM.from_pretrained("princeton-nlp/Sheared-LLaMA-2.7B", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
    elif model_name == "tinyllama": 
        model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
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
        large_model.set_msece_loss(use_mse_loss = False, ce_loss_only = True) 
        large_model.set_addonsmallmodel(small_model) 
        large_model.set_inference_setting("setting3") 
        large_model.set_walpha(0.5) 
        large_model.set_slidingwindowlength(sliding_window_length = args.kernel_size, addonmodel_start = args.kernel_size + 1) 
        large_model.set_tokenizer_bos_id(bos_id = tokenizer.bos_token_id, pad_id = tokenizer.pad_token_id) 
        large_model.set_cosinesimilarity(False) 
        
        large_model.config.pad_token_id = tokenizer.pad_token_id 
        small_model.config.pad_token_id = tokenizer.pad_token_id 
        
        large_model.model.eval() 
        large_model.addonsmallmodel.eval() 
    else: 
        raise ValueError("model_name is not recognized") 

training_args = TrainingArguments(
    output_dir = dir_models, 
    per_device_eval_batch_size = 64, 
    do_train = False, 
    do_eval = True, 
    label_names = ["labels"], 
) 

if args.model_name == "debugging": 
    model = large_model 
    
trainer = CustomTrainer( 
    args = training_args, 
    model = model, 
    data_collator = data_collator, 
    # experiment_setting = args.experiment_setting, 
    # eval_mode = False, 
    time_hash = hash_of_time, 
    # dtype = model.dtype, 
    # dtype = model.dtype, 
    model_name = model_name, 
    text_eval = "just_evaluation_{}.txt".format(hash_of_time), 
    tokenizer = tokenizer, 
    n = args.kernel_size, 
    # time_hash = hash_of_time, 
    commit_hash = commit_hash 
) 

results = trainer.evaluate(eval_dataset = datasetnew) 
print(results) 
# model.save_pretrained("../model_checkpoints/llama-160m_deciphering_{}_{}_{}".format(args.model_name, args.experiment_setting, commit_hash)) 

