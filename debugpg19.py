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
from transformers.models.llama.modeling_llama import LlamaWeirdLargeIntermediate 
from transformers.models.llama.modeling_llama import LlamaWeirdLargeTest 
from transformers.models.llama.modeling_llama import LlamaWeirdLargeFullCoverage 
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model 
import time 
from torch.utils.data import random_split 
from transformers import BitsAndBytesConfig 
from packaging import version 
# import torch.nn.parallel.distributed.DistributedDataParallel as DDP 

from torch.nn import CrossEntropyLoss 

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
parser.add_argument("--setting0usedq", action = "store_true") 
parser.add_argument("--full_sequence_length_layer_pos", type = int, default = 10) 
parser.add_argument("--label_adjustment", action = "store_true") 
parser.add_argument("--secondlast", action = "store_true") 
parser.add_argument("--using_prefix", action = "store_true") 
parser.add_argument("--max_length", type = int, default = 260) 
parser.add_argument("--batch_size", type = int, default = 8) 

args = parser.parse_args() 

# model_name = "openllama3b" 
# model_name = args.model_name 

if "lovelace" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/yangzho6/model_checkpoints/" 
    # dir_c4llmsynthesized = "/home/yangzho6/c4llm_synthesized/" 
    # dir_c4llmsynthesized = "/home/yangzho6/c4llm_synthesized/tinyllama/" 
    dir_c4llmsynthesized = "/home/yangzho6/c4llm_synthesized/llama2_7b_topkna/" 
    # dir_c4llmsynthesized = "/home/beidic/yangzho6/c4llm_synthesized/" 
    dir_c4 = "/home/yangzho6/c4_parts/downloads/" 
    # dir_sdata = "/home/yangzho6/slimpajama/SlimPajama-627B/test/chunk1/" 
elif "ada" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/beidic/yangzho6/model_checkpoints/" 
    dir_c4llmsynthesized = "/home/beidic/yangzho6/c4llm_synthesized/" 
else: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    # dir_models = "/home/yangzho6/model_checkpoints/" 
    dir_models = "/fsx-storygen/beidic/yang/model_checkpoints/" 
    # dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
    # dir_sdata = "/fsx-storygen/beidic/yang/c4llm_synthesized/" 
    dir_c4llmsynthesized = "/fsx-storygen/beidic/yang/c4llm_synthesized/" 
    dir_c4 = "/fsx-storygen/beidic/yang/c4_parts/downloads/" 

class CustomTrainer(Trainer): 
    def __init__(
            self, 
            experiment_setting = "setting0", 
            tokenizer = None, 
            commit_hash = None, 
            eval_mode = False, 
            time_hash = None, 
            dtype = None, 
            model_name = None, 
            text_eval = None, 
            kernel_size = 7, 
            *args, 
            **kwargs, 
    ): 
        super().__init__(*args, **kwargs) 
        # self.large_model = large_model 
        # self.generation_config = GenerationConfig(return_dict_in_generate = True) 
        # self.time_checkpoint = time.time() 
        self.time_checkpoint = 0 
        self.iteration_count = 0 
        self.experiment_setting = experiment_setting 
        self.tokenizer = tokenizer 
        self.commit_hash = commit_hash 
        self.eval_mode = eval_mode 
        self.time_hash = time_hash 
        self.dtype = dtype 
        self.model_name = model_name 
        self.text_eval = text_eval 
        self.n = kernel_size 
        
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
    
    def training_step(self, model, inputs): 
        model.train() 
        inputs = self._prepare_inputs(inputs) 
        '''
        for k, v in inputs.items(): 
            if isinstance(v, tuple): 
                print(k, len(v)) 
            elif isinstance(v, torch.Tensor): 
                print(k, v.shape) 
            else: 
                print(k, v) 
        ''' 
        '''
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device) 
        ''' 
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device) 
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, evaluation_mode = False) 

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training 
        
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss) 
        
        self.iteration_count += 1 
        print(colored("the training iteration count is {}".format(self.iteration_count), "red")) 
        return loss.detach() / self.args.gradient_accumulation_steps 
    
    def downsample_vectors(self, listoflasthiddenstates, kernel_size = 4): 
        downsampled_vectors = [] 
        shape = listoflasthiddenstates[0].shape 
        device = listoflasthiddenstates[0].device 
        for i in range(len(listoflasthiddenstates)): 
            sum = torch.zeros(shape, device = device) 
            if i % kernel_size == kernel_size - 1: 
                sum += listoflasthiddenstates[i] 
                downsampled_vectors.append(sum/kernel_size) 
                sum.mul_(0.) 
            else: 
                sum += listoflasthiddenstates[i] 
        return downsampled_vectors 

    def compute_loss(self, model, inputs, return_outputs = False, evaluation_mode = True): 
        labels = None 
        input_ids = inputs["input_ids"] 
        attention_mask = inputs["attention_mask"] 
        label2 = inputs["labels"] 
        print("type of the model is {}".format(type(model))) 
        outputs = model(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            labels = label2, 
            output_hidden_states = True, 
            output_attentions = True, 
            return_dict = True, 
        ) 
        '''
        for differentsampleidx in range(1): 
            print("the input ids is {}".format(input_ids[differentsampleidx])) 
            print("the attention mask is {}".format(attention_mask[differentsampleidx])) 
            print("the input is {}".format(self.tokenizer.decode(input_ids[differentsampleidx]))) 
            prediction = outputs.logits.argmax(dim = -1)[differentsampleidx] 
            # print("the prediction is {}".format(self.tokenizer.decode(prediction))) 
            # wordspredicted = self.tokenizer.decode(prediction) 
            for i in range(input_ids.shape[1] - 1): 
                worddecoded = self.tokenizer.decode(prediction[i], skip_special_tokens = False) 
                if input_ids[differentsampleidx][i + 1] == prediction[i]: 
                    print(colored(worddecoded, "green"), end = " ") 
                else: 
                    print(colored(worddecoded, "red"), end = " ") 
            print() 
        print("the loss of the batch is {}".format(outputs.loss)) 
        
        time.sleep(1) 
        ''' 
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
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
        
        print(colored("rank {} and the loss is {}".format(self.accelerator.state.process_index, loss), "yellow" if evaluation_mode is False else "cyan")) 
        if self.accelerator.is_main_process and has_wandb and evaluation_mode is False and self.iteration_count % 20 == 0: 
            if len(self.optimizer.param_groups) > 1: 
                wandb.log({"loss": loss, 
                        "group1.lr": self.optimizer.param_groups[0]["lr"], 
                        "group2.lr": self.optimizer.param_groups[1]["lr"], 
                        # "iteration_count": self.iteration_count * 50 
                        "iteration_count": self.iteration_count 
                }) 
            else: 
                wandb.log({"loss": loss, 
                        "group1.lr": self.optimizer.param_groups[0]["lr"], 
                        "iteration_count": self.iteration_count 
                }) 
        
        if self.accelerator.is_main_process and self.iteration_count % 1000 == 0 and evaluation_mode is False and has_wandb: 
            print(colored("generating images ... at iteration {}".format(self.iteration_count), "yellow")) 
            for layer in [0, 6, 11]: 
                for head in [0, 6, 11]: 
                    '''
                    if isinstance(outputs.attentions, tuple): 
                        print("the attention mask have shape {}".format(len(outputs.attentions))) 
                        print("the attention mask first element has shape {}".format(outputs.attentions[0].shape)) 
                    else: 
                        print("the attention mask has shape {}".format(outputs.attentions.shape)) 
                    ''' 
                    # SimpleSmallModel.plot_attention_map(outputs.attentions, 0, 0, 144, "testing_attention_map.jpg") 
                    plot_name = "testing_attention_map_{}_{}_{}.jpg".format(self.commit_hash, self.time_hash, self.experiment_setting) 
                    SimpleSmallModel.plot_attention_map(outputs.attentions, layer, head, input_ids.shape[1], plot_name) 
                    # print(outputs.attentions[0][0][0][64]) 
                    # time.sleep(0.1) # ensure the file is written to disk 
                    field_name = "layer{}_head{}".format(layer, head) 

                    try: 
                        wandb.log({field_name: wandb.Image(plot_name)}) 
                    except Exception as e: 
                        print(f"An error has occured during logging attention map: {e}") 
        
        return (loss, outputs) if return_outputs else loss 

    def local_compute_metrics_weird(
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

    def local_compute_metrics(
        self, 
        logits, 
        labels, 
        loss, 
        input_attention_mask, 
        outside_step, 
    ): 
        from sklearn.metrics import accuracy_score 
        
        logits = logits[:, :-1, :] 
        input_attention_mask = input_attention_mask[:, 1:] 
        labels = labels[:, 1:] 
        preds = torch.argmax(logits, dim = -1) 
        
        perplexity = torch.exp(loss).mean().item() 
        indices_to_keep = input_attention_mask == 1 # only for debugging purposes 
        total_valid_tokens = torch.sum(indices_to_keep.view(-1), dim = 0).item() 
        interest_token_count = torch.sum(indices_to_keep[:, 63 :].reshape(-1), dim = 0).item() # check whether 63 makes sense and make it more general if it is correct or not 
        correct_words = torch.sum((preds[indices_to_keep] == labels[indices_to_keep]).view(-1), dim = 0).item() 
        interest_correct_count = torch.sum(((preds * indices_to_keep)[:, 63: ] == (labels * indices_to_keep)[:, 63: ]).view(-1), dim = 0).item() 
        print("correct words: {} and total words: {}".format(correct_words, total_valid_tokens)) 
        print("interest correct words: {} and interest total words: {}".format(interest_correct_count, interest_token_count)) 
        return {"perplexity": perplexity, "correct_words": correct_words, "total_words": total_valid_tokens, "interest_correct_words": interest_correct_count, "interest_total_words": interest_token_count} 
    
    def evaluation_loop(
        self, 
        dataloader: DataLoader, 
        description: str, 
        prediction_loss_only: Optional[bool] = None, 
        ignore_keys: Optional[List[str]] = None, 
        metric_key_prefix: str = "eval", 
    ) -> EvalLoopOutput: 
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
        sum_of_perplexity = 0 # used to compute the average perplexity 
        total_loss = 0 # used to compute the correct perplexity 
        interest_total_words = 0 
        interest_correct_words = 0 
        
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
            interest_total_words += local_metrics["interest_total_words"] 
            interest_correct_words += local_metrics["interest_correct_words"] 

            if is_torch_tpu_available():
                xm.mark_step()
        
        if self.accelerator.is_main_process: 
            print("rank {} total_loss before aggregation is {}".format(self.accelerator.state.process_index, total_loss)) 
        # all gather the metrics 
        aggregated_loss = self.gather_function(torch.tensor(total_loss).reshape(1, -1).to(local_device)) 
        if self.accelerator.is_main_process: 
            print("rank {} total_loss after aggregation is {}".format(self.accelerator.state.process_index, aggregated_loss)) 
        total_loss = self.gather_function(torch.tensor(total_loss).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).div(self.accelerator.state.num_processes).item() 
        total_correct_words = self.gather_function(torch.tensor(total_correct_words).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).item() 
        total_words = self.gather_function(torch.tensor(total_words).reshape(-1, 1).to(local_device)).view(-1).sum(dim = -1).item() 
        sum_of_perplexity = self.gather_function(torch.tensor(sum_of_perplexity).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).item() 
        interest_total_words = self.gather_function(torch.tensor(interest_total_words).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).item() 
        interest_correct_words = self.gather_function(torch.tensor(interest_correct_words).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).item() 
        
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
        global_interest_accuracy = interest_correct_words / interest_total_words 
        all_losses = total_loss / total_num_steps 
        
        metrics = {"perplexity": global_perplexity, "accuracy": global_accuracy, "interest_accuracy": global_interest_accuracy} 
        # if self.accelerator.is_main_process: 
        #     print(colored(metrics, "magenta")) 
        #     wandb.log({"global_eval_perplexity": global_perplexity, "global_eval_accuracy": global_accuracy, "global_eval_interest_accuracy": global_interest_accuracy}) 
        
        metrics = denumpify_detensorize(metrics) 
        
        if all_losses is not None: 
            metrics[f"{metric_key_prefix}_loss"] = all_losses 
        if hasattr(self, "jit_compilation_time"): 
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time 
        
        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples) 

model_type = "use_large_model" 
# model_type = "use_small_model" 

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 

if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
tokenizer.padding_side = "left" 
# tokenizer.padding_side = "right" 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 

training_args = TrainingArguments(
    output_dir = dir_models, 
    # per_device_eval_batch_size = args.batch_size, 
    per_device_eval_batch_size = 1, 
    do_train = False, 
    do_eval = True, 
    label_names = ["labels"], 
) 

# if args.model_name == "debugging" or args.model_name == "debugging2" or args.model_name == "debugging3": 

def get_dataset(datasetname, max_length): 
    if datasetname == "c4llm_synthesized": 
        # datasetnew = load_dataset('json', data_files = dfiles, split = "train[:10000]") 
        dfiles = [] 
        if "lovelace" in hostname: 
            # filename = "c4synthesized_file1_kernel7_0.json" 
            filename = "c4synthesized_file1_1_0.json" 
            # dfiles.append(dir_c4llmsynthesized + "{}/".format("tinyllama") + filename) 
            dfiles.append(dir_c4llmsynthesized + filename) 
            datasetnew = load_dataset("json", data_files = dfiles, split = "train[:10000]") 
        else: 
            filename = "c4synthesized_file1_kernel7_{}_combined.json".format(7) 
            dfiles.append(dir_c4llmsynthesized + "{}_topk{}/".format("tinyllama", "na") + filename) 
            datasetnew = load_dataset("json", data_files = dfiles, split = "train[:10000]") 
    elif datasetname == "c4": 
        dfiles = [] 
        # filename = "c4_file1.json" 
        # filename = "c4_file15.json" 
        filename = "c4_file150.json" 
        dfiles.append(dir_c4 + filename) 
        datasetnew = load_dataset("json", data_files = dfiles, split = "train[:10000]") 
    elif datasetname == "pg19": 
        if max_length == 2048: 
            numexample = 1000 
        elif max_length == 1024: 
            numexample = 2000 
        elif max_length == 4096: 
            numexample = 500 
        else: 
            numexample = 10000 
        datasetnew = load_dataset('emozilla/pg19', split = "train[:{}]".format(numexample)) 
    elif datasetname == "cnn_dailymail": # we need to use special processing for this dataset 
        datasetnew = load_dataset("cnn_dailymail", "3.0.0", split = "test[:10000]") 
    elif datasetname == "openwebtext": 
        datasetnew = load_dataset("Skylion007/openwebtext", split = "train[:10000]") 
    elif datasetname == "xsum": # we need to use special processing for this dataset 
        datasetnew = load_dataset("xsum", split = "test[:10000]") 
    else: 
        raise ValueError("dataset_name is not recognized") 

    def encode_with_truncationspecialized(examples): 
        if args.using_prefix: 
            tokdictionary = tokenizer(examples['text'][: 5000 * (max_length // 256)], padding = "max_length", max_length = max_length, 
                                 return_attention_mask = True, return_tensors = "pt", truncation = True, 
                                 add_special_tokens = True) 
        else: 
            tokdictionary = tokenizer(examples['text'][100000 : 100000 + (5000 * (max_length // 256))], padding = "max_length", max_length = max_length, 
                            return_attention_mask = True, return_tensors = "pt", truncation = True, 
                            add_special_tokens = True) 
        newdictionary = {} 
        newdictionary['input_ids'] = tokdictionary['input_ids'].squeeze(0) 
        newdictionary['attention_mask'] = tokdictionary['attention_mask'].squeeze(0) 
        return newdictionary 

    def encode_with_truncation(examples): 
        # tokdictionary = tokenizer(examples['text'][100000 : 100000 + 3000], padding = "max_length", max_length = 260, 
        #                  return_attention_mask = True, return_tensors = "pt", truncation = True, 
        #                  add_special_tokens = True) 
        tokdictionary = tokenizer(examples['text'], padding = "max_length", max_length = max_length, 
                                return_attention_mask = True, return_tensors = "pt", truncation = True, 
                                add_special_tokens = True) 
        newdictionary = {} 
        newdictionary['input_ids'] = tokdictionary['input_ids'].squeeze(0) 
        newdictionary['attention_mask'] = tokdictionary['attention_mask'].squeeze(0) 
        return newdictionary 
    
    def encode_text_summary(examples): # cnn_dailymail uses "article" 
        tokdictionary = tokenizer(examples['article'], padding = "max_length", max_length = max_length, 
                                return_attention_mask = True, return_tensors = "pt", truncation = True, 
                                add_special_tokens = True) 
        newdictionary = {} 
        newdictionary['input_ids'] = tokdictionary['input_ids'].squeeze(0) 
        newdictionary['attention_mask'] = tokdictionary['attention_mask'].squeeze(0) 
        return newdictionary 
    
    def encode_text_summary_xsum(examples): # xsum uses "document" 
        tokdictionary = tokenizer(examples["document"], padding = "max_length", max_length = max_length, 
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
    if datasetname == "pg19": 
        datasetnew = datasetnew.map(encode_with_truncationspecialized, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask"]) 
    elif datasetname == "xsum": 
        datasetnew = datasetnew.map(encode_text_summary_xsum, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask"]) 
    elif datasetname == "cnn_dailymail": 
        datasetnew = datasetnew.map(encode_text_summary, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask"]) 
    else: 
        datasetnew = datasetnew.map(encode_with_truncation, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask", "text"]) 
    # datasetnew = datasetnew.map(unflatten_list_func, num_proc = 8) 

    # datasetnew = datasetnew.map(unflatten_list_func, num_proc = 8) 
    return datasetnew 

# onedataset = get_dataset("pg19", args.max_length) 
# results = trainer.evaluate(onedataset) 
# ce_loss = results["eval_loss"] 
# ppl = results["eval_perplexity"] 
# print("ce_loss {} ppl {}".format(ce_loss, ppl)) 

trainer = CustomTrainer(
    args = training_args, 
    model = model, 
    data_collator = data_collator, 
    time_hash = hash_of_time, 
    text_eval = "just_evaluation_{}.txt".format(hash_of_time), 
    tokenizer = tokenizer, 
    commit_hash = commit_hash, 
    eval_dataset = get_dataset("pg19", args.max_length), 
) 

accumulate_loss = torch.zeros((args.max_length - 1, )).to(torch_device).float() 
accumulate_count = torch.zeros((args.max_length - 1, )).to(torch_device).float() 
sum = torch.zeros((1,)).to(torch_device).float() 
for i, batch in enumerate(tqdm(trainer.get_eval_dataloader())): 
    input_ids = batch["input_ids"].to(torch_device) 
    attention_mask = batch["attention_mask"].to(torch_device) 
    original_attention_mask = batch["attention_mask"] 
    labels = batch["labels"].to(torch_device) 
    batch_size, seq_len = original_attention_mask.shape 
    addedon_length = (seq_len - 7 - 1) // 7 
    
    large_input_ids = input_ids 
    small_input_ids = input_ids 
    
    original_attention_mask2 = torch.cat((original_attention_mask, torch.ones((batch_size, addedon_length), dtype = torch.long).to(small_input_ids.device)), dim = 1) 
    logits = None 
    for i in range(2): 
        past_key_values = None 
        with torch.no_grad(): 
            outputs = model(
                input_ids = input_ids[: , i * args.max_length/2 : (i + 1) * args.max_length/2], 
                attention_mask = attention_mask[: , i * args.max_length/2 : (i + 1) * args.max_length/2], 
                labels = None, 
                use_cache = (i == 1), 
                past_key_values = past_key_values, 
            ) 
            temp_logits = outputs.logits 
            if logits is None: 
                logits = temp_logits 
            else: 
                logits = torch.cat((logits, temp_logits), dim = 1) 
            past_key_values = outputs.past_key_values 
    print("the shape of logits is {}".format(logits.shape)) 
    exit(0) 
    logits = outputs.logits 
    logits = logits[..., :-1, :].contiguous() 
    labels = labels[..., 1:].contiguous() 
    
    ce_loss = CrossEntropyLoss(reduction = "none") 
    
    loss = ce_loss(logits.view(-1, logits.shape[-1]), labels.view(-1)) 
    # print(loss) 
    # mask = (loss != 0).float() 
    # print(loss.sum(0)/mask.sum(0)) 
    # if i == 10: 
        # exit(0) 
    # continue 
    sum += loss.sum(0) 
    mask = loss != 0 
    mask = mask.float() 
    accumulate_loss += loss 
    accumulate_count += mask 

accumulate_loss /= accumulate_count 
print("accumulate_loss is {}".format(accumulate_loss)) 
# print("sum is {}".format(sum)) 
