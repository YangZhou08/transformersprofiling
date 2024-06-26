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
from src.transformers.models.llama.modeling_llama import LlamaWeirdLarge3 
from src.transformers.models.llama.modeling_llama import SimpleSmallModel2 
from src.transformers.models.llama.modeling_llama import LlamaForCausalLM2 
from src.transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model 
import time 
from torch.utils.data import random_split 
from src.transformers import BitsAndBytesConfig 
from packaging import version 
# import torch.nn.parallel.distributed.DistributedDataParallel as DDP 

import datetime 
import os 
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union 

if TYPE_CHECKING: 
    import optuna 

# # cache_dir = "/home/bc20/yang/" 
# dir_dataset = "/home/yangzho6/c4_parts" 
# dir_models = "/home/yangzho6/model_checkpoints2" 
# dir_sdata = "/home/yangzho6/c4llm_synthesized/" 

# torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
rank = os.environ.get("RANK") 
print("the rank is {}".format(rank)) 
if rank is None: 
    rank = 0 
torch_device = 'cuda:{}'.format(rank) if torch.cuda.is_available() else 'cpu' 

# Set a global seed for reproducibility
seed_value = 42
from src.transformers import set_seed 
set_seed(seed_value) 

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False 

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
    is_torch_npu_available, 
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

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt" 

import warnings 
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False 
from src.transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments 
import random 

logger = logging.get_logger(__name__) 

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help') 

parser.add_argument("--group1lr", type = float, default = 2e-4) 
parser.add_argument("--group2lr", type = float, default = 2e-3) 
parser.add_argument("--experiment_setting", type = str, default = "setting0") 
parser.add_argument("--eval_mode", action="store_true", default = False) 
parser.add_argument("--embedding_pretrained", action = "store_true", default = False) 
parser.add_argument("--input_condensed", action = "store_true", default = False) 
parser.add_argument("--kernel_size", type = int, default = 4) 
parser.add_argument("--use_plain_model", action = "store_true", default = False) 
parser.add_argument("--model_name", type = str, default = "openllama3b") 
parser.add_argument("--resume_from_checkpoint", type = str, default = None) 
parser.add_argument("--use_past", action = "store_true") 
parser.add_argument("--debug", action = "store_true") 
parser.add_argument("--gradient_accumulation_steps", type = int, default = 4) 
parser.add_argument("--embedding_reinitialization_type", type = str, default = None) 
parser.add_argument("--cosine_similarity", action = "store_true") 
parser.add_argument("--use_old_checkpoint", action = "store_true") 
parser.add_argument("--batch_size", type = int, default = 64) 
parser.add_argument("--num_epochs", type = int, default = 1) 
parser.add_argument("--usedatasettype", type = str, choices = ["synthesized", "c4"], default = "synthesized") 
parser.add_argument("--path_d", type = int, default = None) # this is the argument that has to be specified 
parser.add_argument("--wandbsession", type = str, default = None) 

args = parser.parse_args() 
if args.embedding_pretrained: 
    args.group2lr = None # we enforce it 
print(args) 

if "lovelace" in hostname: 
    dir_models = "/home/yangzho6/model_checkpoints/" 
    dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
    datapath_c4 = "/home/yangzho6/c4_parts/downloads/" 
elif "ada" in hostname: 
    dir_models = "/home/beidic/yangzho6/model_checkpoints/" 
    dir_sdata = "/home/beidic/yangzho6/c4llm_synthesized/" 
else: 
    dir_models = "/fsx-storygen/beidic/yang/model_checkpoints/" 
    dir_sdata = "/fsx-storygen/beidic/yang/c4llm_synthesized/" 
    datapath_c4 = "/fsx-storygen/beidic/hanshi/data/c4/" 

model_name = args.model_name 
text_eval = "evaluation_printout_{}_{}_{}.txt".format(commit_hash, hash_of_time, model_name) 

class CustomTrainer(Trainer): 
    def __init__(
        self, 
        tokenizer = None, 
        commit_hash = None, 
        time_hash = None, 
        dtype = None, 
        model_name = None, 
        text_eval = None, 
        use_past = False, 
        *args, 
        **kwargs, 
    ): 
        super().__init__(*args, **kwargs) 
        self.time_checkpoint = 0 
        self.iteration_count = 0 
        self.tokenizer = tokenizer 
        self.commit_hash = commit_hash 
        self.time_hash = time_hash 
        self.dtype = dtype 
        self.model_name = model_name 
        self.text_eval = text_eval 
        self.use_past = use_past 
        
        if self.args.resume_from_checkpoint is not None: 
            self.time_checkpoint = int(self.args.resume_from_checkpoint.split("-")[-1]) 
            print(colored("resuming from checkpoint {}".format(self.time_checkpoint), "yellow")) 
            print(colored("the learning rate is {}".format(self.optimizer.param_groups[0]["lr"]), "yellow")) 
            print(colored("the step count is {}".format(self.state.global_step), "yellow")) 
            if self.iteration_count == 0: 
                self.iteration_count = 4 * self.state.global_step 
    
    def training_step(self, model, inputs): 
        model.train() 
        inputs = self._prepare_inputs(inputs) 

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
        
        if self.accelerator.is_main_process and self.iteration_count % 1000 == 0 and evaluation_mode is False and has_wandb and not args.debug: 
            print(colored("generating images ... at iteration {}".format(self.iteration_count), "yellow")) 
            for layer in [0, 6, 11]: 
                for head in [0, 6, 11]: 
                    plot_name = "testing_attention_map_{}_{}.jpg".format(self.commit_hash, self.time_hash) 
                    SimpleSmallModel.plot_attention_map(outputs.attentions, layer, head, input_ids.shape[1], plot_name) 
                    field_name = "layer{}_head{}".format(layer, head) 

                    try: 
                        wandb.log({field_name: wandb.Image(plot_name)}) 
                    except Exception as e: 
                        print(f"An error has occured during logging attention map: {e}") 
        
        return (loss, outputs) if return_outputs else loss 

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
        if self.accelerator.is_main_process: 
            print(colored(metrics, "magenta")) 
            wandb.log({"global_eval_perplexity": global_perplexity, "global_eval_accuracy": global_accuracy, "global_eval_interest_accuracy": global_interest_accuracy}) 
        
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

class CustomDataset: 
    def __init__(self, data_dir, tokenizer = None, max_length = 256, kernel_size = 7, input_condensed = True, use_c4 = False, istraining = True): 
        self.synthesize_dir = data_dir 
        dfiles = [] 
        topk = None 
        print(colored("hostname is {}".format(hostname), "yellow")) 
        if not use_c4: 
            if "lovelace" in hostname: 
                filename = "c4synthesized_file1_kernel7_0.json" 
                dfiles.append(self.synthesize_dir + "{}/".format(model_name) + filename) 
            else: 
                for i in range(0, 8): 
                    filename = "c4synthesized_file1_kernel7_{}_combined.json".format(i) 
                    dfiles.append(self.synthesize_dir + "{}_topk{}/".format(model_name, topk if topk is not None else "na") + filename) 
        else: 
            if istraining: 
                if "lovelace" in hostname: 
                    for i in range(1, 9): 
                        filename = "c4_file{}.json".format(i) 
                        dfiles.append(self.synthesize_dir + filename) 
                else: 
                    print(colored("using c4 files {} to {}".format(args.path_d * 8, args.path_d * 8 + 8), "yellow")) 
                    for i in range(args.path_d * 8, args.path_d * 8 + 8): 
                    # for i in range(0, 8): 
                        filename = "c4_file{}.json".format(i) 
                        dfiles.append(self.synthesize_dir + filename) 
            else: 
                if "lovelace" in hostname: 
                    filename = "c4_file9.json" 
                    dfiles.append(self.synthesize_dir + filename) 
                else: 
                    filename = "c4_file{}.json".format(100) 
                    dfiles.append(self.synthesize_dir + filename) 
        
        if args.debug: 
            self.dataset = load_dataset('json', data_files = dfiles, split = "train[:2000]") 
        else: 
            if istraining: 
                self.dataset = load_dataset('json', data_files = dfiles, split = "train") 
            else: 
                self.dataset = load_dataset('json', data_files = dfiles, split = "train[:200000]") 
        
        self.dict_kernel_maxlength = {2 : 64, 3 : 63, 4 : 64, 5 : 65, 6 : 66, 7 : 70, 10 : 70} 
        self.kernel_size = kernel_size 
        self.input_condensed = input_condensed 
        self.use_c4 = use_c4 
        
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
        
        if self.tokenizer is not None: 
            encoded_text = self.tokenizer(
                item["text"], 
                add_special_tokens = True, 
                padding = "max_length", 
                max_length = self.max_length, 
                return_attention_mask = True, 
                return_tensors = "pt", 
                truncation = True, 
            ) 
            
            item['input_ids'] = encoded_text['input_ids'].squeeze(0) 
            item['attention_mask'] = encoded_text['attention_mask'].squeeze(0) 
        
        return item 
    
    def split(self, train_size): 
        if isinstance(train_size, float): 
            train_size = int(train_size * len(self)) 
        eval_size = len(self) - train_size 
        return random_split(self, [train_size, eval_size]) 
                
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 

if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
tokenizer.padding_side = "left" # we pad from the left 
'''
dfiles = [] 
print(colored("hostname is {}".format(hostname), "yellow")) 
topk = None 
if "ada" in hostname: 
    for i in range(0, 2): 
        filename = "c4synthesized_file1_kernel7_{}.json".format(i) 
        dfiles.append(dir_sdata + "{}/".format(model_name) + filename) 
elif "lovelace" in hostname: 
    filename = "c4synthesized_file1_kernel7_0.json" 
    dfiles.append(dir_sdata + "{}/".format(model_name) + filename) 
else: 
    for i in range(0, 8): 
        filename = "c4synthesized_file1_kernel7_{}_combined.json".format(i) 
        dfiles.append(dir_sdata + "{}_topk{}/".format(model_name, topk if topk is not None else "na") + filename) 

if not args.debug: 
    onedataset = load_dataset('json', data_files = dfiles, split = "train") 
else: 
    onedataset = load_dataset('json', data_files = dfiles, split = "train[:2000]") 

d = onedataset.train_test_split(test_size = 0.1) 
def encode_with_truncation(examples): 
    return tokenizer(examples["text"], padding = "max_length", max_length = 260, 
                     return_attention_mask = True, return_tensors = "pt", truncation = True) 
train_dataset = d["train"].map(encode_with_truncation, batched = True, num_proc = 8) 
test_dataset = d["test"].map(encode_with_truncation, batched = True, num_proc = 8) 

train_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 
test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 
''' 
if args.usedatasettype == "synthesized": 
    datasetnew = CustomDataset(max_length = 260, data_dir = dir_sdata, tokenizer = tokenizer, kernel_size = 7, input_condensed = args.input_condensed) 
    train_dataset, test_dataset = datasetnew.split(0.98) 
else: 
    train_dataset = CustomDataset(max_length = 260, data_dir = datapath_c4, tokenizer = tokenizer, kernel_size = 7, input_condensed = args.input_condensed, use_c4 = True, istraining = True) 
    test_dataset = CustomDataset(max_length = 260, data_dir = datapath_c4, tokenizer = tokenizer, kernel_size = 7, input_condensed = args.input_condensed, use_c4 = True, istraining = False) 

if args.model_name == "tinyllama": 
    model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models).to(torch_device).to(torch.bfloat16) 
elif args.model_name == "small_model": 
    # small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models) 
    model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).to(torch_device).to(torch.bfloat16) 

model.train() 
model.config.pad_token_id = tokenizer.pad_token_id 

training_weights_group = [] 
for k, v in model.named_parameters(): 
    training_weights_group.append(v) 
print("length of the weight group is {}".format(len(training_weights_group))) 

custom_optimizer = torch.optim.AdamW([
    {"params": training_weights_group, "lr": args.group1lr}, 
]) 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

model_path = dir_models + "{}_{}_{}/".format(args.model_name, commit_hash, hash_of_time) 

training_args = TrainingArguments(
    output_dir = model_path, 
    overwrite_output_dir=True,      
    num_train_epochs=args.num_epochs, 
    per_device_train_batch_size = args.batch_size if not args.debug else 10,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    per_device_eval_batch_size= args.batch_size if not args.debug else 10, 
    # logging_steps = 100 if not args.debug else 1, 
    logging_steps = 500 if not args.debug else 1, 
    # save_steps = 100 if not args.debug else 1, 
    save_steps = 500 if not args.debug else 1, 
    learning_rate = 2e-5, 
    save_total_limit=5,
    warmup_steps = 100, 
    label_names = ["labels"], 
    remove_unused_columns = True, 
    save_strategy = "steps", 
    evaluation_strategy = "steps", 
    lr_scheduler_type = "cosine", 
) 
print(colored("resume_from_checkpoint is {}".format(args.resume_from_checkpoint), "red")) 
weightmodelfirst = next(model.parameters()) 
# print(weightmodelfirst.dtype) 
print(colored(weightmodelfirst.dtype, "red")) 

trainer = CustomTrainer(
    model = model, 
    args = training_args, 
    # train_dataset = train_dataset, 
    train_dataset = train_dataset, 
    # eval_dataset = test_dataset, 
    eval_dataset = test_dataset, 
    data_collator = data_collator, 
    optimizers = (custom_optimizer, None), 
    # optimizers = (custom_optimizer, None), 
    tokenizer = tokenizer, 
    time_hash = hash_of_time, 
    commit_hash = commit_hash, 
    text_eval = model_path + text_eval, 
) 

if trainer.accelerator.is_main_process and has_wandb: 
    today = datetime.date.today() 
    wandblogconfigs = training_args.to_dict() 
    wandblogconfigs["git_commit"] = commit_hash 
    wandblogconfigs["time_hash"] = hash_of_time 
    wandb.init(project = "chunkedlargefinetuning", 
               config = wandblogconfigs, 
               name = "plainmodel{}_{}".format(today, args.model_name), 
               id = args.wandbsession, 
               resume = True if args.resume_from_checkpoint is not None else False 
    ) 

torch.autograd.set_detect_anomaly(True) 

if args.path_d is None: 
    raise ValueError("please specify the path to the checkpoint file") 

if args.path_d > 0 and args.resume_from_checkpoint is None: 
    raise ValueError("please specify the path to the checkpoint file") 

if args.resume_from_checkpoint is not None: 
    assert args.wandbsession is not None, "please specify the wandb session name" 
    trainer.train(resume_from_checkpoint = args.resume_from_checkpoint) 
else: 
    trainer.train() 

wandb.finish() 
