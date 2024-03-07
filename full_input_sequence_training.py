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

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 

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
parser.add_argument("--finetune_checkpoint", type = str, default = None) 

args = parser.parse_args() 
if args.embedding_pretrained: 
    args.group2lr = None # we enforce it 
print(args) 

if "lovelace" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/yangzho6/model_checkpoints/" 
    dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
elif "ada" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/beidic/yangzho6/model_checkpoints/" 
    dir_sdata = "/home/beidic/yangzho6/c4llm_synthesized/" 
else: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/yangzho6/model_checkpoints/" 
    dir_sdata = "/home/yangzho6/c4llm_synthesized/" 

# has_wandb = False # disable for debugging 
# model_name = "openllama3b" 
# model_name = "shearedllama2_7b" 
model_name = args.model_name 
text_eval = "evaluation_printout_{}_{}_{}.txt".format(commit_hash, hash_of_time, model_name) 

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
            input_condensed = False, 
            sliding_window_length = 7, 
            use_past = False, 
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
        self.input_condensed = input_condensed 
        self.text_eval = text_eval 
        self.sliding_window_length = sliding_window_length 
        self.use_past = use_past 
        
        if self.args.resume_from_checkpoint is not None: 
            self.time_checkpoint = int(self.args.resume_from_checkpoint.split("-")[-1]) 
            print(colored("resuming from checkpoint {}".format(self.time_checkpoint), "yellow")) 
            print(colored("the learning rate is {}".format(self.optimizer.param_groups[0]["lr"]), "yellow")) 
            print(colored("the step count is {}".format(self.state.global_step), "yellow")) 
            if self.iteration_count == 0: 
                self.iteration_count = 4 * self.state.global_step 
        if input_condensed: 
            self.large_model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models) 
            self.large_model_embeds = self.large_model.get_input_embeddings().to(torch_device) 
    
    def naive_grouping(self, input_ids): 
        # embedding_searched = self.model.embed_tokens(input_ids) 
        embedding_searched = self.large_model_embeds(input_ids) 
        # print("embedding_searched shape {} {}".format(embedding_searched.shape, embedding_searched.dtype)) 
        seq_length = embedding_searched.shape[1] 
        
        # assert seq_length % 7 == 0, "seq_length is not divisible by 7" 
        assert seq_length % self.sliding_window_length == 0, "seq_length is not divisible by sliding_window_length" 
        # added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // 7, embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
        added_tensor = torch.zeros((embedding_searched.shape[0], seq_length // self.sliding_window_length, embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
        # for i in range(seq_length // 7): 
        for i in range(seq_length // self.sliding_window_length): 
            sum = torch.zeros((embedding_searched.shape[0], embedding_searched.shape[2])).to(input_ids.device).to(embedding_searched.dtype) 
            # for j in range(7): 
            for j in range(self.sliding_window_length): 
                # sum += embedding_searched[:, i * 7 + j, :] 
                sum += embedding_searched[:, i * self.sliding_window_length + j, :] 
                # sum /= 7. 
                sum /= float(self.sliding_window_length) 
                # print("sum dtype {}".format(sum.dtype)) 
            added_tensor[:, i, :] = sum 
        # print("added_tensor shape {}".format(added_tensor.shape)) 
        
        return added_tensor 
    
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """ 
        print(colored("resume from checkpoint: {}".format(resume_from_checkpoint), "yellow")) 
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None
        '''
        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")
        ''' 
        if (
            resume_from_checkpoint is not None
            and not is_sagemaker_mp_enabled()
            and not self.is_deepspeed_enabled
            and not self.is_fsdp_enabled
        ):
            self._load_from_checkpoint(resume_from_checkpoint)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        ) 
        print("resume_from_checkpoint is {}".format(resume_from_checkpoint)) 
        
        if args.push_to_hub:
            try:
                # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                hf_hub_utils.enable_progress_bars()
        else:
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
    
    '''
    def _save_checkpoint(self, model, trial, metrics = None): 
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"
        
        print(colored("running updated save checkpoint, now with more identifiable names", "cyan")) 
        # Save model checkpoint
        changed_checkpoint_prefix = "{}largemodel{}kernelsize{}date{}".format("SimpleSmallModel" if isinstance(self.model, SimpleSmallModel) else "LlamaModel", model_name, self.model.sliding_window_length, hash_of_time) 
        # checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}" 
        checkpoint_folder = changed_checkpoint_prefix + "-{}".format(self.state.global_step) 

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.is_deepspeed_enabled:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.model_wrapped.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.fsdp or self.is_fsdp_enabled:
            if self.is_fsdp_enabled:
                save_fsdp_optimizer(
                    self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, output_dir
                )
            else:
                # FSDP has a different interface for saving optimizer states.
                # Needs to be called on all ranks to gather all states.
                # full_optim_state_dict will be deprecated after Pytorch 2.2!
                full_osd = self.model.__class__.full_optim_state_dict(self.model, self.optimizer)
                torch.save(full_osd, os.path.join(output_dir, OPTIMIZER_NAME))

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
        elif self.args.should_save and not self.is_deepspeed_enabled and not (self.fsdp or self.is_fsdp_enabled):
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

        # Save SCHEDULER & SCALER
        is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and not isinstance(
            self.lr_scheduler, DeepSpeedSchedulerWrapper
        )
        if (
            self.args.should_save
            and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler)
            and not is_torch_tpu_available()
        ):
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        if is_torch_npu_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                rng_states["npu"] = torch.npu.random.get_rng_state_all()
            else:
                rng_states["npu"] = torch.npu.random.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
    ''' 
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
        '''
        for name, parameters in model.named_parameters(): 
            if name == "embed_tokens.weight": 
                # print(colored("{} has gradient {}".format(name, parameters.grad.data[1][: 100]), "light_magenta")) 
                for i in range(parameters.grad.data.shape[0]): 
                    if (parameters.grad.data[i] != 0).any(): 
                        print(colored("row {} has gradient that is numerically not zero {}".format(i, parameters.grad.data[i][: 20]), "light_magenta")) 
            else: 
                print(colored("{} has gradient {}".format(name, parameters.grad.data.view(-1)[: 10]), "light_magenta")) 
            print("the gradient of {} contains nan or not Ture or False: {}".format(name, torch.isnan(parameters.grad.data.view(-1).any()))) 
        ''' 
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
        '''
        for k, v in inputs.items(): 
            if isinstance(v, tuple): 
                print(k, len(v)) 
            elif isinstance(v, torch.Tensor): 
                if k == "condensed_embeds": 
                    print(k, v.shape) 
                else: 
                    print(k, v) 
            else: 
                print(k, v) 
        ''' 
        # print("attention_mask: {}".format(inputs["attention_mask"])) 
        input_ids = inputs["input_ids"] 
        attention_mask = inputs["attention_mask"] 
        label2 = inputs["labels"] 
        # print("the optimizer parameter group list 0 is {} learning rate is {}".format(len(self.optimizer.param_groups[0]['params']), self.optimizer.param_groups[0]['lr'])) 
        # print("the optimizer parameter group list 1 is {} learning rate is {}".format(len(self.optimizer.param_groups[1]['params']), self.optimizer.param_groups[1]['lr'])) 
        # print("the input ids are {}".format(input_ids[0])) 
        # print("labels are {}".format(labels[0])) 
        print("type of the model is {}".format(type(model))) 
        if isinstance(getattr(model, "module", model), SimpleSmallModel) or isinstance(model, SimpleSmallModel) == True and not self.use_past: 
            if self.input_condensed: 
                condensed_embeds = self.naive_grouping(input_ids[:, 64:]).to(self.dtype) # condensed_embeds shape is (28, d_model) 
            else: 
                condensed_embeds = inputs["condensed_embeds"].to(self.dtype) 
            print(colored("the shape of condensed_embeds is {}".format(condensed_embeds.shape), "yellow")) 
            batch_size, seq_len = attention_mask.shape 
            addedon_length = condensed_embeds.shape[1] 
            # print("get the input sentence: {}".format(tokenizer.decode(input_ids[0]))) 
            attention_mask = torch.cat((attention_mask, torch.ones((batch_size, addedon_length), dtype = torch.long).to(input_ids.device)), dim = 1) 
            
            # print("condensed_embeds dtype is {}".format(condensed_embeds.dtype)) 
            # print("condensed_embeds is {}".format(condensed_embeds)) 
            # print("input_ids are {}".format(input_ids)) 
            # outputs = model(input_ids = large_outputs.sequences, attention_mask = attention_mask, labels = large_outputs.sequences, condensed_embeds = downsampled_vectors) 
            if self.accelerator.is_main_process: 
                print("printing out the experiment_setting: {} eval_mode: {}".format(self.experiment_setting, self.eval_mode)) 
            outputs = model(
                input_ids = input_ids, 
                attention_mask = attention_mask, 
                labels = label2, 
                condensed_embeds = condensed_embeds, 
                output_hidden_states = True, 
                output_attentions = True, 
                return_dict = True, 
                # condensed_fashion = "ground_truth", 
                iteration_count = self.iteration_count, 
                # eval_mode = True, 
                experiment_setting = self.experiment_setting, 
                # eval_model = self.eval_mode, 
                eval_mode = self.eval_mode, 
            ) 
            
            # visualize attention map 
            # print("the input ids are {}".format(input_ids))
            '''
            if isinstance(outputs.attentions, tuple): 
                print("the attention mask have shape {}".format(len(outputs.attentions))) 
                print("the attention mask first element has shape {}".format(outputs.attentions[0].shape)) 
            else: 
                print("the attention mask has shape {}".format(outputs.attentions.shape)) 
            SimpleSmallModel.plot_attention_map(outputs.attentions, 0, 0, 144, "testing_attention_map.jpg") 
            print(outputs.attentions[0][0][0][64]) 
            
            if isinstance(outputs.hidden_states, tuple): 
                print("the hidden states have shape {}".format(len(outputs.hidden_states))) 
                print("the hidden states first element has shape {}".format(outputs.hidden_states[0].shape)) 
            for i in range(len(outputs.hidden_states)): 
                print(outputs.hidden_states[i][0][64][: 10]) 
            exit(0) 
            ''' 
        elif isinstance(getattr(model, "module", model), SimpleSmallModel2) or isinstance(model, SimpleSmallModel2) and self.use_past: 
            if self.input_condensed: 
                condensed_embeds = self.naive_grouping(input_ids[:, 64:]).to(self.dtype) # condensed_embeds shape is (28, d_model) 
            else: 
                condensed_embeds = inputs["condensed_embeds"].to(self.dtype) 
            print(colored("the shape of condensed_embeds is {}".format(condensed_embeds.shape), "yellow")) 
            condensed_embeds = condensed_embeds[:, : -1, :] # condensed_embeds has shape (batch_size, number of condensed tokens, d_model) 
            assert condensed_embeds.shape[1] == (input_ids.shape[1] - 64 - self.sliding_window_length) // self.sliding_window_length 
            batch_size, seq_len = attention_mask.shape 
            addedon_length = condensed_embeds.shape[1] 
            # print("get the input sentence: {}".format(tokenizer.decode(input_ids[0]))) 
            attention_mask = torch.cat((attention_mask, torch.ones((batch_size, addedon_length), dtype = torch.long).to(input_ids.device)), dim = 1) 
            
            # print("condensed_embeds dtype is {}".format(condensed_embeds.dtype)) 
            # print("condensed_embeds is {}".format(condensed_embeds)) 
            # print("input_ids are {}".format(input_ids)) 
            # outputs = model(input_ids = large_outputs.sequences, attention_mask = attention_mask, labels = large_outputs.sequences, condensed_embeds = downsampled_vectors) 
            if self.accelerator.is_main_process: 
                print("printing out the experiment_setting: {} eval_mode: {}".format(self.experiment_setting, self.eval_mode)) 
            outputs = model(
                input_ids = input_ids, 
                attention_mask = attention_mask, 
                labels = label2, 
                condensed_embeds = condensed_embeds, 
                output_hidden_states = True, 
                output_attentions = True, 
                return_dict = True, 
                # condensed_fashion = "ground_truth", 
                iteration_count = self.iteration_count, 
                # eval_mode = True, 
                experiment_setting = self.experiment_setting, 
                # eval_model = self.eval_mode, 
                eval_mode = self.eval_mode, 
            ) 
        else: 
            outputs = model(
                input_ids = input_ids, 
                attention_mask = attention_mask, 
                labels = label2, 
                # condensed_embeds = condensed_embeds, 
                output_hidden_states = True, 
                output_attentions = True, 
                return_dict = True, 
                # eval_mode = True, 
            ) 
        
        # print(outputs.hidden_states[0].shape) 
        # print(outputs.hidden_states[0][0][0][: 10]) 
        # print(len(outputs.hidden_states)) 
        # print(outputs.attentions[0][0]) 
        
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
                    SimpleSmallModel.plot_attention_map(outputs.attentions, layer, head, input_ids.shape[1] + addedon_length, plot_name) 
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
        
        # inspect the hidden states here 

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

        logits = logits[:, :-1, :] 
        # input_attention_mask = input_attention_mask[:, :-1] 
        input_attention_mask = input_attention_mask[:, 1:] 
        labels = labels[:, 1:] 
        preds = torch.argmax(logits, dim = -1) 
        if self.accelerator.is_main_process and outside_step == 0: 
            print("*** evaluating at step {} ***".format(self.iteration_count)) 
            # f = open("key_notes{}.md".format(self.commit_hash), "a") 
            # f.write("writing key notes at step {}".format(self.iteration_count)) 
            mask_correctness = (preds[: 20, 63 :] == labels[: 20, 63 :]).to(torch.bool) 
            # print(mask_correctness.shape) 
            pred_outputs = preds[: 20] 
            write_out_text = [] 
            for i in range(len(pred_outputs)): 
                prediction_prefix = self.tokenizer.decode(pred_outputs[i][: 63]) 
                prediction_text = "the prediction is: {}".format(prediction_prefix) 
                for j in range(mask_correctness.shape[1]): 
                    if mask_correctness[i][j]: 
                        prediction_text += colored(self.tokenizer.decode(pred_outputs[i][63 + j]), "green") + " " 
                    else: 
                        prediction_text += colored(self.tokenizer.decode(pred_outputs[i][63 + j]), "red") + " " 
                print(prediction_text) 
                print() 
                # print(labels[i]) 
                mask_filtered = labels[i][input_attention_mask[i] == 1] 
                mask_filtered[mask_filtered == -100] = 0 
                labels_outputs1 = self.tokenizer.decode(mask_filtered[: 63]) 
                label_text = "the label is: {}".format(colored(labels_outputs1, "yellow")) 
                print(label_text, end = " ") 
                labels_outputs2 = self.tokenizer.decode(mask_filtered[63 :]) 
                write_out_text.append("the prediction is: " + prediction_prefix + " " + prediction_text + "\n" + "the label is: " + label_text + " " + labels_outputs2 + "\n") 
                print(colored(labels_outputs2, "cyan")) 
                print() 
                print() 
                # wandb.log({"key notes: ": prediction_text + label_text}) 
                # f.write(prediction_text + "\n" + label_text + "\n") 
                
            with open(self.text_eval, "a") as f: 
                f.write("*** at step {} {}".format(self.iteration_count, self.state.global_step)) 
                f.write("\n") 
                for i, text in enumerate(write_out_text): 
                    f.write("example {}/{}\n".format(i, len(write_out_text))) 
                    f.write(text) 
                    f.write("\n") 
                f.write("\n") 
            # f.write("\n") 
            # f.close() 
            # self.artifact.add_file("key_notes{}.md".format(self.commit_hash), name = "key_notes.md") 
            # wandb.log_artifact(self.artifact) 
        if self.accelerator.state.num_processes > 1: 
            # torch.distributed.barrier() # I found that barrier() works, but it still not as good as wait_for_everyone() 
            self.accelerator.wait_for_everyone() 
                
        # print("the shape of preds is {}".format(preds.shape)) 
        # use loss to compute perplexity 
        perplexity = torch.exp(loss).mean().item() 
        # print("the perplexity is {}".format(perplexity)) 
        # use preds to compute accuracy 
        indices_to_keep = input_attention_mask == 1 # only for debugging purposes 
        total_valid_tokens = torch.sum(indices_to_keep.view(-1), dim = 0).item() 
        # print("shape of indices_to_keep: {}".format(indices_to_keep.shape)) 
        interest_token_count = torch.sum(indices_to_keep[:, 63 :].reshape(-1), dim = 0).item() # check whether 63 makes sense and make it more general if it is correct or not 
        # accuracy = accuracy_score(labels[indices_to_keep], preds[indices_to_keep]) 
        correct_words = torch.sum((preds[indices_to_keep] == labels[indices_to_keep]).view(-1), dim = 0).item() 
        # print("shape of indices_to_keep: {}".format(indices_to_keep.shape)) 
        # interest_correct_count = torch.sum((preds[indices_to_keep][:, 63 :] == labels[indices_to_keep][:, 63 :]).view(-1), dim = 0).item() 
        interest_correct_count = torch.sum(((preds * indices_to_keep)[:, 63: ] == (labels * indices_to_keep)[:, 63: ]).view(-1), dim = 0).item() 
        print("correct words: {} and total words: {}".format(correct_words, total_valid_tokens)) 
        print("interest correct words: {} and interest total words: {}".format(interest_correct_count, interest_token_count)) 
        # use preds to compute f1 score 
        # f1 = precision_recall_fscore_support(labels, preds, average = "weighted") 
        return {"perplexity": perplexity, "correct_words": correct_words, "total_words": total_valid_tokens, "interest_correct_words": interest_correct_count, "interest_total_words": interest_token_count} 
    
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
        if self.accelerator.is_main_process: 
            print(colored(metrics, "magenta")) 
            wandb.log({"global_eval_perplexity": global_perplexity, "global_eval_accuracy": global_accuracy, "global_eval_interest_accuracy": global_interest_accuracy}) 

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

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples) 
        
class CustomDataset: 
    def __init__(self, data_dir, tokenizer = None, max_length = 256, kernel_size = 7, input_condensed = True): 
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
        self.dict_kernel_maxlength = {2 : 64, 3 : 63, 4 : 64, 5 : 65, 6 : 66, 7 : 70, 10 : 70} 
        self.kernel_size = kernel_size 
        self.input_condensed = input_condensed 
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
            if not self.input_condensed: 
                print("we are ")
                tensor = torch.load(item["condensed_token_path"]) 
        except IOError as e: 
            print(colored("///IOError occured replacing with an empty tensor///", "red")) 
            tensor = torch.zeros((28, 2560 if model_name == "shearedllama2_7b" else 3200), dtype = torch.float32) 
        
        if self.tokenizer is not None: 
            encoded_text = self.tokenizer( 
                item["text"], 
                # add_special_tokens = False, 
                add_special_tokens = True, 
                padding = "max_length", 
                # max_length = 64 + self.dict_kernel_maxlength[self.kernel_size], 
                max_length = self.max_length, 
                return_attention_mask = True, 
                return_tensors = "pt", 
                truncation = True, 
            ) 
            
            item['input_ids'] = encoded_text['input_ids'].squeeze(0)  # remove the batch dimension
            item['attention_mask'] = encoded_text['attention_mask'].squeeze(0)  # remove the batch dimension 
        
        if not self.input_condensed: 
            item["condensed_embeds"] = tensor 
        else: 
            # turn this field into a placeholder 
            item["condensed_embeds"] = torch.zeros((28, ))
        # print(colored("the shape of condensed_embeds is {}".format(tensor.shape), "yellow")) 
        # item["input_ids"] = torch.tensor(item["input_ids"]) 
        # item["attention_mask"] = torch.tensor(item["attention_mask"]) 

        return item 

    def split(self, train_size): 
        if isinstance(train_size, float): 
            train_size = int(train_size * len(self)) 
        eval_size = len(self) - train_size 
        return random_split(self, [train_size, eval_size]) 

# defining tokenizer 
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir) 
# if args.resume_from_checkpoint is None: 
#     tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
# else: 
#     tokenizer = AutoTokenizer.from_pretrained(args.resume_from_checkpoint) 
tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
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

'''
# backup dataset 
onedataset = load_dataset('json', data_files = '/home/yangzho6/c4llm_synthesized/c4synthesized_file1.json', split = "train") 
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

datasetnew = CustomDataset(max_length = 260, data_dir = dir_sdata, tokenizer = tokenizer, kernel_size = kernel_size, input_condensed = args.input_condensed) 
# datasetnew.preprocess_dataset() 
train_set, test_set = datasetnew.split(0.98)     # 712k * 0.95 = 676k 712k * 0.05 = 36k 
                                                 # 356k * 0.99 = 352k 356k * 0.01 = 3.6k 
                                                 # 5 * 356k = 1780000, 1780000 * 0.98 = 1744400, 1780000 * 0.02 = 35600 

if (not args.use_plain_model or args.resume_from_checkpoint is not None) and not args.use_past and not args.finetune_checkpoint: 
    print(colored("we use custom small", "cyan")) 
    # handling simplesmallmodel 
    # small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = cache_dir).to(torch_device) 
    # small_config = LlamaConfig.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
    small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models) 

    small_state_dict_for_model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).state_dict() 
    if model_name == "openllama3b": 
        small_model = SimpleSmallModel(small_config, hostname = hostname, sliding_window_length = kernel_size, target_model_dim = 3200) 
    elif model_name == "shearedllama2_7b": 
        small_model = SimpleSmallModel(small_config, hostname = hostname, sliding_window_length = kernel_size, target_model_dim = 2560) 
    else: 
        small_model = SimpleSmallModel(small_config, hostname = hostname, sliding_window_length = kernel_size, target_model_dim = 2048) 

    new_state_dict = {} 

    for key in small_state_dict_for_model.keys(): 
        new_key = key 
        if 'lm_head' in key: 
            print("got here found the following key {}".format(key)) 
        if 'model.' in key: 
            new_key = key[6 :] 
        print(new_key) 
        new_state_dict[new_key] = small_state_dict_for_model[key] 
    if args.embedding_pretrained: 
        new_state_dict["embed_projection.weight"] = torch.load("linearprojectionweighttesting.pt") 

    try: 
        small_model.load_state_dict(new_state_dict) 
    except RuntimeError as r: 
        print(colored(r, "yellow")) 

    small_model = small_model.to(torch_device) 
    # small_model = small_model.to(torch.bfloat16).to(torch_device) 
    small_model.train() 

    # custom_lr_scheduler = torch.optim.lr_scheduler.LambdaLR 
elif args.finetune_checkpoint: 
    print(colored("we use finetune model", "cyan")) 
    small_model = SimpleSmallModel.from_pretrained(args.finetune_checkpoint, hostname = hostname, sliding_window_length = kernel_size, target_model_dim = 2048).to(torch_device) 
    small_model.train() 
elif args.use_plain_model and not args.use_past: 
    print(colored("we use plain model", "cyan")) 
    # alternative pretrained model 
    # small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m").to(torch_device) 
    # config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
    # print(config) 
    # small_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", cache_dir = dir_models).to(torch_device) 
    # small_model = AutoModelForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).to(torch_device) 
    small_model = LlamaCausalLMWeirdTwo.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).to(torch_device) 
    # small_model = LlamaCausalLMWeirdTwo.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).to(torch_device) 
    small_model.train() 
elif args.use_past: 
    print(colored("SimpleSmallModel with past", "cyan")) 
    small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models) 

    small_state_dict_for_model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).state_dict() 
    small_model = SimpleSmallModel2(small_config, hostname = hostname, sliding_window_length = kernel_size, target_model_dim = 2048) 
    new_state_dict = {} 

    for key in small_state_dict_for_model.keys(): 
        new_key = key 
        if 'lm_head' in key: 
            print("got here found the following key {}".format(key)) 
        if 'model.' in key: 
            new_key = key[6 :] 
        print(new_key) 
        new_state_dict[new_key] = small_state_dict_for_model[key] 
    if args.embedding_pretrained: 
        new_state_dict["embed_projection.weight"] = torch.load("linearprojectionweighttesting.pt") 

    try: 
        small_model.load_state_dict(new_state_dict) 
    except RuntimeError as r: 
        print(colored(r, "yellow")) 

    small_model = small_model.to(torch_device) 
    # small_model = small_model.to(torch.bfloat16).to(torch_device) 
    small_model.train() 
else: 
    raise ValueError("please specify whether you want to use plain model or custom model") 

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
model_path = dir_models + "{}_{}_{}_{}_{}/".format(model_name, args.experiment_setting, args.kernel_size, commit_hash, hash_of_time) 
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint 
    # resume_from_checkpoint="./model_output/checkpoint-500", 
    # resume_from_checkpoint = args.resume_from_checkpoint, 
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=5,            # number of training epochs, feel free to tweak
    per_device_train_batch_size = 64,  # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=4,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=128,  # evaluation batch size
    # logging_steps=1, 
    logging_steps = 500,            # evaluate, log and save model checkpoints every 1000 step
    # save_steps=1000, 
    # save_steps = 2000, 
    save_steps = 500, 
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
    lr_scheduler_type = "cosine", 
) 
print(colored("resum_from_checkpoint is {}".format(args.resume_from_checkpoint), "red")) 
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

trainer = CustomTrainer( 
    model = small_model, 
    args = training_args, 
    train_dataset = train_set, 
    eval_dataset = test_set, 
    # train_dataset = train_dataset, 
    # eval_dataset = test_dataset, 
    data_collator = data_collator, 
    # compute_metrics = compute_metrics, 
    optimizers = (custom_optimizer, None), 
    experiment_setting = args.experiment_setting, 
    tokenizer = tokenizer, 
    eval_mode = args.eval_mode, 
    time_hash = hash_of_time, 
    # dtype = torch.bfloat16, # TODO find a way to automatically do it 
    dtype = weightmodelfirst.dtype, 
    model_name = model_name, 
    text_eval = model_path + text_eval, 
    input_condensed = args.input_condensed, 
    sliding_window_length = args.kernel_size, 
    use_past = args.use_past, 
) 

max_length = 128 
if trainer.accelerator.is_main_process and has_wandb: 
    project_setting = args.experiment_setting if args.eval_mode is False else "finetuning" 
    today = datetime.date.today() 
    wandblogconfigs = {**(training_args.to_dict()), **(args.__dict__)} 
    wandblogconfigs["git_commit"] = commit_hash 
    wandblogconfigs["time_hash"] = hash_of_time 
    wandblogconfigs["model_name"] = model_name 
    wandblogconfigs["texteval"] = model_path + text_eval 
    # wandb.init(project = "llm160m", config = training_args, name="{}_{}".format(today, project_setting)) 
    wandb.init(project = "llm160m", config = wandblogconfigs, name = "{}_{}_{}".format(today, project_setting, "custom" if args.use_plain_model is False else "plain")) 

print("experiment-setting is {}".format(trainer.experiment_setting)) 
# print("***** Using input condensed tokens {} *****".format("yes" if trainer.input_condensed else "no")) 
print("***** Using input condensed tokens {} *****".format("yes" if args.input_condensed else "no")) 

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

trainer.train(resume_from_checkpoint = args.resume_from_checkpoint) 

wandb.finish() 
