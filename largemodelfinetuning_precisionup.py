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
from src.transformers.models.llama.modeling_llama import LlamaWeirdLarge 
from src.transformers.models.llama.modeling_llama import LlamaWeirdLarge2 
from src.transformers.models.llama.modeling_llama import LlamaWeirdLarge3 
from src.transformers.models.llama.modeling_llama import LlamaWeirdLargeTest 
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
    dir_models = "/fsx-storygen/beidic/yang/model_checkpoints/" 
    # dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
    dir_sdata = "/fsx-storygen/beidic/yang/c4llm_synthesized/" 

logger = logging.get_logger(__name__) 

parser = argparse.ArgumentParser() 
parser.add_argument("--kernel_size", type = int, default = 7) 
parser.add_argument("--use_pretrained_small_model", action = "store_true") 
parser.add_argument("--finetuned_small_model_checkpoint", type = str, default = None) 
parser.add_argument("--finetuned_large_model_checkpoint", type = str, default = None) 
parser.add_argument("--large_model", type = str, default = "openllama3b") 
parser.add_argument("--use_mse_loss", action = "store_true") 
parser.add_argument("--resume_from_checkpoint", type = str, default = None) 
parser.add_argument("--num_epochs", type = int, default = 5) 
parser.add_argument("--freeze_small_model", action = "store_true") 
parser.add_argument("--freeze_large_model", action = "store_true") 
parser.add_argument("--ce_loss_only", action = "store_true") 
parser.add_argument("--topk", type = int, default = None) 
parser.add_argument("--batch_size", type = int, default = 64) 
parser.add_argument("--debug", action = "store_true") 
parser.add_argument("--experiment_setting", type = str, default = "setting0") 
parser.add_argument("--alpha", type = float, default = 0.5) 
parser.add_argument("--lr", type = float, default = 5e-5) 
parser.add_argument("--gradient_accumulation_steps", type = int, default = 4) 
parser.add_argument("--embedding_reinitialization_type", type = str, default = None) 
parser.add_argument("--cosine_similarity", action = "store_true") 
parser.add_argument("--use_old_checkpoint", action = "store_true") 
parser.add_argument("--use_new_small_model_checkpoint", action = "store_true") 
parser.add_argument("--autoregressive_baseline", action = "store_true") 
parser.add_argument("--group_compress", action = "store_true") 

args = parser.parse_args() 
model_name = args.large_model 
if args.use_pretrained_small_model: 
    assert args.finetuned_small_model_checkpoint is not None 
text_eval = "evaluating_printout_{}_{}_{}.txt".format(commit_hash, hash_of_time, model_name) 

assert not (args.freeze_small_model and args.freeze_large_model) 
assert not (args.use_mse_loss and args.freeze_large_model) 

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
        # large_input_ids = inputs["large_input_ids"] # (batch_size, 203) 
        large_input_ids = inputs["input_ids"] # (batch_size, 203) 
        small_input_ids = inputs["input_ids"] # (batch_size, 203) 
        attention_mask = inputs["attention_mask"] 
        # attention_mask = inputs["attention_mask_chunk"] 
        if "condensed_embeds" in inputs.keys(): 
            condensed_embeds_labels = inputs["condensed_embeds"] # (batch_size, 28, 3200) 
            condensed_embeds_labels = condensed_embeds_labels.to(self.model.small_model_dtype) 
        else: 
            condensed_embeds_labels = None 
        if isinstance(self.model, LlamaWeirdLarge3): 
            # attention_mask = torch.ones((large_input_ids.shape[0], condensed_embeds_labels.shape[1] + 1), dtype = torch.long).to(large_input_ids.device) 
            # attention_mask = torch.ones((large_input_ids.shape[0], condensed_embeds_labels.shape[1] + 2), dtype = torch.long).to(large_input_ids.device) # sequence length is 204, one bos, 29 more tokens, so 30 in total, we have 28 condensed tokens 
            attention_mask = torch.ones((large_input_ids.shape[0], (large_input_ids.shape[1] - 1) // self.n + 1), dtype = torch.long).to(large_input_ids.device) 
        original_attention_mask = inputs["attention_mask"] # (batch_size, 203) 
        label2 = inputs["labels"] # (batch_size, 203) 
        print("shape of large_input_ids {} shape of small_input_ids {}".format(large_input_ids.shape, small_input_ids.shape)) 
        print("shape of output_attention_mask {}".format(original_attention_mask.shape)) 
        
        batch_size, seq_len = original_attention_mask.shape 
        # addedon_length = (seq_len - 8) // self.n 
        addedon_length = (seq_len - self.n - 1) // self.n 
        # addedon_length = 28 
        original_attention_mask = torch.cat((original_attention_mask, torch.ones((batch_size, addedon_length), dtype = torch.long).to(small_input_ids.device)), dim = 1) 
        # original_attention_mask = torch.cat((torch.ones((batch_size, addedon_length), dtype = torch.long).to(small_input_ids.device), original_attention_mask), dim = 1) 
        
        if isinstance(self.model, LlamaWeirdLarge3): 
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
        elif isinstance(self.model, LlamaWeirdLarge) or isinstance(self.model, LlamaWeirdLargeTest): 
            outputs = model(
                large_input_ids = large_input_ids, 
                small_input_ids = small_input_ids, 
                attention_mask = attention_mask, 
                output_hidden_states = True, 
                output_attentions = True, 
                return_dict = True, 
                # condensed_embed_labels = None, 
                original_attention_mask = original_attention_mask, 
                labels = label2, 
                condensed_embed_labels = condensed_embeds_labels, 
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
        if self.accelerator.is_main_process and has_wandb and self.iteration_count % 500 == 0 and not args.debug: 
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

class CustomDataset: 
    # def __init__(self, data_dir, tokenizer = None, max_length = 256, kernel_size = 7): 
    def __init__(self, data_dir, large_tokenizer = None, small_tokenizer = None, max_length = 256, kernel_size = 7, topk = None, prompt_length = 64): 
        # self.synthesize_dir = "/home/yangzho6/c4llm_synthesized/" 
        self.synthesize_dir = data_dir 
        # self.dataset = load_dataset('json', data_files = self.synthesize_dir + "c4synthesized_file1.json", split = "train") 
        # self.dataset = load_dataset('json', data_files = [self.synthesize_dir + 'c4synthesized_file1.json', self.synthesize_dir + 'c4synthesized_file2.json'], split="train") 
        dfiles = [] 
        print(colored("hostname is {}".format(hostname), "yellow")) 
        if "ada" in hostname: 
            for i in range(0, 2): 
                # filename = "c4synthesized_file1_kernel{}_{}.json".format(kernel_size, i) 
                # filename = "c4synthesized_file1_kernel{}_{}.json".format(kernel_size, i) 
                filename = "c4synthesized_file1_kernel7_{}.json".format(i) 
                dfiles.append(self.synthesize_dir + "{}/".format(model_name) + filename) 
        elif "lovelace" in hostname: 
            # filename = "c4synthesized_file1_kernel{}_{}.json".format(kernel_size, 0) 
            filename = "c4synthesized_file1_kernel7_0.json" 
            dfiles.append(self.synthesize_dir + "{}/".format(model_name) + filename) 
        else: 
            for i in range(0, 8): 
                # filename = "c4synthesized_file1_kernel{}_{}_combined.json".format(kernel_size, i) 
                filename = "c4synthesized_file1_kernel7_{}_combined.json".format(i) 
                dfiles.append(self.synthesize_dir + "{}_topk{}/".format(model_name, topk if topk is not None else "na") + filename) 
        
        if not args.debug: 
            self.dataset = load_dataset('json', data_files = dfiles, split = "train") 
        else: 
            self.dataset = load_dataset('json', data_files = dfiles, split = "train[:2000]") 
        # self.dataset = load_dataset('json', data_files = dfiles, split = "train[:2000]") 
        self.dict_kernel_maxlength = {2 : 64, 3 : 63, 4 : 64, 5 : 65, 6 : 66, 7 : 70, 10 : 70} 
        self.kernel_size = kernel_size 
        # self.dataset = self.dataset["train"][0: 5120] 

        # self.tokenizer = tokenizer 
        self.large_tokenizer = large_tokenizer 
        self.small_tokenizer = small_tokenizer 
        self.max_length = max_length 
        self.prompt_length = prompt_length 
    
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
            if model_name == "shearedllama2_7b": 
                dmodel = 2560 
            elif model_name == "openllama3b": 
                dmodel = 3200 
            elif model_name == "tinyllama": 
                dmodel = 2048 
            # tensor = torch.zeros((expected_condensed_token_length, dmodel), dtype = torch.float32) 
            tensor = torch.zeros((28, dmodel), dtype = torch.float32) 
            print(colored("///IOError occured replacing with an empty tensor///", "red")) 
            # tensor = torch.zeros((28, dmodel), dtype = torch.float32) 
        
        # expected_condensed_token_length = (self.max_length - self.prompt_length) // self.kernel_size 
        # tensor = torch.zeros((expected_condensed_token_length, dmodel), dtype = torch.float32) 
        
        if self.large_tokenizer is not None and self.small_tokenizer is not None: 
            large_encoded_text = self.large_tokenizer( 
                item["text"], # 6 word-level tokens + BOS to be the first chunk 
                # add_special_tokens = False, 
                add_special_tokens = True, 
                padding = "max_length", 
                # max_length = 64 + self.dict_kernel_maxlength[self.kernel_size], 
                max_length = self.max_length, 
                return_attention_mask = True, 
                return_tensors = "pt", 
                truncation = True, 
            ) 
            # item['large_input_ids'] = large_encoded_text['input_ids'][0].squeeze(0)  # remove the batch dimension 
            input_idsfull = large_encoded_text['input_ids'].squeeze(0) # remove the batch dimension 
            # if input_idsfull[57] == 2 or input_idsfull[57] == 1: # if the first token is </s> or <s> 
            if input_idsfull[self.prompt_length - self.kernel_size] == 2 or input_idsfull[self.prompt_length - self.kernel_size] == 1: # if the first token is </s> or <s> 
                head_token = torch.tensor([2], dtype = torch.long) # pad with </s> eos token 
                head_mask = torch.zeros((1, ), dtype = torch.long) # attention mask starts with 0 
            else: 
                head_token = torch.ones((1, ), dtype = torch.long) # pad with <s> bos token 
                head_mask = torch.ones((1, ), dtype = torch.long) # attention mask starts with 1 
            # item['large_input_ids'] = torch.cat((head_token, input_idsfull[57 :]), dim = 0) 
            item['large_input_ids'] = torch.cat((head_token, input_idsfull[(self.prompt_length - self.kernel_size) :]), dim = 0) 
            small_encoded_text = self.small_tokenizer(
                item["text"], # 6 word-level tokens + BOS to be the first chunk 
                # add_special_tokens = False, 
                add_special_tokens = True, 
                padding = "max_length", 
                # max_length = 64 + self.dict_kernel_maxlength[self.kernel_size],
                max_length = self.max_length, 
                return_attention_mask = True, 
                return_tensors = "pt", 
                truncation = True, 
            ) 
            input_idsfull2 = small_encoded_text['input_ids'].squeeze(0) # remove the batch dimension 
            # if input_idsfull2[57] == 2 or input_idsfull2[57] == 1: # if the first token is </s> or <s> 
            if input_idsfull2[self.prompt_length - self.kernel_size] == 2 or input_idsfull2[self.prompt_length - self.kernel_size] == 1: # if the first token is </s> or <s> 
                head_token2 = torch.tensor([2], dtype = torch.long) # pad with </s> eos token 
                head_mask2 = torch.zeros((1, ), dtype = torch.long) # attention mask starts with 0 
            else: 
                head_token2 = torch.ones((1, ), dtype = torch.long) # pad with <s> bos token 
                head_mask2 = torch.ones((1, ), dtype = torch.long) # attention mask starts with 1 
            # item['input_ids'] = torch.cat((head_token2, input_idsfull2[57 :]), dim = 0) 
            item['input_ids'] = torch.cat((head_token2, input_idsfull2[(self.prompt_length - self.kernel_size) :]), dim = 0) 
            # item['attention_mask'] = torch.cat((head_mask2, small_encoded_text['attention_mask'].squeeze(0)[57 :]), dim = 0) 
            item['attention_mask'] = torch.cat((head_mask2, small_encoded_text['attention_mask'].squeeze(0)[(self.prompt_length - self.kernel_size) :]), dim = 0) 
            
            # print("input_ids is {}, the length is {}".format(item["input_ids"], item["input_ids"].shape[0])) 
        
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
    
    def static_split(self, train_size): 
        if isinstance(train_size, float): 
            train_size = int(train_size * len(self)) 
        eval_size = len(self) - train_size 
        train_indices = list(range(0, train_size)) 
        eval_indices = list(range(train_size, len(self))) 
        
        train_set = CustomDatasetSubset(self, train_indices) 
        eval_set = CustomDatasetSubset(self, eval_indices) 
        return train_set, eval_set 

class CustomDatasetSubset: 
    def __init__(self, dataset, indices): # indices here is a list of ints 
        self.dataset = dataset 
        self.indices = indices 
    
    def __len__(self): 
        return len(self.indices) 
    
    def __getitem__(self, idx): 
        actual_idx = self.indices[idx] 
        return self.dataset.__getitem__(actual_idx) 
    
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
# large_tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models) 
# large_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
large_tokenizer = LlamaTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models) 
small_tokenizer = LlamaTokenizer.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
# tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models) 
# tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
# tokenizers = [large_tokenizer, small_tokenizer] 
# for tokenizer in tokenizers: 
tokenizer = small_tokenizer 
if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
tokenizer.padding_side = "left" 
# tokenizer.padding_side = "right" 

kernel_size = args.kernel_size 
# datasetnew = CustomDataset(max_length = 203, data_dir = dir_sdata, tokenizer = tokenizer, kernel_size = kernel_size) 
# datasetnew = CustomDataset(max_length = 260, data_dir = dir_sdata, large_tokenizer = large_tokenizer, small_tokenizer = small_tokenizer, kernel_size = kernel_size, topk = args.topk) 
# datasetnew = CustomDataset(max_length = 260, data_dir = dir_sdata, tokenizer = tokenizer, kernel_size = kernel_size, input_condensed = False) 
# train_dataset, test_dataset = datasetnew.split(0.98) 
# the max_length assignment is subject to change 
max_length_lookup = {2 : 260, 3 : 259, 4 : 260, 5 : 259, 6 : 262, 7 : 260, 8 : 264} 
# datasetnew = CustomDataset(max_length = max_length_lookup[kernel_size], data_dir = dir_sdata, large_tokenizer = large_tokenizer, small_tokenizer = small_tokenizer, kernel_size = kernel_size, topk = args.topk, prompt_length = 64) 
dfiles = [] 
if "ada" in hostname: 
    for i in range(0, 2): 
        # filename = "c4synthesized_file1_kernel{}_{}.json".format(kernel_size, i) 
        # filename = "c4synthesized_file1_kernel{}_{}.json".format(kernel_size, i) 
        filename = "c4synthesized_file1_kernel7_{}.json".format(i) 
        dfiles.append(dir_sdata + "{}/".format(model_name) + filename) 
elif "lovelace" in hostname: 
    # filename = "c4synthesized_file1_kernel{}_{}.json".format(kernel_size, 0) 
    filename = "c4synthesized_file1_kernel7_0.json" 
    dfiles.append(dir_sdata + "{}/".format(model_name) + filename) 
else: 
    for i in range(0, 8): 
        # filename = "c4synthesized_file1_kernel{}_{}_combined.json".format(kernel_size, i) 
        filename = "c4synthesized_file1_kernel7_{}_combined.json".format(i) 
        topk = None 
        dfiles.append(dir_sdata + "{}_topk{}/".format(model_name, topk if topk is not None else "na") + filename) 

if not args.debug: 
    onedataset = load_dataset('json', data_files = dfiles, split = "train") 
else: 
    onedataset = load_dataset('json', data_files = dfiles, split = "train[:2000]") 

d = onedataset.train_test_split(test_size = 0.98) 
def encode_with_truncation(examples): 
    new_item = {} 
    encoded_dict = tokenizer(examples["text"], padding = "max_length", max_length = 260, 
                     return_attention_mask = True, return_tensors = "pt", truncation = True, add_special_tokens = True) 
    new_item["input_ids"] = encoded_dict["input_ids"].squeeze(0) 
    new_item["attention_mask"] = encoded_dict["attention_mask"].squeeze(0) 
    return new_item 
# train_dataset = d["train"].map(encode_with_truncation, batched = True, num_proc = 8) 
train_dataset = d["train"].map(encode_with_truncation, num_proc = 8) 
# test_dataset = d["test"].map(encode_with_truncation, batched = True, num_proc = 8) 
test_dataset = d["test"].map(encode_with_truncation, num_proc = 8) 

# train_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 
# test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 

# if not args.use_pretrained_small_model: 
#     train_set, test_set = datasetnew.split(0.98) 
# else: 
#     train_set, test_set = datasetnew.static_split(0.98) 

# for i in range(0, 2): 
#     item = train_set[i] 
#     print("the shape of condensed_embeds is {}".format(item["condensed_embeds"].shape)) 

for i in range(0, 5): 
    example = train_dataset[i] 
    print("The input ids is {}".format(example["input_ids"])) 
    print("The attention mask is {}".format(example["attention_mask"])) 
    # print("The text is {}".format(example["text"])) 

# TODO change the following code to use the checkpoint of the best trained window 7 model 
small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models) 

if args.large_model == "openllama3b": 
    large_dim = 3200 
elif args.large_model == "shearedllama2_7b": 
    large_dim = 2560 
elif args.large_model == "tinyllama": 
    large_dim = 2048 
else: 
    large_dim = 4096 

if args.use_new_small_model_checkpoint and not args.use_pretrained_small_model: 
    small_state_dict_for_model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).state_dict() 
    print(colored("not using pretrained small model", "green")) 
    # small_model = SimpleSmallModel(small_config, hostname = hostname, sliding_window_length = 7, target_model_dim = large_dim) 
    small_model = SimpleSmallModel(small_config, hostname = hostname, sliding_window_length = args.kernel_size, target_model_dim = large_dim).to(torch.bfloat16) 

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
    small_model.train() 
elif args.use_pretrained_small_model: 
    print(colored("using pretrained small model", "red")) 
    small_model = SimpleSmallModel(small_config, sliding_window_length = args.kernel_size, hostname = hostname, target_model_dim = large_dim).to(torch.bfloat16).to(torch_device) 
    # I found that the weights need to be loaded again once the large model is loaded 
    small_model.eval() 

if args.large_model == "openllama3b": 
    large_model = LlamaWeirdLarge2.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models, sliding_window_length = 7, addonsmallmodel = small_model, use_mse_loss = args.use_mse_loss, ce_loss_only = args.ce_loss_only).to(torch.bfloat16).to(torch_device) 
elif args.large_model == "tinyllama": 
    # large_model = LlamaWeirdLarge2.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models, sliding_window_length = 7, addonsmallmodel = small_model, use_mse_loss = args.use_mse_loss).to(torch.bfloat16).to(torch_device) 
    if args.finetuned_large_model_checkpoint is not None: 
        print(colored("Using the found checkpoint {}".format(args.finetuned_large_model_checkpoint), "yellow")) 
        large_model = LlamaWeirdLarge3.from_pretrained(args.finetuned_large_model_checkpoint).to(torch.bfloat16).to(torch_device) 
    else: 
        if args.use_old_checkpoint: 
            # print(colored("Using an earlier checkpoint", "yellow")) 
            print(colored("Using the very beginning checkpoint", "yellow")) 
            large_model = LlamaWeirdLarge3.from_pretrained("TinyLlama/TinyLlama-1.1B-step-50K-105b", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
            # NOTE this line now loads in both the large and the small model weights into the model 
        elif args.autoregressive_baseline or args.group_compress: 
            if args.autoregressive_baseline: 
                print(colored("autoregressive baseline", "green")) 
                large_model = LlamaWeirdLarge.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models).to(torch_device) 
                large_model.set_hidden_states_compression_scheme("autoregressive_baseline") 
            elif args.group_compress: 
                print(colored("group compress", "green")) 
                large_model = LlamaWeirdLargeTest.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
                # large_model.set_hidden_states_compression_scheme("group_compress") 
            else: 
                raise ValueError("no compression scheme specified") 
        else: 
            large_model = LlamaWeirdLarge3.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
    large_model.set_msece_loss(args.use_mse_loss, args.ce_loss_only) 
    if args.use_new_small_model_checkpoint: 
        print(colored("using new small model checkpoint", "yellow")) 
        # large_model.set_addonsmallmodel(small_model) 
        small_state_dict_for_model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).state_dict() 
        large_model.set_addonsmallmodel_statedict(small_state_dict_for_model) 
    if args.finetuned_small_model_checkpoint is not None: 
        if "setting0" in args.finetuned_small_model_checkpoint: 
            large_model.set_inference_setting("setting0") 
        elif "setting3" in args.finetuned_small_model_checkpoint: 
            large_model.set_inference_setting("setting3") 
        else: 
            raise ValueError("settingnumber has be in the finetuned_large_model_checkpoint") 
    elif args.experiment_setting is not None: 
        large_model.set_inference_setting(args.experiment_setting) 
    large_model.set_walpha(args.alpha) 
    # large_model.set_slidingwindowlength(args.kernel_size, addonmodel_start = 64) 
    large_model.set_slidingwindowlength(args.kernel_size) 
    large_model.set_cosinesimilarity(args.cosine_similarity) 
    if args.embedding_reinitialization_type is not None: 
        print(colored(args.embedding_reinitialization_type, "red")) 
        large_model.reinitialize_embeddings(type = args.embedding_reinitialization_type) 
    large_model.addonsmallmodel.set_criticalpath(hostname = hostname) 
# large_model = LlamaWeirdLarge.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models, sliding_window_length = 7, addonsmallmodel = small_model, use_mse_loss = True).to(torch.bfloat16).to(torch_device) 
# large_model.set_smallmodelfull() # this function has proven to be very important 
# large_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
# large_model = LlamaForCausalLM.from_pretrained("princeton-nlp/Sheared-LLaMA-2.7B", cache_dir = dir_models) 
large_model.train() 
# large_model.set_addonsmallmodel(small_model) 
if args.use_pretrained_small_model: 
    print(colored("using pretrained small model: {}".format(args.finetuned_small_model_checkpoint), "green")) 
    small_model_state_dict = SimpleSmallModel.from_pretrained(args.finetuned_small_model_checkpoint, sliding_window_length = args.kernel_size, hostname = hostname, target_model_dim = large_dim).state_dict() 
    '''
    new_state_dict = {} 

    for key in small_model_state_dict.keys(): 
        new_key = "addonsmallmodel." + key 
        print(new_key) 
        new_state_dict[new_key] = small_model_state_dict[key] 
    ''' 
    large_model.addonsmallmodel.load_state_dict(small_model_state_dict) 
    large_model.addonsmallmodel.eval() 

large_model.config.pad_token_id = large_tokenizer.pad_token_id 
if args.use_new_small_model_checkpoint: 
    small_model.config.pad_token_id = small_tokenizer.pad_token_id 
else: 
    large_model.addonsmallmodel.config.pad_token_id = small_tokenizer.pad_token_id 

large_model.model.train() 
large_model.addonsmallmodel.train() 

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

param_group = [] 
param_group2 = [] 
for name, param in large_model.named_parameters(): 
    if "addonsmallmodel." in name: 
        param.requires_grad = False 
    else: 
        if not args.freeze_large_model: 
            print(colored("large model parameters {}".format(name), "blue")) 
            if args.embedding_reinitialization_type is None: 
                param.requires_grad = True 
                param_group.append(param) 
            else: 
                param.requires_grad = True 
                if "embed_tokens" in name: 
                    param_group2.append(param) 
                else: 
                    param_group.append(param) 
        else: 
            param.requires_grad = False 

small_model = large_model.addonsmallmodel 

for name, param in small_model.named_parameters(): 
    # print(colored("small model parameters {}".format(name), "yellow")) 
    # if args.use_pretrained_small_model: 
    if args.freeze_small_model or args.use_mse_loss: 
        param.requires_grad = False 
        print(colored("freezing small model parameters {}".format(name), "cyan")) 
    else: 
        print(colored("small model parameters {}".format(name), "blue")) 
        param.requires_grad = True 
        if "embed_projection" in name: 
            param_group2.append(param) 
        else: 
            param_group.append(param) 
print("length of param_group {}".format(len(param_group))) 
print("length of param_group2 {}".format(len(param_group2))) 
if args.embedding_reinitialization_type is not None: 
    print("length of param_group2 {}".format(len(param_group2))) 

# if args.embedding_reinitialization_type is None: 
#     custom_optimizer = torch.optim.AdamW(param_group, lr = args.lr) 
#     # custom_optimizer = torch.optim.AdamW(param_group, lr = 2e-5) 
#     # custom_optimizer = torch.optim.AdamW(param_group, lr = 2e-4) 
# else: 
#     custom_optimizer = torch.optim.AdamW([
#         {"params": param_group, "lr": args.lr}, 
#         {"params": param_group2, "lr": args.lr * 10}
#     ]) 
    
custom_optimizer = torch.optim.AdamW([
    {"params": param_group, "lr": args.lr}, 
    {"params": param_group2, "lr": args.lr * 10}
]) 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

# model_path = "/home/bc20/yang" 
# model_path = "/home/yangzho6/model_checkpoints" 
model_path = dir_models + "largemodel{}_{}_{}/".format(args.large_model, commit_hash, hash_of_time) 
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    # evaluation_strategy="steps",    # evaluate each `logging_steps` steps 
    overwrite_output_dir=True,      
    num_train_epochs=args.num_epochs,            # number of training epochs, feel free to tweak
    per_device_train_batch_size = args.batch_size if not args.debug else 10, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=args.gradient_accumulation_steps,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size= args.batch_size if not args.debug else 10,  # evaluation batch size
    # logging_steps=1, 
    logging_steps = 100 if not args.debug else 1,       # evaluate, log and save model checkpoints every 1000 step
    # save_steps=1000, 
    # save_steps = 2000, 
    save_steps = 100 if not args.debug else 1, 
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
    warmup_steps = 100, 
    label_names = ["labels"], 
    remove_unused_columns = True, 
    save_strategy = "steps", 
    evaluation_strategy = "steps", 
) 
print(colored("resume_from_checkpoint is {}".format(args.resume_from_checkpoint), "red")) 

trainer = CustomTrainer(
    model = large_model, 
    args = training_args, 
    train_dataset = train_dataset, 
    # train_dataset = train_set, 
    eval_dataset = test_dataset, 
    # eval_dataset = test_set, 
    data_collator = data_collator, 
    optimizers = (custom_optimizer, None), 
    tokenizer = tokenizer, 
    time_hash = hash_of_time, 
    commit_hash = commit_hash, 
    text_eval = model_path + text_eval, 
    n = args.kernel_size, 
) 

if trainer.accelerator.is_main_process and has_wandb: 
    today = datetime.date.today() 
    wandblogconfigs = training_args.to_dict() 
    wandblogconfigs["git_commit"] = commit_hash 
    wandblogconfigs["time_hash"] = hash_of_time 
    wandb.init(project = "chunkedlargefinetuning", config = wandblogconfigs, name = "large_small_ce{}_{}".format(today, "unmasked")) 

torch.autograd.set_detect_anomaly(True) 

if args.resume_from_checkpoint is not None: 
    trainer.train(resume_from_checkpoint = args.resume_from_checkpoint) 
else: 
    trainer.train() 

wandb.finish() 
