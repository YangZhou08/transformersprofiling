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
from src.transformers.models.llama.modeling_llama import LlamaWeirdLarge 
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

parser = argparse.ArgumentParser() 
parser.add_argument("--use_pretrained_small_model", action = "store_true") 

args = parser.parse_args() 

class CustomTrainer(Trainer): 
    def __init__(self, n = 7, tokenizer = None, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.n = n 
        self.tokenizer = tokenizer 
        # self.start_idx = start_idx 
        self.iteration_count = 0 
    
    def _set_signature_columns_if_needed(self): 
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names)) 
        self._signature_columns += ["attention_mask_chunk"] 
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None 
        
        print(colored("iteration_count {}".format(self.iteration_count), "yellow")) 
        
        input_ids = inputs["input_ids"] 
        attention_mask = inputs["attention_mask_chunk"] 
        original_attention_mask = inputs["attention_mask"] 
        label2 = inputs["labels"] 
        
        batch_size, seq_len = original_attention_mask.shape 
        addedon_length = seq_len // self.n 
        original_attention_mask = torch.cat((original_attention_mask, torch.ones((batch_size, addedon_length), dtype = torch.long).to(input_ids.device)), dim = 1) 
        
        outputs = model(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            output_hidden_states = True, 
            output_attentions = True, 
            return_dict = True, 
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
            l2_distance = outputs["l2_distance"] if isinstance(outputs, dict) else outputs[-1] 
        
        print(colored("rank {} loss {}".format(self.accelerator.state.process_index, loss), "yellow")) 
        print(colored("rank {} l2_distance {}".format(self.accelerator.state.process_index, l2_distance), "yellow")) 
        if self.accelerator.is_main_process and self.iteration_count % 1000 == 0 and has_wandb: 
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
        # print("length of logits {}".format(len(logits))) 
        # print("logits[0].shape {}".format(logits[0].shape)) 
        # print("logits[1].shape {}".format(logits[1].shape))
        assert len(logits) == 2 
        l2dist = logits[1].reshape(-1) 
        logits = logits[0] 
        # print(l2dist) 
        logits = logits[:, :-1, :] 
        # input_attention_mask = input_attention_mask[:, :-1] 
        input_attention_mask = input_attention_mask[:, 1:] 
        labels = labels[:, 1:] 
        preds = torch.argmax(logits, dim = -1) 
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
                print(colored(labels_output, "cyan")) 
                print() 
                print() 
        
        if self.accelerator.state.num_processes > 1: 
            self.accelerator.wait_for_everyone() 
            
        perplexity = torch.exp(loss).mean().item() 
        indices_to_keep = input_attention_mask == 1 # not sure whether we need this 
        total_valid_tokens = torch.sum(indices_to_keep.view(-1), dim = 0).item() 
        correct_words = torch.sum((preds[indices_to_keep] == labels[indices_to_keep]).view(-1), dim = 0).item() 
        print("correct words: {} and total words: {}".format(correct_words, total_valid_tokens)) 
        return {"perplexity": perplexity, "correct_words": correct_words, "total_words": total_valid_tokens, "l2_distance": l2dist.item()} 
                
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
            if self.model.use_mse_loss != True: 
                local_metrics = self.local_compute_metrics(logits, labels, loss, inputs["attention_mask"], step) 
                total_correct_words += local_metrics["correct_words"] 
                total_words += local_metrics["total_words"] 
                sum_of_perplexity += local_metrics["perplexity"] 
                l2_distance += local_metrics["l2_distance"] 

            if is_torch_tpu_available(): 
                xm.mark_step() 
            
        if self.accelerator.is_main_process: 
            print("rank {} total_loss before aggregation is {}".format(self.accelerator.state.process_index, total_loss)) 
        aggregated_loss = self.gather_function(torch.tensor(total_loss).reshape(1, -1).to(local_device)) 
        if self.accelerator.is_main_process: 
            print("rank {} total_loss after aggregation is {}".format(self.accelerator.state.process_index, aggregated_loss)) 
        total_loss = self.gather_function(torch.tensor(total_loss).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).div(self.accelerator.state.num_processes).item() 
        if self.model.use_mse_loss != True: 
            total_correct_words = self.gather_function(torch.tensor(total_correct_words).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).item() 
            total_words = self.gather_function(torch.tensor(total_words).reshape(-1, 1).to(local_device)).view(-1).sum(dim = -1).item() 
            sum_of_perplexity = self.gather_function(torch.tensor(sum_of_perplexity).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).item() 
            l2_distance = self.gather_function(torch.tensor(l2_distance).reshape(1, -1).to(local_device)).view(-1).sum(dim = -1).div(self.accelerator.state.num_processes).item() 
        
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
        
        if not self.model.use_mse_loss: 
            global_perplexity = np.exp(total_loss / total_num_steps) 
            global_accuracy = total_correct_words / total_words 
            all_losses = total_loss / total_num_steps 
            l2_distance = l2_distance / total_num_steps 

            metrics = {"perplexity": global_perplexity, "accuracy": global_accuracy, "l2_distance": l2_distance} 
            if self.accelerator.is_main_process: 
                print(colored(metrics, "magenta")) 
                wandb.log({"global_eval_perplexity": global_perplexity, "global_eval_accuracy": global_accuracy, "l2_distance": l2_distance}) 
        else: 
            wandb.log({"global_eval_loss": total_loss / total_num_steps}) 
        
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

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
# tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models) 
# tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
tokenizer.padding_side = "left" 

list_of_datasets = ["c4_file{}.json".format(i) for i in range(1, 6)] 
list_of_datasets = [dir_unprocessed_dataset + path for path in list_of_datasets] 
onedataset = load_dataset("json", data_files = list_of_datasets, split = "train[:1000]") 
d = onedataset.train_test_split(test_size = 0.001) # 0.995 for training, 0.005 for testing 

def encode_with_truncation(examples): 
    # return tokenizer(examples["text"], truncation = True, padding = "max_length", 
                    #  max_length = max_length, return_special_tokens_mask = True) 
    return tokenizer(examples["text"], padding = "max_length", max_length = 259, 
                     return_attention_mask = True, return_tensors = "pt", truncation = True) 

train_dataset = d["train"].map(encode_with_truncation, batched = True, num_proc = 8) 
test_dataset = d["test"].map(encode_with_truncation, batched = True, num_proc = 8) 

# TODO change the following code to use the checkpoint of the best trained window 7 model 
small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models) 

small_state_dict_for_model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).state_dict() 
small_model = SimpleSmallModel(small_config, hostname = hostname, sliding_window_length = 7, target_model_dim = 3200) 

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
small_model.eval() # at start we avoid training the small model 

# large_model = LlamaWeirdLarge.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models, sliding_window_length = 7, addonsmallmodel = small_model, use_mse_loss = False).to(torch.bfloat16).to(torch_device) 
large_model = LlamaWeirdLarge.from_pretrained("openlm-research/open_llama_3b_v2", cache_dir = dir_models, sliding_window_length = 7, addonsmallmodel = small_model, use_mse_loss = False).to(torch.bfloat16).to(torch_device) 
# large_model.set_smallmodelfull() # this function has proven to be very important 
# large_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
# large_model = LlamaForCausalLM.from_pretrained("princeton-nlp/Sheared-LLaMA-2.7B", cache_dir = dir_models) 
large_model.train() 
# large_model.set_addonsmallmodel(small_model) 

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

train_dataset = train_dataset.map(group_attention_map_chunked_generation, batched = True, num_proc = 8) 
test_dataset = test_dataset.map(group_attention_map_chunked_generation, batched = True, num_proc = 8) 

for i in range(10): 
    example = train_dataset[i] 
    input_ids = example["input_ids"] 
    for j in range(len(input_ids)): 
        if j != 0 and j % 7 == 0: 
            end = " | " 
        else: 
            end = " " 
        print(input_ids[j], end = end) 
    print() 
    print("attention_mask_chunk {}".format(example["attention_mask_chunk"])) 

# large_model = large_model.to(torch_device) 

train_dataset.set_format(type = "torch", columns = ["attention_mask_chunk", "input_ids", "attention_mask"]) 
test_dataset.set_format(type = "torch", columns = ["attention_mask_chunk", "input_ids", "attention_mask"]) 

for i in range(10): 
    example = train_dataset[i] 
    input_ids = example["input_ids"] 
    for j in range(input_ids.shape[0]): 
        if j != 0 and j % 7 == 0: 
            end = " | " 
        else: 
            end = " " 
        print(input_ids[j].item(), end = end) 
    print() 
    print("attention_mask_chunk {}".format(example["attention_mask_chunk"])) 

param_group = [] 
for name, param in large_model.named_parameters(): 
    if "addonsmallmodel." in name: 
        param.requires_grad = False 
    else: 
        print(name) 
        param.requires_grad = True 
        param_group.append(param) 
print("length of param_group {}".format(len(param_group))) 
for name, param in small_model.named_parameters(): 
    # print(colored("small model parameters {}".format(name), "yellow")) 
    if args.use_pretrained_small_model: 
        param.requires_grad = False 
    else: 
        param.requires_grad = True 
        param_group.append(param) 

custom_optimizer = torch.optim.AdamW(param_group, lr = 5e-5) 
# custom_optimizer = torch.optim.AdamW(param_group, lr = 1e-4) 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

# model_path = "/home/bc20/yang" 
# model_path = "/home/yangzho6/model_checkpoints" 
model_path = dir_models + "smallmodelopenllamav3" 
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    # evaluation_strategy="steps",    # evaluate each `logging_steps` steps 
    overwrite_output_dir=True,      
    num_train_epochs=5,            # number of training epochs, feel free to tweak
    per_device_train_batch_size = 10, # the training batch size, put it as high as your GPU memory fits
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

trainer = CustomTrainer(
    model = large_model, 
    args = training_args, 
    train_dataset = train_dataset, 
    eval_dataset = test_dataset, 
    data_collator = data_collator, 
    optimizers = (custom_optimizer, None), 
    tokenizer = tokenizer, 
) 

if trainer.accelerator.is_main_process and has_wandb: 
    today = datetime.date.today() 
    wandblogconfigs = training_args.to_dict() 
    wandblogconfigs["git_commit"] = commit_hash 
    wandblogconfigs["time_hash"] = hash_of_time 
    wandb.init(project = "chunkedlargefinetuning", config = wandblogconfigs, name = "large_small_ce{}_{}".format(today, "unmasked")) 

torch.autograd.set_detect_anomaly(True) 

trainer.train() 

wandb.finish() 
