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

def log_dict_converterc(filename, preproc, tokenizer): 
    import ast 

    with open(filename, "r") as f: 
        data = f.read() 

        words = ast.literal_eval(data) 

        data = {tuple(pairs): count for pairs, count in words} 
        if not preproc: 
            return data 
        else: 
            # first take all the keys out 
            keys = list(data.keys()) 

            # then we tokenize them 
            assert tokenizer is not None 
            output_keys = [] 
            for idx, key in enumerate(keys): 
                local_tensor = [] 
                for seg in key: 
                    if seg == "<0x0A>": 
                        seg = "\n" 
                    output_tokenized_keys = tokenizer(seg, add_special_tokens = False, return_attention_mask = False, return_tensors = "pt") 
                    # local_tensor.append(output_tokenized_keys["input_ids"].squeeze(0)) 
                    tensorofinterest = output_tokenized_keys["input_ids"].squeeze(0) 
                    # if local_tensor.shape[0] == 1: 
                    if tensorofinterest.shape[0] != 1: 
                        # assert local_tensor.shape[0] == 2 
                        assert tensorofinterest.shape[0] == 2 
                        if tensorofinterest[0] == 29871: 
                            # print(seg, tensorofinterest) 
                            tensorofinterest = tensorofinterest[1:] 
                    local_tensor.append(tensorofinterest) 
                tokencat = torch.cat(local_tensor, dim = 0) 
                if tokencat.shape[0] != 3: 
                    for i in range(tokencat.shape[0] - 2): 
                        cat1 = tokencat[i : i + 3] 
                        output_keys.append(cat1) 
                else: 
                    output_keys.append(tokencat) 
                '''
                print(local_tensor) 
                for seg in local_tensor: 
                    for i in range(seg.shape[0]): 
                        print(tokenizer.decode(seg[i])) 
                ''' 
                # output_keys.append(output_tokenized_keys["input_ids"].squeeze(1)) 
            output_keys = torch.stack(output_keys, dim = 0) 
            return output_keys 

class CustomTrainer(Trainer): 
    def __init__(self, common_n_gram_list, use_filtered_hot_labels = True, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.common_n_gram_list = common_n_gram_list 
        self.use_filtered_hot_labels = use_filtered_hot_labels 
        self.training_mode = True 
    
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
            original_model_output = True, 
        ) 
        
        print("outputs have shape {}".format(len(outputs))) 
        print(colored("model running loss: {}".format(outputs[0].item()), "yellow")) 
        if has_wandb: 
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
    
    def local_compute_metrics(
        self, 
        logits, 
        labels, 
        loss, 
        input_attention_mask, 
        outside_step, 
    ): 
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        print(colored("printing out the type of logits {}".format(type(logits)), "red")) 
        logits = logits[:, :-1, :] 
        # input_attention_mask = input_attention_mask[:, :-1] 
        input_attention_mask = input_attention_mask[:, 1:] 
        labels = labels[:, 1:] 
        preds = torch.argmax(logits, dim = -1) 
        if outside_step == 0: 
            print("*** evaluating at step {} ***".format(self.iteration_count)) 
            # visualize the actual output prediction during the evaluation 
            if self.use_filtered_hot_labels: 
                pass 
            else: 
                pass 
        
        indices_to_keep = input_attention_mask == 1 
        total_valid_tokens = torch.sum(indices_to_keep.view(-1), dim = 0).item() 
        
        # computing the average acceptance length 
        
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
        sum_of_perplexity = 0 # used to compute the average perplexity 
        total_loss = 0 # used to compute the correct perplexity 
        interest_total_words = 0 
        interest_correct_words = 0 

        observed_num_examples = 0 
        total_num_steps = len(dataloader) 
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

        metrics = {"accuracy": global_accuracy} 
        print(colored(metrics, "magenta")) 
        wandb.log({"global_eval_accuracy": global_accuracy}) 

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
    return tokenizer(examples["text"], padding = "max_length", max_length = 256, 
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

param_group = [] 
module_projection_name = "output_n_projection.weight" 
model = LlamaCausalLMWeirdTwo.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
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

custom_optimizer = torch.optim.AdamW(param_group, lr = 8e-5) 


# for llama model we need to add the padding token 
model.config.pad_token_id = tokenizer.pad_token_id 
# print(small_model.embed_projection.weight.dtype) 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

# model_path = "/home/bc20/yang" 
# model_path = "/home/yangzho6/model_checkpoints" 
model_path = dir_models 
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=5,            # number of training epochs, feel free to tweak
    per_device_train_batch_size = 40, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=4,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=40,  # evaluation batch size
    # logging_steps=1, 
    logging_steps = 2,            # evaluate, log and save model checkpoints every 1000 step
    # save_steps=1000, 
    # save_steps = 2000, 
    save_steps = 2, 
    # learning_rate=5e-7, 
    # learning_rate=5e-5, 
    # learning_rate=2e-4, 
    learning_rate = 1e-3, 
    # learning_rate = 1e-4, 
    # learning_rate = 5e-6, 
    # learning_rate = 0, 
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=5,            # whether you don't have much space so you let only 3 model weights saved in the disk 
    # lr_scheduler_type = "cosine", 
    warmup_steps = 25, 
    label_names = ["labels"], 
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

hot_1000_3_grams = log_dict_converterc("partial_c4_hot1000.txt", preproc = True, tokenizer = tokenizer) 

trainer = CustomTrainer(
    model = model, 
    args = training_args, 
    train_dataset = train_dataset, 
    eval_dataset = test_dataset, 
    data_collator = data_collator, 
    optimizers = (custom_optimizer, None), 
    common_n_gram_list = hot_1000_3_grams, 
    use_filtered_hot_labels = False, 
) 


torch.autograd.set_detect_anomaly(True) 

trainer.train() 

wandb.finish() 

