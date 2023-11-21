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

# cache_dir = "/home/bc20/yang/" 
dir_dataset = "/home/yangzho6/c4_parts" 
dir_models = "/home/yangzho6/model_checkpoints2" 
dir_sdata = "/home/yangzho6/c4llm_synthesized/" 

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

logger = logging.get_logger(__name__) 

class CustomTrainer(Trainer): 
    def __init__(self, experiment_setting = "setting0", *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        # self.large_model = large_model 
        # self.generation_config = GenerationConfig(return_dict_in_generate = True) 
        # self.time_checkpoint = time.time() 
        self.time_checkpoint = 0 
        self.iteration_count = 0 
        self.experiment_setting = experiment_setting 
    
    def training_step(self, model, inputs): 
        model.train() 
        self.iteration_count += 1 
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
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

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

    def compute_loss(self, model, inputs, return_outputs = False): 
        torch.cuda.synchronize() 
        print(colored("time elasped in the last iteration is {}".format(time.time() - self.time_checkpoint)), "red") 
        self.time_checkpoint = time.time() 
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
        labels = inputs["labels"] 
        # print("the input ids are {}".format(input_ids[0])) 
        # print("labels are {}".format(labels[0])) 
        if not isinstance(model, SimpleSmallModel): 
            outputs = model(
                input_ids = input_ids, 
                attention_mask = attention_mask, 
                labels = labels, 
                # condensed_embeds = condensed_embeds, 
                output_hidden_states = True, 
                output_attentions = True, 
                return_dict = True, 
                # eval_mode = True, 
            ) 
        else: 
            condensed_embeds = inputs["condensed_embeds"] 
            batch_size, seq_len = attention_mask.shape 
            addedon_length = condensed_embeds.shape[1] 
            # print("get the input sentence: {}".format(tokenizer.decode(input_ids[0]))) 
            attention_mask = torch.cat((attention_mask, torch.ones((batch_size, addedon_length), dtype = torch.long).to(input_ids.device)), dim = 1) 
            
            # print("condensed_embeds dtype is {}".format(condensed_embeds.dtype)) 
            # print("condensed_embeds is {}".format(condensed_embeds)) 
            # print("input_ids are {}".format(input_ids)) 
            # outputs = model(input_ids = large_outputs.sequences, attention_mask = attention_mask, labels = large_outputs.sequences, condensed_embeds = downsampled_vectors) 
            outputs = model(
                input_ids = input_ids, 
                attention_mask = attention_mask, 
                labels = labels, 
                condensed_embeds = condensed_embeds, 
                output_hidden_states = True, 
                output_attentions = True, 
                return_dict = True, 
                # condensed_fashion = "ground_truth", 
                iteration_count = self.iteration_count, 
                # eval_mode = True, 
                experiment_setting = self.experiment_setting, 
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
        
        # print(outputs.hidden_states[0].shape) 
        # print(outputs.hidden_states[0][0][0][: 10]) 
        # print(len(outputs.hidden_states)) 
        # print(outputs.attentions[0][0]) 
        
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            ) 
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
        print(colored("the loss is {}".format(loss), "yellow")) 
        if has_wandb: 
            wandb.log({"loss": loss, 
                       "group1.lr": self.optimizer.param_groups[0]["lr"], 
                       "group2.lr": self.optimizer.param_groups[1]["lr"], 
                       "iteration_count": self.iteration_count * 50 
            }) 
        if self.iteration_count % 300 == 0: 
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
                    SimpleSmallModel.plot_attention_map(outputs.attentions, layer, head, 144, "testing_attention_map.jpg") 
                    # print(outputs.attentions[0][0][0][64]) 
                    field_name = "layer{}_head{}".format(layer, head) 
                    wandb.log({field_name: wandb.Image("testing_attention_map.jpg")}) 

        # inspect the hidden states here 

        return (loss, outputs) if return_outputs else loss 

    def local_compute_metrics(
            self, 
            logits, 
            labels, 
            loss, 
            input_attention_mask, 
    ): 
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        logits = logits[:, :-1, :] 
        input_attention_mask = input_attention_mask[:, :-1] 
        labels = labels[:, 1:] 
        preds = torch.argmax(logits, dim = -1) 
        print("the shape of preds is {}".format(preds.shape)) 
        # use loss to compute perplexity 
        perplexity = torch.exp(loss).mean().item() 
        print("the perplexity is {}".format(perplexity)) 
        # use preds to compute accuracy 
        indices_to_keep = input_attention_mask == 1 
        total_valid_tokens = torch.sum(indices_to_keep.view(-1), dim = 0).item() 
        # accuracy = accuracy_score(labels[indices_to_keep], preds[indices_to_keep]) 
        correct_words = torch.sum((preds[indices_to_keep] == labels[indices_to_keep]).view(-1), dim = 0).item() 
        print("correct words: {} and total words: {}".format(correct_words, total_valid_tokens)) 
        # use preds to compute f1 score 
        # f1 = precision_recall_fscore_support(labels, preds, average = "weighted") 
        return {"perplexity": perplexity, "correct_words": correct_words, "total_words": total_valid_tokens} 

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
        '''
        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")
        ''' 
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

        observed_num_examples = 0
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
            print(ignore_keys) 
            print(colored("the loss is {}".format(loss), "yellow")) 
            # print(colored("the shape of logits is {} {}".format(logits.shape, "yellow"))) 
            # print(colored("the shape of logits if {} {}".format(len(logits), logits[0].shape), "yellow")) 
            print(colored("the shape of logits is {}".format(logits.shape), "yellow")) 
            print(colored("the shape of labels is {}".format(labels.shape), "yellow")) 
            total_loss += loss.item() 
            local_metrics = self.local_compute_metrics(logits, labels, loss, inputs["attention_mask"]) 
            total_correct_words = local_metrics["correct_words"] 
            total_words = local_metrics["total_words"] 
            sum_of_perplexity += local_metrics["perplexity"] 

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
        
        global_perplexity = np.exp(total_loss / num_samples) 
        global_accuracy = total_correct_words / total_words 
        metrics = {"perplexity": global_perplexity, "accuracy": global_accuracy} 
        wandb.log({"global_eval_perplexity": global_perplexity, "global_eval_accuracy": global_accuracy}) 

        '''
        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}
        ''' 
        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples) 

class CustomDataset: 
    def __init__(self, data_dir, tokenizer = None, max_length = 128): 
        # self.synthesize_dir = "/home/yangzho6/c4llm_synthesized/" 
        self.synthesize_dir = data_dir 
        self.dataset = load_dataset('json', data_files = self.synthesize_dir + "c4synthesized_file1.json") 
        # self.dataset = load_dataset('json', data_files = self.synthesize_dir + "c4synthesized_file1copy.json") 
        self.dataset = self.dataset["train"] 

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
                max_length = 128, 
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

# defining tokenizer 
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir) 
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
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
datasetnew = CustomDataset(data_dir = dir_sdata, tokenizer = tokenizer) 
train_set, test_set = datasetnew.split(0.99) 

# handling simplesmallmodel 
# small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = cache_dir).to(torch_device) 
# small_config = LlamaConfig.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
small_config = LlamaConfig.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models) 

small_state_dict_for_model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).state_dict() 
small_model = SimpleSmallModel(small_config) 

new_state_dict = {} 

for key in small_state_dict_for_model.keys(): 
    new_key = key 
    if 'lm_head' in key: 
        print("got here found the following key {}".format(key)) 
    if 'model.' in key: 
        new_key = key[6 :] 
    print(new_key) 
    new_state_dict[new_key] = small_state_dict_for_model[key] 

try: 
    small_model.load_state_dict(new_state_dict) 
except RuntimeError as r: 
    print(colored(r, "yellow")) 

pretraining_weights_group = []
newly_initialized_group = [] 
for k, v in small_model.named_parameters(): 
    if "embed_projection" in k: 
        print(k) 
        newly_initialized_group.append(v) 
    else: 
        pretraining_weights_group.append(v) 
print(len(pretraining_weights_group), len(newly_initialized_group)) 

custom_optimizer = torch.optim.AdamW([
    {"params": pretraining_weights_group, "lr": 2e-4}, 
    {"params": newly_initialized_group, "lr": 2e-3}, 
]) 

def _lr_scheduler_rewriting(current_step, *, num_warmup_steps: int, num_training_steps: int): 
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

# custom_lr_scheduler = torch.optim.lr_scheduler.LambdaLR 

small_model = small_model.to(torch_device) 
small_model.train() 

# alternative pretrained model 
# small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m").to(torch_device) 
# config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
# print(config) 
# small_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", cache_dir = dir_models).to(torch_device) 
# small_model = AutoModelForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models).to(torch_device) 
# small_model.train() 

# for llama model we need to add the padding token 
small_model.config.pad_token_id = tokenizer.pad_token_id 
# print(small_model.embed_projection.weight.dtype) 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

# model_path = "/home/bc20/yang" 
model_path = "/home/yangzho6/model_checkpoints" 
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=5,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=128, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=4,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=256,  # evaluation batch size
    logging_steps=1,             # evaluate, log and save model checkpoints every 1000 step
    # save_steps=1000, 
    # save_steps = 2000, 
    save_steps = 1, 
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
    eval_accumulation_steps = 2, 
) 


max_length = 128 
if has_wandb: 
    wandb.init(project = "llm160m", config = training_args, name="sequencelength{}kernelsize{}learning_rate{}".format(max_length, 4, training_args.learning_rate)) 

weightmodelfirst = next(small_model.parameters()) 
# print(weightmodelfirst.dtype) 
print(colored(weightmodelfirst.dtype, "red")) 

def compute_metrics(p): 
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits = p.predictions 
    labels = p.label_ids 
    probs = torch.softmax(torch.tensor(logits), dim = -1) 
    loss = nn.CrossEntropyLoss()(torch.tensor(logits), torch.tensor(labels)).item() 
    perplexity = torch.exp(torch.tensor(loss)).item() 

    pred = torch.argmax(probs, dim = -1) 

    wandb.log({"evaluation_acc": accuracy_score(p.labels_ids, pred), 
                "evaluation_f1": precision_recall_fscore_support(p.label_ids, pred, average = 'weighted'), 
                "evaluation_perplexity": perplexity, 
    }) 

    return {
        'accuracy': accuracy_score(p.labels_ids, pred), 
        'f1': precision_recall_fscore_support(p.label_ids, pred, average = 'weighted'), 
        'perplexity': perplexity,
    } 

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
    experiment_setting = "setting2", 
) 



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
