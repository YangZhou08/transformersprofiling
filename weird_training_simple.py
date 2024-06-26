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
import numpy as np 

from termcolor import colored 
from src.transformers import Trainer, TrainingArguments 
from torch import nn 
from src.transformers import DataCollatorForLanguageModeling 
from src.transformers.generation.utils import GenerationConfig 
from src.transformers.models.llama.modeling_llama import LlamaForCausalLM, SimpleSmallModel 
from src.transformers.models.llama.modeling_llama import LlamaForCausalLMWeird 
import time 

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False 
# has_wandb = False 

from torch.optim import AdamW 
import torch.optim as optim 

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

if is_apex_available():
    from apex import amp 

class CustomTrainer(Trainer): 
    def __init__(self, large_model, togetherForming, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.large_model = large_model 
        self.generation_config = GenerationConfig(return_dict_in_generate = True) 
        # self.time_checkpoint = time.time() 
        self.time_checkpoint = 0 
        if "tokenizer" in kwargs: 
            self.tokenizer = kwargs["tokenizer"] 
        self.iteration_count = 0 
        self.togetherForming = togetherForming 
    
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
    
    def training_step(self, model, inputs): 
        model.train() 
        self.iteration_count += 1 
        inputs = self._prepare_inputs(inputs) 
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
        
        for name, parameters in model.named_parameters(): 
            if name == "lm_head_different.weight": 
                print(colored("{} has gradient {}".format(name, parameters.grad.data.view(-1)[: 10]), "light_magenta")) 
                print(colored("{} has weights {}".format(name, parameters.data.view(-1)[: 10]), "light_magenta")) 
            elif name == "embed_projection.weight": 
                print(colored("{} has gradient {}".format(name, parameters.grad.data.view(-1)[: 10]), "light_magenta")) 
                print(colored("{} has weights {}".format(name, parameters.data.view(-1)[: 10]), "light_magenta")) 
        
        return loss.detach() / self.args.gradient_accumulation_steps 

    def compute_loss(self, model, inputs, return_outputs = False): 
        torch.cuda.synchronize() 
        self.iteration_count += 1 
        print(colored("time elasped in the last iteration is {}".format(time.time() - self.time_checkpoint), "red")) 
        self.time_checkpoint = time.time() 
        labels = None 
        for k, v in inputs.items(): 
            if k == "input_ids": 
                print("the first batch contains elements: ", end = "") 
                print(colored(self.tokenizer.decode(v[0]), "yellow")) 
                print("the first batch contains element {}".format(v[0])) 
            
            if isinstance(v, tuple): 
                print(k, len(v)) 
            elif isinstance(v, torch.Tensor): 
                print(k, v.shape) 
            else: 
                print(k, v) 
        
        print("attention_mask: {}".format(inputs["attention_mask"])) 
        with torch.no_grad(): 
            input_ids = inputs["input_ids"] 
            attention_mask = inputs["attention_mask"] 
            labels = inputs["labels"] 

            # large_outputs = self.large_model.generate(input_ids = input_ids, max_length = 128) 
            large_outputs = self.large_model.generate(input_ids = input_ids, attention_mask = attention_mask, do_sample = False, output_hidden_states = True, return_dict_in_generate = True, max_length = 128) 
            print("the large model output sequence is: ", end = "") 
            print(colored(self.tokenizer.decode(large_outputs.sequences[0]), "green")) 
            hidden_states_of_interest = large_outputs.hidden_states[-1][-1][:, -1, :] 
            '''
            large_model_output = self.large_model.lm_head(hidden_states_of_interest) 
            print("we got here : {}".format(torch.argmax(large_model_output, dim = -1))) 
            print("we should have : {}".format(large_outputs.sequences[:, -1])) 
            ''' 
        attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.shape[0], 127 - attention_mask.shape[1]), device = attention_mask.device)), dim = 1) 
        hidden_states_of_interest = hidden_states_of_interest.to(torch.float) 
        outputs = model(input_ids = large_outputs.sequences[:, :-1], attention_mask = attention_mask, added_condensed_token = hidden_states_of_interest, return_dict = True) 
        print("shape of the smll model logits: {}".format(outputs.logits.shape)) 
        print(self.togetherForming) 
        if self.togetherForming == "average": 
            loss = torch.nn.CrossEntropyLoss()(outputs.logits[:, -1, :], large_outputs.sequences[:, -1]) 
        elif self.togetherForming == "concatenation": 
            loss = torch.nn.CrossEntropyLoss()(outputs.logits, large_outputs.sequences[:, -1]) 
        else: 
            raise ValueError("togetherForming must be either average or concatenation") 

        # outputs = model(input_ids = large_outputs.sequences, attention_mask = attention_mask, labels = large_outputs.sequences, condensed_embeds = downsampled_vectors) 
        # outputs = model(input_ids = large_outputs.sequences[:, :-1], attention_mask = attention_mask, added_condensed_token = None) 
        '''
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            ) 
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
        ''' 
        print(colored("the loss is {}".format(loss), "yellow")) 

        if has_wandb: 
            if len(self.optimizer.param_groups) == 1: 
                wandb.log({"loss": loss, 
                           "group1.lr": self.optimizer.param_groups[0]["lr"], 
                           "iteration_count": self.iteration_count * 50 
                }) 
            else: 
                wandb.log({"loss": loss, 
                        "group1.lr": self.optimizer.param_groups[0]["lr"], 
                        "group2.lr": self.optimizer.param_groups[1]["lr"], 
                        "iteration_count": self.iteration_count * 50 
                }) 

        return (loss, outputs) if return_outputs else loss 

class CustomDataset: 
    def __init__(self, data_dir, tokenizer = None, max_length = 128): 
        # self.synthesize_dir = "/home/yangzho6/c4llm_synthesized/" 
        self.synthesize_dir = data_dir 
        self.dataset = load_dataset('json', data_files = self.synthesize_dir + "c4synthesized_file1.json") 

        self.tokenizer = tokenizer 
        self.max_length = max_length 
    
    def __len__(self): 
        return len(self.dataset) 
    
    def __getitem__(self, idx): 
        item = self.dataset[idx] 
        tensor = torch.load(item["condensed_token_path"]) 

        if self.tokenizer is not None: 
            encoded_text = self.tokenizer(
                item['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            item['input_ids'] = encoded_text['input_ids'].squeeze(0)  # remove the batch dimension
            item['attention_mask'] = encoded_text['attention_mask'].squeeze(0)  # remove the batch dimension 
        
        item["condensed_embeds"] = tensor 

        return item 

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help') 

parser.add_argument("--group1lr", type = float, default = 5e-4) 
parser.add_argument("--group2lr", type = float, default = 1) 
parser.add_argument("--togetherforming", type = str, default = "concatenation") 
parser.add_argument("--freeze_pretrained", action = "store_true", default = False) 

args = parser.parse_args() 
print(args) 

dir_dataset = "/home/yangzho6/c4_parts" 
dir_models = "/home/yangzho6/model_checkpoints" 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
onedataset = load_dataset('json', data_files = '/home/yangzho6/c4_parts/downloads/c4_file1.json', split = "train") 

d = onedataset.train_test_split(test_size = 0.1) 
print(d["train"], d["test"]) 

print() 

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

# small_model = LlamaForCausalLMWeird.from_pretrained("JackFram/llama-160m", cache_dir = dir_models, adding_mode = "average" if args.togetherforming == "average" else "concatenate").to(torch_device) 
small_model = LlamaForCausalLMWeird.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models, adding_mode = "average" if args.togetherforming == "average" else "concatenate").to(torch_device) 
small_model.train() 

large_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
configs = large_model.config 

large_model.eval() 

small_model.config.pad_token_id = tokenizer.pad_token_id 
small_model.train() 

max_length = 64 

def encode_with_truncation(examples): 
    return tokenizer(examples["text"], truncation = True, padding = "max_length", 
                     max_length = max_length, return_special_tokens_mask = True) 

train_dataset = d['train'].map(encode_with_truncation, batched = True, num_proc = 4) 
test_dataset = d['test'].map(encode_with_truncation, batched = True, num_proc = 4) 

print("The model max length is {}".format(small_model.config.max_position_embeddings)) 

train_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 
test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

model_path = "/home/yangzho6/model_checkpoints" 
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=10,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=50, # the training batch size, put it as high as your GPU memory fits
    # gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
) 

pretraining_weights_group = []
newly_initialized_group = [] 

for k, v in small_model.named_parameters(): 
    print(k) 
    if k == "lm_head_different.weight" or k == "embed_projection.weight": 
        newly_initialized_group.append(v) 
    else: 
        if args.freeze_pretrained: 
            v.requires_grad = False 
        else: 
            pretraining_weights_group.append(v) 

print(colored("length of pretraining weights group is {}".format(len(pretraining_weights_group)), "red")) 
print(colored("length of newly initialized weights group is {}".format(len(newly_initialized_group)), "red")) 

if args.freeze_pretrained: 
    custom_optimizer = AdamW([
        {"params": newly_initialized_group, "lr": args.group2lr}, 
    ]) 
else: 
    custom_optimizer = AdamW([
        {"params": pretraining_weights_group, "lr": args.group1lr}, 
        {"params": newly_initialized_group, "lr": args.group2lr}, 
    ]) 

max_st = training_args.num_train_epochs * (len(train_dataset)//training_args.per_device_train_batch_size) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(custom_optimizer, T_max = max_st, eta_min = 1e-6) 

weightmodelfirst = next(small_model.parameters()) 
print(weightmodelfirst.dtype) 

name = None 
if len(custom_optimizer.param_groups) == 1: 
    name = "weirdtaskwithgroup1learningrate{}togetherform{}".format(custom_optimizer.param_groups[0]["lr"], args.togetherforming) 
else: 
    name = "weirdtaskwithgroup1learningrate{}group2learningrate{}togetherform{}".format(custom_optimizer.param_groups[0]["lr"], custom_optimizer.param_groups[1]["lr"], args.togetherforming) 

if has_wandb: 
    wandb.init(project = "llm160m",
        config = {**(training_args.to_dict()), **(args.__dict__)}, 
        name = "concatenationwithpretrainedfrozend", 
    ) 

trainer = CustomTrainer( 
    large_model = large_model, 
    model = small_model, 
    args = training_args, 
    train_dataset = train_dataset, 
    eval_dataset = test_dataset, 
    data_collator = data_collator, 
    tokenizer = tokenizer, 
    optimizers = (custom_optimizer, scheduler), 
    togetherForming = args.togetherforming, 
) 

trainer.train() 
