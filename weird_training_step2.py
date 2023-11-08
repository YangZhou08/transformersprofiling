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
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        # self.large_model = large_model 
        # self.generation_config = GenerationConfig(return_dict_in_generate = True) 
        # self.time_checkpoint = time.time() 
        self.time_checkpoint = 0 
    
    def training_step(self, model, inputs): 
        model.train() 
        inputs = self._prepare_inputs(inputs) 
        for k, v in inputs.items(): 
            if isinstance(v, tuple): 
                print(k, len(v)) 
            elif isinstance(v, torch.Tensor): 
                print(k, v.shape) 
            else: 
                print(k, v) 
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
        for k, v in inputs.items(): 
            if isinstance(v, tuple): 
                print(k, len(v)) 
            elif isinstance(v, torch.Tensor): 
                print(k, v.shape) 
            else: 
                print(k, v) 
        
        # print("attention_mask: {}".format(inputs["attention_mask"])) 
        input_ids = inputs["input_ids"] 
        attention_mask = inputs["attention_mask"] 
        labels = inputs["labels"] 
        condensed_embeds = inputs["condensed_embeds"] 
        batch_size, seq_len = attention_mask.shape 
        addedon_length = condensed_embeds.shape[1] 
        attention_mask = torch.cat((attention_mask, torch.ones((batch_size, addedon_length), dtype = torch.long).to(input_ids.device)), dim = 1) 
        
        print("condensed_embeds dtype is {}".format(condensed_embeds.dtype)) 
        print("condensed_embeds is {}".format(condensed_embeds)) 
        # outputs = model(input_ids = large_outputs.sequences, attention_mask = attention_mask, labels = large_outputs.sequences, condensed_embeds = downsampled_vectors) 
        outputs = model(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            labels = labels, 
            condensed_embeds = condensed_embeds, 
            output_hidden_states = True, 
            output_attentions = True, 
            return_dict = True 
        ) 

        print(outputs.hidden_states[0].shape) 
        print(outputs.hidden_states[0][0][0][: 10]) 
        print(outputs.attentions[0][0]) 
        
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            ) 
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
        print(colored("the loss is {}".format(loss), "yellow")) 

        return (loss, outputs) if return_outputs else loss 

class CustomDataset: 
    def __init__(self, data_dir, tokenizer = None, max_length = 128): 
        # self.synthesize_dir = "/home/yangzho6/c4llm_synthesized/" 
        self.synthesize_dir = data_dir 
        self.dataset = load_dataset('json', data_files = self.synthesize_dir + "c4synthesized_file1.json") 
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
                item['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt', 
                return_special_tokens_mask = True 
            ) 
            item['input_ids'] = encoded_text['input_ids'].squeeze(0)  # remove the batch dimension
            item['attention_mask'] = encoded_text['attention_mask'].squeeze(0)  # remove the batch dimension 
        
        item["condensed_embeds"] = tensor 

        return item 

from src.transformers import BitsAndBytesConfig 

# cache_dir = "/home/bc20/yang/" 
dir_dataset = "/home/yangzho6/c4_parts" 
dir_models = "/home/yangzho6/model_checkpoints" 
dir_sdata = "/home/yangzho6/c4llm_synthesized/" 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
onedataset = load_dataset('json', data_files = '/home/yangzho6/c4_parts/downloads/c4_file1.json', split = "train[:1000]") 
# onedataset = load_dataset("c4", "en", split = "train", cache_dir = dir_dataset) 

# d = onedataset.train_test_split(test_size = 0.1) 
# print(d["train"], d["test"]) 

print() 

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir) 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
# tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
# print("the tokenizer pad token id is {}".format(tokenizer.pad_token_id)) 
# tokenizer.pad_token = "[PAD]" 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 
datasetnew = CustomDataset(data_dir = dir_sdata, tokenizer = tokenizer) 

# small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = cache_dir).to(torch_device) 
small_config = LlamaConfig.from_pretrained("JackFram/llama-160m", cache_dir = dir_models) 
'''
print("print out configurations") 
for k, v in small_config.__dict__.items(): 
    print(k, v) 
''' 
small_state_dict_for_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = dir_models).state_dict() 
small_model = SimpleSmallModel(small_config) 
'''
print("we expect the following keys") 
print(len(small_model.state_dict().keys())) 
for key, _ in small_model.named_parameters(): 
    print(key) 

print() 
print("from the pretrained model, we found the following keys") 
print(type(small_state_dict_for_model)) 
print(len(small_state_dict_for_model.keys())) 
''' 
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
small_model = small_model.to(torch_device) 
small_model.train() 

small_model.config.pad_token_id = tokenizer.pad_token_id 
small_model.train() 

# max_length = small_model.config.max_position_embeddings 
max_length = 64 
# def encode_with_truncation(examples): 
    # return tokenizer(examples["text"], truncation=True, padding="max_length",
                #    max_length=max_length, return_special_tokens_mask=True) 
def encode_with_truncation(examples): 
    return tokenizer(examples["text"], truncation = True, padding = "max_length", 
                     max_length = max_length, return_special_tokens_mask = True) 

# train_dataset = d['train'].map(encode_with_truncation, batched = True, num_proc = 4) 
# test_dataset = d['test'].map(encode_with_truncation, batched = True, num_proc = 4) 

print("The model max length is {}".format(small_model.config.max_position_embeddings)) 

# train_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 
# test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

# model_path = "/home/bc20/yang" 
model_path = "/home/yangzho6/model_checkpoints" 
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=50,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=2, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
) 

weightmodelfirst = next(small_model.parameters()) 
print(weightmodelfirst.dtype) 

trainer = CustomTrainer( 
    model = small_model, 
    args = training_args, 
    train_dataset = datasetnew, 
    data_collator = data_collator, 
) 

trainer.train() 
