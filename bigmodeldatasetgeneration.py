import torch 
import argparse 
# import contexttimer 

import datasets 
from datasets import load_dataset 

from src.transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from src.transformers.models.llama.modeling_llama import LlamaForCausalLM 
from src.transformers import GPTNeoXForCausalLM 

from tqdm import tqdm
# from sampling.utils import norm_logits, sample 

import torch.nn.functional as F 

from src.transformers.generation.logits_process import LogitsProcessorList 
import time 
import numpy as np 

from termcolor import colored 

from src.transformers import BitsAndBytesConfig 
from src.transformers import Trainer, TrainingArguments 
from src.transformers import DataCollatorForLanguageModeling 
from src.transformers.generation.utils import GenerationConfig 

import os 
import json 

# cache_dir = "/home/bc20/yang/transformersprofiling" 
dir_dataset = "/home/yangzho6/c4_parts" 
dir_models = "/home/yangzho6/model_checkpoints" 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
onedataset = load_dataset('json', data_files = '/home/yangzho6/c4_parts/downloads/c4_file1.json', split = "train[:1000]") 
# onedataset = load_dataset('json', data_files = "/home/bc20/yang/transformersprofiling/downloads/c4_subset.json", split = "train") 

# onedataset = load_dataset("c4", "en", split = "train", cache_dir = cache_dir) 

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir) 
# small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", revision = "step3000", cache_dir = "/rscratch/zhendong/yang_tasc").to(torch_device) 
# small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", revision = "step3000", cache_dir = cache_dir).to(torch_device) 
# small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b", revision = "step3000", cache_dir = cache_dir).to(torch_device) 
# small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir).to(torch_device) 
# quant_config = BitsAndBytesConfig( 
#     load_in_8bit = True, 
#     llm_int8_has_fp16_weight = True, 
# ) 

class CustomTrainer(Trainer): 
    def __init__(self, large_model = None, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.large_model = large_model 
        self.generation_config = GenerationConfig(return_dict_in_generate = True) 
        # self.time_checkpoint = time.time() 
        self.time_checkpoint = 0 
    
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
        
        print("attention_mask: {}".format(inputs["attention_mask"])) 
        with torch.no_grad(): 
            input_ids = inputs["input_ids"] 
            attention_mask = inputs["attention_mask"] 
            labels = inputs["labels"] 
            top_k = 10
            top_p = 0.9 

            temperature = 1 

            large_outputs = self.large_model.generate(input_ids = input_ids, max_length = 128, do_sample = True, top_k = top_k, top_p = top_p, temperature = temperature, output_hidden_states = True, return_dict_in_generate = True) 
            # print("the shape of the sequence is {}".format(large_outputs.sequences.shape)) 
            # print("output last hidden states list has length {}".format(len(large_outputs.hidden_states))) 
            # print("output last hidden states list first element has shape {}".format([len(large_outputs.hidden_states[i]) for i in range(len(large_outputs.hidden_states))])) 
            # print("each token output hiddens states has shape {}".format(large_outputs.hidden_states[-1][-1].shape)) 
            list_of_last_hidden_states = [token_hidden_states[-1][:, -1, :] for token_hidden_states in large_outputs.hidden_states] 
            print(colored("sequences of the large model output sequence has shape {}".format(large_outputs.sequences.shape), "yellow")) 
            downsampled_vectors = self.downsample_vectors(list_of_last_hidden_states) 
            assert len(downsampled_vectors) == 64/4 
            # print("each dim of downsampled_vectors is {}".format(downsampled_vectors[0].shape)) 
            downsampled_vectors = torch.stack(downsampled_vectors, dim = 1) 
            print("downsampled vector dimension is {}".format(downsampled_vectors.shape)) 
            attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.shape[0], 80), device = attention_mask.device)), dim = 1) #TODO make it more general 
            # print("shape of the downsampled vectors is {} hidden states dim {}".format(len(downsampled_vectors), downsampled_vectors[0].shape)) 
        
        outputs = model(input_ids = large_outputs.sequences, attention_mask = attention_mask, labels = large_outputs.sequences, condensed_embeds = downsampled_vectors) 
        
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            ) 
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
        print("the loss is {}".format(loss)) 

        return (loss, outputs) if return_outputs else loss 

small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", cache_dir = dir_models).to(torch_device) 
small_model.eval() 

# train_dataset = onedataset["train"] 
# validation_dataset = onedataset["validation"] 

# for i in range(10): 
#     print(onedataset[i]) 

# d = onedataset.train_test_split(test_size = 0.1) 
# print(d["train"], d["test"]) 
# print(d["train"]) 

print() 

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir) 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
# tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
# print("the tokenizer pad token id is {}".format(tokenizer.pad_token_id)) 
# tokenizer.pad_token = "[PAD]" 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

large_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
large_model.eval() 

# max_length = small_model.config.max_position_embeddings 
max_length = 64 
# def encode_with_truncation(examples): 
    # return tokenizer(examples["text"], truncation=True, padding="max_length",
                #    max_length=max_length, return_special_tokens_mask=True) 
def encode_with_truncation(examples): 
    return tokenizer(examples["text"], truncation = True, padding = "max_length", 
                     max_length = max_length, return_special_tokens_mask = True) 

# train_dataset = onedataset["train"].map(encode_with_truncation, batched = True, num_proc = 4) 
train_dataset = onedataset.map(encode_with_truncation, batched = True, num_proc = 4) 
# train_dataset = d['train'].map(encode_with_truncation, batched = True, num_proc = 4) 
# test_dataset = d['test'].map(encode_with_truncation, batched = True, num_proc = 4) 

print("The model max length is {}".format(small_model.config.max_position_embeddings)) 

train_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 
# test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

# model_path = "/home/bc20/yang" 
model_path = "/home/yangzho6/model_checkpoints" 
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=50,            # number of training epochs, feel free to tweak
    per_device_train_batch_size= 100, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
) 

trainer = CustomTrainer( 
    large_model = large_model, 
    model = small_model, 
    args = training_args, 
    train_dataset = train_dataset, 
    # eval_dataset = test_dataset, 
    data_collator = data_collator, 
) 

synthesized_dir_path = "/home/yangzho6/c4llm_synthesized/" 
synthesized_data_path = "/home/yangzho6/c4llm_synthesized/tensor_dir/" 

os.makedirs(synthesized_data_path, exist_ok = True) 
json_file_name = "c4synthesized_file1.json" 
json_file1 = open(synthesized_dir_path + json_file_name, "a") 

train_dataloader = trainer.get_train_dataloader() 
for step, inputs in enumerate(train_dataloader): 
    inputs = trainer._prepare_inputs(inputs) 
    input_ids = inputs["input_ids"] 
    attention_mask = inputs["attention_mask"] 
    labels = inputs["labels"] 
    top_k = 10
    top_p = 0.9 

    temperature = 1 

    large_outputs = large_model.generate(input_ids = input_ids, max_length = 128, do_sample = True, top_k = top_k, top_p = top_p, temperature = temperature, output_hidden_states = True, return_dict_in_generate = True) 
    # tensor_file_path = os.path.join(synthesized_data_path, "ct_{}.pt".format(step)) 
    list_of_last_hidden_states = [token_hidden_states[-1][:, -1, :] for token_hidden_states in large_outputs.hidden_states] 
    downsampled_vectors = trainer.downsample_vectors(list_of_last_hidden_states) 
    downsampled_vectors = torch.stack(downsampled_vectors, dim = 1) 
    # print("downampled_vector has shape {}".format(downsampled_vectors.shape)) 
    print("downampled_vector has shape {}".format(downsampled_vectors.shape)) 
    textsynthesized = tokenizer.batch_decode(large_outputs.sequences) 
    # print(colored("the text synthesized is {}".format(textsynthesized[49]), "yellow")) 
    print("shape of condensed_token shape is {}".format(downsampled_vectors[0].shape)) 
    # break 
    
    for i in range(downsampled_vectors.shape[0]): 
        print(i) 
        example_downsampled_vector = downsampled_vectors[i].clone() 
        tensor_file_path = os.path.join(synthesized_data_path, "ct_{}.pt".format(step * 100 + i)) 
        '''
        torch.save(example_downsampled_vector, tensor_file_path) 
        ''' 
        example_synthesized = textsynthesized[i] 
        # print("the text synthesized is {}".format(example_synthesized)) 
        # print input sequences 
        print(tokenizer.decode(input_ids[i])) 
        # print("raw output in text") 
        print(colored("the text synthesized is {}".format(textsynthesized[i]), "green")) 
        # outputs = tokenizer.encode(textsynthesized[i], add_special_tokens = False, padding = False) 
        outputs = large_outputs.sequences[i] 
        print("length of the input_ids is {}".format(outputs.shape)) 
        print("the input_ids after the tokenizer is {}".format(outputs)) 
        seq_len = len(outputs) 
        for i in range(seq_len): 
            if outputs[i] == 1: 
                outputs = outputs[i :] 
                break 
        print("the input_ids that should be adjusted is {}".format(outputs)) 
        # what should be stored in the dataset 
        new_output = tokenizer.decode(outputs) 
        print("the input setence is {}".format(new_output)) 

        # what should be loaded in from the dataset after tokenizer 
        new_output = tokenizer.encode_plus(new_output, add_special_tokens = False, padding = "max_length", max_length = 128, return_attention_mask = True, return_tensors = "pt") 
        print("the input_ids that is corrected is {}".format(new_output["input_ids"])) 
        print("the length of the input_ids is {}".format(new_output["input_ids"].shape)) 
        print("the attention mask we got is {}".format(new_output["attention_mask"])) 
        '''
        example_data = {
            "text": example_synthesized, 
            "condensed_token_path": tensor_file_path, 
        } 
        json_file1.write(json.dumps(example_data) + "\n") 
        ''' 
    
json_file1.close() 
