import torch 
import argparse 

from datasets import load_dataset 
import sys 
import os 
current_dir = os.path.dirname(__file__) 
parent_dir = os.path.dirname(current_dir) 
src_folder = os.path.join(parent_dir, "src") 
sys.path.append(src_folder) 

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from transformers import GPTNeoXForCausalLM 
from transformers import AutoConfig 
from transformers import LlamaConfig, LlamaPreTrainedModel 
from transformers import LlamaTokenizer 
from transformers.models.llama.modeling_llama import LlamaForCausalLM 
# from transformers.models.llama.modeling_llama import LlamaWeirdLargeTest 
from transformers.models.llama.modeling_llama import LlamaWeirdLargeRecoveringModeOn 
from transformers.models.llama.modeling_llama import LlamaForCausalLM2 
from transformers.models.llama.modeling_llama import SimpleSmallModel 
from transformers import Trainer, TrainingArguments 
from transformers import DataCollatorForLanguageModeling 
from torch.utils.data import random_split 
from torch.nn import CrossEntropyLoss 
from termcolor import colored 

from tqdm import tqdm 

import torch.nn.functional as F 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
from transformers.griffin.llamastats import get_llama_griffinstats 
from transformers.griffin.llama_chunk_redirecting import get_llama_griffintwo 

import socket 

hostname = socket.gethostname() 
print("Hostname: ", hostname) 

torch.set_printoptions(precision=3) 
import matplotlib.pyplot as plt 
import numpy as np 

##### Useful dataset path ##### 
if "lovelace" in hostname: 
    dir_models = "/home/yangzho6/model_checkpoints/" 
    dir_c4llmsynthesized = "/home/yangzho6/c4llm_synthesized/llama2_7b_topkna/" 
    dir_c4 = "/home/yangzho6/c4_parts/downloads/" 
elif "ada" in hostname: 
    dir_models = "/home/beidic/yangzho6/model_checkpoints/" 
    dir_c4llmsynthesized = "/home/beidic/yangzho6/c4llm_synthesized/" 
else: 
    dir_models = "/fsx-storygen/beidic/yang/model_checkpoints/" 
    dir_c4llmsynthesized = "/fsx-storygen/beidic/yang/c4llm_synthesized/" 
    dir_c4 = "/fsx-storygen/beidic/yang/c4_parts/downloads/" 

##### Loading the tokenizer ##### 
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
tokenizer.padding_side = "left" 

def jaccard_similarity(vector1, vector2):
    # Ensure the vectors are numpy arrays
    vector1 = vector1.cpu().numpy() 
    vector2 = vector2.cpu().numpy() 
    
    # Compute the intersection and union
    intersection = np.sum((vector1 == 1) & (vector2 == 1))
    union = np.sum((vector1 == 1) | (vector2 == 1))
    
    # Compute the Jaccard similarity
    if union == 0:
        return 1.0 if np.array_equal(vector1, vector2) else 0.0
    return intersection / union 

##### Argument parsing ##### 
parser = argparse.ArgumentParser() 
parser.add_argument("--densitychose", type = float, default = 0.5) 
parser.add_argument("--loading_from_checkpoint", type = str, default = None) 
parser.add_argument("--kernelsize", type = int, default = 2) 
parser.add_argument("--experiment_setting", type = str, default = "setting0") 
parser.add_argument("--use_dataset", type = str, default = "pg19") 

args = parser.parse_args() 
'''
##### Loading the model ##### 
large_model = LlamaWeirdLargeTest.from_pretrained(args.loading_from_checkpoint, cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
large_model.set_sliding_window_length(args.kernelsize) 
large_model.addonsmallmodel.set_criticalpath(hostname = hostname) 
large_model.set_msece_loss(use_mse_loss = False, ce_loss_only = True) 
large_model.to(torch.bfloat16).to(torch_device) 
large_model.set_inference_setting(args.experiment_setting) 
large_model.set_walpha(0.5) 
large_model.set_slidingwindowlength(args.kernelsize) 
large_model.set_tokenizer_bos_id(bos_id = tokenizer.bos_token_id, pad_id = tokenizer.pad_token_id) 
large_model.set_cosinesimilarity(False) 
large_model.config.pad_token_id = tokenizer.pad_token_id 
large_model.addonsmallmodel.config.pad_token_id = tokenizer.pad_token_id 
large_model.model.eval() 
large_model.addonsmallmodel.eval() 
model = large_model 
''' 
# large_model = LlamaWeirdLargeTest.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
# large_model = LlamaWeirdLargeRecoveringModeOn.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
# large_model, loading_information = LlamaWeirdLargeRecoveringModeOn.from_pretrained("/home/yangzho6/model_checkpoints/recoveringkernelsize4setting0checkpoint-1500", output_loading_info = True) 
# large_model.set_sliding_window_length(args.kernelsize) 

# small_model_state_dict = SimpleSmallModel.from_pretrained("YangZhoumill/llama_160m_deciphering_tinyllama_setting0_01da4cb_hf", target_model_dim = 2048, cache_dir = dir_models).state_dict() 
# large_model.set_addonsmallmodel_statedict(small_model_state_dict) 

# large_model.addonsmallmodel.set_criticalpath(hostname = hostname) 
# large_model.set_msece_loss(use_mse_loss = False, ce_loss_only = True) 
# large_model.to(torch.bfloat16).to(torch_device) 
# large_model.set_inference_setting(args.experiment_setting) 
# large_model.set_walpha(0.5) 
# large_model.set_slidingwindowlength(sliding_window_length = args.kernelsize, addonmodel_start = args.kernelsize + 1) 
# large_model.set_tokenizer_bos_id(bos_id = tokenizer.bos_token_id, pad_id = tokenizer.pad_token_id) 
# large_model.set_cosinesimilarity(False) 

density = args.densitychose 
# config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf") 
# large_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models).to(torch.bfloat16) 
config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf") 
# large_model = LlamaForCausalLM2.from_pretrained("meta-llama/Llama-2-7b-hf") 
large_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf") 
large_model.config.mode = "gen" 
large_model.config.chunksize = 8 
large_model.config.selection_method = "topk" 

schedule = [density for _ in range(config.num_hidden_layers)] 

# large_model = get_llama_griffintwo(large_model, schedule) 
large_model = get_llama_griffinstats(large_model, schedule) 

large_model.config.pad_token_id = tokenizer.pad_token_id 
# large_model.addonsmallmodel.config.pad_token_id = tokenizer.pad_token_id 
# large_model.model.eval() 
# large_model.addonsmallmodel.eval() 
large_model.eval() 
model = large_model 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

def get_dataset(datasetname, max_length): 
    if datasetname == "c4llm_synthesized": 
        # datasetnew = load_dataset('json', data_files = dfiles, split = "train[:10000]") 
        dfiles = [] 
        if "lovelace" in hostname: 
            # filename = "c4synthesized_file1_kernel7_0.json" 
            filename = "c4synthesized_file1_1_0.json" 
            # dfiles.append(dir_c4llmsynthesized + "{}/".format("tinyllama") + filename) 
            dfiles.append(dir_c4llmsynthesized + filename) 
            datasetnew = load_dataset("json", data_files = dfiles, split = "train[:100]") 
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
        datasetnew = load_dataset("json", data_files = dfiles, split = "train[:100]") 
    elif datasetname == "pg19": 
        datasetnew = load_dataset('emozilla/pg19', split = "train[:100]") 
    elif datasetname == "cnn_dailymail": # we need to use special processing for this dataset 
        datasetnew = load_dataset("cnn_dailymail", "3.0.0", split = "test[:10000]") 
    elif datasetname == "openwebtext": 
        datasetnew = load_dataset("Skylion007/openwebtext", split = "train[:10000]") 
    elif datasetname == "xsum": # we need to use special processing for this dataset 
        datasetnew = load_dataset("xsum", split = "test[:10000]") 
    elif datasetname == "gsm8k": 
        datasetnew = load_dataset("gsm8k", "main", split = "train[:10000]") 
    else: 
        raise ValueError("dataset_name is not recognized") 

    def encode_with_truncationspecialized(examples): 
        # tokdictionary = tokenizer(examples['text'][100000 : 100000 + 3000], padding = "max_length", max_length = max_length, 
        #                 return_attention_mask = True, return_tensors = "pt", truncation = True, 
        #                 add_special_tokens = True) 
        tokdictionary = tokenizer(examples['text'], padding = "max_length", max_length = 260, 
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
    
    def encode_text_summary_gsm8k(examples): 
        # tokdictionary = tokenizer( 
        tokdictionary = tokenizer("Question: " + examples["question"] + "\n " + "Answer: " + examples["answer"], padding = "max_length", 
                                  max_length = max_length, return_attention_mask = True, return_tensors = "pt", 
                                  add_special_tokens = False) 
        newdictionary = {} 
        newdictionary['input_ids'] = tokdictionary['input_ids'].squeeze(0) 
        newdictionary['attention_mask'] = tokdictionary['attention_mask'].squeeze(0) 
        return newdictionary 

    def unflatten_list_func(examples): 
        examples['input_ids'] = examples['input_ids'].squeeze(0) 
        examples['attention_mask'] = examples['attention_mask'].squeeze(0) 

    if datasetname == "pg19": 
        datasetnew = datasetnew.map(encode_with_truncationspecialized, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask"]) 
    elif datasetname == "xsum": 
        datasetnew = datasetnew.map(encode_text_summary_xsum, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask"]) 
    elif datasetname == "cnn_dailymail": 
        datasetnew = datasetnew.map(encode_text_summary, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask"]) 
    elif datasetname == "gsm8k": 
        datasetnew = datasetnew.map(encode_text_summary_gsm8k, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask"]) 
    else: 
        datasetnew = datasetnew.map(encode_with_truncation, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask", "text"]) 

    return datasetnew 

# datasetname = "gsm8k" 
datasetname = args.use_dataset 
listmaxl = {1 : 259, 2 : 259, 3 : 259, 4 : 257, 5 : 256, 7 : 260, 10 : 261} 
# 259 = 256 + 3
eval_dataset = get_dataset(datasetname, max_length = 1024) # 101 = 98 + 3 

training_args = TrainingArguments(
    output_dir = dir_models, 
    per_device_eval_batch_size = 2 if datasetname == "gsm8k" else 1, 
    do_train = False, 
    do_eval = True, 
    label_names = ["labels"], 
) 

trainer = Trainer(
    args = training_args, 
    model = model, 
    data_collator = data_collator, 
    eval_dataset = eval_dataset, 
) 

sum = torch.zeros((1,)).to(torch_device).float() 

seq_level_jaccard_sim_collection = {5 : [], 15 : [], 25 : []} 
for i, batch in enumerate(tqdm(trainer.get_eval_dataloader())): 
    input_ids = batch["input_ids"].to(torch_device) 
    attention_mask = batch["attention_mask"].to(torch_device) 
    if datasetname == "gsm8k": 
        input_ids = torch.cat((input_ids[0], input_ids[1]), dim = 0) 
        input_ids = input_ids.unsqueeze(0) 
        attention_mask = torch.cat((attention_mask[0], attention_mask[1]), dim = 0) 
        attention_mask = attention_mask.unsqueeze(0) 
    
    original_attention_mask = batch["attention_mask"] 
    # (batch_size, 203) 
    labels = batch["labels"].to(torch_device) 
    batch_size, seq_len = original_attention_mask.shape 
    # addedon_length = (seq_len - 7 - 1) // 7 
    addedon_length = (seq_len - args.kernelsize - 1) // args.kernelsize 
    
    large_input_ids = input_ids 
    small_input_ids = input_ids 
    
    original_attention_mask2 = torch.cat((original_attention_mask, torch.ones((batch_size, addedon_length), dtype = torch.long).to(small_input_ids.device)), dim = 1) 
    for i, l in enumerate(model.model.layers): 
        l.mlp.resetgenerationiterateingcount() 
    with torch.no_grad(): 
        output = model.generate(input_ids, 
                                attention_mask = attention_mask, 
                                max_length = 1256, 
                                return_dict_in_generate = True, 
                                # do_sample = False, 
                                do_sample = True, 
                                use_cache = True, 
        ) 
        print("input_ids shape {} output.sequences shape {}".format(input_ids.shape, output.sequences.shape)) 
        for i, l in enumerate(model.model.layers): 
            if l.mlp.savingintermediatestates is not None: 
                layerjaccardsimilarity = [] # this line is for clearing previous list 
                # l.mlp.visualizecolormap(l.mlp.savingactivations, "layer{}_intermediateac.png".format(i)) 
                # l.mlp.seqlenbyintermediate(l.mlp.savingintermediatestates, "layer{}_intermediate.png".format(i)) 
                
                for j in range(1, l.mlp.savingintermediatestates.shape[0]): 
                    similarity = jaccard_similarity(l.mlp.savingintermediatestates[j - 1], l.mlp.savingintermediatestates[j]) 
                    layerjaccardsimilarity.append(similarity) 
                
                avgjaccardsimilarity = np.mean(layerjaccardsimilarity) 
                if not np.isnan(avgjaccardsimilarity): 
                    seq_level_jaccard_sim_collection[i].append(avgjaccardsimilarity.item()) 

        for key in seq_level_jaccard_sim_collection.keys(): 
            if len(seq_level_jaccard_sim_collection[key]) > 1: 
                print("Layer {} average Jaccard similarity {}".format(key, seq_level_jaccard_sim_collection[key][-1])) 
                print("Layer {} average Jaccard similarity {}".format(key, np.mean(seq_level_jaccard_sim_collection[key]))) 
            # print("Layer {} average Jaccard similarity {}".format(key, np.mean(seq_level_jaccard_sim_collection[key]))) 
                
        for i in range(output.sequences.shape[0]): 
            print(colored(tokenizer.decode(output.sequences[i][:101]), "blue"), end = "") 
            print(colored(tokenizer.decode(output.sequences[i][101:]), "green")) 
            print("\n", end = "") 

for key in seq_level_jaccard_sim_collection.keys(): 
    print("length of collected samples {}".format(len(seq_level_jaccard_sim_collection[key]))) 
    print("Layer {} average Jaccard similarity {}".format(key, np.mean(seq_level_jaccard_sim_collection[key]))) 
