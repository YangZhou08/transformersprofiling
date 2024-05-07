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
from transformers import LlamaConfig, LlamaPreTrainedModel 
from transformers import LlamaTokenizer 
from transformers.models.llama.modeling_llama import LlamaForCausalLM 
from transformers.models.llama.modeling_llama import LlamaWeirdLargeTest 
from transformers import Trainer, TrainingArguments 
from transformers import DataCollatorForLanguageModeling 
from torch.utils.data import random_split 
from torch.nn import CrossEntropyLoss 

from tqdm import tqdm 

import torch.nn.functional as F 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 

import socket 

hostname = socket.gethostname() 
print("Hostname: ", hostname) 

torch.set_printoptions(precision=3) 

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

##### Argument parsing ##### 
parser = argparse.ArgumentParser() 
parser.add_argument("--loading_from_checkpoint", type = str, default = None) 
parser.add_argument("--kernelsize", type = int, default = 2) 

args = parser.parse_args() 

##### Loading the model ##### 
large_model = LlamaWeirdLargeTest.from_pretrained(args.loading_from_checkpoint, cache_dir = dir_models).to(torch.bfloat16).to(torch_device) 
large_model.set_sliding_window_length(args.kernelsize) 
large_model.addonsmallmodel.set_criticalpath(hostname = hostname) 
large_model.set_msece_loss(use_mse_loss = False, ce_loss_only = True) 
large_model.to(torch.bfloat16).to(torch_device) 
large_model.set_inference_setting("setting0") 
large_model.set_walpha(0.5) 
large_model.set_slidingwindowlength(args.kernelsize) 
large_model.set_tokenizer_bos_id(bos_id = tokenizer.bos_token_id, pad_id = tokenizer.pad_token_id) 
large_model.set_cosinesimilarity(False) 
large_model.config.pad_token_id = tokenizer.pad_token_id 
large_model.addonsmallmodel.config.pad_token_id = tokenizer.pad_token_id 
large_model.model.eval() 
large_model.addonsmallmodel.eval() 
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
            datasetnew = load_dataset("json", data_files = dfiles, split = "train[:10000]") 
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
        datasetnew = load_dataset("json", data_files = dfiles, split = "train[:5000]") 
    elif datasetname == "pg19": 
        datasetnew = load_dataset('emozilla/pg19', split = "train[:10000]") 
    elif datasetname == "cnn_dailymail": # we need to use special processing for this dataset 
        datasetnew = load_dataset("cnn_dailymail", "3.0.0", split = "test[:10000]") 
    elif datasetname == "openwebtext": 
        datasetnew = load_dataset("Skylion007/openwebtext", split = "train[:10000]") 
    elif datasetname == "xsum": # we need to use special processing for this dataset 
        datasetnew = load_dataset("xsum", split = "test[:10000]") 
    else: 
        raise ValueError("dataset_name is not recognized") 

    def encode_with_truncationspecialized(examples): 
        tokdictionary = tokenizer(examples['text'][100000 : 100000 + 3000], padding = "max_length", max_length = max_length, 
                        return_attention_mask = True, return_tensors = "pt", truncation = True, 
                        add_special_tokens = True) 
        # tokdictionary = tokenizer(examples['text'], padding = "max_length", max_length = 260, 
        #                          return_attention_mask = True, return_tensors = "pt", truncation = True, 
        #                          add_special_tokens = True) 
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
    else: 
        datasetnew = datasetnew.map(encode_with_truncation, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask", "text"]) 

    return datasetnew 

datasetname = "c4" 
listmaxl = {1 : 259, 2 : 259, 3 : 259, 4 : 257, 5 : 256, 7 : 260, 10 : 261} 
eval_dataset = get_dataset(datasetname, listmaxl[2]) 

training_args = TrainingArguments(
    output_dir = dir_models, 
    per_device_eval_batch_size = 1, 
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
for i, batch in enumerate(tqdm(trainer.get_eval_dataloader())): 
    input_ids = batch["input_ids"].to(torch_device) 
    attention_mask = batch["attention_mask"].to(torch_device) 
    original_attention_mask = batch["attention_mask"] # (batch_size, 203) 
    labels = batch["labels"].to(torch_device) 
    batch_size, seq_len = original_attention_mask.shape 
    # addedon_length = (seq_len - 7 - 1) // 7 
    addon_length = (seq_len - args.kernelsize - 1) // args.kernelsize 
    
    large_input_ids = input_ids 
    small_input_ids = input_ids 
    
    original_attention_mask2 = torch.cat((original_attention_mask, torch.ones((batch_size, addedon_length), dtype = torch.long).to(small_input_ids.device)), dim = 1) 
    with torch.no_grad(): 
        outputs = model(
            large_input_ids = large_input_ids, 
            small_input_ids = small_input_ids, 
            attention_mask = original_attention_mask, 
            output_hidden_states = True, 
            output_attentions = True, 
            return_dict = True, 
            original_attention_mask = original_attention_mask2, 
            labels = labels, 
            condensed_embed_labels = None, 
            label_adjustment = False, 
            usingsecondtolastvectors = False, 
            autoregressive_first_element = True, 
        ) 
        loss = outputs.loss 
        print("loss: ", loss) 

sum += loss 
print("score: ", sum/len(trainer.get_eval_dataloader())) 
