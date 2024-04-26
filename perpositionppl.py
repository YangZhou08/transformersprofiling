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

if "lovelace" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/yangzho6/model_checkpoints/" 
    # dir_c4llmsynthesized = "/home/yangzho6/c4llm_synthesized/" 
    # dir_c4llmsynthesized = "/home/yangzho6/c4llm_synthesized/tinyllama/" 
    dir_c4llmsynthesized = "/home/yangzho6/c4llm_synthesized/llama2_7b_topkna/" 
    # dir_c4llmsynthesized = "/home/beidic/yangzho6/c4llm_synthesized/" 
    dir_c4 = "/home/yangzho6/c4_parts/downloads/" 
    # dir_sdata = "/home/yangzho6/slimpajama/SlimPajama-627B/test/chunk1/" 
elif "ada" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/beidic/yangzho6/model_checkpoints/" 
    dir_c4llmsynthesized = "/home/beidic/yangzho6/c4llm_synthesized/" 
else: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    # dir_models = "/home/yangzho6/model_checkpoints/" 
    dir_models = "/fsx-storygen/beidic/yang/model_checkpoints/" 
    # dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
    # dir_sdata = "/fsx-storygen/beidic/yang/c4llm_synthesized/" 
    dir_c4llmsynthesized = "/fsx-storygen/beidic/yang/c4llm_synthesized/" 
    dir_c4 = "/fsx-storygen/beidic/yang/c4_parts/downloads/" 

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 

if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
tokenizer.padding_side = "left" 
# tokenizer.padding_side = "right" 

kernel_size = 7 

class CustomDataset: 
    # def __init__(self, data_dir, tokenizer = None, max_length = 256, kernel_size = 7): 
    def __init__(self, data_dir, large_tokenizer = None, small_tokenizer = None, max_length = 256, kernel_size = 7, topk = None, prompt_length = 64, use_minipile = False, in_training = True): 
        # self.synthesize_dir = "/home/yangzho6/c4llm_synthesized/" 
        self.synthesize_dir = data_dir 
        # self.dataset = load_dataset('json', data_files = self.synthesize_dir + "c4synthesized_file1.json", split = "train") 
        # self.dataset = load_dataset('json', data_files = [self.synthesize_dir + 'c4synthesized_file1.json', self.synthesize_dir + 'c4synthesized_file2.json'], split="train") 
        dfiles = [] 
        print(colored("hostname is {}".format(hostname), "yellow")) 
        if not use_minipile: 
            if "ada" in hostname: 
                for i in range(0, 2): 
                    # filename = "c4synthesized_file1_kernel{}_{}.json".format(kernel_size, i) 
                    # filename = "c4synthesized_file1_kernel{}_{}.json".format(kernel_size, i) 
                    filename = "c4synthesized_file1_kernel7_{}.json".format(i) 
                    dfiles.append(self.synthesize_dir + "{}/".format(model_name) + filename) 
            elif "lovelace" in hostname: 
                # filename = "c4synthesized_file1_kernel{}_{}.json".format(kernel_size, 0) 
                filename = "c4synthesized_file1_kernel7_0.json" 
                dfiles.append(self.synthesize_dir + "{}/".format("tinyllama") + filename) 
            else: 
                for i in range(0, 8): 
                    # filename = "c4synthesized_file1_kernel{}_{}_combined.json".format(kernel_size, i) 
                    filename = "c4synthesized_file1_kernel7_{}_combined.json".format(i) 
                    dfiles.append(self.synthesize_dir + "{}_topk{}/".format(model_name, topk if topk is not None else "na") + filename) 
            
            self.dataset = load_dataset('json', data_files = dfiles, split = "train[:10000]") 
            # self.dataset = load_dataset('json', data_files = dfiles, split = "train[:2000]") 
        else: 
            if in_training: 
                self.dataset = load_dataset("JeanKaddour/minipile", split = "train") 
            else: 
                self.dataset = load_dataset("JeanKaddour/minipile", split = "test") 
        self.use_minipile = use_minipile 
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
        
        if not self.use_minipile: 
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
        else: 
            tensor = torch.zeros((28, 2048), dtype = torch.float32) 
        
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

model = LlamaForCausalLM.from_pretrained("Cheng98/llama-160m", cache_dir = dir_models) 
model = model.to(torch_device) 
model.eval() 

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
        filename = "c4_file15.json" 
        dfiles.append(dir_c4 + filename) 
        datasetnew = load_dataset("json", data_files = dfiles, split = "train[:10000]") 
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

    # datasetnew = datasetnew.map(encode_with_truncation, batched = True, num_proc = 8) 
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
    # datasetnew = datasetnew.map(unflatten_list_func, num_proc = 8) 

    # datasetnew = datasetnew.map(unflatten_list_func, num_proc = 8) 
    return datasetnew 

datasetname = "c4" 
listmaxl = {1 : 259, 2 : 259, 3 : 259, 4 : 257, 5 : 256, 7 : 260, 10 : 261} 
eval_dataset = get_dataset(datasetname, listmaxl[7]) 

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

for i, batch in enumerate(trainer.get_eval_dataloader(eval_dataset)): 
    print("i is {}".format(i)) 
    input_ids = batch["input_ids"].to(torch_device) 
    attention_mask = batch["attention_mask"].to(torch_device) 
    labels = batch["labels"].to(torch_device) 
    outputs = model(input_ids = input_ids, 
                    attention_mask = attention_mask, 
                    labels = None) 
    logits = outputs.logits 
    logits = logits[..., :-1, :].contiguous() 
    labels = labels[..., 1:].contiguous() 
    print("logits shape is {}".format(logits.shape)) 
    
    ce_loss = CrossEntropyLoss(reduction = "none") 
    
    loss = ce_loss(logits.view(-1, logits.shape[-1]), labels.view(-1)) 
    print("loss.shape is {}".format(loss.shape)) 
    # print("loss is {}".format(loss)) 
    
    
