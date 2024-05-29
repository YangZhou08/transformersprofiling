import torch 
import argparse 
# import contexttimer 

from datasets import load_dataset 

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 

from termcolor import colored 
from transformers import Trainer, TrainingArguments 
from transformers import DataCollatorForLanguageModeling 
from transformers import LlamaForCausalLM 

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False 
# has_wandb = False 

from torch.optim import AdamW 

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

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
onedataset = load_dataset("wikitext", "wikitext-2-v1", split = "train") # wikitext is loaded, can be replaced with other dataset 

d = onedataset.train_test_split(test_size = 0.1) 
print(d["train"], d["test"]) 

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code = True) 

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", dtype = "auto", trust_remote_code = True) 

max_length = 256 

def encode_with_truncation(examples): 
    return tokenizer(examples["text"], truncation = True, padding = "max_length", 
                     max_length = max_length, return_special_tokens_mask = True) 

train_dataset = d['train'].map(encode_with_truncation, batched = True, num_proc = 4) # num_proc is for accelerating the dataset processing, increase for faster speed 
test_dataset = d['test'].map(encode_with_truncation, batched = True, num_proc = 4) 

train_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 
test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask']) 

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 

model_path = "/home/yangzho6/model_checkpoints" # change to your own path 
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

weights_group_default = [] 

for name, param in model.named_parameters(): 
    weights_group_default.append(param) 

custom_optimizer = AdamW([
    {"params": weights_group_default, "lr": 2e-4}]) 

max_st = training_args.num_train_epochs * (len(train_dataset)//training_args.per_device_train_batch_size) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(custom_optimizer, T_max = max_st, eta_min = 1e-6) # change to another scheduler if needed 

name = None 
if len(custom_optimizer.param_groups) == 1: 
    name = "weirdtaskwithgroup1learningrate{}togetherform{}".format(custom_optimizer.param_groups[0]["lr"], args.togetherforming) 
else: 
    name = "weirdtaskwithgroup1learningrate{}group2learningrate{}togetherform{}".format(custom_optimizer.param_groups[0]["lr"], custom_optimizer.param_groups[1]["lr"], args.togetherforming) 

if has_wandb: 
    wandb.init(project = "phi-2", 
        config = {**(training_args.to_dict()), **(args.__dict__)}, 
        name = "simpletrainingscript") 

trainer = Trainer( 
    model = model, 
    args = training_args, 
    train_dataset = train_dataset, 
    eval_dataset = test_dataset, 
    data_collator = data_collator, 
    optimizers = (custom_optimizer, scheduler), 
) 

trainer.train() 
