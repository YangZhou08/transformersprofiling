import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
# import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM 

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False 

model = AutoModelForCausalLM.from_pretrained(
    # "facebook/opt-125m", 
    "Cheng98/llama-160m", 
    # load_in_8bit=True, 
    # device_map='auto',
) 

# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
# tokenizer = AutoTokenizer.from_pretrained("Cheng98/llama-160m") 
tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m") 

if has_wandb: 
    wandbrunname = "sequencelength{}kernelsize{}learning_rate{}".format(128, 4, 2e-4) 
    wandb.init(project="llm160m", name=wandbrunname)

if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
tokenizer.padding_side = "left" 

import transformers
from datasets import load_dataset
# data = load_dataset("Abirate/english_quotes") 
# data = load_dataset()
data = load_dataset('json', data_files = '/home/yangzho6/c4llm_synthesized/c4synthesized_file1.json') 
data = data.map(lambda samples: tokenizer(samples['text']), batched=True) 

print(data) 
# data = data.train_test_split(test_size = 0.1) 

trainer = transformers.Trainer(
    model=model, 
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100, 
        max_steps=5000, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs'
        report_to='wandb' if has_wandb else 'none', 
        run_name=wandbrunname if has_wandb else None, 
    ) 
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
