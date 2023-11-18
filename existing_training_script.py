import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
# import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m", 
    # load_in_8bit=True, 
    # device_map='auto',
) 

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

import transformers
from datasets import load_dataset
# data = load_dataset("Abirate/english_quotes") 
# data = load_dataset()
data = load_dataset('json', data_files = '/home/yangzho6/c4llm_synthesized/c4synthesized_file1.json') 
data = data.map(lambda samples: tokenizer(samples['text']), batched=True) 
# data = data.train_test_split(test_size = 0.1) 

trainer = transformers.Trainer(
    model=model, 
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100, 
        max_steps=200, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
