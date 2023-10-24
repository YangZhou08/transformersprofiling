import torch
import argparse
import contexttimer 
import torch.nn.functional as F 

from src.transformers import AutoTokenizer, AutoModelForCausalLM 
from torch.profiler import ProfilerActivity 
from src.transformers.models.gpt_neox.modeling_gpt_neox import SimpleSmallModel 

# from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2 
import tqdm 

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", cache_dir = "/rscratch/zhendong/yang_tasc") 
simple_small_model = SimpleSmallModel.from_pretrained("EleutherAI/pythia-70m", cache_dir = "/rscratch/zhendong/yang_tasc") 
simple_small_model.eval() 

inputs_embeds = torch.randn(1, 2, 512) 
word_seq = "Peter want to marry a German woman" 
input_ids = tokenizer.encode(word_seq, return_tensors = "pt").to(torch_device) 

if isinstance(input_ids, torch.Tensor): 
    print("input_ids is a Tensor") 
    # input_ids = input_ids["input_ids"] 
else: 
    print("type of input_ids is {}".format(type(input_ids))) 
    input_ids = input_ids["input_ids"] 
    attention_mask = input_ids["attention_mask"] 
    position_ids = torch.arange(0, input_ids.shape[-1], dtype = torch.long, device = input_ids.device).view(1, -1) 

outputs = simple_small_model(input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids, inputs_embeds = inputs_embeds) 

print(outputs.logits.shape) 
