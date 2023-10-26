import torch
import argparse
import contexttimer 
import torch.nn.functional as F 

from src.transformers import AutoTokenizer, AutoModelForCausalLM 
from torch.profiler import ProfilerActivity 
# from src.transformers.models.gpt_neox.modeling_gpt_neox import SimpleSmallModel 
from src.transformers.models.llama.modeling_llama import SimpleSmallModel 

# from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2 
import tqdm 

cache_dir = "/home/bc20/yang/transformersprofiling" 
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", cache_dir = "/rscratch/zhendong/yang_tasc") 
# simple_small_model = SimpleSmallModel.from_pretrained("EleutherAI/pythia-70m", cache_dir = "/rscratch/zhendong/yang_tasc").to(torch_device) 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = cache_dir) 
simple_small_model = SimpleSmallModel.from_pretrained("JackFram/llama-160m", cache_dir = cache_dir).to(torch_device) 

simple_small_model.eval() 

inputs_embeds = torch.randn(2, 2, 4096).to(torch_device) 
word_seq = "Peter want to marry a German woman" 
input_ids = tokenizer.encode(word_seq, return_tensors = "pt").to(torch_device) 
input_ids = torch.cat([input_ids, input_ids], dim = 0) 
later_input_ids = "," 
later_input_ids = tokenizer.encode(later_input_ids, return_tensors = "pt").to(torch_device) 
later_input_ids = torch.cat([later_input_ids, later_input_ids], dim = 0) 
past_key_values = None 

if isinstance(input_ids, torch.Tensor): 
    print("input_ids is a Tensor") 
    # input_ids = input_ids["input_ids"] 
else: 
    print("type of input_ids is {}".format(type(input_ids))) 
    input_ids = input_ids["input_ids"] 
    attention_mask = input_ids["attention_mask"] 
    position_ids = torch.arange(0, input_ids.shape[-1], dtype = torch.long, device = input_ids.device).view(1, -1) 

simple_small_model.config.use_cache = False 
outputs = simple_small_model(context_input_ids = input_ids, inputs_embeds = inputs_embeds, later_input_ids = later_input_ids) 

print(outputs.logits.shape) 
