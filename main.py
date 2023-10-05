import torch 
import argparse 
import contexttimer 

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 

from tqdm import tqdm
from sampling.utils import norm_logits, sample

def run(): 
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    from transformers import FlaxT5EncoderModel, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small") # TODO: need a better solution 
    
    small_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(torch_device) 
    
    input_ids = tokenizer.encode("I am new to huggingface transformers", return_tensors = "pt").to(torch_device) 
    
    small_model(input_ids) 
