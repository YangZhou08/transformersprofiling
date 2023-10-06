import torch 
import argparse 
# import contexttimer 

from src.transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from src.transformers import FlaxT5EncoderModel, T5Tokenizer 

from tqdm import tqdm
# from sampling.utils import norm_logits, sample 

import torch.nn.functional as F 

# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def sample(probs : torch.Tensor, num_samples: int = 1, random_seed = None):
    if random_seed:
        torch.manual_seed(random_seed)
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    if (idx_next.item() == 0):
        raise RuntimeError
    return idx_next


def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True) 
    return x_max / x_max_sum

def run(): 
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    # torch_device = 'cpu' 
    
    # from transformers import FlaxT5EncoderModel, T5Tokenizer 
    # tokenizer = AutoTokenizer("t5-small", trust_remote_code = True) 
    tokenizer = AutoTokenizer.from_pretrained("t5-small", trust_remote_code = True) 
    
    # tokenizer = T5Tokenizer.from_pretrained("google/mt5-small") # TODO: need a better solution 
    
    small_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", cache_dir = "/rscratch/zhendong/yang_tasc").to(torch_device) 
    small_model.eval() 
    
    input_ids = tokenizer.encode("I am new to huggingface transformers", return_tensors = "pt").to(torch_device) 
    
    pad_token_id = tokenizer.pad_token_id
    decoder_input_ids = torch.full((input_ids.shape[0], 1), pad_token_id, dtype=torch.long).to(input_ids.device) 
    x = decoder_input_ids 
    eos_token_id = tokenizer.eos_token_id 
    
    encoder_outputs = small_model.get_encoder()(input_ids) 
    
    n = 0 
    top_k = 10
    top_p = 0.9 
    
    temperature = 1 
    past_key_values = None 
    
    while n < 10: 
        outputs = small_model(decoder_input_ids = x, encoder_outputs = encoder_outputs, past_key_values = past_key_values) 
        
        # outputs = small_model(input_ids = input_ids, decoder_input_ids = decoder_input_ids) 
        
        last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
        past_key_values = outputs.past_key_values
        idx_next = sample(last_p) 
        x = torch.cat((x, idx_next), dim=1) 
        print("one token ahead") 
        n += 1 
    
    generatedText = tokenizer.decode(x[0], skip_special_tokens = True) 
    print("next generated token: {}".format(generatedText)) 
    
    # last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p) 

if __name__ == "__main__": 
    run() 
