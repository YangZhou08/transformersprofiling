import torch 
import argparse 
# import contexttimer 

from src.transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from src.transformers import GPTNeoXForCausalLM 

from tqdm import tqdm
# from sampling.utils import norm_logits, sample 

import torch.nn.functional as F 

from src.transformers.generation.logits_process import LogitsProcessorList 

# set_logger("/rscratch/zhendong/yang_tasc/transformersprofiling/simple_tb3b_log.txt") 
cache_dir = "/rscratch/zhendong/yang_tasc" 

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
    
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b-deduped", cache_dir = cache_dir) 
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", revision = "step3000", cache_dir = cache_dir) 
    
    # small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", revision = "step3000", cache_dir = "/rscratch/zhendong/yang_tasc").to(torch_device) 
    small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b", cache_dir = "/rscratch/zhendong/yang_tasc").to(torch_device) 
    # small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped", revision = "step3000", cache_dir = cache_dir).to(torch_device) 
    small_model.eval() 
    
    # word_prefix = "summarize: " 
    # word_prefix = "translate English to German: " 
    # word_seq = "I am new to huggingface transformers" 
    word_seq = "Peter want to marry a German woman" 
    # word_seq = "I am a student." 
    # word_seq = "I am currently playing with chatGPT to write a furniture assembly plan to train a robot." 
    # word_seq = "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battlefield of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live." 
    # word_seq = "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America." 
    # word_seq = "The Apollo 11 mission in 1969 marked a monumental achievement for humanity. American astronauts Neil Armstrong and Buzz Aldrin became the first humans to walk on the moon, with Armstrong's famous words: 'That's one small step for man, one giant leap for mankind." 
    # word_suffix = " In the previous sentence, what did Neil Armstrong say?" 
    # word_seq = word_prefix + word_seq 
    
    # input_ids = tokenizer.encode(word_seq, return_tensors = "pt").to(torch_device) 
    input_ids2 = tokenizer(word_seq, return_tensors = "pt").to(torch_device) 
    attention_mask = None 
    print(input_ids2) 
    
    pad_token_id = tokenizer.pad_token_id 
    eos_token_id = tokenizer.eos_token_id 
    # decoder_input_ids = torch.full((input_ids.shape[0], 1), pad_token_id, dtype=torch.long).to(input_ids.device) 
    print("input: {}".format(word_seq)) 
    
    n = 0 
    top_k = 10
    top_p = 0.9 
    
    temperature = 1 
    past_key_values = None 
    
    print("--------- What should be the actual output ---------") 
    input_ids = small_model.generate(**input_ids2, max_length = 30) 
    print("input: {}".format(word_seq)) 
    generatedText = tokenizer.decode(input_ids[0], skip_special_tokens = True) 
    print("generatedText: {}".format(generatedText)) 
    print() 
    
    if isinstance(input_ids2, torch.Tensor): 
        print("input_ids is a Tensor") 
        # input_ids = input_ids["input_ids"] 
    else: 
        print("type of input_ids is {}".format(type(input_ids2))) 
        input_ids = input_ids2["input_ids"] 
        attention_mask = input_ids2["attention_mask"] 
        position_ids = torch.arange(0, input_ids.shape[-1], dtype = torch.long, device = input_ids.device).view(1, -1) 
    
    generated_sequence = input_ids 
    past_output = None 
    while n < 23: 
        # outputs = small_model(decoder_input_ids = x, encoder_outputs = encoder_outputs, past_key_values = past_key_values) 
        # outputs = small_model(**input_ids, past_key_values = past_key_values) 
        # outputs = small_model(input_ids = input_ids, past_key_values = past_key_values) # , attention_mask = attention_mask) 
        # outputs = small_model(**input_ids, past_key_values = past_key_values) 
        print("input_ids get is {}".format(input_ids.shape)) 
        # print("attention_mask get is {}".format(attention_mask)) 
        # print("previous round posision_ids is {}".format(position_ids)) 
        # outputs = small_model(input_ids = input_ids, past_key_values = past_key_values, use_cache = True, attention_mask = attention_mask, position_ids = position_ids) 
        outputs = small_model(input_ids = input_ids, past_key_values = past_key_values, use_cache = True) 
        
        if n == 0: 
            past_output = outputs.logits 
        else: 
            print((outputs.logits[:, : -1, :] - past_output).norm()) 
        print(outputs.logits.shape) # (batch_size, seq_len, vocab_size) 
        # print(outputs.attention_mask) 
        # print(outputs.position_ids) 
        # print(outputs) 
        # last_p = outputs.logits.argmax(-1)[:, -1].unsqueeze(-1) # argmax (batch_size, seq_len), after [:, -1] -> (batch_size, ), after unsqueeze(-1) -> (batch_size, 1) 
        next_token_logits = outputs.logits[:, -1, :] 
        next_tokens = torch.argmax(next_token_logits, dim = -1) 
        
        print("****** {} iteration {} ******".format(n, next_tokens)) 
        
        past_key_values = outputs.past_key_values 
        # idx_next = sample(last_p) 
        
        if next_tokens.item() == eos_token_id: 
            break 
        
        # print("{}".format(tokenizer.decode(idx_next[0], skip_special_tokens = True))) 
        # input_ids.input_ids = torch.cat(input_ids.input_ids, idx_next, dim = 1) 
        # input_ids = torch.cat([input_ids, next_tokens[:, None]], dim = -1) 
        generated_sequence = torch.cat([generated_sequence, next_tokens[:, None]], dim = -1) 
        input_ids = next_tokens.unsqueeze(1) 
        n += 1 
        # attention_mask = torch.cat((attention_mask, torch.ones(attention_mask.shape[0], 1, dtype = torch.long).to(torch_device)), dim = 1) 
        # position_ids = torch.tensor([generated_sequence.shape[-1] - 1]).to(torch_device).view(1, -1) 
        print() 
    print("input: {}".format(word_seq)) 
    generatedText = tokenizer.decode(generated_sequence[0], skip_special_tokens = True) 
    print("generatedText: {}".format(generatedText)) 
    
    print() 
    
    # last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p) 

if __name__ == "__main__": 
    run() 
