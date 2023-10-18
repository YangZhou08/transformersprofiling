import torch 
import argparse 
# import contexttimer 

from src.transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from src.transformers import FlaxT5EncoderModel, T5Tokenizer, T5Config 
from src.transformers.models.t5.modeling_t5 import T5BiLDModel 

from tqdm import tqdm
# from sampling.utils import norm_logits, sample 

import torch.nn.functional as F 

cache_dir = "/rscratch/zhendong/yang_tasc" 

def run(): 
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    # torch_device = 'cpu' 
    
    # from transformers import FlaxT5EncoderModel, T5Tokenizer 
    # tokenizer = AutoTokenizer("t5-small", trust_remote_code = True) 
    tokenizer = AutoTokenizer.from_pretrained("t5-3b", cache_dir = "/rscratch/zhendong/yang_tasc") 
    
    # tokenizer = T5Tokenizer.from_pretrained("google/mt5-small") # TODO: need a better solution 
    
    small_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", cache_dir = "/rscratch/zhendong/yang_tasc").to(torch_device) 
    small_model.eval() 
    # large_model = AutoModelForSeq2SeqLM.from_pretrained("t5-3b", cache_dir = "/rscratch/zhendong/yang_tasc").to(torch_device) 
    large_model = AutoModelForSeq2SeqLM.from_pretrained("t5-large", cache_dir = "/rscratch/zhendong/yang_tasc").to(torch_device) 
    large_model.eval() 
    
    model = T5BiLDModel(large = large_model, small = small_model) # num_small_iter, fallback_threshold, rollback_threshold 
    
    word_prefix = "translate English to German: " 
    word_seq = "I am new to huggingface transformers" 
    word_seq = "Peter want to marry a German woman" 
    word_seq = "I am a student." 
    word_seq = "I am currently playing with chatGPT to write a furniture assembly plan to train a robot." 
    # word_seq = "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battlefield of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live." 
    # word_seq = "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America." 
    word_seq = "The Apollo 11 mission in 1969 marked a monumental achievement for humanity. American astronauts Neil Armstrong and Buzz Aldrin became the first humans to walk on the moon, with Armstrong's famous words: 'That's one small step for man, one giant leap for mankind." 
    # word_suffix = " In the previous sentence, what did Neil Armstrong say?" 
    word_seq = word_prefix + word_seq 
    
    input_ids = tokenizer.encode(word_seq, return_tensors = "pt").to(torch_device) 
    
    pad_token_id = tokenizer.pad_token_id
    decoder_input_ids = torch.full((input_ids.shape[0], 1), pad_token_id, dtype=torch.long).to(input_ids.device) 
    x = decoder_input_ids 
    eos_token_id = tokenizer.eos_token_id 
    
    # encoder_output_small = small_model.get_encoder()(input_ids) 
    # encoder_output_large = large_model.get_encoder()(input_ids) 
    '''
    n = 0 
    past_key_values = None 
    
    while n < 35: 
        outputs = small_model(decoder_input_ids = x, encoder_outputs = encoder_outputs, past_key_values = past_key_values) 
        
        # outputs = small_model(input_ids = input_ids, decoder_input_ids = decoder_input_ids) 
        
        # last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p) 
        print(outputs.logits.shape) # (batch_size, seq_len, vocab_size) 
        # print(outputs) 
        last_p = outputs.logits.argmax(-1)[:, -1].unsqueeze(-1) # argmax (batch_size, seq_len), after [:, -1] -> (batch_size, ), after unsqueeze(-1) -> (batch_size, 1) 
        
        past_key_values = outputs.past_key_values 
        # idx_next = sample(last_p) 
        idx_next = last_p 
        
        # if idx_next.item() == eos_token_id: 
            # break 

        # print("{}".format(tokenizer.decode(idx_next[0], skip_special_tokens = True))) 
        x = torch.cat((x, idx_next), dim=1) 
        n += 1 
    ''' 
    # model.generate(input_ids = x, max_length = 10, pad_token_id = eos_token_id, eos_token_id = eos_token_id, 
    output_ids = model.generate(input_ids = x, max_length = 30, pad_token_id = eos_token_id, eos_token_id = eos_token_id, do_sample = False) 
    print("input: {}".format(word_seq)) 
    # generatedText = tokenizer.decode(x[0], skip_special_tokens = True) 
    generatedText = tokenizer.decode(output_ids[0], skip_special_tokens = True) 
    print("generatedText: {}".format(generatedText)) 
    
    # last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p) 

if __name__ == "__main__": 
    run() 
