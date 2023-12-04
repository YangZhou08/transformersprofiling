import torch 
from src.transformers import AutoTokenizer 

def log_dict_converter(filename, preproc, tokenizer): 
    import ast 

    with open(filename, "r") as f: 
        data = f.read() 

        words = ast.literal_eval(data) 

        data = {tuple(pairs): count for pairs, count in words} 
        if preproc: 
            return data 
        else: 
            # first take all the keys out 
            keys = list(data.keys()) 

            # then we tokenize them 
            assert tokenizer is not None 
            output_keys = [] 
            for key in keys: 
                output_tokenized_keys = tokenizer(key, add_special_tokens = False, return_attention_mask = False, return_tensors = "pt") 
                output_keys.append(output_tokenized_keys["input_ids"].squeeze(1)) 
            output_keys = torch.stack(output_keys, dim = 0) 
            print(output_keys.shape) 
            return output_keys 

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = "/home/yangzho6/model_checkpoints") 
datadict = log_dict_converter("partial_c4_hot1000.txt", preproc = True, tokenizer = tokenizer) 
'''
print(len(datadict)) 
for key in datadict.keys(): 
    print(key, datadict[key]) 
''' 
print(datadict) 
