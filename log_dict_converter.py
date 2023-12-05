import torch 
from src.transformers import AutoTokenizer 

def log_dict_converter(filename, preproc, tokenizer): 
    import ast 

    with open(filename, "r") as f: 
        data = f.read() 

        words = ast.literal_eval(data) 

        data = {tuple(pairs): count for pairs, count in words} 
        if not preproc: 
            return data 
        else: 
            # first take all the keys out 
            keys = list(data.keys()) 

            # then we tokenize them 
            assert tokenizer is not None 
            output_keys = [] 
            for key in keys: 
                print("got here, keys are {}".format(key)) 
                # output_tokenized_keys = tokenizer(key, add_special_tokens = False, return_attention_mask = False, return_tensors = "pt") 
                local_tensor = [] 
                for seg in key: 
                    if seg == "<0x0A>": 
                        seg = "\n" 
                    output_tokenized_keys = tokenizer(seg, add_special_tokens = False, return_attention_mask = False, return_tensors = "pt") 
                    # local_tensor.append(output_tokenized_keys["input_ids"].squeeze(0)) 
                    tensorofinterest = output_tokenized_keys["input_ids"].squeeze(0) 
                    # if local_tensor.shape[0] == 1: 
                    if tensorofinterest.shape[0] == 1: 
                        print(seg, local_tensor) 
                    else: 
                        # assert local_tensor.shape[0] == 2 
                        assert tensorofinterest.shape[0] == 2 
                        if tensorofinterest[0] == 29871: 
                            # print(seg, tensorofinterest) 
                            tensorofinterest = tensorofinterest[1:] 
                            print(seg, tensorofinterest) 
                    local_tensor.append(tensorofinterest) 
                print(local_tensor) 
                output_keys.append(torch.cat(local_tensor, dim = 0)) 
                '''
                print(local_tensor) 
                for seg in local_tensor: 
                    for i in range(seg.shape[0]): 
                        print(tokenizer.decode(seg[i])) 
                ''' 
                # output_keys.append(output_tokenized_keys["input_ids"].squeeze(1)) 
            print("got here, length of the output_keys is {}".format(len(output_keys))) 
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
print(datadict.shape) 
