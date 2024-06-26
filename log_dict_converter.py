import torch 
from src.transformers import AutoTokenizer 
from termcolor import colored 

def parallel_processing_of_labels(filename, tokenizer): 
    pass 

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
            for idx, key in enumerate(keys): 
                print(idx) 
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
                tokencat = torch.cat(local_tensor, dim = 0) 
                if tokencat.shape[0] != 3: 
                    print(colored(tokencat, "red")) 
                    for i in range(tokencat.shape[0] - 2): 
                        cat1 = tokencat[i : i + 3] 
                        print(cat1) 
                        output_keys.append(cat1) 
                else: 
                    output_keys.append(tokencat) 
                print() 
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

def log_dict_converterc(filename, preproc, tokenizer): 
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
            for idx, key in enumerate(keys): 
                local_tensor = [] 
                for seg in key: 
                    if seg == "<0x0A>": 
                        seg = "\n" 
                    output_tokenized_keys = tokenizer(seg, add_special_tokens = False, return_attention_mask = False, return_tensors = "pt") 
                    # local_tensor.append(output_tokenized_keys["input_ids"].squeeze(0)) 
                    tensorofinterest = output_tokenized_keys["input_ids"].squeeze(0) 
                    # if local_tensor.shape[0] == 1: 
                    if tensorofinterest.shape[0] != 1: 
                        # assert local_tensor.shape[0] == 2 
                        assert tensorofinterest.shape[0] == 2 
                        if tensorofinterest[0] == 29871: 
                            # print(seg, tensorofinterest) 
                            tensorofinterest = tensorofinterest[1:] 
                    local_tensor.append(tensorofinterest) 
                tokencat = torch.cat(local_tensor, dim = 0) 
                if tokencat.shape[0] != 3: 
                    for i in range(tokencat.shape[0] - 2): 
                        cat1 = tokencat[i : i + 3] 
                        output_keys.append(cat1) 
                else: 
                    output_keys.append(tokencat) 
                '''
                print(local_tensor) 
                for seg in local_tensor: 
                    for i in range(seg.shape[0]): 
                        print(tokenizer.decode(seg[i])) 
                ''' 
                # output_keys.append(output_tokenized_keys["input_ids"].squeeze(1)) 
            output_keys = torch.stack(output_keys, dim = 0) 
            return output_keys 

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = "/home/yangzho6/model_checkpoints") 
datadict = log_dict_converter("partial_c4_hot1000.txt", preproc = True, tokenizer = tokenizer) 
datadicc = log_dict_converterc("partial_c4_hot1000.txt", preproc = True, tokenizer = tokenizer) 
torch.set_printoptions(threshold = 10_000) 
'''
print(len(datadict)) 
for key in datadict.keys(): 
    print(key, datadict[key]) 
''' 
print(datadict) 
# find = torch.tensor([[322, 372, 338]]) 
# find = torch.tensor([263, 1353, 310]) 
find = torch.tensor([12220,  4684,  1996]) 
print(torch.where(torch.all(datadict == find, dim = 1))) 
print(datadict.shape) 
print(torch.unique((datadicc == datadict).view(-1))) 
