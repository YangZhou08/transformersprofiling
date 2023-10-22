import torch
import argparse
import contexttimer

from src.transformers import AutoTokenizer, AutoModelForCausalLM 
from torch.profiler import ProfilerActivity

# from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2 
import tqdm 

# my local models
MODELZOO = {
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama1b": "/share_nfs/fangjiarui/root/code/hf_models/TinyLlama-1.1B-step-50K-105b",
    "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    # https://huggingface.co/huggyllama/llama-13b
    "llama13b": None,
    "bloom-560m": "/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
    "bloom7b": "/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1",
    "baichuan-7b": "/share_nfs/duanqiyuan/models/source_models/hf/baichuan-7B",
    "baichuan-13b": "/share_nfs/duanqiyuan/models/source_models/hf/Baichuan-13B-Base", 
    "pythia-160M": "EleutherAI/pythia-160m", 
    "pythia-70M": "EleutherAI/pythia-70m", 
    "pythia-2.8b": "EleutherAI/pythia-2.8b",    
} 

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Suggest at least five related search terms to \"Mạng neural nhân tạo\".")
    # parser.add_argument('--approx_model_name', type=str, default=MODELZOO["bloom-560m"]) 
    parser.add_argument('--approx_model_name', type = str, default = MODELZOO['pythia-70M']) 
    # parser.add_argument('--target_model_name', type=str, default=MODELZOO["bloom7b"]) 
    parser.add_argument('--target_model_name', type = str, default = MODELZOO['pythia-2.8b']) 
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed')
    args = parser.parse_args()
    return args


def benchmark(fn, print_prefix, use_profiler=True, *args, **kwargs):
    TEST_TIME = 10
    profile_filename = f"./profile_logs/{print_prefix}"
    
    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME): 
                    output = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens") 

@torch.no_grad()
def speculative_sampling_v2(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                # below is a line with LogitsProcessor and sampling, however for now, we just use greedy decoding 
                # next_tok = sample(norm_logits(q[:, -1, :], 
                                #   temperature, top_k, top_p), random_seed = random_seed) 
                next_tok = torch.argmax(q[:, -1, :], dim = -1) 
                # x = torch.cat((x, next_tok), dim=1)
                x = torch.cat([x, next_tok[:, None]], dim = 1) 
            '''
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = target_model(x).logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)
            ''' 
            p = target_model(x).logits 
            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            n = prefix_len - 1 
            '''
            for i in range(gamma):
                r = torch.rand(1, device = p.device)
                j = x[:, prefix_len + i]
                
                if r < torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                else:
                    # reject
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]), random_seed = random_seed)
                    is_all_accept = False
                    break
            ''' 
            # vectorize all the operations 
            r = torch.rand(gamma, device = p.device) 
            j = x[:, prefix_len : prefix_len + gamma] # (batch_size, gamma) 
            
            p_seg = p[:, prefix_len - 1 : prefix_len + gamma - 2, j] 
            q_seg = q[:, prefix_len - 1 : prefix_len + gamma - 2, j] 
            pdivq = torch.div(p_seg, q_seg) 
            pdivq = torch.minimum(torch.ones_like(pdivq).to(pdivq.device), pdivq) 
            criterion = (pdivq - r.unsqueeze(1)) > 0 
            
            if criterion.any(): 
                n = criterion.to(torch.int).argmax() # hack, just to find the first 
                n += 1 # NOTE note sure whether this is needed 
            
            
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(p[:, -1, :], random_seed = random_seed)
            
            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(n - pbar.n)

    return prefix


def generate(input_text, approx_model_name, target_model_name, num_tokens=40, random_seed = None, verbose = False, use_benchmark = True):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, cache_dir = "/rscratch/zhendong/yang_tasc").to(torch_device) 
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, cache_dir = "/rscratch/zhendong/yang_tasc").to(torch_device) 
    print("finish loading models")
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 10
    top_p = 0.9
    '''
    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"large (target) model autoregressive_sampling: {generated_text}")
    
    TEST_TIME = 10
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_large", True,
                  input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_small", True,
                  input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    ''' 
    torch.manual_seed(123)
    output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"deepmind's speculative_sampling: {generated_text}")   
    '''
    torch.manual_seed(123)
    output = speculative_sampling(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"google's speculative_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(speculative_sampling, "SP", True,
                  input_ids, small_model, large_model, max_len = num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
    ''' 
if __name__ == "__main__":
    args = parse_arguments()
    
    generate(args.input, args.approx_model_name, args.target_model_name, random_seed = args.seed, verbose=args.verbose)
