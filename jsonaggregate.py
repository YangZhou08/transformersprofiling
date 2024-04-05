import json 
import glob 
import argparse 
import subprocess 

parser = argparse.ArgumentParser() 
parser.add_argument("--task_id", type = int, default = 0) 
parser.add_argument("--kernel_size", type = int, default = 7) 
parser.add_argument("--model_name", type = str, default = "openllama3b") 
parser.add_argument("--topk", type = int, default = None) 
args = parser.parse_args() 

model_name = args.model_name 
# synthesized_dir_path = "/fsx-storygen/beidic/yang/c4llm_synthesized/{}_topk{}/".format(model_name, args.topk if args.topk is not None else "na") 
synthesized_dir_path = "/fsx-storygen/beidic/yang/c4llm_synthesized/llama2_7b_topkna/" 
# synthesized_dir_path = "/home/yangzho6/slimpajama/SlimPajama-627B/test/chunk1/" 

json_file_list = [synthesized_dir_path + "c4synthesized_file1_{}_{}.json".format(args.task_id, i) for i in range(8)] 
# json_file_list = [synthesized_dir_path + "example_holdout_{}.jsonl".format(i) for i in range(6282)] 
with open(synthesized_dir_path + "c4synthesized_file1_{}_combined.json".format(args.task_id), "w") as f: 
# with open(synthesized_dir_path + "example_holdout_{}combined.jsonl".format(args.task_id), "w") as f: 
    for json_file in json_file_list: 
        with open(json_file, "r") as f2: 
            for line in f2.readlines(): 
                f.write(line) 
        print("Done with {}".format(json_file)) 

# result = subprocess.run(["wc", "-l", synthesized_dir_path + "c4synthesized_file1_kernel{}_{}_combined.json".format(args.kernel_size, args.task_id)], stdout = subprocess.PIPE) 
result = subprocess.run(["wc", "-l", synthesized_dir_path + "c4synthesized_file1_{}_combined.json".format(args.task_id)], stdout = subprocess.PIPE) 
# result = subprocess.run(["wc", "-l", synthesized_dir_path + "example_holdout_{}combined.jsonl".format(args.task_id)], stdout = subprocess.PIPE) 
print(result.stdout.split()[0]) 
