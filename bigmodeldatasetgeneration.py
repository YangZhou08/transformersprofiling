import datasets 
from datasets import load_dataset 

onedataset = load_dataset("c4", "en", split="train[:1%]", cache_dir = "/rscratch/zhendong/yang_tasc") 

# train_dataset = onedataset["train"] 
# validation_dataset = onedataset["validation"] 

for i in range(100): 
    print(onedataset[i]) 
