import datasets 
from datasets import load_dataset 

onedataset = load_dataset("wikitext", "wikitext-103-v1", cache_dir = "/rscratch/zhendong/yang_tasc") 

train_dataset = onedataset["train"] 
validation_dataset = onedataset["validation"] 

print(train_dataset[0]) 
