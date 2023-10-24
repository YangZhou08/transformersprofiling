import datasets 
from datasets import load_dataset 

onedataset = load_dataset('json', data_files = '/rscratch/zhendong/yang_tasc/downloads/c4_subset.json', split = 'train') 

# train_dataset = onedataset["train"] 
# validation_dataset = onedataset["validation"] 

for i in range(10): 
    print(onedataset[i]) 
