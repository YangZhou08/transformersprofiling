import torch 

labels = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]] 
labels = torch.tensor(labels) 
addonmodel_start = 2 
sliding_window_length = 1 

copy_idx = [addonmodel_start + (sliding_window_length * i) for i in range(6)] 
print("copy_idx: ", copy_idx) 
labels_addition = labels[:, copy_idx] 
newlabels = labels[:, : addonmodel_start] 
old_label_count = addonmodel_start 
for i in range(labels_addition.shape[1]): 
    newlabels = torch.cat([newlabels, labels_addition[:, i].unsqueeze(1)], dim = 1) 
    if old_label_count < labels.shape[1]: 
        # newlabels = torch.cat([newlabels, labels[:, old_label_count : min(old_label_count + self.sliding_window_length, labels.shape[1])]], dim = 1) 
        newlabels = torch.cat([newlabels, labels[:, old_label_count : min(old_label_count + sliding_window_length, labels.shape[1])]], dim = 1) 
    old_label_count += sliding_window_length 

print(newlabels) 
