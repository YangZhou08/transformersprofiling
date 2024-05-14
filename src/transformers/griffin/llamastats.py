# Adapted from Hugging Face implementation

import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import numpy as np 

import sys 
import os 
current_dir = os.path.dirname(__file__) 
parent_dir = os.path.dirname(current_dir) 
sys.path.append(current_dir) 
sys.path.append(parent_dir) 

from utils import select_neurons 

def get_llama_griffinstats(model,  k_schedule): 
    config = model.config
    for i, l in enumerate(model.model.layers):
        new_mlp = LlamaMLP(config, k_schedule[i], i) 

        new_mlp.gate_proj = l.mlp.gate_proj
        new_mlp.up_proj = l.mlp.up_proj
        new_mlp.down_proj = l.mlp.down_proj
        new_mlp.act_fn = l.mlp.act_fn

        if config.selection_method == 'magnitude':
            assert k_schedule[i] > 0.0
            gate_stat = l.mlp.gate_proj.weight.data.norm(dim=1)
            up_stat = l.mlp.up_proj.weight.data.norm(dim=1)
            stat = (gate_stat * up_stat).unsqueeze(0)
            _, indices = torch.topk(stat, int(stat.shape[1] * new_mlp.k_factor), dim=-1)
            new_mlp.prepare_reduced_weights(indices)
            new_mlp.mag_mask = torch.ones(stat.shape[-1], dtype=bool)
            new_mlp.mag_mask[indices[0]] = False

        l.mlp = new_mlp
    
    return model

class LlamaMLP(nn.Module):
    def __init__(self, config, k_factor, index): 
        super().__init__()
        self.config = config 
        self.layer_index = index 
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu
        
        self.k_factor = k_factor
        self.mode = config.mode
        assert self.mode in ['gen', 'class'] 
        self.savingintermediatestates = None 
        
    def getdense(self, indextensor, shape): 
        # Convert the tensor to a numpy array for visualization 
        tensorinput = torch.zeros(shape, dtype = torch.int32).to("cpu") 
        for i in range(len(indextensor)): 
            tensorinput[indextensor[i]] = 1 
        assert tensorinput.shape[0] == 1 
        if self.savingintermediatestates is not None: 
            self.savingintermediatestates = torch.cat([self.savingintermediatestates, tensorinput], dim = 0) 
        else: 
            self.savingintermediatestates = tensorinput 
    
    def seqlenbyintermediate(self, tensorinput, filename): 
        # assert tensorinput.shape[1] == 111008 
        # this is the intermediate 
        
        array = tensorinput.cpu().numpy() 

        # Create a colormap: 0 -> white, 1 -> green
        cmap = plt.colors.ListedColormap(['white', 'green'])

        # Create the plot
        fig, ax = plt.figure(figsize=(10, 20)) 
        car = ax.imshow(array, cmap=cmap, interpolation='nearest') 

        # Remove grid lines
        ax.grid(False)

        # Remove axis ticks
        ax.set_xticks([]) 
        ax.set_yticks([]) 

        # Display the plot
        
        plt.savefig(filename, bbox_inches='tight') 

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            k_factor = self.k_factor
            if self.mode == 'gen':
                int_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

                # GRIFFIN Expert Selection
                if self.config.selection_method != 'magnitude' and k_factor > 0.0: ### 
                    # print("shape of int_states {}".format(int_states.shape)) 
                    k = int(int_states.shape[-1] * k_factor) 
                    # print("int_states.norm(dim=-1).shape {}".format(int_states.norm(dim = -1).shape)) 
                    # print("int_states.norm(dim=-1).unsqueeze(-1).shape {}".format(int_states.norm(dim=-1).unsqueeze(-1).shape)) 
                    neuron_stat = ((int_states / int_states.norm(dim=-1).unsqueeze(-1))).norm(dim=1) # B, D 
                    # print("(int_states / int_states.norm(dim=-1).unsqueeze(-1)).shape {}".format((int_states / int_states.norm(dim=-1).unsqueeze(-1)).shape)) 
                    # print("neuron_stat.shape {}".format(neuron_stat.shape)) 
                    topk_weight, topk_indices = select_neurons(neuron_stat, self.config.selection_method, k) 
                    if self.layer_index in [5, 15, 25]: 
                        # self.seqlenbyintermediate(topk_indices, int_states.shape, "intermediate_layer_{}_{}.png".format(self.layer_index, self.config.selection_method)) 
                        self.getdense(topk_indices, int_states.shape) 
                        print("shape of self.savingintermediatestates {}".format(self.savingintermediatestates.shape)) 
                    # print("topk_indices.shape {}".format(topk_indices.shape)) 
                    
                down_proj = self.down_proj(int_states) 

        return down_proj
