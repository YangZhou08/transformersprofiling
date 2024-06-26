# Adapted from Hugging Face implementation

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys 
import os 
current_dir = os.path.dirname(__file__) 
parent_dir = os.path.dirname(current_dir) 
sys.path.append(current_dir) 
sys.path.append(parent_dir) 

from utils import select_neurons 

def get_llama_griffinthree(model, k_schedule, chunksize): 
    config = model.config
    for i, l in enumerate(model.model.layers):
        new_mlp = LlamaMLP(config, k_schedule[i], chunksize) 

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
    def __init__(self, config, k_factor, chunksize): 
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu 
        
        self.k_factor = k_factor
        self.mode = config.mode
        assert self.mode in ['gen', 'class'] 
        self.chunksize = chunksize 
        self.generationiterationcount = -1 
        self.neuron_stat = None 
        
    def resetgenerationiterateingcount(self): 
        self.generationiterationcount = -1 
        self.neuron_stat = None 


    def prepare_reduced_weights(self, topk_indices):
        assert topk_indices.shape[0] == 1 # Batch size 1
        
        self.gate_proj_reduced = nn.Linear(self.gate_proj.weight.data.shape[1], len(topk_indices), bias=False)
        self.up_proj_reduced = nn.Linear(self.up_proj.weight.data.shape[1], len(topk_indices), bias=False)
        self.down_proj_reduced = nn.Linear(len(topk_indices), self.down_proj.weight.data.shape[0], bias=False)
        topk_indices = topk_indices[0] 

        self.gate_proj_reduced.weight.data = self.gate_proj.weight.data[topk_indices]
        self.up_proj_reduced.weight.data = self.up_proj.weight.data[topk_indices]
        self.down_proj_reduced.weight.data = self.down_proj.weight.data[:, topk_indices]
    

    def forward(self, x):
        self.generationiterationcount += 1 
        
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
                if self.generationiterationcount % self.chunksize == 0:                                                                                                                                                                                                      
                    int_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

                    # GRIFFIN Expert Selection
                    if self.config.selection_method != 'magnitude' and k_factor > 0.0: 
                        k = int(int_states.shape[-1] * k_factor)
                        neuron_stat = ((int_states / int_states.norm(dim=-1).unsqueeze(-1))).norm(dim=1) # B, D 
                        if self.neuron_stat is None: 
                            self.neuron_stat = neuron_stat 
                        else: 
                            self.neuron_stat = torch.sqrt(torch.pow(self.neuron_stat, 2) + torch.pow(neuron_stat, 2)) 
                        # topk_weight, topk_indices = select_neurons(neuron_stat, self.config.selection_method, k) 
                        topk_weight, topk_indices = select_neurons(self.neuron_stat, self.config.selection_method, k) 
                        # print("at number {}, we re-select neurons".format(self.generationiterationcount)) 
                        self.prepare_reduced_weights(topk_indices)
                        
                    down_proj = self.down_proj(int_states)

                else:
                    down_proj =self.down_proj_reduced(self.act_fn(self.gate_proj_reduced(x)) * self.up_proj_reduced(x)) 
            else: 
                raise ValueError("Invalid mode") 

        return down_proj 
