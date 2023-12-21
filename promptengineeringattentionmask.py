import torch 
import matplotlib.pyplot as plt 
import numpy as np 
from src.transformers import AutoTokenizer 
import warnings 
# from ...modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask 
from src.transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask 
from typing import List, Optional, Tuple, Union 


def _modify_decoder_attention_mask(combined_attention_mask, dtype, mask_list_pos, start_idx = None, kernel_size = None): 
        ''' 
        Modify the attention mask based on the mask_list_pos, which is a list of position for separation 
        in the modeling_llama.py, this function is of name _modify_decoder_attention_mask_for_hardest 
        
        inputs: 
            combined_attention_mask: the attention mask to be modified (usually the original attention mask generated by the tokenier) 
            dtype: the dtype of the attention mask 
            mask_list_pos: the list of position for separation 
            start_idx: the starting position for separation 
            kernel_size: the sliding window size for separation 
        
        This function modifies the attention mask in-place 
        '''
        mask_shape = combined_attention_mask.shape # (batch_size, 1, tgt_seq_len, src_seq_len) 
        seq_len = mask_shape[-1] 
        start_idx = start_idx 
        kernel_size = kernel_size 
        
        # row dimensional masking 
        # row_idx_masked_out = [start_idx + i * (kernel_size + 1) for i in range((seq_len - start_idx) / (kernel_size + 1))] 
        row_mask = torch.zeros(mask_shape[-2], mask_shape[-1], device = combined_attention_mask.device) # NOTE currently, this line only works for training 
        # row_mask[row_idx_masked_out] = 1 
        row_mask[mask_list_pos] = 1 

        condensed_token_idx_list = mask_list_pos 
        for i in range(len(condensed_token_idx_list) - 1): 
            # if i == 0: 
            #     row_mask[condensed_token_idx_list[i] : condensed_token_idx_list[-1], condensed_token_idx_list[i] : condensed_token_idx_list[i]] = 1 
            # else: 
            if i == 0: 
                continue 
            row_mask[condensed_token_idx_list[i] : condensed_token_idx_list[-1], condensed_token_idx_list[i - 1] : condensed_token_idx_list[i]] = 1 
        
        # print("row mask shape {}".format(row_mask.shape)) 
        row_mask = row_mask[None, None, :, :].expand(mask_shape).to(torch.bool) 
        row_mask = row_mask.to(device = combined_attention_mask.device) 

        combined_attention_mask.masked_fill_(row_mask, torch.finfo(dtype).min) 

def visualize_attention_mask(sequence_length, tensor, filename): 
        # code generated by GPT-4 
        '''
        # Create a tensor for demonstration purposes (different from the plot function, this function is called outside the model's forward prop and visualize the statically generated attention mask) 
        # In your case, replace this with the actual tensor
        tensor = torch.full((sequence_length, sequence_length), float('-inf'))
        mask = torch.rand(sequence_length, sequence_length) > 0.5  # Random mask for demo
        tensor[mask] = 0.0
        ''' 
        # Convert to numpy for visualization
        tensor_np = tensor.cpu().clone().numpy() 

        # Replace -inf with 1 and 0 with 0 for visualization purposes
        # visual_tensor = np.where(tensor_np == float('-inf'), 1, 0) 
        visual_tensor = np.where(tensor_np < 0, 1, 0) 
        # print(visual_tensor) 

        # Create the plot
        fig, ax = plt.subplots(figsize=(30, 30)) 
        cmap = plt.cm.colors.ListedColormap(['black', 'lightblue'])
        bounds = [-0.5, 0.5, 1.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        # Display the data
        cbar = ax.imshow(visual_tensor, cmap=cmap, norm=norm)

        # Set the background color to white
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Add gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        # ax.set_xticks(np.arange(-0.5, sequence_length, 1)) 
        # ax.set_yticks(np.arange(-0.5, sequence_length, 1)) 
        tick_positions = np.arange(0, sequence_length, 1)
        ax.set_xticks(tick_positions - 0.5)  # Shift the tick positions to be centered between the gridlines
        ax.set_yticks(tick_positions - 0.5)  # Same shift for y-ticks

        # Label the axes
        ax.set_xticklabels(np.arange(0, sequence_length))
        ax.set_yticklabels(np.arange(0, sequence_length))

        # Set the tick labels for both axes
        plt.xticks(rotation=90)
        ax.tick_params(axis=u'both', which=u'both',length=0)

        # Set axis limits to make the grid fit the image correctly
        ax.set_xlim(-0.5, sequence_length-0.5)
        ax.set_ylim(sequence_length-0.5, -0.5)

        # Show the color bar
        plt.colorbar(cbar, ticks=[0, 1], orientation='vertical', shrink=0.8, aspect=20)

        # Save the plot
        plt.savefig(filename, format='jpg', bbox_inches='tight') 
        # print("we got here") 
        plt.close() 

def plot_attention_map(attention_maps, layer_num, head_num, seq_length, filename):
        """
        Plots the attention map for a specific layer and head and saves it to a file. (Different from visualization function above, this function is called inside the model's forward prop) 
        Also, this function is for visualizing the attention map not attention mask so you can also see other attention score being visualized. 

        :param attention_maps: A nested list or array of attention maps from the transformer model.
        :param layer_num: The layer number to visualize.
        :param head_num: The head number to visualize.
        :param seq_length: The sequence length of the model.
        :param filename: The filename to save the plot.
        """

        import matplotlib.colors as mcolors

        # Extract the specific attention map
        # attention_map = attention_maps[layer_num][head_num] 
        attention_map = attention_maps[layer_num][0][head_num].cpu().detach().numpy() 
        
        # Create a mask for exact zeros
        zero_mask = attention_map == 0
        '''
        # Create a custom colormap
        colors = [(plt.cm.bwr(i)) for i in range(256)]
        # midpoint = np.where(np.linspace(-1, 1, 256) == 0)[0][0] 
        midpoint = np.abs(np.linspace(-1, 1, 256)).argmin() 
        colors[midpoint] = (0, 0, 0, 1)
        new_colormap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)
        ''' 
        # Define a new color dictionary
        cdict = {
            'red':   ((0.0, 0.0, 0.0),   # Black
                    (0.25, 1.0, 1.0),  # Red
                    (0.5, 1.0, 1.0),   # Yellow (1.0, 1.0, 0.0) -> Red + Green
                    (0.75, 0.0, 0.0),  # Green
                    (1.0, 0.0, 0.0)),  # Blue

            'green': ((0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 1.0, 1.0),   # Yellow
                    (0.75, 1.0, 1.0),  # Green
                    (1.0, 0.0, 0.0)),

            'blue':  ((0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 0.0, 0.0),   # Yellow has no blue component
                    (0.75, 0.0, 0.0),  # Green
                    (1.0, 1.0, 1.0))   # Blue
        }

        custom_cmap = mcolors.LinearSegmentedColormap('custom_colormap', cdict)
        new_colormap = custom_cmap 

        # Normalization
        max_val = np.max(attention_map)
        norm = mcolors.Normalize(vmin=0, vmax=max_val)
        '''
        # Normalization
        max_val = np.max(np.abs(attention_map))
        norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
        ''' 
        # Create a custom colormap
        fig, ax = plt.subplots(figsize=(100, 60)) 
        '''
        colors = [(0, 0, 0)] + [(plt.cm.bwr(i)) for i in range(256)]
        new_colormap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors, N=257)
        new_colormap.set_under('black')  # for values under the min value
        
        # Replace -inf with a value smaller than the minimum of the colormap
        attention_map = np.where(attention_map == -np.inf, -np.finfo(np.float32).max, attention_map)
        ''' 
        # Plotting
        # cbar = ax.imshow(attention_map, norm = norm, cmap=new_colormap, aspect='auto', interpolation='nearest', vmin=-1, vmax=1) 
        cbar = ax.imshow(attention_map, cmap=new_colormap, norm=norm, aspect='auto', interpolation='nearest') 
        ax.imshow(zero_mask, cmap=mcolors.ListedColormap(['none', 'gold']), aspect='auto', interpolation='nearest', alpha=0.5) 
        ax.set_title(f'Attention Map: Layer {layer_num}, Head {head_num}')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Sequence Position')
        ax.set_xticks(range(seq_length))
        ax.set_yticks(range(seq_length)) 
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, ha = "right") 

        plt.colorbar(cbar, orientation = "vertical") 

        # Save to file
        plt.savefig(filename, bbox_inches='tight')
        plt.close() 

# the following function is defined by the llama model and is copied here only for demonstration purposes 
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils.AttentionMaskConverter._prepare_4d_attention_mask"
    )
    # return AttentionMaskConverter._prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)
    return AttentionMaskConverter._expand_mask(mask = mask, dtype = dtype, tgt_len = tgt_len) 


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.llama.modeling_llama.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    ) 

# this function happens after inputs_embeds has been made, so there shouldn't be problem related to condensed tokens 
def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length): 
    combined_attention_mask = None 
    if input_shape[-1] > 1: 
        combined_attention_mask = _make_causal_mask(
            input_shape, 
            inputs_embeds.dtype, 
            device = inputs_embeds.device, 
            past_key_values_length = past_key_values_length, 
        ) 
        # print("combined attention mask shape {}".format(combined_attention_mask.shape)) 
    
    if attention_mask is not None: 

        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len = input_shape[-1]).to( #008000 
            inputs_embeds.device 
        ) 
        # print("expanded_attn_mask has shape {}".format(expanded_attn_mask.shape)) 
        # print("combined_attention_mask has shape {}".format(combined_attention_mask.shape)) 
        # print("expanded attention mask shape {}".format(expanded_attn_mask.shape)) 
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask 
        ) 
    
    return combined_attention_mask 

if __name__ == "__main__": 
    text_prompt = "template: Identify the sentiment of the review: {Positive,Negative}\nReview: <X> Sentiment: <Label> Text: A solid and refined piece of moviemaking imbued with passion and attitude. Label: Positive. Identify the sentiment of the review: {Positive,Negative}\nReview:Identify the sentiment of the review:A solid and refined piece of moviemaking imbued with passion and attitude.Sentiment: " 
    
    # have an example llama2 tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf") 
    if tokenizer.pad_token is not None: 
        print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
    else: 
        tokenizer.pad_token = tokenizer.eos_token 
        print("We now use eos_token as pad token") 
    tokenizer.padding_side = "left" 
    
    inputs = tokenizer(text_prompt, return_tensors = "pt", return_attention_mask = True) 
    inputattentionmask = inputs["attention_mask"] 
    inputattentionmask = inputattentionmask.to(dtype = torch.float32) 
    (batch_size, seq_length) = inputs["input_ids"].shape 
    inputs["input_ids"] = inputs["input_ids"].to(dtype = torch.float32) # just to have a input of torch.float32 type 
    
    past_key_value_length = 0 
    
    attention_mask = _prepare_decoder_attention_mask( 
        inputattentionmask, 
        (batch_size, seq_length), 
        inputs["input_ids"], # we only need the device 
        past_key_value_length, 
    ) 
    
    # attention mask has shape (1, 1, seq_length, seq_length) 
    visualize_attention_mask(seq_length, attention_mask[0][0], "before_modification_prompt.jpg") 
    
    mask_list_pos = list(range(5, seq_length, 7)) 
    print("mask_list_pos {}".format(mask_list_pos)) 
    
    # modify the attention mask 
    _modify_decoder_attention_mask(
        attention_mask, 
        dtype = inputs["input_ids"].dtype, 
        mask_list_pos = mask_list_pos, 
        start_idx = 0, 
    ) 
        
    visualize_attention_mask(seq_length, attention_mask[0][0], "after_modification_prompt.jpg") 
