import torch
import einops
import gc
import copy
import plotly.graph_objects as go
from dataset.load_data import load_wikidata_queries
from collections import defaultdict
import numpy as np
from typing import List, Union, Optional, Literal, Tuple
from torch import Tensor
import torch.nn.functional as F
from fancy_einsum import einsum
from jaxtyping import Float
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import transformer_lens.patching as patching
from functools import partial
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from adjustText import adjust_text
from tqdm import tqdm
import json

import random
random_seed = 42
random.seed(random_seed)

### Hooks
def steer_sae_latents(activation, hook, direction, pos, coeff_value=1):
        
        if activation.shape[1]==1:
            # generating
            return activation

        #activation[:, :, :] += norm_res_streams.unsqueeze(-1).unsqueeze(-1)*direction
        if pos != 'all':
            if coeff_value == 'norm':
                # TODO: simplify this!
                if isinstance(pos[0], list):
                    for batch_idx, p in enumerate(pos):
                        norm_res_streams = torch.norm(activation[batch_idx, p, :], dim=-1)
                        activation[batch_idx, p, :] += direction.unsqueeze(0)*norm_res_streams.unsqueeze(-1)
                else:
                    norm_res_streams = torch.norm(activation[:, pos, :], dim=-1)
                    activation[:, pos, :] += direction.unsqueeze(0)*norm_res_streams.unsqueeze(-1)
            else:
                if isinstance(pos[0], list):
                    for batch_idx, p in enumerate(pos):
                        activation[batch_idx, p, :] += direction.unsqueeze(0)*coeff_value
                else:
                    activation[:, pos, :] += direction.unsqueeze(0)*coeff_value
        else:
            # All positions at once
            if coeff_value == 'norm':
                norm_res_streams = torch.norm(activation[:, :, :], dim=-1)
                activation[:, :, :] += direction.unsqueeze(0)*norm_res_streams.unsqueeze(-1)
            else:
                activation[:, :, :] += direction.unsqueeze(0)*coeff_value

        return activation

def ablate_sae_latents(activation, hook, direction, pos):
        # Project away a direction from the residual stream
        if activation.shape[1]==1:
            # generating
            return activation

        if pos != 'all':
            if isinstance(pos[0], list):
                for batch_idx, p in enumerate(pos):
                    activation[batch_idx, p, :] -= (activation[batch_idx, p, :] @ direction.unsqueeze(-1)) * direction
            else:
                activation[:, pos, :] -= (activation[:, pos, :] @ direction.unsqueeze(-1)) * direction
        else:
            # activation -> batch, seq_len, d_model
            # direction -> d_model
            activation[:, :, :] -= (activation[:, :, :] @ direction.unsqueeze(-1)) * direction

        return activation


def generation_steered_latents(model, ids, pos, steering_latents=None, ablate_latents=None, feature_type='latents', coeff_value=1, max_new_tokens=30):
    assert feature_type in ["latents", "hidden"], feature_type
    if feature_type == "hidden":
        raise NotImplementedError

    if steering_latents is not None:
        # TODO: change this
        steer_fwd_hooks = [(f"blocks.{layer}.hook_resid_pre", partial(steer_sae_latents,
                                                pos=pos,
                                                direction=direction,
                                                coeff_value=coeff_value))
                            for layer, latent_idx, mean_act, direction in steering_latents]
    else:
        steer_fwd_hooks = []

    if ablate_latents is not None:
        ablate_fwd_hooks = [(f"blocks.{layer}.hook_resid_pre", partial(ablate_sae_latents,
                                                pos=pos,
                                                direction=direction))
                            for layer, latent_idx, mean_act, direction in ablate_latents]
    else:
        ablate_fwd_hooks = []
    
    for hook_filter, hook_fn in steer_fwd_hooks:
        model.add_hook(hook_filter, hook_fn, "fwd")

    for hook_filter, hook_fn in ablate_fwd_hooks:
        model.add_hook(hook_filter, hook_fn, "fwd")
    generations = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    steered_generations = [model.to_string(generation) for generation in generations]
    return steered_generations

def cache_steered_latents(model, ids, pos, steering_latents: List[Tuple[int, float, Tensor]]=None, ablate_latents: List[Tuple[int, float, Tensor]]=None, coeff_value=1):
        
    if steering_latents is not None:
        steer_fwd_hooks = [(f"blocks.{layer}.hook_resid_pre", partial(steer_sae_latents,
                                                pos=pos,
                                                direction=direction,
                                                coeff_value=mean_act if coeff_value == 'mean' else coeff_value))
                            for layer, latent_idx, mean_act, direction in steering_latents]
    else:
        steer_fwd_hooks = []

    if ablate_latents is not None:
        ablate_fwd_hooks = [(f"blocks.{layer}.hook_resid_pre", partial(ablate_sae_latents,
                                                pos=pos,
                                                direction=direction))
                            for layer, latent_idx, mean_act, direction in ablate_latents]
    else:
        ablate_fwd_hooks = []
    
    for hook_filter, hook_fn in steer_fwd_hooks:
        model.add_hook(hook_filter, hook_fn, "fwd")

    for hook_filter, hook_fn in ablate_fwd_hooks:
        model.add_hook(hook_filter, hook_fn, "fwd")
    output = model.run_with_cache(ids, return_type="logits")
    return output



######
def get_batch_pos(batch_entity_pos, pos_type, tokenized_prompts):
    """
    Computes returns the positions to se for steering in the tokenized prompts.

    Args:
        batch_entity_pos (List[List[int]]): A list of lists where each sublist contains the positions of entity tokens in a prompt.
        pos_type (str): The type of position to compute. It can be one of the following:
            - 'all': Use all positions.
            - 'entity': Use the positions of the entity tokens.
            - 'entity_and_eoi': Use the positions of the entity tokens and the next 6 tokens.
            - 'entity_to_end': Use the positions of the entity tokens and all tokens to the end of the sequence.
            - 'entity_last': Use only the last position of the entity tokens.
            - 'entity_last_to_end': Use the last position of the entity tokens and all tokens to the end of the sequence.
        tokenized_prompts (List[List[int]]): A list of lists where each sublist contains the tokenized prompt.

    Returns:
        List[List[int]]: A list of lists where each sublist contains the computed positions.
    """

    if pos_type == 'all':
        batch_pos = pos_type

    elif pos_type == 'entity':
        batch_pos = batch_entity_pos

    elif pos_type == 'entity_and_eoi':
        # add one extra position to consider "?"
        batch_pos = []
        for pos_ in batch_entity_pos:
            last_pos = pos_[-1]
            for j in range(1,7):
                pos_.append(last_pos+j)
            batch_pos.append(pos_)
    
    elif pos_type == 'entity_to_end':
        batch_pos = []
        for idx, pos_ in enumerate(batch_entity_pos):
            # pos_ starts with the entity tokens, now we add the rest of the tokens to the end
            len_seq = len(tokenized_prompts[idx])
            last_pos = pos_[-1]
            for j in range(1,len_seq-last_pos):
                pos_.append(last_pos+j)
            batch_pos.append(pos_)

    elif pos_type == 'entity_last':
        batch_pos = [[entity_pos_[-1]] for entity_pos_ in batch_entity_pos]

    elif pos_type == 'entity_last_to_end':
        batch_pos = []
        # For each entity position
        for idx, pos_ in enumerate(batch_entity_pos):
            new_pos_ = copy.deepcopy(pos_)
            # We start new positions only from the last entity token
            last_pos = new_pos_[-1]
            new_pos_ = [last_pos]
            len_seq = len(tokenized_prompts[idx])
            for j in range(1,len_seq-last_pos):
                new_pos_.append(last_pos+j)
            batch_pos.append(new_pos_)
    else:
        raise ValueError(f"Invalid pos: {pos_type}")
    

    return batch_pos
            
def steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type: Literal['all', 'entity', 'entity_last', 'entity_and_eoi', 'entity_to_end', 'entity_last_to_end']='all',
                                 steering_latents=None, ablate_latents=None, feature_type='latents', coeff_value=100, max_new_tokens=30, orig_generations=True, batch_size=4):

    original_generations_full = []
    steered_generations_full = []
    N = min(N, len(tokenized_prompts))
    tokenized_prompts = tokenized_prompts[:N]
    pos_entities = pos_entities[:N]
    for i in range(0, len(tokenized_prompts), batch_size):
        batch_tokenized_prompts = tokenized_prompts[i:i+batch_size]
        batch_entity_pos = copy.deepcopy(pos_entities[i:i+batch_size])
        model.reset_hooks()

        if orig_generations == True:
            generations = model.generate(batch_tokenized_prompts, max_new_tokens=max_new_tokens, do_sample=False)
            original_generations_full.extend([model.to_string(generation) for generation in generations])

        batch_pos = get_batch_pos(batch_entity_pos, pos_type, batch_tokenized_prompts)

        if i == 0 and pos_type != 'all':
            for batch_idx in range(len(batch_tokenized_prompts)):
                print('Example of tokens we are steering on:', model.to_str_tokens(batch_tokenized_prompts[batch_idx, batch_pos[batch_idx]]))
                
        model.reset_hooks()
        
        steered_generations = generation_steered_latents(model, batch_tokenized_prompts, pos=batch_pos,
                                            steering_latents=steering_latents,
                                            ablate_latents=ablate_latents,
                                            feature_type=feature_type,
                                            coeff_value=coeff_value,
                                            max_new_tokens=max_new_tokens)
        steered_generations_full.extend(steered_generations)

        torch.cuda.empty_cache()
    if orig_generations == True:
        return original_generations_full, steered_generations_full
    else:
        return None, steered_generations_full

# Logit diff analysis
def compute_metric(model, original_logits, metric: Literal['logit_diff', 'logprob', 'prob']='logit_diff'):

    YES_TOKEN = model.to_single_token(' Yes')
    NO_TOKEN = model.to_single_token(' No')

    if metric == 'logit_diff':
        logit_diff = original_logits[:, -1, YES_TOKEN].detach() - original_logits[:, -1, NO_TOKEN].detach()
        return logit_diff
    elif metric == 'logprob':
        logprob = F.log_softmax(original_logits[:, -1], dim=-1)[:, [YES_TOKEN, NO_TOKEN]].detach()
        return logprob
    elif metric == 'prob':
        prob = F.softmax(original_logits[:, -1], dim=-1)[:, [YES_TOKEN, NO_TOKEN]].detach()
        return prob
    
def compute_logit_diff_original(model, N, tokenized_prompts, metric: Literal['logit_diff', 'logprob', 'prob'], batch_size=4):
    # Logit diff with original inputs (no steering nor ablations)
    
    model.reset_hooks()
    result_metric_full = []
    for i in range(0, min(N, len(tokenized_prompts)), batch_size):
        batch_tokenized_prompts = tokenized_prompts[i:i+batch_size]
        
        original_logits, original_cache = model.run_with_cache(batch_tokenized_prompts, return_type="logits")
        del original_cache

        result_metric_full.append(compute_metric(model, original_logits, metric=metric))

    return torch.cat(result_metric_full)


def compute_logit_diff_steered(model, N, tokenized_prompts, metric: Literal['logit_diff', 'logprob', 'prob'], pos_entities, pos_type: Literal['all', 'entity', 'entity_last', 'entity_and_eoi']='all', steering_latents=None, ablate_latents=None, coeff_value=Union[Literal['norm', 'mean'], int], batch_size=4):
    # Logit diff with steered (or ablated) latents

    steered_logit_diff_full = []
    N = min(N, len(tokenized_prompts))
    tokenized_prompts = tokenized_prompts[:N]
    pos_entities = pos_entities[:N]
    for i in range(0, len(tokenized_prompts), batch_size):
        batch_tokenized_prompts = tokenized_prompts[i:i+batch_size]
        batch_entity_pos = copy.deepcopy(pos_entities[i:i+batch_size])

        batch_pos = get_batch_pos(batch_entity_pos, pos_type, batch_tokenized_prompts)

        if i == 0:
            print('Example of tokens we are steering on:', model.to_str_tokens(batch_tokenized_prompts[0,batch_pos[0]]))
        
        model.reset_hooks()
        steered_logits, steered_cache = cache_steered_latents(model, batch_tokenized_prompts, pos=batch_pos,
                                                steering_latents=steering_latents,
                                                ablate_latents=ablate_latents,
                                                coeff_value=coeff_value)
        
        steered_logit_diff_full.append(compute_metric(model, steered_logits, metric=metric))
        del steered_logits,steered_cache

        torch.cuda.empty_cache()
    
    return torch.cat(steered_logit_diff_full)