# Activation Patching Analysis

# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')
# %%
import sys
import os
sys.path.append("../..")
sys.path.append("./")
sys.path.append("../")

import torch
from collections import defaultdict
import plotly.graph_objects as go
import plotly.io as pio
import gc
import einops
from neel_plotly import line, imshow, scatter
import numpy as np
from functools import partial
from transformer_lens import HookedTransformer
from mech_interp_utils import entity_type_to_token_pos, load_data
from mech_interp_utils import compute_patching, compute_metric, entity_type_to_base_relations
from dataset.load_data import load_wikidata_queries
import random
random.seed(42)

# %%
#model_alias = 'gemma-2-9b'
model_alias = 'gemma-2-2b'
model = HookedTransformer.from_pretrained_no_processing(
    model_alias,
)
tokenizer = model.tokenizer
model_alias = model_alias.replace('/','_')

# %%
entity_type = 'movie'
tokenized_prompts_dict = {}
answers_dict = {}
queries = load_wikidata_queries(model_alias)
for known_label in ['known', 'unknown']:
    prompt_template = 'The {entity_type} {entity_name} {relation}'
    tokenized_prompts, pos_entities, formatted_instructions, answers_label = load_data(model, queries, entity_type, tokenizer,
                                                                            known_label, prompt_template, conversation=None,
                                                                            relations_model_type='base',
                                                                            fixed_config='both')
    
    tokenized_prompts_dict[known_label] = tokenized_prompts
    answers_dict[known_label] = answers_label

# %%
# Get all relation types
relation_types = []
for prompt_type in entity_type_to_base_relations.keys():
    if entity_type in prompt_type:
        relation_types.append(entity_type_to_base_relations[prompt_type])

# Order by relation type, corrupted and clean have to have the same relation type
final_tokenized_prompts_dict = defaultdict(list)
final_answers_dict = defaultdict(list)

taken_known_ids = []
for tok_prompt, answer in zip(tokenized_prompts_dict['unknown'], answers_dict['unknown']):
    for relation in relation_types:
        if relation in tokenizer.decode(tok_prompt):
            for known_idx, (tok_prompt_known, answer_known) in enumerate(zip(tokenized_prompts_dict['known'], answers_dict['known'])):
                if relation in tokenizer.decode(tok_prompt_known) and known_idx not in taken_known_ids:
                    final_tokenized_prompts_dict['unknown'].append(tok_prompt)
                    final_tokenized_prompts_dict['known'].append(tok_prompt_known)
                    final_answers_dict['unknown'].append(answer)
                    final_answers_dict['known'].append(answer_known)
                    taken_known_ids.append(known_idx)
                    break

# %%
# Get base, src, answer tokens
patching_type = 'heads_last_pos'
N = 32
batch_size = 32
base_tokens = torch.stack(final_tokenized_prompts_dict['known'][:N])
src_tokens = torch.stack(final_tokenized_prompts_dict['unknown'][:N])
answer_token_indices = torch.tensor([[model.to_single_token(model.to_str_tokens(final_answers_dict['known'][i], prepend_bos=False)[0]), model.to_single_token(model.to_str_tokens(final_answers_dict['unknown'][i], prepend_bos=False)[0])] for i in range(min(N, len(final_answers_dict['known'])))], device=model.cfg.device)

# %%
# Get baseline logit diffs over dataset to define metric
metric_fn = compute_metric(model, base_tokens, src_tokens, answer_token_indices, batch_size=batch_size)

# %%
patch_results = compute_patching(model, base_tokens, src_tokens, patching_type, answer_token_indices, metric_fn, batch_size=16)

# %%
y_labels = [f'{str(layer)}' for layer in range(model.cfg.n_layers-1,-1,-1)]
entity_token_pos = entity_type_to_token_pos[entity_type]
entity_toks = [f'entity_tok {j+1}' for j in range(entity_token_pos - 2)]
relation_toks = [f'relation_tok {j+1}' for j in range(3)]

patching_plot_sentence = ['<bos>', 'The', f'{entity_type}'] + entity_toks + relation_toks

# %%
if patching_type=='resid_streams':
    fig = imshow(torch.flip(patch_results, dims=[0]), 
            yaxis="Layer", 
            xaxis="Position",
            x=[f"{tok} {i}" for i, tok in enumerate(patching_plot_sentence)],
            y=y_labels,
            title=f"Residual Stream Activation Patching ({entity_type})",
            return_fig=True)
    fig.update_xaxes(tickangle=45)
    width = 500
#every_block_result = patching.get_act_patch_block_every(model, src_tokens, base_cache, metric)
elif patching_type=='full':
    fig = imshow(torch.flip(patch_results, dims=[1]), facet_col=0,
                    y=y_labels,
                    facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
                    title=f"Activation Patching Per Block ({entity_type})", xaxis="Position", yaxis="Layer",
                    zmax=1, zmin=-1, x= patching_plot_sentence,
                    return_fig=True
                    )
    width = 800
    fig.update_xaxes(tickangle=45)
elif patching_type == 'heads_last_pos':
    fig = imshow(torch.flip(patch_results, dims=[0]), 
                    yaxis="Layer", 
                    xaxis="Head",
                    x=[f'{head}' for head in range(model.cfg.n_heads)],
                    y=y_labels,
                    title=f"Attn Head Output (Last Pos) ({entity_type})",
                    return_fig=True)
    width=350
elif patching_type == 'heads_all_pos':
    fig = imshow(torch.flip(patch_results, dims=[0]), 
    yaxis="Layer", 
    xaxis="Head",
    x=[f'{head}' for head in range(model.cfg.n_heads)],
    y=y_labels,
    title=f"Attn Head Output (All Pos) ({entity_type})",
    return_fig=True)
    width=350

fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

fig.show()
os.makedirs('./plots/patching_results', exist_ok=True)
pio.write_image(fig, f'./plots/patching_results/{model_alias}_{entity_type}_{patching_type}.png',scale=5, width=width, height=500)

# %%
