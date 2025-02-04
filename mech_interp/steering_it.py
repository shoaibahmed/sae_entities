# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

# %%
import sys
sys.path.append("../..")
sys.path.append("../dataset")
sys.path.append("./")
sys.path.append("../")

import os
import json
import torch
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import einops
from typing import List, Tuple, Literal, Union
from torch import Tensor
from colorama import Fore
import textwrap
import copy
import gc
import numpy as np
from functools import partial
from utils.hf_models.model_factory import construct_model_base
from transformer_lens import HookedTransformer, ActivationCache
from dataset.load_data import load_wikidata_queries
from utils.utils import compute_cosim
from mech_interp_utils import load_data
from utils.utils import find_string_in_tokens, slice_to_list
from mech_interp_utils import html_colors
from mech_interp_utils import load_steering_latents, load_latents

from hooks_utils import compute_logit_diff_steered, compute_logit_diff_original
from hooks_utils import steered_and_orig_generations
from utils.utils import model_alias_to_model_name, model_is_chat_model, paper_plot

import random
random_seed = 42
random.seed(random_seed)
# %%
# def load_model(model_alias):
#     model = HookedTransformer.from_pretrained_no_processing(
#                 'gemma-2-2b-it' if model_alias == 'gemma-2-2b' else 'gemma-2-9b-it',
#             )
#     tokenizer = model.tokenizer
#     tokenizer.padding_side = 'left'
#     return model, tokenizer
# %%

model_alias = 'meta-llama/Llama-3.1-8B'#'gemma-2-9b'
#model_alias = 'gemma-2-2b'
prompt_template = 'Are you sure you know the {entity_type} {entity_name}? Answer yes or no.'


conversation =[
      {
        "role": "user",
        "content": ""
      },
      {
        "role": "assistant",
        "content": "Answer:"
      }
    ]


# %%
n_devices = torch.cuda.device_count()

model = HookedTransformer.from_pretrained_no_processing(
    model_alias + '-it' if 'gemma' in model_alias else model_alias + '-Instruct'
)
model.set_use_attn_result(True)
tokenizer = model.tokenizer
tokenizer.padding_side = 'left'
model_alias = model_alias.replace('/','_')

# %%
# conversation =[
#       {
#         "role": "user",
#         "content": ""
#       },
#       {
#         "role": "assistant",
#         "content": "Answer:"
#       }
#     ]
# conversation[0]['content'] = 'Are you sure you know the player LeBron James?'
# tokenizer.apply_chat_template(conversation, tokenize=False)


# %%

tokenized_prompts_dict_entity_type = {}
pos_entities_dict_entity_type = {}
formatted_instructions_dict_entity_type = {}
queries = load_wikidata_queries(model_alias)
for entity_type in ['player', 'movie', 'song', 'city']:
    tokenized_prompts_dict_entity_type[entity_type] = {}
    pos_entities_dict_entity_type[entity_type] = {}
    formatted_instructions_dict_entity_type[entity_type] = {}
    for known_label in ['unknown', 'known']:
        tokenized_prompts_dict_entity_type[entity_type][known_label], pos_entities_dict_entity_type[entity_type][known_label], formatted_instructions_dict_entity_type[entity_type][known_label], _ = load_data(model, queries, entity_type, tokenizer,
                                                                                                                                                                                                                    known_label, prompt_template, conversation,
                                                                                                                                                                                                                    relations_model_type='it')

gc.collect()
torch.cuda.empty_cache()
# YES_TOKEN = model.to_single_token('yes') # 'yes'
# NO_TOKEN = model.to_single_token('no') # 'no'
YES_TOKEN = model.to_single_token(' Yes') # ' Yes'
NO_TOKEN = model.to_single_token(' No') # ' No'

# %%
known_label = 'unknown'
pos_type = 'entity_last'
N=16
max_new_tokens = 10
batch_size = 16
top_latents = {'known': 0, 'unknown': 0}
coeff_values = {'known': 15, 'unknown': 20}
split = 'test'


# %%
known_latent, unknown_latent, random_latents_known, random_latents_unknown = load_latents(model_alias, top_latents,
                                                                                          filter_with_pile=True,
                                                                                          random_n_latents=5)

# %%
entity_type = 'player'
tokenized_prompts, pos_entities, formatted_instructions = tokenized_prompts_dict_entity_type[entity_type][known_label], pos_entities_dict_entity_type[entity_type][known_label], formatted_instructions_dict_entity_type[entity_type][known_label]


original_generations_full, steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type='entity_last',
                                                                                steering_latents=unknown_latent, ablate_latents=None,
                                                                                coeff_value=coeff_values['unknown'], max_new_tokens=max_new_tokens,
                                                                                orig_generations=True, batch_size=batch_size)


for i in range(len(original_generations_full)):
    clean_completion = original_generations_full[i].replace(formatted_instructions[i], '').replace('<bos>', '').replace('<pad>','')
    steered_completion = steered_generations_full[i].replace(formatted_instructions[i], '').replace('<bos>', '').replace('<pad>','')
    print(Fore.WHITE + f"QUESTION:")
    print(textwrap.fill(repr(formatted_instructions[i].replace('<start_of_turn>user\n', '').replace('<end_of_turn>\n<start_of_turn>model','').replace('<bos>','')), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.GREEN + f"ORIGINAL COMPLETION:")
    print(textwrap.fill(repr(clean_completion.replace('<eos>', '').replace('\n<end_of_turn>', '').replace('<end_of_turn>\n<start_of_turn>model\n','')), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RED + f"STEERED COMPLETION:")
    print(textwrap.fill(repr(steered_completion.replace('<eos>', '').replace('\n<end_of_turn>','')), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RESET)


# %%



# %%
# Iterate over topk values
def multi_line_top_k_latents_effect_plot(results_dict, xticks_labels: List[str], title: str, metric: Literal['logit_diff', 'logprob', 'prob'], token_check=None):
    # Custom colors
    colors = [html_colors['blue_matplotlib'], html_colors['orange_matplotlib'], html_colors['brown_D3'], html_colors['grey_drawio']]
    
    fig = go.Figure()

    for i, topk in enumerate(results_dict.keys()):
        logit_diff_results = [logit_diff.cpu().numpy() for logit_diff in results_dict[topk]]
        means = [np.mean(logit_diff) for logit_diff in logit_diff_results]
        errors = [np.std(logit_diff) for logit_diff in logit_diff_results]

        fig.add_trace(go.Scatter(
            x=xticks_labels,
            y=means,
            error_y=dict(type='data', array=errors, visible=True),
            mode='lines+markers',
            name=f'{topk.capitalize()}',
            line=dict(color=colors[i % len(colors)])
        ))
    if metric == 'logit_diff':
        y_title = 'Logit Difference (Yes - No)'
    elif metric == 'logprob':
        y_title = f'Log Probability ({token_check.capitalize()} token)'
    elif metric == 'prob':
        y_title = f'Probability ({token_check.capitalize()} token)'
    # Customize the plot
    fig.update_layout(
        xaxis_title='',
        yaxis_title=y_title,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(xticks_labels))),
            ticktext=xticks_labels,
            ticklabelposition='outside bottom',
            tickfont=dict(size=12),
        ),
        yaxis_title_font=dict(size=16),
        #legend_title_text='Top k Latents',
        title=dict(
            text=title,
            font=dict(size=16),
            y=0.86,  # Lower the title position
        ),
    )

    fig.show()

    return fig
# %%
prompt_template = 'Are you sure you know the {entity_type} {entity_name}? Answer yes or no.'

conversation =[
      {
        "role": "user",
        "content": ""
      },
      {
        "role": "assistant",
        "content": "Answer:"
      }
    ]

known_label = 'unknown'
pos_type = 'entity_last'
N=100
max_new_tokens = 10
batch_size = 8
top_latents = {'known': 0, 'unknown': 0}
coeff_values = {'known': 15, 'unknown': 20}
split = 'test'
metric = 'logit_diff'

# %%
known_latent, unknown_latent, random_latents_known, random_latents_unknown = load_latents(model_alias, top_latents,
                                                                                          filter_with_pile=True,
                                                                                          random_n_latents=5)

# %%
for known_label in ['known','unknown']:
    if known_label == 'known':
        steering_latents = unknown_latent
    elif known_label == 'unknown':
        steering_latents = known_latent
    results_dict = defaultdict(list)
    for entity_type in ['player', 'movie', 'city', 'song']:
        complement_known_label = 'unknown' if known_label == 'known' else 'known'
        token_check: Literal['yes', 'no'] = 'no' if known_label == 'known' else 'yes'
        tokenized_prompts, pos_entities, formatted_instructions = tokenized_prompts_dict_entity_type[entity_type][known_label], pos_entities_dict_entity_type[entity_type][known_label], formatted_instructions_dict_entity_type[entity_type][known_label]
        

        N = min(N, len(tokenized_prompts))
        steered_result_list = []
        rdm_steered_result_list = []
        
        orig_results = compute_logit_diff_original(model, N, tokenized_prompts, metric=metric, batch_size=4)
        
        print('RANDOM STEERING LATENTS')
        rdm_steered_result = compute_logit_diff_steered(model, N, tokenized_prompts, metric, pos_entities, pos_type=pos_type,
                                                        steering_latents=[random_latents_unknown[0]], ablate_latents=None,
                                                        coeff_value=coeff_values[complement_known_label], batch_size=batch_size)
        
        print('STEERING LATENTS')
        steered_result = compute_logit_diff_steered(model, N, tokenized_prompts, metric, pos_entities, pos_type=pos_type,
                                                        steering_latents=steering_latents, ablate_latents=None,
                                                        coeff_value=coeff_values[complement_known_label], batch_size=batch_size)
        
        if metric != 'logit_diff':
            if token_check == 'yes':
                # Do you know known entities? we check the prob of the YES token
                orig_results_ = orig_results[:,0]
                rdm_steered_result = rdm_steered_result[:,0]
                steered_result = steered_result[:,0]
            elif token_check == 'no':
                # Do you know unknown entities? we check the prob of the NO token
                orig_results_ = orig_results[:,1]
                rdm_steered_result = rdm_steered_result[:,1]
                steered_result = steered_result[:,1]
        else:
            orig_results_ = orig_results

        results_dict[entity_type].append(orig_results_)
        results_dict[entity_type].append(rdm_steered_result)
        results_dict[entity_type].append(steered_result)

    fig = multi_line_top_k_latents_effect_plot(results_dict,
                                xticks_labels=['Original', f'Random SAE latent', f'{complement_known_label.capitalize()} SAE latent'],
                                title=f'Steering {known_label.capitalize()} Prompts with {complement_known_label.capitalize()} SAE latent',
                                metric=metric,
                                token_check=token_check)

    os.makedirs('plots/do_you_know', exist_ok=True)

    fig = paper_plot(fig, tickangle=0)
    pio.write_image(fig, f'plots/do_you_know/{model_alias}_from_{known_label}_to_{complement_known_label}_{metric}_{top_latents[complement_known_label]}_coeff{coeff_values[complement_known_label]}_pos{pos_type}.png',
                    scale=5, width=500, height=400)
    pio.write_image(fig, f'plots/do_you_know/{model_alias}_from_{known_label}_to_{complement_known_label}_{metric}_{top_latents[complement_known_label]}_coeff{coeff_values[complement_known_label]}_pos{pos_type}.pdf',
                    scale=5, width=500, height=400)

# %%
