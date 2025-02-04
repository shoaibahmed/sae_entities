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
import matplotlib.pyplot as plt
import einops
from typing import List, Tuple, Literal, Union
from torch import Tensor
from colorama import Fore
import textwrap
import gc
import copy
import torch.nn.functional as F
import numpy as np
from functools import partial
from utils.hf_models.model_factory import construct_model_base
from utils.utils import model_alias_to_model_name
from transformer_lens import HookedTransformer, ActivationCache
from utils.sae_utils import load_sae
from mech_interp_utils import load_data, load_steering_latents
from mech_interp_utils import html_colors, load_latents
from utils.generation_utils import is_unknown, is_generation_refusal
from hooks_utils import steered_and_orig_generations
from utils.utils import tl_orthogonalize_gemma_weights, paper_plot
from dataset.load_data import load_wikidata_queries
import random
random_seed = 42
random.seed(random_seed)
# %%
def run_generations(model, N, tokenized_prompts, max_new_tokens, batch_size=8):
    original_generations_full = []
    N = min(N, len(tokenized_prompts))
    tokenized_prompts = tokenized_prompts[:N]
    model.reset_hooks()
    for i in range(0, len(tokenized_prompts), batch_size):
        batch_tokenized_prompts = tokenized_prompts[i:i+batch_size]
        generations = model.generate(batch_tokenized_prompts, max_new_tokens=max_new_tokens, do_sample=False)
        original_generations_full.extend([model.to_string(generation) for generation in generations])
    torch.cuda.empty_cache()
    return original_generations_full

def plot_counter_refusal(counter_refusal, save, **kwargs):
    counter_refusal_category_to_plot_label = {'original': 'Original generation', 'steered_known': 'Steering known latent',
                                           'steered_unknown': 'Steering unknown latent', 'orthogonalized_unknown': 'Orthogonalized model\nUnknown latent',
                                           'steered_known_random': 'Random steering\nKnown latent setting', 'steered_unknown_random': 'Random steering\nUnknown latent setting'}
    
    colors_dict = {'original': html_colors['blue_matplotlib'], 'steered_known': html_colors['dark_green_drawio'],
                   'steered_unknown': html_colors['brown_D3'], 'orthogonalized_unknown': html_colors['orange_drawio'],
                   'steered_known_random': html_colors['dark_green_drawio'], 'steered_unknown_random': html_colors['brown_D3']}
    
    hatch_dict = {'original': None, 'steered_known': None, 'steered_unknown': None,
                  'orthogonalized_unknown': '///', 'steered_known_random': '..', 'steered_unknown_random': '..'}
    
    entity_types = list(counter_refusal.keys())

    categories = list(counter_refusal[entity_types[0]].keys())

    category_counts = {}
    for category in categories:
        category_counts[category] = [counter_refusal[et][category] for et in entity_types]

    x = np.arange(len(entity_types))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6.5, 2.75), dpi=500)
    
    bar_positions = np.arange(len(entity_types)) * 1.5  # Increase spacing between groups
    width = 0.2  # Reduce bar width

    # Calculate offsets based on number of categories
    num_categories = len(categories)
    offsets = np.linspace(-(num_categories-1)*width/2, (num_categories-1)*width/2, num_categories)

    for cat_idx, category in enumerate(categories):
        yerrs = []
        values = []
        for i in range(len(category_counts[category])):
            if isinstance(category_counts[category][i], list):
                yerrs.append(np.array(category_counts[category][i]).std())
                values.append(np.array(category_counts[category][i]).mean())
            else:
                yerrs.append(0)
                values.append(category_counts[category][i])
        ax.bar(bar_positions + offsets[cat_idx], values, width, yerr=yerrs, label=counter_refusal_category_to_plot_label[category],
               color=colors_dict[category], alpha=1, edgecolor='black', linewidth=1, hatch=hatch_dict[category])


    ax.set_ylabel('Refusal Rate')
    #ax.set_title('Refusal Counts by Entity Type: Original vs Orthogonalized')
    ax.set_xticks(bar_positions)
    entity_types = [entity.capitalize() for entity in entity_types]
    ax.set_xticklabels(entity_types, rotation=0)
    #ax.legend()
    # Adjust the x positions to spread the bars more
    
    
    # Add light grid lines
    ax.grid(True, linestyle=(0, (5, 10)))
    ax.set_axisbelow(True)

    # Set y-axis to show only integers
    y_max = 101#max(max(steered_unknown_counts), max(steered_unknown_counts))
    ax.set_yticks(range(int(y_max) + 1))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Set y-axis ticks to be smaller
    ax.tick_params(axis='y', which='major', labelsize=8)
    # Make legend transparent
    #legend = ax.legend(framealpha=0.3)
    #ax.legend(loc='center left')
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")


    #plt.setp(legend.get_texts(), fontsize=8)  # Reduce font size of legend text

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.ylim(0,100)
    os.makedirs('./plots/refusal_analysis', exist_ok=True)
    if save == True:
        plt.savefig(f"./plots/refusal_analysis/v2_{kwargs['model_alias']}_k{kwargs['top_latents']['known']}_u{kwargs['top_latents']['unknown']}_{kwargs['pos_type']}_k{kwargs['coeff_values']['known']}_u{kwargs['coeff_values']['unknown']}_{kwargs['split']}_{kwargs['known_label']}.png", dpi=500, transparent=True)
        plt.savefig(f"./plots/refusal_analysis/v2_{kwargs['model_alias']}_k{kwargs['top_latents']['known']}_u{kwargs['top_latents']['unknown']}_{kwargs['pos_type']}_k{kwargs['coeff_values']['known']}_u{kwargs['coeff_values']['unknown']}_{kwargs['split']}_{kwargs['known_label']}.pdf", transparent=True)

def load_model(model_alias):
    model_alias = model_alias.replace('_','/')
    model_to_load = model_alias+'-it' if 'gemma' in model_alias.lower() else model_alias+'-Instruct'
    model = HookedTransformer.from_pretrained_no_processing(
                model_to_load,
            )
    tokenizer = model.tokenizer
    tokenizer.padding_side = 'left'
    return model, tokenizer



def count_refusals(generations):
    counter_refusals = 0
    for generation in generations:
        pos_end_of_instruction = generation.find(end_of_instruction_string)
        generation = generation[pos_end_of_instruction+len(end_of_instruction_string):]
        if is_generation_refusal(generation) == True:
            counter_refusals += 1
    return counter_refusals
##### Experiments original model and steering
# %%
# # Load model
# model_alias = 'gemma-2-2b'
# model = HookedTransformer.from_pretrained_no_processing(
#     'gemma-2-2b-it',
# )
# #model.set_use_attn_result(False)
# tokenizer = model.tokenizer
# tokenizer.padding_side = 'left'

# %%
model_alias = 'gemma-2-2b'

prompt_template = '{relation} the {entity_type} {entity_name}?'

conversation =[
      {
        "role": "user",
        "content": ""
      },
      {
        "role": "assistant",
        "content": ""
      }
    ]

end_of_instruction_string = '<end_of_turn>\n<start_of_turn>model\n'

# %%
model, tokenizer = load_model(model_alias)
model_alias = model_alias.replace('/','_')
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

del model
gc.collect()
torch.cuda.empty_cache()

# %%

known_label = 'unknown'
pos_type = 'entity_last_to_end'
N=100
max_new_tokens = 30
batch_size = 100
top_latents = {'known': 0, 'unknown': 0}
coeff_values = {'known': 15, 'unknown': 20}
split = 'test'

categories = ['original', 'steered_unknown', 'steered_unknown_random', 'orthogonalized_unknown', 'steered_known', 'steered_known_random']
# for idx in range(1,5):
known_latent, unknown_latent, random_latents_known, random_latents_unknown = load_latents(model_alias, top_latents,
                                                                                          random_n_latents=10,
                                                                                          filter_with_pile=True)
print(unknown_latent)
print(known_latent)
counter_refusal = {}
params_args = {'model_alias': model_alias, 'coeff_values': coeff_values, 'split': split, 'top_latents': top_latents, 'pos_type': pos_type, 'known_label': known_label}

for e_idx, entity_type in enumerate(['player', 'city', 'movie', 'song']):
    counter_refusal[entity_type] = {}
    for category in categories:
        counter_refusal[entity_type][category] = 0

    if e_idx > 0:
        if 'orthogonalized_unknown' in categories or 'orthogonalized_known' in categories:
            # Load model again if orthogonalized (and deleted) before
            model, tokenizer = load_model(model_alias)
    else:
        # Always load model first thing
        model, tokenizer = load_model(model_alias)
    

    tokenized_prompts, pos_entities, formatted_instructions = tokenized_prompts_dict_entity_type[entity_type][known_label], pos_entities_dict_entity_type[entity_type][known_label], formatted_instructions_dict_entity_type[entity_type][known_label]

    if 'original' in categories:
        original_generations_full, known_steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type=pos_type,
                                                                                        steering_latents=known_latent, ablate_latents=None,
                                                                                        coeff_value=coeff_values['known'], max_new_tokens=max_new_tokens, batch_size=batch_size)

        counter_refusal[entity_type]['original'] = count_refusals(original_generations_full)
        counter_refusal[entity_type]['steered_known'] = count_refusals(known_steered_generations_full)

    if 'steered_unknown' in categories:
        _, unknown_steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type=pos_type,
                                                                                        steering_latents=unknown_latent, ablate_latents=None,
                                                                                        coeff_value=coeff_values['unknown'], max_new_tokens=max_new_tokens,
                                                                                        orig_generations=False, batch_size=batch_size)
        
        counter_refusal[entity_type]['steered_unknown'] = count_refusals(unknown_steered_generations_full)

    if 'steered_known_random' in categories:
        random_latents_counter = []
        for random_latent in random_latents_known:
            _, random_steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type=pos_type,
                                                                                        steering_latents=[random_latent], ablate_latents=None,
                                                                                        coeff_value=coeff_values['known'], max_new_tokens=max_new_tokens,
                                                                                        orig_generations=False, batch_size=batch_size)
            
            random_latents_counter.append(count_refusals(random_steered_generations_full))

        counter_refusal[entity_type]['steered_known_random'] = random_latents_counter

    if 'steered_unknown_random' in categories:
        random_latents_counter = []
        for random_latent in random_latents_unknown:
            _, random_steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type=pos_type,
                                                                                    steering_latents=[random_latent], ablate_latents=None,
                                                                                    coeff_value=coeff_values['unknown'], max_new_tokens=max_new_tokens,
                                                                                    orig_generations=False, batch_size=batch_size)
            random_latents_counter.append(count_refusals(random_steered_generations_full))

        counter_refusal[entity_type]['steered_unknown_random'] = random_latents_counter
    
    if 'orthogonalized_unknown' in categories or 'orthogonalized_known' in categories:
        for ortho_idx, orhogonalization_type in enumerate(['unknown']):
            if orhogonalization_type == 'unknown':
                direction = unknown_latent[0][-1]
            else:
                direction = known_latent[0][-1]
            if ortho_idx == 1:
                # Load model again
                model, tokenizer = load_model(model_alias)
            
            tl_orthogonalize_gemma_weights(model, direction=direction)

            # Run generations with orthogonalized model
            orthogonalized_generations_full = run_generations(model, N, tokenized_prompts, max_new_tokens, batch_size)
            torch.cuda.empty_cache()

            counter_refusal[entity_type][f'orthogonalized_{orhogonalization_type}'] = count_refusals(orthogonalized_generations_full)

            del model
        gc.collect()
        torch.cuda.empty_cache()

    print('counter_refusal', counter_refusal)
    # del model
    # gc.collect()
    # torch.cuda.empty_cache()

# Save counter_refusal results
save_path = f'results/refusal_analysis/{model_alias}'
os.makedirs(save_path, exist_ok=True)
with open(os.path.join(save_path, f'counter_refusal_{entity_type}_{known_label}.json'), 'w') as f:
    json.dump(counter_refusal, f)

plot_counter_refusal(counter_refusal, save=True,**params_args)




# %%
### Test ###
# %%
model, tokenizer = load_model(model_alias)

# %%
entity_type = 'city'
known_label = 'unknown'
N = 100
max_new_tokens = 30
batch_size = 50
top_latents = {'known': 0, 'unknown': 0}
coeff_values = {'known': 20, 'unknown': 20}
model_alias = model_alias.replace('/','_')
# %%
known_latent, unknown_latent, random_latents_known, random_latents_unknown = load_latents(model_alias, top_latents,
                                                                                          filter_with_pile=True,
                                                                                          random_n_latents=10)
# %%
# unknown_latent = load_steering_latents('movie', label='unknown', topk=1,
#                                         #layers_range=[unknown_latent[0]],
#                                         specific_latents=[(16,9583)],
#                                         model_alias=model_alias,
#                                         random_latents=False)


# %%
tokenized_prompts, pos_entities, formatted_instructions = tokenized_prompts_dict_entity_type[entity_type][known_label], pos_entities_dict_entity_type[entity_type][known_label], formatted_instructions_dict_entity_type[entity_type][known_label]

counter_refusals = {}


original_generations_full, steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type='entity_last_to_end',
                                                                                steering_latents=[random_latents_known[-2]], ablate_latents=None,
                                                                                coeff_value=coeff_values['known'], max_new_tokens=max_new_tokens,
                                                                                orig_generations=True, batch_size=batch_size)

counter_refusals['original'] = count_refusals(original_generations_full)
counter_refusals['steered'] = count_refusals(steered_generations_full)


# counter_refusals['steered_known_random'] = []
# counter_refusals['steered_unknown_random'] = []
# for random_latent in random_latents_known:
#     _, random_steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type='entity_last_to_end',
#                                                                                 steering_latents=[random_latent], ablate_latents=None,
#                                                                                 coeff_value=coeff_values['known'], max_new_tokens=max_new_tokens,
#                                                                                 orig_generations=False, batch_size=batch_size)

#     counter_refusals['steered_known_random'].append(count_refusals(random_steered_generations_full))

# for random_latent in random_latents_unknown:
#     _, random_steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type='entity_last_to_end',
#                                                                                 steering_latents=[random_latent], ablate_latents=None,
#                                                                                 coeff_value=coeff_values['unknown'], max_new_tokens=max_new_tokens,
#                                                                                 orig_generations=False, batch_size=batch_size)

#     counter_refusals['steered_unknown_random'].append(count_refusals(random_steered_generations_full))

for i in range(len(original_generations_full)):
    clean_completion = original_generations_full[i].replace(formatted_instructions[i], '').replace('<bos>', '').replace('<pad>','')
    steered_completion = steered_generations_full[i].replace(formatted_instructions[i], '').replace('<bos>', '').replace('<pad>','')
    print(Fore.WHITE + f"QUESTION:")
    print(textwrap.fill(repr(formatted_instructions[i].replace('<start_of_turn>user\n', '').replace('<end_of_turn>\n<start_of_turn>model','').replace('<bos>','')), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.GREEN + f"ORIGINAL COMPLETION:")
    print(textwrap.fill(repr(clean_completion.replace('<eos>', '').replace('\n<end_of_turn>', '').replace('<end_of_turn>\n<start_of_turn>model\n','')), width=100, initial_indent='\t', subsequent_indent='\t'))
    print('is refusal:', is_generation_refusal(clean_completion))
    #counter_refusals['original'] += is_generation_refusal(clean_completion)
    print(Fore.RED + f"STEERED COMPLETION:")
    print(textwrap.fill(repr(steered_completion.replace('<eos>', '').replace('\n<end_of_turn>','')), width=100, initial_indent='\t', subsequent_indent='\t'))
    print('is refusal:', is_generation_refusal(steered_completion))
    #counter_refusals['steered'] += is_generation_refusal(steered_completion)
    print(Fore.RESET)


counter_refusals
# %%
