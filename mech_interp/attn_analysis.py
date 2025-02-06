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

import json
import torch
import os
from collections import defaultdict
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import einops
import copy
import numpy as np
from scipy import stats
import einops
from fancy_einsum import einsum
from typing import List, Union, Optional, Tuple, Literal
from torch import Tensor
from jaxtyping import Float
import torch.nn.functional as F
#from circuitsvis.attention import attention_heads
from neel_plotly import line, imshow, scatter
from mech_interp_utils import get_attn_to_entity_tok_mean, get_valid_entities, create_prompts_and_answers, entity_type_to_token_pos
from mech_interp_utils import load_steering_latents, visualize_attention_patterns, plot_heads_scatter_entity_types
from dataset.load_data import load_wikidata_queries
import random
random.seed(42)
from transformer_lens import HookedTransformer
from utils.utils import find_string_in_tokens, slice_to_list
from mech_interp_utils import plot_heads_boxplot, plot_heads_boxplot_entity_types, html_colors, load_latents
from hooks_utils import cache_steered_latents, get_batch_pos

# %%
# Attn analysis
def compute_attn_original(model, N, tokenized_prompts, pos_entities, pos_type:Literal['all', 'entity', 'entity_last', 'entity_and_eoi', 'entity_to_end', 'entity_last_to_end']='all', type_pattern: Literal['attn_weights', 'value_weighted']='attn_weights', batch_size=4):

    model.reset_hooks()
    N = min(N, len(tokenized_prompts))
    tokenized_prompts = tokenized_prompts[:N]
    pos_entities = pos_entities[:N]
    mean_orig_attn_list = []
    for i in range(0, len(tokenized_prompts), batch_size):
        batch_tokenized_prompts = tokenized_prompts[i:i+batch_size]
        batch_entity_pos = copy.deepcopy(pos_entities[i:i+batch_size])

        batch_pos = get_batch_pos(batch_entity_pos, pos_type, batch_tokenized_prompts)
        
        original_logits, original_cache = model.run_with_cache(batch_tokenized_prompts, return_type="logits")
        del original_logits

        clean_mean_attn = get_attn_to_entity_tok_mean(model, original_cache, batch_pos, model.cfg.n_layers, type_pattern=type_pattern)

        mean_orig_attn_list.append(clean_mean_attn)

        del original_cache

    return torch.cat(mean_orig_attn_list, 0)


def compute_attn_steered(model, N, tokenized_prompts, pos_entities, pos_type: Literal['all', 'entity', 'entity_last', 'entity_and_eoi', 'entity_to_end', 'entity_last_to_end']='all', steering_latents=None, ablate_latents=None, coeff_value=Union[Literal['norm', 'mean'], int], type_pattern: Literal['attn_weights', 'value_weighted']='attn_weights', batch_size=4):

    mean_steered_attn_list = []
    N = min(N, len(tokenized_prompts))
    tokenized_prompts = tokenized_prompts[:N]
    pos_entities = pos_entities[:N]
    for i in range(0, len(tokenized_prompts), batch_size):
        batch_tokenized_prompts = tokenized_prompts[i:i+batch_size]
        batch_entity_pos = copy.deepcopy(pos_entities[i:i+batch_size])

        batch_pos = get_batch_pos(batch_entity_pos, pos_type, batch_tokenized_prompts)

        # if i == 0 and pos_type != 'all':
        #     for batch_idx in range(len(batch_tokenized_prompts)):
        #         print('Example of tokens we are steering on:', model.to_str_tokens(batch_tokenized_prompts[batch_idx, batch_pos[batch_idx]]))
        
        model.reset_hooks()
        steered_logits, steered_cache = cache_steered_latents(model, batch_tokenized_prompts, pos=batch_pos,
                                                steering_latents=steering_latents,
                                                ablate_latents=ablate_latents,
                                                coeff_value=coeff_value)
        

        del steered_logits

        steered_mean_attn = get_attn_to_entity_tok_mean(model, steered_cache, batch_pos, model.cfg.n_layers, type_pattern=type_pattern)
        mean_steered_attn_list.append(steered_mean_attn)

        torch.cuda.empty_cache()
        del steered_cache
    
    return torch.cat(mean_steered_attn_list, 0)

def load_data(model_alias, entity_type, tokenizer, split='test'):

    prompt_template = 'The {entity_type} {entity_name} {relation}'
    entity_token_pos = entity_type_to_token_pos[entity_type]
    queries = load_wikidata_queries(model_alias)
    valid_entities = get_valid_entities(queries, tokenizer, entity_type, entity_token_pos, split=split, fixed_length=False)
    # Tokenize prompts and get entities positions
    pos_entities_dict = {}
    tokenized_prompts_dict = {}
    for known_label in ['known', 'unknown']:
        prompts_label, _, entities_label = create_prompts_and_answers(tokenizer, queries, entity_type, known_label, valid_entities,
                                                                                prompt_template, relations_model_type='base')

        tokenized_prompts_dict[known_label] = model.to_tokens(prompts_label)
        pos_entities_dict[known_label] = []
        for i in range(len(tokenized_prompts_dict[known_label])):
            entity = entities_label[i]
            if 'Lou ! Journal infime' in entity:
                entity = entity.replace('Lou ! Journal infime', 'Lou! Journal infime')
            slice_entity_pos = find_string_in_tokens(entity, tokenized_prompts_dict[known_label][i], tokenizer)
            # TODO: this shouldn't be necessary
            list_entity_pos = slice_to_list(slice_entity_pos)
            pos_entities_dict[known_label].append(list_entity_pos)

    return tokenized_prompts_dict, pos_entities_dict, entity_token_pos

def plot_head_differences(values, heads, width=20):
    # Calculate mean and std for each (layer, head) combination
    means = []
    stds = []
    labels = []
    y_max, y_min = 0, 0
    for layer, head in heads:
        mean_value = np.mean(values[:, layer, head])
        std_value = np.std(values[:, layer, head])
        means.append(mean_value)
        stds.append(std_value)
        labels.append(f"L{layer}H{head}")
        y_max = max(y_max, mean_value + std_value)
        y_min = min(y_min, mean_value - std_value)
    
    y_max = max(abs(y_max), abs(y_min)) + 0.02
    
    # Create the bar plot
    plt.figure(figsize=(width, 6))
    x = np.arange(len(heads))
    bars = plt.bar(x, means, yerr=stds, capsize=5)

    
    # Customize the plot
    plt.xlabel('Head')
    plt.ylabel('Attention Score Difference')
    #plt.title('Differences Across Layer-Head Combinations with Standard Deviation')

    # Show only every 5th tick
    tick_positions = x[::5]
    tick_labels = [labels[i] for i in range(0, len(labels), 5)]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    #plt.xticks(x, labels, rotation=45)
    # Make x and y axis labels and ticks bigger
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=13)
    plt.xlabel('Head', fontsize=15)
    plt.ylabel('Attention Score Difference', fontsize=15)

    plt.xlim(-0.7, len(heads) - 0.7)
    plt.ylim(-y_max, y_max)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    # Add grid for better readability
    plt.grid(True, axis='y', alpha=0.3)

    # Make background transparent
    plt.gca().set_facecolor('none')
    plt.gcf().set_facecolor('none')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    
    return plt.gcf()


def compare_means(sample1, sample2, alpha=0.05):
    """
    Performs a one-tailed Welch's t-test to determine if sample1's mean is 
    significantly larger than sample2's mean.
    
    Parameters:
    sample1, sample2: arrays or lists of numerical values
    alpha: significance level (default 0.05)
    
    Returns:
    dict containing test results and descriptive statistics
    """
    # Calculate descriptive statistics
    stats_dict = {
        'mean1': np.mean(sample1),
        'mean2': np.mean(sample2),
        'std1': np.std(sample1, ddof=1),
        'std2': np.std(sample2, ddof=1),
        'n1': len(sample1),
        'n2': len(sample2)
    }
    
    # Perform Welch's t-test (not assuming equal variances)
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)
    
    # Convert to one-tailed p-value
    # If t-stat is positive (mean1 > mean2), we use p_value/2
    # If t-stat is negative (mean1 < mean2), we use 1 - p_value/2
    one_tailed_p = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)
    
    # Add test results to dictionary
    stats_dict.update({
        't_statistic': t_stat,
        'p_value': one_tailed_p,
        'significant': one_tailed_p < alpha
    })
    
    return stats_dict

def test_mean_difference(data, alternative=Literal['greater', 'less', 'two-sided'], alpha=0.05):
    """
    Performs a one-sample t-test to determine if the mean is different from zero.
    
    Parameters:
    data (array-like): Sample data to test
    alternative (str): Type of test to perform: 'two-sided', 'greater', or 'less'
    alpha (float): Significance level, default is 0.05
    
    Returns:
    dict: Dictionary containing test results including:
        - t_statistic: The t-statistic
        - p_value: The p-value
        - mean: Sample mean
        - ci: Confidence interval
        - significant: Boolean indicating if result is significant
        - effect_size: Cohen's d effect size
        - test_type: String indicating the type of test performed
    """
    # Perform t-test
    t_stat, p_val = stats.ttest_1samp(data, popmean=0, alternative=alternative)
    
    # Calculate effect size (Cohen's d)
    effect_size = np.mean(data) / np.std(data, ddof=1)
    
    results = {
        't_statistic': t_stat,
        'p_value': p_val,
        'mean': np.mean(data),
        'significant': p_val < alpha,
        'effect_size': effect_size,
        'test_type': alternative
    }
    
    return results

def compute_attn_original_vs_steered(known_label, steering_latent, pos_type, type_pattern, batch_size,
                                     tokenized_prompts_dict_entity_type, pos_entities_dict_entity_type,
                                     scatter_plot=False, **scatter_plot_kwargs):
    mean_attn_dict = {}
    mean_attn_dict['Original'] = {}
    mean_attn_dict['Steered'] = {}
    for entity_type in ['player', 'movie', 'song', 'city']:
        tokenized_prompts_dict = tokenized_prompts_dict_entity_type[entity_type]
        pos_entities_dict = pos_entities_dict_entity_type[entity_type]
        # Check
        model.to_str_tokens(tokenized_prompts_dict[known_label][0])
        
        tokenized_prompts = tokenized_prompts_dict[known_label]
        pos_entities = pos_entities_dict[known_label]

        # Original generation cache
        orig_attn = compute_attn_original(model, N, tokenized_prompts, pos_entities, pos_type=pos_type, type_pattern=type_pattern, batch_size=batch_size)
        # Steered generation cache
        steered_attn = compute_attn_steered(model, N, tokenized_prompts, pos_entities, pos_type=pos_type,
                                                steering_latents=[steering_latent], ablate_latents=None,
                                                coeff_value=coeff_value,
                                                type_pattern=type_pattern,
                                                batch_size=batch_size)

        mean_attn_dict['Original'][entity_type] = orig_attn
        mean_attn_dict['Steered'][entity_type] = steered_attn


    if scatter_plot:
        fig = plot_heads_scatter_entity_types(mean_attn_dict['Original'], mean_attn_dict['Steered'],
                        'Original', 'Steered', scatter_plot_kwargs['scatter_plot_heads'],
                        scatter_plot_kwargs['head_colors'],
                        f'Attention Scores for Original vs Steered Entities')

        pio.write_image(fig, f'plots/attn_plots/{model_alias}_known_latent_attn_original_vs_steered_entity_types_coeff_{coeff_value}from_{known_label}_{pos_type}_{str(heads)}.png', scale=5, width=600, height=400)
        pio.write_image(fig, f'plots/attn_plots/{model_alias}_known_latent_attn_original_vs_steered_entity_types_coeff_{coeff_value}from_{known_label}_{pos_type}_{str(heads)}.pdf', scale=5, width=600, height=400)
    difference_dict = {}
    for entity_type in ['player', 'movie', 'song', 'city']:
        difference_dict[entity_type] = (mean_attn_dict['Steered'][entity_type] - mean_attn_dict['Original'][entity_type])

    return difference_dict

# %%
model_alias = 'gemma-2-2b'#'gemma-2-9b'
n_devices = torch.cuda.device_count()

model = HookedTransformer.from_pretrained_no_processing(
    model_alias
)
model.set_use_attn_result(True)
tokenizer = model.tokenizer
tokenizer.padding_side = 'left'
model_alias = model_alias.replace('/','_')

# %%
tokenized_prompts_dict_entity_type = {}
pos_entities_dict_entity_type = {}
for entity_type in ['player', 'movie', 'song', 'city']:# , 'city', 'movie', 'song'
    tokenized_prompts_dict_entity_type[entity_type], pos_entities_dict_entity_type[entity_type], _ = load_data(model_alias, entity_type, tokenizer)
 # %%
 
######
# Attention to entity (original vs steering)
######
batch_size = 16
N = 100
type_pattern = 'attn_weights'
pos_type = 'entity_last' # position to steer
all_heads = True
split = 'test'
random_n_latents = 10
top_latents = [0, 1, 2] # top 3 SAE latents
top_latents = [0]

# For extracting plots Figure 4 (d) and (e)
scatter_plot = True
scatter_plot_kwargs = {}
scatter_plot_kwargs['scatter_plot_heads'] = [[20,3], [18,5]]
scatter_plot_kwargs['head_colors'] = [html_colors['blue_matplotlib'], html_colors['orange_matplotlib']]

for filter_with_pile in [True]:
    for idx in top_latents:
        top_latents = {'known': idx, 'unknown': idx}
        coeff_values = {'known': 100 if 'gemma' in model_alias else 20, 'unknown': 100 if 'gemma' in model_alias else 20}

        known_latent, unknown_latent, random_latents_known, random_latents_unknown = load_latents(model_alias, top_latents,
                                                                                                random_n_latents=random_n_latents,
                                                                                                filter_with_pile=filter_with_pile)

        layer_start = {}
        layer_start['known'] = unknown_latent[0][0]
        layer_start['unknown'] = known_latent[0][0]

        # Select relevant heads
        heads_dict = defaultdict(list)
        heads = []
        if all_heads == False:
            if model_alias == 'gemma-2-2b':
                for known_label in ['known', 'unknown']:
                    heads_dict[known_label] = [[15,5], [18,5], [20,3], [25,4]]
            else:
                for known_label in ['known', 'unknown']:
                    heads_dict[known_label] = [[25,2], [26,2], [29,14], [33,7], [37,12], [39,7]]
        else:
            for known_label in ['known', 'unknown']:
                for layer in range(layer_start[known_label], model.cfg.n_layers):
                    for head in range(model.cfg.n_heads):
                        heads_dict[known_label].append([layer, head])
        head_colors = [html_colors['blue_matplotlib'], html_colors['orange_matplotlib'], html_colors['brown_D3'], html_colors['dark_red_drawio']]
        os.makedirs('plots/attn_plots', exist_ok=True)


        for known_label in ['known', 'unknown']:
            coeff_value = coeff_values['unknown'] if known_label == 'known' else coeff_values['known']
            steering_latent = unknown_latent[0] if known_label == 'known' else known_latent[0]


            difference_dict = compute_attn_original_vs_steered(known_label, steering_latent, pos_type, type_pattern,
                                                               batch_size, tokenized_prompts_dict_entity_type, pos_entities_dict_entity_type,
                                                               scatter_plot=scatter_plot, **scatter_plot_kwargs)

            # Concatenate tensors from all entity types
            all_values = []
            for entity_type in difference_dict:
                values = difference_dict[entity_type]  # Shape: [n, n_layers, n_heads]
                all_values.append(values)
            concatenated = np.concatenate(all_values, axis=0)  # Shape: [total_n, n_layers, n_heads]

            data_save = {}
            for layer, head in heads_dict[known_label]:
                values_head = concatenated[:, layer, head]
                data_save[f'L{layer}H{head}'] = values_head
            
            top_latent_idx = top_latents['unknown'] if known_label == 'known' else top_latents['known']
            # Save concatenated values to file
            os.makedirs('./attn_steering_values', exist_ok=True)
            save_path = f'./attn_steering_values/{model_alias}_coeff_{coeff_value}from_{known_label}_{top_latent_idx}_{pos_type}_pile_filtering_{filter_with_pile}.npy'
            np.save(save_path, data_save)

            # Plot barplot
            if all_heads == True:
                width = 20 if model_alias == 'gemma-2-2b' else 50
            else:
                width = 6
            fig = plot_head_differences(concatenated, heads_dict[known_label], width=width)
            if all_heads == True:
                heads_str = 'all_heads'
            else:
                heads_str = str(heads)
            
            plt.savefig(f'plots/attn_plots/{model_alias}_all_heads_coeff_{coeff_value}from_{known_label}_{top_latent_idx}_{pos_type}_pile_filtering_{filter_with_pile}.png', transparent=True, dpi=500)
            plt.savefig(f'plots/attn_plots/{model_alias}_all_heads_coeff_{coeff_value}from_{known_label}_{top_latent_idx}_{pos_type}_pile_filtering_{filter_with_pile}.pdf', transparent=True)


        # Random steering
        for known_label in ['known', 'unknown']:
            list_difference_dict = []
            for random_latent_idx in range(random_n_latents):
                coeff_value = coeff_values['unknown'] if known_label == 'known' else coeff_values['known']
                steering_latent = random_latents_unknown[random_latent_idx] if known_label == 'known' else random_latents_known[random_latent_idx]
                print(steering_latent)

                difference_dict = compute_attn_original_vs_steered(known_label, steering_latent, pos_type, type_pattern, batch_size, tokenized_prompts_dict_entity_type, pos_entities_dict_entity_type)

                # Concatenate tensors from all entity types
                all_values = []
                for entity_type in difference_dict:
                    values = difference_dict[entity_type]  # Shape: [n, n_layers, n_heads]
                    all_values.append(values)
                concatenated = np.concatenate(all_values, axis=0)  # Shape: [total_n, n_layers, n_heads]
                #plot_head_differences(concatenated, heads, width=width)

                data_save = {}
                for layer, head in heads_dict[known_label]:
                    values_head = concatenated[:, layer, head]
                    data_save[f'L{layer}H{head}'] = values_head
                list_difference_dict.append(data_save)
            
            # Save concatenated values to file
            top_latent_idx = top_latents['unknown'] if known_label == 'known' else top_latents['known']
            os.makedirs('./attn_steering_values', exist_ok=True)
            save_path = f'./attn_steering_values/{model_alias}_random_coeff_{coeff_value}from_{known_label}_{top_latent_idx}_{pos_type}_pile_filtering_{filter_with_pile}.npy'
            np.save(save_path, list_difference_dict)

# %%
def load_attn_diff_results(model_alias, coeff_value, known_label, top_latent_idx, pos_type, filter_with_pile):
        random_results_path = f'./attn_steering_values/{model_alias}_random_coeff_{coeff_value}from_{known_label}_{top_latent_idx}_{pos_type}_pile_filtering_{filter_with_pile}.npy'
        random_results = np.load(random_results_path, allow_pickle=True)

        top_latent_results_path = f'./attn_steering_values/{model_alias}_coeff_{coeff_value}from_{known_label}_{top_latent_idx}_{pos_type}_pile_filtering_{filter_with_pile}.npy'
        top_latent_results = np.load(top_latent_results_path, allow_pickle=True)

        return random_results, top_latent_results


# %%
# Statistical significance tests
def compute_significance_test_against_random(random_results, top_latent_results, known_label):
    significant_count = 0
    for random_idx in range(len(random_results)):
        all_random_latents_list = []
        all_top_latent_list = []
        for head in top_latent_results[()].keys():
            all_random_latents_list.append(random_results[random_idx][head])
            all_top_latent_list.append(top_latent_results[()][head])
        all_random_latents = np.array(all_random_latents_list).flatten()
        all_top_latent = np.array(all_top_latent_list).flatten()

        if known_label == 'known':
            mean_1 = all_random_latents
            mean_2 = all_top_latent
        else:
            mean_1 = all_top_latent
            mean_2 = all_random_latents
        # Test mean_1 greater than mean_2
        stats_dict = compare_means(mean_1, mean_2, alpha=0.05)

        significant_count += stats_dict['significant']==True

    return significant_count




pos_type = 'entity_last'
filter_with_pile = True
significant_count_dict = {}
for model_alias in ['gemma-2-2b']:#, 'gemma-2-9b', 'meta-llama/Llama-3.1-8B']:
    model_alias = model_alias.replace('/','_')
    significant_count_dict[model_alias] = {}
    for known_label in ['known', 'unknown']:
        significant_count_dict[model_alias][known_label] = {}
        significant_count_dict[model_alias][known_label][filter_with_pile] = {}
        for top_latent_idx in range(3):
            # We pick top 3 latents
            coeff_value = 100 if 'gemma' in model_alias else 20

            random_results, top_latent_results = load_attn_diff_results(model_alias, coeff_value, known_label, top_latent_idx, pos_type, filter_with_pile)

            significant_count = compute_significance_test_against_random(random_results, top_latent_results, known_label)
            significant_count_dict[model_alias][known_label][filter_with_pile][top_latent_idx] = significant_count
            print(f'Significant count {model_alias} {known_label} {filter_with_pile}: {significant_count}')


# %%

head = 'L23H0'
for random_idx in range(len(random_results)):
    random_latent = random_results[random_idx][head]
    top_latent = top_latent_results[()][head]
    if known_label == 'known':
        mean_1 = random_latent
        mean_2 = top_latent
    else:
        mean_1 = top_latent
        mean_2 = random_latent

    # Test mean_1 greater than mean_2
    #stats_dict = compare_means(mean_1, mean_2, alpha=0.05)
    stats_dict = test_mean_difference(mean_2, alternative='less', alpha=0.05)
    print()
    print(repr(stats_dict))

# %%
# Known vs Unknown attention patterns all entity types
N = 100
batch_size = 16
type_pattern = 'attn_weights'
mean_attn_dict = {}
pos_type = 'entity_last' # position to steer

for known_label in ['known', 'unknown']:
    mean_attn_dict[known_label] = {}
    for entity_type in ['player', 'city', 'movie', 'song']:
        tokenized_prompts_dict, pos_entities_dict, entity_token_pos = load_data(model_alias, entity_type, tokenizer)
        tokenized_prompts = tokenized_prompts_dict[known_label]
        pos_entities = pos_entities_dict[known_label]
        orig_attn = compute_attn_original(model, N, tokenized_prompts, pos_entities, pos_type=pos_type, type_pattern=type_pattern, batch_size=batch_size)
        mean_attn_dict[known_label][entity_type] = orig_attn

# %%
# Select relevant heads
if model_alias == 'gemma-2-2b':
    heads = [[20,3], [18,5]]
else:
    heads = [[25,1], [25,2], [37,12]]

attn_heads_A = mean_attn_dict['known']
attn_heads_B = mean_attn_dict['unknown']
head_colors = [html_colors['green_drawio'], html_colors['dark_red_drawio']]
# %%
os.makedirs('plots/attn_plots', exist_ok=True)
fig = plot_heads_boxplot_entity_types(attn_heads_A, attn_heads_B,
                    'Known', 'Unknown', heads, head_colors,
                    f'Attention Scores for Known vs Unknown entities')

pio.write_image(fig, f'plots/attn_plots/{model_alias}_attn_known_vs_unknown_entities_{pos_type}_{type_pattern}_{str(heads)}.pdf', scale=7, width=600, height=400)

# %%
# Known vs Unknown attention patterns
N = 100
mean_attn_dict = {}
for known_label in ['known', 'unknown']:
    tokenized_prompts = tokenized_prompts_dict[known_label]
    pos_entities = pos_entities_dict[known_label]
    orig_attn = compute_attn_original(N, tokenized_prompts, entity_token_pos, type_pattern, batch_size=4)
    mean_attn_dict[known_label] = orig_attn

# %%
if model_alias == 'gemma-2-2b':
    heads = [[20,3], [18,5]]
else:
    heads = [[25,2], [25,1], [29,14], [37,12]]

attn_heads_A = mean_attn_dict['known']
attn_heads_B = mean_attn_dict['unknown']
plot_heads_boxplot(attn_heads_A, attn_heads_B,
                    'Known', 'Unknown', heads,
                    f'Attention scores for Known vs Unknown {entity_type} entities')






# %%
# Attn patterns plots
head_name_to_number = {}
head_names_list = [f"L{layer}H{head}" for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
for i, head in enumerate(head_names_list):
    head_name_to_number[head] = i

heads = ['L20H3', 'L18H5', 'L15H5']
heads_number = [head_name_to_number[head] for head in heads]

# %%
known_label = 'known'
type_pattern = 'attn_weights'
viz_attn_list = []
N = 100
batch_size = 1
prompt_template = 'The {entity_type} {entity_name} {relation}'

#tokenized_prompts_dict, pos_entities_dict, entity_token_pos = load_data(model_alias, entity_type, tokenizer, known_label, prompt_template)

entity_type = 'song'
tokenized_prompts_dict = {}
answers_dict = {}
for known_label in ['known', 'unknown']:
    prompt_template = 'The {entity_type} {entity_name} {relation}'
    tokenized_prompts, pos_entities, formatted_instructions, answers_label = load_data(model, model_alias, entity_type, tokenizer,
                                                                            known_label, prompt_template, conversation=None,
                                                                            relations_model_type='base',
                                                                            fixed_config='both')
    
    tokenized_prompts_dict[known_label] = tokenized_prompts
    answers_dict[known_label] = answers_label

# %%

prompts = tokenized_prompts_dict[known_label]#random.sample(prompts_label[known_type][:N], 8)

for i in range(0, min(N,len(prompts)), batch_size):
    prompts_batch = prompts[i:i+batch_size]
    #prompts_batch = [prompt.replace('The movie ', '') for prompt in prompts_batch]
    model.reset_hooks()
    clean_logits, clean_cache = model.run_with_cache(prompts_batch)

    #base_tokens = tokenizer(prompts_batch, padding=True, truncation=True, return_tensors="pt").input_ids

    #base_logits, base_cache = model.run_with_cache(base_tokens[0,:])
    viz_attn = visualize_attention_patterns(
        model,
        type_pattern,
        heads_number,
        clean_cache,
        f"",
        html=False
    )
viz_attn_list.append(viz_attn)
viz_attn_stack = torch.cat(viz_attn_list, 0)

# %%
#tokens_plot = [f"{tok} {i}" for i, tok in enumerate(str_tokens)]
# %%
tokens_plot = ['<bos> 0', 'The 1', '{entity_type} 2', '{entity_name} 3', '{entity_name} 4', '{relation} 5', '{relation} 6', '{relation} 7']
entity_token_pos = entity_type_to_token_pos[entity_type]
entity_toks = [f'entity_tok {j+1}' for j in range(entity_token_pos - 2)]
relation_toks = [f'relation_tok {j+1}' for j in range(3)]
patching_plot_sentence = ['<bos>', 'The', f'{entity_type}'] + entity_toks + relation_toks

#str_tokens = model.to_str_tokens(tokenizer(prompts[0], padding=True, truncation=True, return_tensors="pt").input_ids[0])
str_tokens = model.to_str_tokens(prompts[0])
head_idx = 0
fig = imshow(
    viz_attn_stack[:,head_idx].mean(0),
    x=patching_plot_sentence,
    y=patching_plot_sentence,
    title="",
    xaxis="", 
    yaxis="",
    return_fig=True,
)
fig.show()
#pio.write_image(fig, f'plots/{heads[head_idx]}_{type_pattern}_entity_type_{entity_type}.png',scale=6, width=500, height=250)


# %%


def unembed_proj(model, activation):
    projections = einsum(
            f"batch d_model, d_model vocab_size \
            -> batch vocab_size",
            activation * model.ln_final.w,
            model.unembed.W_U)
    torch.cuda.empty_cache()
    return projections


# %%
entity_type = 'song'
tokenized_prompts_dict, pos_entities_dict, entity_token_pos = load_data(model_alias, entity_type, tokenizer)

# %%
known_label = 'known'
prompts = tokenized_prompts_dict[known_label]

head_idx = 14
layer_idx = 29
N=8
batch_size = 8

random_ids = random.sample(range(prompts.shape[0]), batch_size)
prompts = prompts[random_ids]

model.reset_hooks()
clean_logits, clean_cache = model.run_with_cache(prompts)

decoded_prompts = [model.to_string(prompt)+'\n' for prompt in prompts]
for i, prompt in enumerate(decoded_prompts):
    print(f'{i}: {prompt.replace("<pad>", "")}')

# [batch, pos, head_index, d_model]
out_attn_head = clean_cache[f'blocks.{layer_idx}.attn.hook_result'][:, -1, head_idx].detach()

out_attn_head_unembed = unembed_proj(model, out_attn_head)

base_logits_idxs = out_attn_head_unembed.topk(20, dim=-1).indices
base_logits_values = out_attn_head_unembed.topk(20, dim=-1).values
for i in range(len(base_logits_idxs)):
    logit_toks = [model.to_str_tokens(x) for x in base_logits_idxs[i]]
    print(logit_toks)

del clean_cache

# %%
