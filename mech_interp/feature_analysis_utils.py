import json
import os
import copy
import gc
import torch
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from utils.hf_models.model_factory import construct_model_base
from utils.utils import model_alias_to_model_name, paper_plot

from dataset.load_data import load_wikidata_queries, load_triviaqa_queries
from mech_interp_utils import get_known_unknown_splits
from mech_interp_utils import get_acts_labels_dict, get_features_layers, get_top_k_features
from mech_interp_utils import plot_all_features, read_layer_features
from mech_interp_utils import html_colors, model_alias_to_sae_repo_id
from sae_entities.utils.activation_cache import CachedDataset

SAE_WIDTH = '16k'

def get_dataloader(model_alias, tokens_to_cache, n_layers, d_model, dataset_name, batch_size=128):
    """
    Get a DataLoader to load cached activations from a pre-computed dataset for a specific model and entity type.

    Args:
        model_alias (str): The alias of the model to use.
        entity_type (str): The type of entity to load data for (e.g., 'movie', 'song', 'player', 'city').
        tokens_to_cache (str): Specifies which tokens to cache (e.g., 'entity', 'model', 'last_eoi').
        n_layers (int): The number of layers in the model.
        batch_size (int): The batch size to use for the DataLoader.

    Returns:
        torch.utils.data.DataLoader: A DataLoader containing the cached dataset for the specified model and entity type.
    """
    entity_shard_size_entities = {'movie': 65370, 'song': 33792, 'player': 22461, 'city': 31616}
    if 'wikidata' in dataset_name:
        entity_type = dataset_name.split('_')[1]
        shard_size = entity_shard_size_entities[entity_type]

    elif 'triviaqa' in dataset_name:
        shard_size = 9961

    elif 'pile' in dataset_name:
        shard_size = 10000
    
    cached_acts_path = '../dataset/cached_activations'
    seq_len = 128 if 'triviaqa' in dataset_name else 64
    n_positions = 1
    foldername = f"{cached_acts_path}/{tokens_to_cache}/{model_alias}_{dataset_name}/{tokens_to_cache}_npositions_{n_positions}_shard_size_{shard_size}"
    dataset = CachedDataset(foldername, range(0,n_layers), d_model, seq_len, n_positions, shard_size=shard_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader

def get_acts_labels_dict_(model_alias, tokenizer, dataloader, sae_layers, **kwargs):
    """
    Get activations and labels dictionary for a given model, dataloader, and evaluation settings.

    Args:
        model_alias (str): Alias of the model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        evaluate_on (str): What to evaluate on ('prompts' or 'entities').
        split (str, optional): Data split to use. Defaults to 'train'.
        free_generation (bool, optional): Whether to use free generation. Defaults to False.
        consider_refusal_label (bool, optional): Whether to consider the refusal label. Defaults to False.
        further_split (bool, optional): Whether to further split errors into known and unknown entities. Defaults to False.

    Returns:
        dict: Dictionary containing activations ('acts') and labels ('labels'), and optionally ('prompts') for each layer.
    """
    dataset_name = kwargs['dataset_name']
    split = kwargs['split']
    if 'pile' in dataset_name:
        acts_labels_dict = {}
        labels = []
        activations_list = []
        for _, (batch_activations, batch_input_ids) in tqdm(enumerate(dataloader), total=len(dataloader)):
            for j in range(len(batch_activations)):
                labels.append(1)
                activations_list.append(batch_activations[j][:,0].to('cuda'))

        labels_full = torch.tensor(labels)
        for layer in sae_layers:
            activations_full = torch.stack([activations[layer] for activations in activations_list], dim=0)
            acts_labels = {'acts': activations_full, 'labels': labels_full}
            acts_labels_dict[layer] = acts_labels
        
        return acts_labels_dict
    
    evaluate_on = kwargs['evaluate_on']
    free_generation = kwargs['free_generation']
    consider_refusal_label = kwargs['consider_refusal_label']
    further_split = kwargs['further_split']

    if 'wikidata' in dataset_name:
        # Wikidata specific processing
        entity_type = dataset_name.split('_')[1]
        entity_type_and_entity_name_format = kwargs['entity_type_and_entity_name_format']
        # Load wikidata dataset queries for an entity type
        queries = load_wikidata_queries(model_alias, free_generation=free_generation, entity_type_and_entity_name_format=entity_type_and_entity_name_format)
        queries = [query for query in queries if query['entity_type'] == entity_type]
        prompts = [q['prompt'] for q in queries]
    elif 'triviaqa' in dataset_name:
        queries = load_triviaqa_queries(model_alias, split=None)
        prompts = [q['prompt'] for q in queries]


    if evaluate_on == 'prompts':
        if 'wikidata' in dataset_name:
            # We first get known and unknown entities
            known_entities, unknown_entities = get_known_unknown_splits(queries, return_type='entity', split=split, consider_refusal_label=False)
            set_entities = known_entities + unknown_entities
            # Get queries of known and unknown entities
            queries = [q for q in queries if q['entity'] in set_entities]
            # Get known (correct) and unknown (error) prompts. That is, prompts with correct or incorrect answers
        known_prompts, unknown_prompts = get_known_unknown_splits(queries, return_type='prompt', split=None, consider_refusal_label=consider_refusal_label)
        print(f"Number of known prompts: {len(known_prompts)}")
        print(f"Number of unknown prompts: {len(unknown_prompts)}")
    elif evaluate_on == 'entities':
        # We split into known and unknown entities
        known_entities, unknown_entities = get_known_unknown_splits(queries, return_type='entity', split=split, consider_refusal_label=consider_refusal_label)
        print(f"Number of known entities: {len(known_entities)}")
        print(f"Number of unknown entities: {len(unknown_entities)}")

    # Get latents for all layers and store them
    if evaluate_on == 'prompts':
        acts_labels_dict = get_acts_labels_dict(model_alias, tokenizer, dataloader, queries, prompts, known_prompts, unknown_prompts, input_data_type=evaluate_on, layers=sae_layers)
        if further_split == True:
            # Update acts_labels_dict[layer]
            # Split errors into errors of known and unknown entities
            for layer in acts_labels_dict.keys():
                labels = acts_labels_dict[layer]['labels']
                prompts = acts_labels_dict[layer]['prompts']
                for i, (prompt, label) in enumerate(zip(prompts, labels)):
                    if label == 1:
                        for entity in known_entities:
                            if entity in prompt:
                                acts_labels_dict[layer]['labels'][i] = 3
                                break

    elif evaluate_on == 'entities':
        acts_labels_dict = get_acts_labels_dict(model_alias, tokenizer, dataloader, queries, prompts, known_entities, unknown_entities, input_data_type=evaluate_on, layers=sae_layers)


    return acts_labels_dict



### Latents scores per layer ###
def get_per_layer_latent_scores(model_alias, tokenizer, n_layers, d_model, sae_layers, save=False, **kwargs):
    """
    Compute latent scores for each layer of the model.

    This function processes the model's activations for a specific entity type,
    computes latent scores using the specified method, and optionally saves the results.

    Args:
        model_alias (str): Alias of the model being analyzed.
        tokenizer: Tokenizer associated with the model.
        n_layers (int): Total number of layers in the model.
        d_model (int): Dimensionality of the model's hidden states.
        sae_layers (list): List of layer indices for which to compute SAE scores.

    """
    dataset_name = kwargs['dataset_name']
    # Fixed parameters
    batch_size = 16
    repo_id = model_alias_to_sae_repo_id[model_alias]

    # # Get dataloader
    dataloader = get_dataloader(model_alias, kwargs['tokens_to_cache'], n_layers, d_model, dataset_name=dataset_name, batch_size=batch_size)
    # Get cached activations and labels
    acts_labels_dict = get_acts_labels_dict_(model_alias, tokenizer, dataloader, sae_layers, **kwargs)
    # Get features info per layer and (optionally) save them as JSON files
    get_features_layers(model_alias, acts_labels_dict, sae_layers, SAE_WIDTH, repo_id, save, **kwargs)



### Scatter plot latent separation scores ###
def scatter_plot_latent_separation_scores_experiment(model_alias, tokenizer, entity_type, tokens_to_cache, n_layers, testing_layers, d_model, entity_type_and_entity_name_format=False):
    # Parameters
    evaluate_on = 'entities' # keep as 'entities' for computing the scatter plot
    scoring_method = 'absolute_difference' # keep this as 'absolute_difference' for computing the scatter plot
    min_activations = None
    split = 'test'
    save = False # save intermediate results from get_features_layers
    repo_id = model_alias_to_sae_repo_id[model_alias]
   
    # Get features scores for all layers in the training set
    feats_layers = read_layer_features(model_alias, testing_layers, entity_type, scoring_method, tokens_to_cache, evaluate_on)
    train_feats_dict = get_top_k_features(feats_layers, k=None)

    # Get features scores for all layers in the test set
    # TODO: add entity_type to dataset_name
    dataloader = get_dataloader(model_alias, tokens_to_cache, n_layers, d_model, f'wikidata_{entity_type}', batch_size=16)
    test_acts_labels_dict = get_acts_labels_dict_(model_alias, tokenizer, dataloader, testing_layers,
                                                  dataset_name=f'wikidata_{entity_type}', evaluate_on=evaluate_on, split=split,
                                                  free_generation=False, consider_refusal_label=False,
                                                  further_split=False, entity_type_and_entity_name_format=entity_type_and_entity_name_format)
    test_feats_layers = get_features_layers(model_alias, test_acts_labels_dict, testing_layers, SAE_WIDTH, repo_id,
                                            save, scoring_method=scoring_method, min_activations=min_activations,
                                            dataset_name=f'wikidata_{entity_type}')

    test_final_feats_dict = get_top_k_features(test_feats_layers, k=None)

    # we always save scatter plots
    fig = plot_all_features(test_final_feats_dict, train_feats_dict, entity_type, k=10, labels=False)
    os.makedirs('./plots/scatter_plots', exist_ok=True)
    fig.savefig(f'./plots/scatter_plots/{model_alias}_feature_activation_frequencies_smaller_{entity_type}_{str(testing_layers)}.png', transparent=True)
    fig.savefig(f'./plots/scatter_plots/{model_alias}_feature_activation_frequencies_smaller_{entity_type}_{str(testing_layers)}.pdf', transparent=True)
    fig.show()

### Searching for the top general latents ###
def get_general_latents(model_alias, entity_types, testing_layers, tokens_to_cache, evaluate_on, scoring_method, filter_with_pile=False):
    """
    Analyze and identify general latents across different entity types.

    This function processes feature scores for all entity types and computes various statistics to identify
    the most general latents. It calculates mean scores, minimum scores, and mean ranks for each latent
    across all entity types, for both known and unknown entities.

    Args:
        model_alias (str): The alias of the model being analyzed.
        entity_types (list): List of entity types to analyze.
        testing_layers (list): List of layers to analyze.
        tokens_to_cache (str): Type of tokens to cache ('model', 'last_eoi', '?', or 'entity').
        evaluate_on (str): What to evaluate on ('prompts' or 'entities').
        scoring_method (str): Method used for scoring ('absolute_difference' or 't_test').
        filter_with_pile (bool): Whether to filter latents with low frequency in the Pile dataset.

    Returns:
        None. Results are saved as JSON files in the specified directory.

    Side effects:
        - Creates a directory structure based on the input parameters.
        - Saves three JSON files for each known/unknown category:
          1. sorted_scores_min: Minimum scores for each latent across entity types.
          2. sorted_scores_mean: Mean scores for each latent across entity types.
          3. sorted_ranks_mean: Mean ranks for each latent across entity types.

    """
    scores = {}
    ranks = {}
    for known_label in ['known', 'unknown']:

        scores[known_label] = defaultdict(list)
        ranks[known_label] = defaultdict(list)

        for entity_type in entity_types:
            feats_layers = read_layer_features(model_alias, testing_layers, entity_type, scoring_method, tokens_to_cache, evaluate_on)
            train_feats_dict = get_top_k_features(feats_layers, k=None)
            for latent_idx in train_feats_dict[known_label].keys():
                latent = train_feats_dict[known_label][latent_idx]
                full_latent_id = f"L{latent['layer']}F{latent['latent_idx']}"
                scores[known_label][full_latent_id].append(latent['score'])
                ranks[known_label][full_latent_id].append(latent_idx)

        scores_mean = {}
        scores_min = {}
        ranks_mean = {}
        for full_latent_id in scores[known_label].keys():
            scores_mean[full_latent_id] = np.mean(scores[known_label][full_latent_id])
            scores_min[full_latent_id] = np.min(scores[known_label][full_latent_id])
            ranks_mean[full_latent_id] = np.mean(ranks[known_label][full_latent_id])

        sorted_scores_mean = dict(sorted(scores_mean.items(), key=lambda item: item[1], reverse=True))
        sorted_ranks_mean = dict(sorted(ranks_mean.items(), key=lambda item: item[1], reverse=False))
        sorted_scores_min = dict(sorted(scores_min.items(), key=lambda item: item[1], reverse=True))

        save_dir = f'./train_latents_layers_{evaluate_on}/{scoring_method}/{model_alias}/{tokens_to_cache}'
        print('saving in', save_dir)
        os.makedirs(save_dir, exist_ok=True)
        for sorted_dict in zip([sorted_scores_min, sorted_scores_mean, sorted_ranks_mean], ['scores_min', 'scores_mean', 'ranks_mean']):
            if filter_with_pile:
                final_sorted_dict = {}
                counter = 0
                feats_layers_pile = read_layer_features(model_alias, testing_layers, 'pile', scoring_method, 'random', 'random')
                for latent_id in list(sorted_dict[0].keys()):
                    layer_latent = int(latent_id[1:latent_id.find('F')])
                    latent_idx = float(latent_id[latent_id.find('F')+1:])
                    for idx in feats_layers_pile[layer_latent]['unknown'].keys():
                        if feats_layers_pile[layer_latent]['unknown'][idx]['latent_idx'] == latent_idx:
                            pile_freq_score = feats_layers_pile[layer_latent]['unknown'][idx]['freq_acts_unknown']
                            break
                    if pile_freq_score < 0.02: # thershold used
                        final_sorted_dict[latent_id] = sorted_dict[0][latent_id]
                        counter += 1
                    if counter >= 20:
                        break
                # Save sorted_scores_min to JSON
                with open(f'{save_dir}/pile_filtered_{sorted_dict[1]}_{known_label}.json', 'w') as f:
                    json.dump(final_sorted_dict, f, indent=4)

            else:
                # Save sorted_scores_min to JSON
                with open(f'{save_dir}/sorted_{sorted_dict[1]}_{known_label}.json', 'w') as f:
                    json.dump(sorted_dict[0], f, indent=4)

        print(f"Saved feature analysis results for {known_label} entities to JSON files.")



# %%
#### Layerwise Latent Scores Analysis ####
def get_layerwise_latent_scores(model_alias, sae_layers, tokens_to_cache, scoring_method, entity_types, top_k):
    """
    Get layerwise latent scores for a given model, tokens to cache, evaluate on, scoring method, and top k.
    """
    evaluate_on = 'entities'

    scores = {}
    scores_min = {}
    top_scores_layers = {}
    minmax_layerwise_scores = {}
    for known_label in ['known', 'unknown']:
        scores_min[known_label] = {}
        top_scores_layers[known_label] = {}
        scores[known_label] = {}

        for layer in sae_layers:
            scores[known_label][layer] = defaultdict(list)

        # Get scores and ranks for all layers. These are computed for each entity type
        for entity_type in entity_types:
            top_scores_layers[known_label][entity_type] = defaultdict(list)
            feats_layers = read_layer_features(model_alias, sae_layers, entity_type, scoring_method, tokens_to_cache, evaluate_on)
            for layer in feats_layers.keys():
                for latent_idx in feats_layers[layer][known_label].keys():
                    latent = feats_layers[layer][known_label][latent_idx]
                    full_latent_id = f"L{latent['layer']}F{latent['latent_idx']}"
                    scores[known_label][layer][full_latent_id].append(latent['score'])
                for i in range(top_k):
                    score = feats_layers[layer][known_label][str(i)]['score']
                    top_scores_layers[known_label][entity_type][layer].append(score)

        # Compute the min score for each latent across entity types. We do it per layer
        for layer in feats_layers.keys():
            scores_min[known_label][layer] = {}
            for full_latent_id in scores[known_label][layer].keys():
                scores_min[known_label][layer][full_latent_id] = np.min(scores[known_label][layer][full_latent_id])

        # Now compute the max of the min scores of all the latents
        minmax_layerwise_scores[known_label] = {}
        min_score_of_top_k_rank = defaultdict(list)
        for layer in scores_min[known_label].keys():
            for i, (latent_id, _) in enumerate(scores_min[known_label][layer].items()):
                min_score_of_top_k_rank[layer].append(scores_min[known_label][layer][latent_id])
            minmax_layerwise_scores[known_label][layer] = np.max(min_score_of_top_k_rank[layer])

    return top_scores_layers, minmax_layerwise_scores


def plot_layerwise_latent_scores(model_alias, sae_layers, top_scores_layers, minmax_layerwise_scores, known_label, top_k):

    entity_types = list(top_scores_layers[known_label].keys())
    colors = [html_colors['blue_drawio'], html_colors['grey_drawio'], html_colors['green_drawio'], html_colors['brown_D3']]  # Add more colors if needed
    final_scores = list(minmax_layerwise_scores[known_label].values())

    # Create the line plot with error bars
    fig = go.Figure()

    for i, entity_type in enumerate(entity_types):
        scores = [np.mean(top_scores_layers[known_label][entity_type][layer]) for layer in sae_layers]
        min_scores = [min(top_scores_layers[known_label][entity_type][layer]) for layer in sae_layers]
        max_scores = [max(top_scores_layers[known_label][entity_type][layer]) for layer in sae_layers]
        error_y = [score - min_score for score, min_score in zip(scores, min_scores)]
        error_y_minus = [max_score - score for score, max_score in zip(scores, max_scores)]
        # Add error bars to the plot
        fig.add_trace(go.Scatter(
            x=sae_layers,
            y=scores,
            error_y=dict(
                type='data',
                array=error_y,
                symmetric=False,
                arrayminus=error_y_minus,
                visible=True,
            ),
            mode='lines+markers',
            opacity=0.6,  # Added opacity to make the line more opaque
            name=entity_type.capitalize(),
            line=dict(color=colors[i])
        ))

    # Add min_score_of_top_k_rank to the plot
    fig.add_trace(go.Scatter(
        x=sae_layers,
        y=final_scores,
        mode='lines+markers',
        name=f'MaxMin',
        line=dict(color=html_colors['red_drawio'], dash='dash')
    ))

    fig.update_layout(
        xaxis_title='Layer',
        yaxis_title='Score',
        legend_title='',
        # legend=dict(
        #     yanchor="top",
        #     y=0.99,
        #     xanchor="left",
        #     x=0.01
        # )
    )

    fig.update_layout(
        title={
            'text': f'Top {top_k} {known_label.capitalize()} Separation Scores Latents',
            'y': 0.75,  # Moves the title closer to the plot (default is 0.9)
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
    fig.update_layout(showlegend=True)

    # Update x-axis limits
    fig.update_xaxes(range=[0.5, len(sae_layers)+1-0.5])
    # Update x-axis to show ticks every 5 layers
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(0, max(sae_layers)+1, 5)),
        ticktext=[str(i) for i in range(0, max(sae_layers)+1, 5)]
    )

    fig = paper_plot(fig, tickangle=0)
    os.makedirs(f'plots/layerwise_evolution', exist_ok=True)
    pio.write_image(fig, f'plots/layerwise_evolution/{model_alias}_entities_top_scores_layers_{known_label}.png',
                        scale=10, width=500, height=315)# width=475, height=300
    
    pio.write_image(fig, f'plots/layerwise_evolution/{model_alias}_entities_top_scores_layers_{known_label}.pdf',
                        scale=10, width=500, height=315)# width=475, height=300

    fig.show()