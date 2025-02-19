# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

import os
import gc
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

import torch

from collections import defaultdict
from utils.hf_models.model_factory import construct_model_base
from utils.utils import model_alias_to_model_name

from feature_analysis_utils import get_dataloader, get_acts_labels_dict_, scatter_plot_latent_separation_scores_experiment
from feature_analysis_utils import get_general_latents, get_layerwise_latent_scores, plot_layerwise_latent_scores
from mech_interp_utils import get_features, format_layer_features


# %%
model_alias = 'gemma-2-2b'
model_alias = model_alias.replace('/','_')
#REPO_ID = model_alias_to_sae_repo_id[model_alias]
ALL_ENTITY_TYPES = ['player', 'song', 'city', 'movie']
model_path = model_alias_to_model_name[model_alias]
# Load model to load tokenizer and config data
model_base = construct_model_base(model_path)
d_model = model_base.model.config.hidden_size
tokenizer = model_base.tokenizer
n_layers = model_base.model.config.num_hidden_layers
del model_base

# %%
# We compute SAE latent scores for all available layers
if model_alias == 'gemma-2b-it':
    LAYERS_WITH_SAE = [13]
elif model_alias == 'gemma-2-9b-it':
    LAYERS_WITH_SAE = [10, 21, 32]
else:
    LAYERS_WITH_SAE = list(range(1, n_layers))

# %%

### Latents scores per layer ###
def get_per_layer_hidden_scores(model_alias, tokenizer, n_layers, d_model, sae_layers, save=False, **kwargs):
    """
    Compute hidden scores for each layer of the model.

    This function processes the model's activations for a specific entity type,
    computes hidden scores using the specified method, and optionally saves the results.

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

    # # Get dataloader
    dataloader = get_dataloader(model_alias, kwargs['tokens_to_cache'], n_layers, d_model, dataset_name=dataset_name, batch_size=batch_size)
    # Get cached activations and labels
    acts_labels_dict = get_acts_labels_dict_(model_alias, tokenizer, dataloader, sae_layers, **kwargs)
    # Get features info per layer and (optionally) save them as JSON files
    get_features_layers(model_alias, acts_labels_dict, sae_layers, save, **kwargs)


def get_features_layers(model_alias, acts_labels_dict, layers, save=True, **kwargs):
    """
    Get the features for each layer using the SAE.

    This function loads the SAE for each specified layer, computes the activations for known and unknown entities,
    and extracts the top features based on the activation differences.

    Args:
        model_alias: The alias of the model.
        entity_type: The type of entity.
        acts_labels_dict: A dictionary containing activations and labels for each layer.
        layers: A list of layer indices for which activations and labels are to be extracted.
        min_activations: A list of minimum activation values for known and unknown entities.

    Returns:
        A dictionary where the keys are layer indices and the values are tuples containing:
            - A list of feature indices.
            - A list of top scores for the features.
    """
    feats_per_layer = {}
    assert 0 not in layers, "Layer 0 is not a valid layer for the SAE"

    scoring_method = kwargs['scoring_method']
    min_activations = kwargs['min_activations'] if 'min_activations' in kwargs else None

    formatted_feats_layer = {}
    for layer in layers:
        acts_labels = acts_labels_dict[layer]
        # Get activations
        model_acts = acts_labels['acts']

        # Get labels
        label_indices_1 = torch.where(acts_labels['labels'] == 1.0)[0]
        label_indices_0 = torch.where(acts_labels['labels'] == 0.0)[0]
        # Divide sae activations into known and unknown
        model_acts_1 = model_acts[label_indices_1, :].detach() # known activations
        model_acts_0 = model_acts[label_indices_0, :].detach() # unknown activations

        model_acts = {'known': model_acts_0, 'unknown': model_acts_1}

        # We return all features
        scores_dict, freq_acts_dict, mean_features_acts = get_features(model_acts,
                                                    metric = scoring_method,
                                                    min_activations=min_activations, # min activations for known and unknown to consider, lower than that we clamp to zero
                                                    )

        feats_per_layer[layer] = (scores_dict, freq_acts_dict, mean_features_acts)
        del model_acts_0, model_acts_1, model_acts
        torch.cuda.empty_cache()
        gc.collect()

    
    if save == True:
        dataset_name = kwargs['dataset_name']
        evaluate_on = kwargs['evaluate_on']
        tokens_to_cache = kwargs['tokens_to_cache']
        folder = f'./train_latents_layers_{evaluate_on}/{scoring_method}/{model_alias}/{tokens_to_cache}'
        os.makedirs(folder, exist_ok=True)
        if 'wikidata' in dataset_name:
            entity_type = dataset_name.split('_')[1]
            filename_prefix = f'{folder}/{entity_type}'
        else:
            filename_prefix = f'{folder}/{dataset_name}'
    else:
        filename_prefix = ''
    formatted_feats_layer = format_layer_features(feats_per_layer, filename_prefix, save=save)

    return formatted_feats_layer


# %%
### Latents scores per layer on a subset of the Pile dataset ###
# You'll need to have precomputed cached activations
# python -m utils.activation_cache --model_alias gemma-2-2b --tokens_to_cache random --batch_size 128 --dataset pile
pile_prompts_experiment = {
    'dataset_name' : 'pile',
    'scoring_method' : 'absolute_difference',
    'tokens_to_cache' : 'random',# Token whose cached activations we want to access
    'free_generation' : False,
    'consider_refusal_label' : False,
    'evaluate_on' : 'random',
    'split' : None,
    'further_split' : False,
    'entity_type_and_entity_name_format' : True,
    }

get_per_layer_hidden_scores(model_alias, tokenizer, n_layers, d_model,
                            LAYERS_WITH_SAE, save=True, **pile_prompts_experiment)

# %%
### Latent scores per layer on Wikidata entities ###
# Known/Unknown entities (base model)
# You'll need to have precomputed cached activations
# python -m utils.activation_cache --model_alias gemma-2-2b --tokens_to_cache entity --batch_size 128 --entity_type_and_entity_name_format

wikidata_prompts_experiment = {
    'dataset_name' : 'wikidata',
    'evaluate_on' : 'entities',
    'scoring_method' : 'absolute_difference',
    'tokens_to_cache' : 'entity',# Token whose cached activations we want to access
    'free_generation' : False,
    'consider_refusal_label' : False,
    'split' : None,
    'further_split' : False,
    'entity_type_and_entity_name_format' : True,
    }
for entity_type in ALL_ENTITY_TYPES:
    wikidata_prompts_experiment['dataset_name'] = f'wikidata_{entity_type}'
    get_per_layer_hidden_scores(model_alias.split('/')[-1], tokenizer, n_layers, d_model,
                                LAYERS_WITH_SAE, save=True, **wikidata_prompts_experiment)

# %%
### Scatter plot latent separation scores ###
testing_layers = LAYERS_WITH_SAE
tokens_to_cache = 'entity'
for entity_type in ALL_ENTITY_TYPES:
    scatter_plot_latent_separation_scores_experiment(model_alias, tokenizer, entity_type,
                                                    tokens_to_cache, n_layers, testing_layers,
                                                    d_model, entity_type_and_entity_name_format=True)


# %%
### Searching for the top general latents ###
tokens_to_cache = 'entity' # 'model' 'last_eoi' '?' 'entity'
evaluate_on = 'entities' # prompts or entities
scoring_method = 'absolute_difference' # 'absolute_difference', 'relative_difference', 't_test'
testing_layers = LAYERS_WITH_SAE
entity_types = ALL_ENTITY_TYPES
get_general_latents(model_alias, entity_types, testing_layers, tokens_to_cache, evaluate_on,
                    scoring_method, filter_with_pile=True)

# %%
#### Layerwise Latent Scores Analysis ####
scoring_method = 'absolute_difference'
top_k = 5
tokens_to_cache = 'entity'
top_scores_layers, minmax_layerwise_scores = get_layerwise_latent_scores(model_alias, LAYERS_WITH_SAE, tokens_to_cache,
                                                                         scoring_method, ALL_ENTITY_TYPES, top_k)

# %%
for known_label in ['known', 'unknown']:
    plot_layerwise_latent_scores(model_alias, LAYERS_WITH_SAE, top_scores_layers,
                                 minmax_layerwise_scores, known_label, top_k)

# %%
