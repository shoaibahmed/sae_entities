# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')
    
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")


from collections import defaultdict
from utils.hf_models.model_factory import construct_model_base
from utils.utils import model_alias_to_model_name

from feature_analysis_utils import get_per_layer_latent_scores, scatter_plot_latent_separation_scores_experiment
from feature_analysis_utils import get_general_latents, get_layerwise_latent_scores, plot_layerwise_latent_scores

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

get_per_layer_latent_scores(model_alias, tokenizer, n_layers, d_model,
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
    get_per_layer_latent_scores(model_alias.split('/')[-1], tokenizer, n_layers, d_model,
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
