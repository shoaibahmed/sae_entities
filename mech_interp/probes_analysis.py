# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

import copy
from tqdm import tqdm

import numpy as np
import torch

import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

from utils.hf_models.model_factory import construct_model_base
from utils.utils import model_alias_to_model_name

from feature_analysis_utils import get_dataloader, get_acts_labels_dict_
from feature_analysis_utils import get_per_layer_latent_scores


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
batch_size = 16
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

feature_type = 'hidden'  # can also be 'latents'
assert feature_type in ["latents", "hidden"], feature_type
get_per_layer_latent_scores(model_alias, tokenizer, n_layers, d_model, LAYERS_WITH_SAE,
                            feature_type=feature_type, save=True, **pile_prompts_experiment)

# %%
# Load the activations for the model -- only computed for the final token of the entity residual stream
dataloader = get_dataloader(model_alias, pile_prompts_experiment['tokens_to_cache'], n_layers, d_model, dataset_name=pile_prompts_experiment["dataset_name"], batch_size=batch_size)
acts_labels_dict_pile = get_acts_labels_dict_(model_alias, tokenizer, dataloader, LAYERS_WITH_SAE, **pile_prompts_experiment)
print("PILE activation size:", acts_labels_dict_pile[LAYERS_WITH_SAE[0]]['acts'].shape, acts_labels_dict_pile[LAYERS_WITH_SAE[0]]['labels'].shape)

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
    get_per_layer_latent_scores(model_alias.split('/')[-1], tokenizer, n_layers, d_model, LAYERS_WITH_SAE,
                                feature_type=feature_type, save=True, **wikidata_prompts_experiment)

# %%
acts_labels_dict_wikidata = {}
for entity_type in ALL_ENTITY_TYPES:
    wikidata_prompts_experiment['dataset_name'] = f'wikidata_{entity_type}'

    # Load the activations for the model -- only computed for the final token of the entity residual stream
    dataloader = get_dataloader(model_alias, wikidata_prompts_experiment['tokens_to_cache'], n_layers, d_model, dataset_name=wikidata_prompts_experiment["dataset_name"], batch_size=batch_size)
    acts_labels_dict_wikidata[entity_type] = get_acts_labels_dict_(model_alias, tokenizer, dataloader, LAYERS_WITH_SAE, **wikidata_prompts_experiment)
    print(f"Wikidata {entity_type} activation size:", acts_labels_dict_wikidata[entity_type][LAYERS_WITH_SAE[0]]['acts'].shape, acts_labels_dict_wikidata[entity_type][LAYERS_WITH_SAE[0]]['labels'].shape)

# %%
# Split the dataset into train / test -- only train on player entity and evaluate on all others
selected_wikidata_entity = "player"
num_ex_wikidata_selected, hidden_dim = acts_labels_dict_wikidata[selected_wikidata_entity][LAYERS_WITH_SAE[0]]['acts'].shape
print(f"selected wikidata entity: {selected_wikidata_entity} / # ex: {num_ex_wikidata_selected} / hidden dim: {hidden_dim}")

train_frac = 0.9
num_train_ex = int(num_ex_wikidata_selected * 0.9)
seed = 10
rng = np.random.default_rng(seed)

all_ex_idx = list(range(num_ex_wikidata_selected))
train_idx = rng.choice(all_ex_idx, size=(num_train_ex,))
eval_idx = [x for x in all_ex_idx if x not in train_idx]
print(f"# examples: {len(all_ex_idx)} / # train: {len(train_idx)} / # eval: {len(eval_idx)}")

# %%
# Define the probe classifier
probe_feature_dim = 32
probe_cls = torch.nn.Sequential(
    torch.nn.Linear(hidden_dim, probe_feature_dim),
    torch.nn.BatchNorm1d(probe_feature_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, probe_feature_dim),
).to("cuda")
base_state_dict = copy.deepcopy(probe_cls.state_dict())  # to ensure same model init
print(probe_cls)

# %%
# Define training hyperparameters
n_epochs = 100
batch_size = 256
loss_fn = torch.nn.BCEWithLogitsLoss()  # binary classification

# %%
# Train the model on the selected examples
for layer in LAYERS_WITH_SAE:
    print(f"Evaluating layer: {layer}")
    data = acts_labels_dict_wikidata[selected_wikidata_entity][layer]
    acts, labels = data["acts"], data["labels"]

    train_data = torch.stack([acts[i] for i in train_idx], dim=0)
    labels = torch.stack([labels[i] for i in train_idx], dim=0)
    print(f"Train data: {train_data.shape} / labels: {labels.shape}")

    # Convert examples into a pytorch dataloader
    tensor_dataset = torch.utils.data.TensorDataset(train_data, labels)
    dataloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

    # Load the base probe classifier
    probe_cls.load_state_dict(base_state_dict)

    # Train the probe classifier
    for epoch in tqdm(range(n_epochs)):
        pass

# %%
# TODO: perform model steering using probes
