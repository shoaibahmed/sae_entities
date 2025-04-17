# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

import os
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
# Define training hyperparameters
n_epochs = 100
batch_size = 256
loss_fn = torch.nn.BCEWithLogitsLoss()  # binary classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Setup checkpoint directory
checkpoint_dir = "checkpoints"
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
    print("Checkpoint directory created:", checkpoint_dir)

# %%
# Define the probe classifier
probe_feature_dim = 32
output_dims = 1  # binary classification
probe_cls = torch.nn.Sequential(
    torch.nn.Linear(hidden_dim, probe_feature_dim),
    torch.nn.BatchNorm1d(probe_feature_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, output_dims),
).to(device)
base_state_dict = copy.deepcopy(probe_cls.state_dict())  # to ensure same model init
print(probe_cls)

# %%

def evaluate_model(probe_cls: torch.nn.Module, eval_loader: torch.utils.data.DataLoader):
    probe_cls.eval()
    total = 0
    num_correct = 0
    for x, y in eval_loader:
        x, y = x.to(device), y.to(device)
        pred = probe_cls(x)
        assert pred.shape == y.shape, f"{pred.shape} != {y.shape}"
        pred = torch.sigmoid(pred) >= 0.5
        correct = pred.to(y.dtype) == y
        num_correct += int(torch.sum(correct))
    acc = 100. * float(num_correct) / total
    print(f"!! Test for epoch: {epoch+1} / total: {total} / correct: {correct} / acc: {acc:.2f}%")


# %%
# Train the model on the selected examples
for layer in LAYERS_WITH_SAE:
    print("="*25)
    print(f"Evaluating layer: {layer}")
    data = acts_labels_dict_wikidata[selected_wikidata_entity][layer]
    acts, labels = data["acts"], data["labels"]

    train_data = torch.stack([acts[i] for i in train_idx], dim=0)
    train_labels = torch.stack([labels[i] for i in train_idx], dim=0)
    print(f"Train data: {train_data.shape} / labels: {train_labels.shape}")

    eval_data = torch.stack([acts[i] for i in eval_idx], dim=0)
    eval_labels = torch.stack([labels[i] for i in eval_idx], dim=0)
    print(f"Eval data: {eval_data.shape} / labels: {eval_labels.shape}")

    # Convert examples into a pytorch dataloader
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    eval_dataset = torch.utils.data.TensorDataset(eval_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Load the base probe classifier
    probe_cls.load_state_dict(base_state_dict)

    # Define the optimizer
    optimizer = torch.optim.AdamW(probe_cls.parameters(), lr=1e-3, weight_decay=1e-4)

    # Train the probe classifier
    for epoch in tqdm(range(n_epochs)):
        probe_cls.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = probe_cls(x)
            assert pred.shape == y.shape, f"{pred.shape} != {y.shape}"
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate test performance
        evaluate_model(probe_cls, eval_loader)

    checkpoint_file = f"entity_{selected_wikidata_entity}_layer_{layer}.pth"
    torch.save(probe_cls.state_dict(), os.path.join(checkpoint_dir, checkpoint_file))

    # evaluate the probe classifier on all entity types
    for entity_type in ALL_ENTITY_TYPES:
        if entity_type == selected_wikidata_entity:
            continue  # already evaluated in the form of the test set

        print("-"*10)
        print(f"Evaluating on entity type: {entity_type}")
        entity_eval_dataset = torch.utils.data.TensorDataset(acts_labels_dict_wikidata[selected_wikidata_entity][layer]["acts"], acts_labels_dict_wikidata[selected_wikidata_entity][layer]["labels"])
        entity_eval_loader = torch.utils.data.DataLoader(entity_eval_dataset, batch_size=batch_size, shuffle=False)
        evaluate_model(probe_cls, entity_eval_loader)

# %%
# TODO: perform model steering using probes
