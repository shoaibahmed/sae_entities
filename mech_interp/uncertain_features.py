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

import json
import os
import copy
import gc
import torch
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from torch import Tensor
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Union, Optional, Literal, Tuple
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from utils.hf_models.model_factory import construct_model_base
from utils.utils import model_alias_to_model_name, paper_plot
from mech_interp_utils import html_colors, model_alias_to_sae_repo_id
from feature_analysis_utils import get_dataloader, get_acts_labels_dict_, get_per_layer_latent_scores
from feature_analysis_utils import get_general_latents
from mech_interp_utils import load_sae, layer_sparisity_widths
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score, roc_auc_score

# %%
def boxplot_latent_activations(sae_acts_entity_type, latent_idx, further_split):
    data = []

    colors = {'Correct': html_colors['green_drawio'], 'Error (Unknown entity)': html_colors['dark_red_drawio'], 'Error (Known entity)': html_colors['red_drawio']}
    # Iterate through each entity type
    for i, entity_type in enumerate(sae_acts_entity_type.keys()):
        latent_acts_1 = sae_acts_entity_type[entity_type]['Error (Unknown entity)'][:, latent_idx]
        latent_acts_2 = sae_acts_entity_type[entity_type]['Error (Known entity)'][:, latent_idx]
        latent_acts_0 = sae_acts_entity_type[entity_type]['Correct'][:, latent_idx]
        if further_split == True:
            for label, acts in [('Correct', latent_acts_0), ('Error (Unknown entity)', latent_acts_1), ('Error (Known entity)', latent_acts_2)]:
                data.append(go.Box(
                    y=acts,
                    name=f'{entity_type.capitalize()} - {label}',
                    marker_color=colors[label],
                    showlegend=False,
                    offsetgroup=i
                ))

        else:
            colors = {'Correct': html_colors['green_drawio'], 'Error': html_colors['dark_red_drawio']}
            # Iterate through each entity type
            latent_acts_1_ = np.concatenate([latent_acts_1, latent_acts_2])
            latent_acts_0_ = latent_acts_0
            for label, acts in [('Correct', latent_acts_0_), ('Error', latent_acts_1_)]:
                data.append(go.Box(
                    y=acts,
                    name=f'{entity_type.capitalize()} - {label}',
                    marker_color=colors[label],
                    showlegend=False,
                    offsetgroup=i
                ))
    # Create the layout
    layout = go.Layout(
        yaxis_title='Activation Value',
        boxmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            tickfont=dict(size=14),
            title=''
        ),
        yaxis=dict(
            gridcolor='rgba(200,200,200,0.3)',
            zeroline=True,
            zerolinecolor='rgba(200,200,200,0.3)',
            zerolinewidth=2
        )
    )

    # Create the figure and show it
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        title={
            'text': f'Error Latent Activations',
            'y': 0.9,  # Moves the title closer to the plot
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 14}  # Increase the font size of the title
        }
    )

    fig = paper_plot(fig, tickangle=45)
    os.makedirs('plots/uncertain_features', exist_ok=True)
    if further_split == True:
        fig.write_image(f"plots/uncertain_features/{model_alias}_latent_{latent_idx}_activation_distribution_by_entity_split.png",
                        scale=12, width=500, height=475)
        fig.write_image(f"plots/uncertain_features/{model_alias}_latent_{latent_idx}_activation_distribution_by_entity_split.pdf",
                        scale=12, width=500, height=475)
    else:
        fig.write_image(f"plots/uncertain_features/{model_alias}_latent_{latent_idx}_activation_distribution_by_entity.png",
                        scale=12, width=475, height=375)
        fig.write_image(f"plots/uncertain_features/{model_alias}_latent_{latent_idx}_activation_distribution_by_entity.pdf",
                        scale=12, width=475, height=375)
    fig.show()

# Function to calculate and print metrics
def calculate_metrics(y_true, scores, threshold):

    auroc = roc_auc_score(y_true, scores)

    y_pred = (scores > threshold).astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print("-----------------------------")

    plot_confusion_matrix(y_true, y_pred)

def find_optimal_threshold(y_true, scores, plot=True):
    """
    Find the optimal threshold for binary classification using ROC curve analysis.
    """
    # Calculate F1 scores for different thresholds
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    f1_scores = [f1_score(y_true, (scores >= threshold).astype(int)) for threshold in thresholds]
    
    # Find the threshold that maximizes F1 score
    optimal_f1_threshold = thresholds[np.argmax(f1_scores)]
    max_f1_score = max(f1_scores)

    if plot:
        # Plot F1 scores against thresholds
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores)
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title(f'F1 Scores vs Thresholds')
        plt.grid(True)
        plt.axvline(x=optimal_f1_threshold, color='r', linestyle='--', label=f'Optimal F1 Threshold: {optimal_f1_threshold:.3f}')
        plt.axhline(y=max_f1_score, color='g', linestyle='--', label=f'Max F1 Score: {max_f1_score:.3f}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    print('F1 optimal')
    calculate_metrics(y_true, scores, optimal_f1_threshold)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    
    # Calculate Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Get predictions using optimal threshold
    #y_pred = (scores >= optimal_threshold).astype(int)
    print('AUROC optimal')
    calculate_metrics(y_true, scores, optimal_threshold)

    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', 
                label=f'Optimal threshold: {optimal_threshold:.3f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve with Optimal Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return optimal_threshold, optimal_f1_threshold

# Function to create and plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    #plt.savefig(f"plots/{model_alias}_{entity_type}_latent_{latent_idx}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

def combine_across_entities(acts_entity_type : dict[str, dict[str, Tensor]], site : Literal['sae', 'residual'], latent_idx : Optional[int]=None):
    all_acts_combined = []
    y_true_combined = []

    for entity_type in acts_entity_type.keys():
        if site == 'sae':
            latent_acts_correct = acts_entity_type[entity_type]['Correct'][:, latent_idx]
            latent_acts_error = acts_entity_type[entity_type]['Error'][:, latent_idx] if 'Error' in acts_entity_type[entity_type] else np.concatenate([
                acts_entity_type[entity_type]['Error (Unknown entity)'][:, latent_idx],
                acts_entity_type[entity_type]['Error (Known entity)'][:, latent_idx]
            ])
        else:
            latent_acts_correct = acts_entity_type[entity_type]['Correct']
            latent_acts_error = acts_entity_type[entity_type]['Error'] if 'Error' in acts_entity_type[entity_type] else np.concatenate([
                acts_entity_type[entity_type]['Error (Unknown entity)'],
                acts_entity_type[entity_type]['Error (Known entity)']
            ])
        
        # Combine 'Correct' and 'Error' activations
        all_acts = np.concatenate([latent_acts_correct, latent_acts_error])
        all_acts_combined.append(all_acts)
        
        # Create true labels (0 for 'Correct', 1 for 'Error')
        y_true = np.concatenate([np.zeros(len(latent_acts_correct)), 
                                np.ones(len(latent_acts_error))])
        y_true_combined.append(y_true)

    # Concatenate all_acts and y_true across entity_type
    all_acts_combined = np.concatenate(all_acts_combined)
    y_true_combined = np.concatenate(y_true_combined)
    return y_true_combined, all_acts_combined

def balance_binary_tensors(y_true_np : np.ndarray, predictions_np : np.ndarray):
    """
    Balance binary classification tensors by subsampling the majority class
    to match the size of the minority class.
    
    Args:
        y_true (torch.Tensor): Binary tensor with ground truth labels (0s and 1s)
        predictions (torch.Tensor): Tensor with prediction scores
        
    Returns:
        tuple: (balanced_y_true, balanced_predictions)
    """    
    # Get indices for each class
    class_0_indices = np.where(y_true_np == 0)[0]
    class_1_indices = np.where(y_true_np == 1)[0]
    
    # Determine the size of the minority class
    min_class_size = min(len(class_0_indices), len(class_1_indices))
    
    # Randomly sample from the majority class
    if len(class_0_indices) > len(class_1_indices):
        class_0_indices = np.random.choice(class_0_indices, min_class_size, replace=False)
    else:
        class_1_indices = np.random.choice(class_1_indices, min_class_size, replace=False)
    
    # Combine indices and sort them to maintain order
    balanced_indices = np.sort(np.concatenate([class_0_indices, class_1_indices]))
    
    # Create balanced tensors
    balanced_y_true = y_true_np[balanced_indices]
    balanced_predictions = predictions_np[balanced_indices]
    
    return balanced_y_true, balanced_predictions

def train_logistic_probe(res_acts_balanced, y_true_balanced):
    # Train logistic regression probe
    X_train = res_acts_balanced
    y_train = y_true_balanced

    # Initialize and train logistic regression model
    lr_probe = LogisticRegression(random_state=42, max_iter=10000)
    lr_probe.fit(X_train, y_train)

    # # Get predictions on test set
    # y_pred = lr_probe.predict(X_test)
    # y_pred_proba = lr_probe.predict_proba(X_test)[:, 1]

    # # Calculate and print metrics
    # print("Logistic Regression Probe Results:")
    # calculate_metrics(y_test, y_pred_proba, threshold=0.5)
    return lr_probe
# %%
model_alias = 'gemma-2b-it'
SAE_WIDTH = '16k'
REPO_ID = model_alias_to_sae_repo_id[model_alias]
if model_alias == 'gemma-2b-it':
    # We remove player as there are not enough examples left
    ALL_ENTITY_TYPES = ['movie', 'city', 'song']
else:
    ALL_ENTITY_TYPES = ['player', 'movie', 'city', 'song']
model_path = model_alias_to_model_name[model_alias]
# Load model to load tokenizer and config data
model_base = construct_model_base(model_path)
d_model = model_base.model.config.hidden_size
tokenizer = model_base.tokenizer
n_layers = model_base.model.config.num_hidden_layers
del model_base

# We compute SAE latent scores for all layers available
if model_alias == 'gemma-2b-it':
    LAYERS_WITH_SAE = [13]
elif model_alias == 'gemma-2-9b-it':
    LAYERS_WITH_SAE = [10, 21, 32]
else:
    LAYERS_WITH_SAE = list(range(1, n_layers))

# %%
### Unknown/Uncertainty Latent Analysis ####
# Correct/Error answers (IT model)
triviaqa_prompts_experiment = {
    'dataset_name' : 'triviaqa',
    'evaluate_on' : 'prompts',
    'scoring_method' : 't_test',
    'tokens_to_cache' : 'model',# Token whose cached activations we want to access
    'free_generation' : True,
    'consider_refusal_label' : True,
    'split' : None,
    'further_split' : False,
    }

wikidata_prompts_experiment = {
    'dataset_name' : 'wikidata',
    'evaluate_on' : 'prompts',
    'scoring_method' : 't_test',
    'tokens_to_cache' : 'model',# Token whose cached activations we want to access
    'free_generation' : True,
    'consider_refusal_label' : True,
    'split' : 'train',
    'further_split' : True,
    'entity_type_and_entity_name_format' :False,
    }

experiment_args = copy.deepcopy(wikidata_prompts_experiment)

# %%
save = True
if experiment_args['dataset_name'] == 'triviaqa':
    get_per_layer_latent_scores(model_alias, tokenizer, n_layers, d_model, LAYERS_WITH_SAE, save=True, **experiment_args)

else:
    for entity_type in ALL_ENTITY_TYPES:
        experiment_args['dataset_name'] = f'wikidata_{entity_type}'
        get_per_layer_latent_scores(model_alias, tokenizer, n_layers, d_model, LAYERS_WITH_SAE, save=True, **experiment_args)

    
# %%
### Searching for the top general latents ###
tokens_to_cache = 'model'
evaluate_on = 'prompts'
scoring_method = 't_test'
testing_layers = LAYERS_WITH_SAE
get_general_latents(model_alias, ALL_ENTITY_TYPES, testing_layers, tokens_to_cache, evaluate_on, scoring_method)


# %%
def compute_acts_for_layer(model_alias, layer, site='sae', **kwargs):
    def compute_res_stream_acts(acts_labels_dict, layer):
        acts_labels = acts_labels_dict[layer]
        # Get labels
        label_indices_error_known = torch.where(acts_labels['labels'] == 3.0)[0].cpu().numpy() # Error (Known entity)
        label_indices_error_unknown = torch.where(acts_labels['labels'] == 1.0)[0].cpu().numpy() # Error (Unknown entity)
        label_indices_correct = torch.where(acts_labels['labels'] == 0.0)[0].cpu().numpy() # Correct
        # Divide sae activations into known and unknown
        res_stream_acts_error_known = acts_labels['acts'][label_indices_error_known, :].cpu().numpy() # Error (Known entity)
        res_stream_acts_error_unknown = acts_labels['acts'][label_indices_error_unknown, :].cpu().numpy() # Error (Unknown entity)
        res_stream_acts_correct = acts_labels['acts'][label_indices_correct, :].cpu().numpy() # Correct

        return {'Correct': res_stream_acts_correct, 'Error (Unknown entity)': res_stream_acts_error_unknown, 'Error (Known entity)': res_stream_acts_error_known}
        
    def compute_sae_acts(model_alias, acts_labels_dict, layer):
        # Load SAE
        if model_alias == 'gemma-2b-it':
            assert layer == 13, "Layer 13 is the only layer for gemma-2b-it"
            sae_id = "gemma-2b-it-res-jb"
        elif model_alias == 'gemma-2-9b-it':
            assert layer in [10, 21, 32], "Layer 10, 21, 32 are the only layers for gemma-2-9b-it"
            sae_sparsity = layer_sparisity_widths[model_alias][layer-1][SAE_WIDTH]
            sae_id = f"layer_{layer-1}/width_{SAE_WIDTH}/average_l0_{str(sae_sparsity)}"
        else:
            sae_sparsity = layer_sparisity_widths[model_alias][layer-1][SAE_WIDTH]
            sae_id = f"layer_{layer-1}/width_{SAE_WIDTH}/average_l0_{str(sae_sparsity)}"
        sae = load_sae(REPO_ID, sae_id)

        acts_labels = acts_labels_dict[layer]
        # Get activations
        sae_acts = sae.encode(acts_labels['acts'])

        # Get labels
        label_indices_error_known = torch.where(acts_labels['labels'] == 3.0)[0].detach().cpu().numpy() # Error (Known entity)
        label_indices_error_unknown = torch.where(acts_labels['labels'] == 1.0)[0].detach().cpu().numpy() # Error (Unknown entity)
        label_indices_correct = torch.where(acts_labels['labels'] == 0.0)[0].detach().cpu().numpy() # Correct
        # Divide sae activations into known and unknown
        sae_acts_error_known = sae_acts[label_indices_error_known, :].detach().cpu().numpy() # Error (Known entity)
        sae_acts_error_unknown = sae_acts[label_indices_error_unknown, :].detach().cpu().numpy() # Error (Unknown entity)
        sae_acts_correct = sae_acts[label_indices_correct, :].detach().cpu().numpy() # Correct

        return {'Correct': sae_acts_correct, 'Error (Unknown entity)': sae_acts_error_unknown, 'Error (Known entity)': sae_acts_error_known}
    
    batch_size = 128

    sae_acts_entity_type = {}
    res_stream_acts_entity_type = {}
    if 'wikidata' in kwargs['dataset_name']:
        for entity_type in ALL_ENTITY_TYPES:
            kwargs.update({'dataset_name': f'wikidata_{entity_type}'})
            dataloader = get_dataloader(model_alias, kwargs['tokens_to_cache'], n_layers, d_model, kwargs['dataset_name'], batch_size=batch_size)
            acts_labels_dict = get_acts_labels_dict_(model_alias, tokenizer, dataloader, LAYERS_WITH_SAE, **kwargs)
            if site == 'sae':
                sae_acts_entity_type[entity_type] = compute_sae_acts(model_alias, acts_labels_dict, layer)
            else:
                res_stream_acts_entity_type[entity_type] = compute_res_stream_acts(acts_labels_dict, layer)

    else:
        dataloader = get_dataloader(model_alias, kwargs['tokens_to_cache'], n_layers, d_model, kwargs['dataset_name'], batch_size=batch_size)
        acts_labels_dict = get_acts_labels_dict_(model_alias, tokenizer, dataloader, LAYERS_WITH_SAE, **kwargs)
        if site == 'sae':
             sae_acts_entity_type['triviaqa'] = compute_sae_acts(model_alias, acts_labels_dict, layer)
        else:
            res_stream_acts_entity_type['triviaqa'] = compute_res_stream_acts(acts_labels_dict, layer)

    return sae_acts_entity_type if site == 'sae' else res_stream_acts_entity_type

# %%
# Train residual stream probe
experiment_args = copy.deepcopy(wikidata_prompts_experiment)
experiment_args.update({'split': 'train'})
layer = 13

res_stream_acts_entity_type = compute_acts_for_layer(model_alias, layer, site = 'residual', **experiment_args)
y_true_combined, all_acts_combined = combine_across_entities(res_stream_acts_entity_type, site='residual')
y_true_balanced, res_acts_balanced = balance_binary_tensors(y_true_combined, all_acts_combined)
lr_probe = train_logistic_probe(res_acts_balanced, y_true_balanced)

del res_acts_balanced, y_true_balanced

# %%
# Validation set
experiment_args = copy.deepcopy(wikidata_prompts_experiment)
experiment_args.update({'split': 'validation'})

# Residual stream
res_stream_acts_entity_type = compute_acts_for_layer(model_alias, layer, site = 'residual', **experiment_args)
y_true_combined, all_acts_combined = combine_across_entities(res_stream_acts_entity_type, site='residual')
y_true_balanced, res_acts_balanced = balance_binary_tensors(y_true_combined, all_acts_combined)
y_pred_proba = lr_probe.predict_proba(res_acts_balanced)[:, 1]
optimal_probe_threshold, optimal_probe_f1_threshold = find_optimal_threshold(y_true_balanced, y_pred_proba)

print(f"Optimal threshold: {optimal_probe_threshold:.3f}, F1 optimal threshold: {optimal_probe_f1_threshold:.3f}")
print(f"Metrics at optimal threshold:")
# %%
# SAE
latent_idx = 3130
sae_acts_entity_type = compute_acts_for_layer(model_alias, layer, site = 'sae', **experiment_args)
y_true_combined, all_acts_combined = combine_across_entities(sae_acts_entity_type, site='sae', latent_idx=latent_idx)
y_true_balanced, predictions_balanced = balance_binary_tensors(y_true_combined, all_acts_combined)
optimal_sae_threshold, optimal_sae_f1_threshold = find_optimal_threshold(y_true_balanced, predictions_balanced, plot=False)

print(f"Optimal threshold: {optimal_sae_threshold:.3f}, F1 optimal threshold: {optimal_sae_f1_threshold:.3f}")
print(f"Metrics at optimal threshold:")

# %%
# Test set
experiment_args = copy.deepcopy(wikidata_prompts_experiment)
experiment_args.update({'split': 'test'})

# Residual stream
res_stream_acts_entity_type = compute_acts_for_layer(model_alias, layer, site = 'residual', **experiment_args)
y_true_combined, all_acts_combined = combine_across_entities(res_stream_acts_entity_type, site='residual')
y_true_balanced, res_acts_balanced = balance_binary_tensors(y_true_combined, all_acts_combined)
y_pred_proba = lr_probe.predict_proba(res_acts_balanced)[:, 1]

calculate_metrics(y_true_balanced, y_pred_proba, optimal_probe_f1_threshold)

# SAE
latent_idx = 3130
sae_acts_entity_type = compute_acts_for_layer(model_alias, layer, site = 'sae', **experiment_args)
y_true_combined, all_acts_combined = combine_across_entities(sae_acts_entity_type, site='sae', latent_idx=latent_idx)
y_true_balanced, sae_acts_balanced = balance_binary_tensors(y_true_combined, all_acts_combined)

y_true_combined, all_acts_combined = combine_across_entities(sae_acts_entity_type, site='sae', latent_idx=latent_idx)
y_true_balanced, predictions_balanced = balance_binary_tensors(y_true_combined, all_acts_combined)

calculate_metrics(y_true_balanced, predictions_balanced, optimal_sae_f1_threshold)

# %%
latent_idx = 3130
further_split = True
boxplot_latent_activations(sae_acts_entity_type, latent_idx, further_split)


# %%
# Generalization to TriviaQA
# Residual stream
res_stream_acts_entity_type = compute_acts_for_layer(model_alias, layer, site = 'residual', **triviaqa_prompts_experiment)
y_true_combined, all_acts_combined = combine_across_entities(res_stream_acts_entity_type, site='residual')
y_true_balanced, res_acts_balanced = balance_binary_tensors(y_true_combined, all_acts_combined)
y_pred_proba = lr_probe.predict_proba(res_acts_balanced)[:, 1]

calculate_metrics(y_true_balanced, y_pred_proba, optimal_probe_f1_threshold)
# %%
# SAE
latent_idx = 3130
sae_acts_entity_type = compute_acts_for_layer(model_alias, layer, site = 'sae', **triviaqa_prompts_experiment)
y_true_combined, all_acts_combined = combine_across_entities(sae_acts_entity_type, site='sae', latent_idx=latent_idx)
y_true_balanced, sae_acts_balanced = balance_binary_tensors(y_true_combined, all_acts_combined)

y_true_combined, all_acts_combined = combine_across_entities(sae_acts_entity_type, site='sae', latent_idx=latent_idx)
y_true_balanced, predictions_balanced = balance_binary_tensors(y_true_combined, all_acts_combined)

# %%
calculate_metrics(y_true_balanced, predictions_balanced, optimal_sae_f1_threshold)

# %%
