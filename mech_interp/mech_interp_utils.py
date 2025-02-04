import os
import torch
import einops
import gc
import plotly.express as px
import plotly.graph_objects as go
from dataset.load_data import load_wikidata_queries
from collections import defaultdict
import numpy as np
import copy
from typing import List, Union, Optional, Literal, Tuple
from torch import Tensor
from fancy_einsum import einsum
from jaxtyping import Float
from transformer_lens import HookedTransformer, ActivationCache
import transformer_lens.patching as patching
from functools import partial
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from adjustText import adjust_text
from tqdm import tqdm
from utils.sae_utils import load_sae, layer_sparisity_widths
from utils.utils import paper_plot, find_string_in_tokens, slice_to_list
import json
import scipy.stats as stats
import time
import random
random_seed = 42
random.seed(random_seed)

model_alias_to_sae_repo_id = {
    "gemma-2-2b": "google/gemma-scope-2b-pt-res",
    "gemma-2-9b": "google/gemma-scope-9b-pt-res",
    "gemma-2-9b-it": "google/gemma-scope-9b-it-res",
    "gemma-2b-it": "gemma-2b-it-res-jb", # Joseph Bloom's SAE
    "meta-llama_Llama-3.1-8B": "llama_scope_lxr_8x", # Joseph Bloom's SAE
    "Llama-3.1-8B": "llama_scope_lxr_8x", # Joseph Bloom's SAE

}


# Entity type relations consist of three tokens
# Relations used in the base models
entity_type_to_base_relations = {
    'city_country': 'is located in',
    'city_elevation': 'is at an elevation of',
    'city_population': 'has a population of',
    'movie_cast': 'has starring',
    'movie_directors': 'is directed by',
    'movie_durations': 'has a duration of',
    'movie_genres': 'has the genre',
    'movie_release_year': 'was released in',
    'movie_screenwriters': 'is written by',
    'player_date_birth': 'was born in the city of',
    'player_place_birth': 'was born in the year',
    'player_teams_list': 'has played for',
    'song_album': "appears in the album",
    'song_genres': 'has the genre',
    'song_performers': 'is sung by',
    'song_publication_year': 'was released in',
}
# Relations used in the chat models
entity_type_to_it_relations = {
    'city_country': 'In which country is',
    'city_elevation': 'What is the elevation of',
    'city_population': 'What is the population of',
    'movie_cast': 'Who is starring in',
    'movie_directors': 'Who directed',
    'movie_durations': 'What is the duration of',
    'movie_genres': 'What is the genre of',
    'movie_release_year': 'What is the release year of',
    'movie_screenwriters': 'Who wrote',
    'player_date_birth': 'When was born',
    'player_place_birth': 'Where was born',
    'player_teams_list': 'In which teams have played',
    'song_album': "In which album appears",
    'song_genres': 'What is the genre of',
    'song_performers': 'Who is the singer of',
    'song_publication_year': 'What is the release year of',    
}

# relations_to_entity_type = {
#     'is directed by': 'movie_directors',
#     'is written by': 'movie_screenwriters',
#     'is located in': 'city_country',
#     'is sung by': 'song_performers',
#     "appears in album": 'song_album',
#     'has played for': 'player_teams_list',
#     'was born in': 'player_place_birth',
# }

# TODO: generalize this to other models
entity_type_to_token_pos = {
    'city': 4,
    'movie': 5,
    'song': 5,
    'player': 4,
}

relation_type_to_token_pos = {
    'city': 3,
    'movie': 3,
    'song': 3,
    'player': 3,
}

html_colors = {
    'darkgreen' : '#138808',
    'green_drawio' : '#82B366',
    'dark_green_drawio' : '#557543',
    'dark_red_drawio' : '#990000',
    'blue_drawio' : '#6C8EBF',
    'orange_drawio' : '#D79B00',
    'red_drawio' : '#FF9999',# 990000
    'grey_drawio' : '#303030',
    'brown_D3' : '#8C564B',
    'orange_matplotlib': '#ff7f0e',
    'blue_matplotlib': '#1f77b4'}

def clean_blocks_labels(label: str) -> str:
    """
    Convert model block label into a more readable format.

    Args:
        label (str): The model component label to be cleaned.

    Returns:
        str: The cleaned label.

    """
    if label == 'embed':
        return 'Emb'
    else:
        component = label.split('_')[1]
        layer = label.split('_')[0]
        if component == 'mlp':
            label = f'MLP{layer}'
        else:
            label = f'Attn{layer}'
        return label

## Entities dataset processing functions

def get_label(query: dict, threshold: int = 3, consider_refusal_label: bool = False):
    if consider_refusal_label == True:
        if query['label'] == 'refusal':
            return 2

    total_correct = sum([1.0 if label == 'correct' else 0.0 for label in query['string_matching_sampled_labels'] ])
    return 1 if total_correct>= threshold else 0
    

def get_known_unknown_splits(queries: list, min_correct: int = 1, return_type: Literal['entity', 'prompt'] = 'entity', split: str = 'train', consider_refusal_label: bool = False):
    """
    Categorize entities as known or unknown based on query results.

    This function processes a list of queries for a specific entity type and categorizes
    the entities as either 'known' or 'unknown' based on the correctness of their
    associated attributes responses.

    Args:
        queries (list): A list of query dictionaries containing information about
                        entity queries and their results.
        entity_type (str): The type of entity to process (e.g., 'city', 'movie', etc.).
        min_correct (int, optional): min_correct+1 is the minimum number of correct responses required
                                   for an entity to be considered 'known'. Defaults to 1.

    Returns:
        tuple: A tuple containing two lists:
               - known (list): Entities/prompts with correct responses above the min_correct.
               - unknown (list): Entities/prompts with no correct responses.

    Note:
        - The function uses the 'get_label' helper function to determine if a query
          response is correct.
        - Entities with some correct responses but below the min_correct are not
          included in either returned list.
    """

    # dictionary with entity and number of correct completions taken from string_matching_sampled_labels, see get_label
    results = {}
    if return_type == 'entity':
        assert consider_refusal_label == False, "Refusal label is not supported when working with entities"
    for query in queries:
        if query[return_type] not in results.keys():
            results[query[return_type]] = 0
        results[query[return_type]] += get_label(query, consider_refusal_label=consider_refusal_label)

    known = []
    unknown = []

    if return_type == 'entity':
        # We use entity rules for classifying into known and unknown
        for entity_name in results.keys():
            if results[entity_name] > min_correct:
                known.append(entity_name)
            elif results[entity_name] == 0:
                unknown.append(entity_name)
    else:
        # Classify prompts into known (correct answer) and unknown (incorrect answer)
        for prompt in results.keys():
            if results[prompt] == 1:
                known.append(prompt)
            elif results[prompt] == 0:
                unknown.append(prompt)

    train_known, test_known = train_test_split(known, test_size=0.5, random_state=random_seed)
    train_unknown, test_unknown = train_test_split(unknown, test_size=0.5, random_state=random_seed)
    # Create validation split
    val_known, test_known = train_test_split(test_known, test_size=0.8, random_state=random_seed)
    val_unknown, test_unknown = train_test_split(test_unknown, test_size=0.8, random_state=random_seed)

    # We return the splits in the order train, validation, test (these can be either list of entities or prompts)
    if split == 'train':
        return train_known, train_unknown
    elif split == 'validation':
        return val_known, val_unknown
    elif split == 'test':
        return test_known, test_unknown
    else:
        return known, unknown

def get_valid_entities(queries, tokenizer, entity_type, entity_token_pos, split='train', fixed_length=False):
    print('queries', len(queries))
    queries = [query for query in queries if query['entity_type'] == entity_type]
    known_entities, unknown_entities = get_known_unknown_splits(queries, return_type='entity', split=split)
    

    valid_entities = defaultdict(list)
    for known_label, entities in zip(['known', 'unknown'], [known_entities, unknown_entities]):
        for entity in entities:
            tokenized_entity = torch.tensor(tokenizer.encode(f'The {entity_type} {entity}')).to('cuda')
            if fixed_length:
                if len(tokenized_entity) == entity_token_pos+1:
                    valid_entities[known_label].append(entity)
            else:
                valid_entities[known_label].append(entity)

    print(f"Number of known entities: {len(valid_entities['known'])}")
    print(f"Number of unknown entities: {len(valid_entities['unknown'])}")

    

    num_entities = min(len(valid_entities['known']), len(valid_entities['unknown']))
    valid_entities['known'] = valid_entities['known'][:num_entities]
    valid_entities['unknown'] = valid_entities['unknown'][:num_entities]

    return valid_entities

def shuffle_lists(list1, list2, list3):
    combined = list(zip(list1, list2, list3))
    random.shuffle(combined)
    return map(list, zip(*combined))

def create_prompts_and_answers(tokenizer, queries, entity_type, known_label, valid_entities, prompt_template, relations_model_type: Literal['base', 'it']='base', fixed_relation_length: bool = False):
    if relations_model_type == 'it':
        entity_type_to_relations = entity_type_to_it_relations
    else:
        entity_type_to_relations = entity_type_to_base_relations
    # Get only those of entity_type
    entity_type_to_relations = {key: value for (key, value) in entity_type_to_relations.items() if key.startswith(entity_type)}
    # If length restriction, leave only those matching the length
    if fixed_relation_length:
        valid_relations = {}
        for prompt_type, relation in entity_type_to_relations.items():
            # We add The because tokenizers are weird and tokenize differently middle of sentence words
            tokenized_realation = torch.tensor(tokenizer.encode(f"The {relation}")).to('cuda')
            if len(tokenized_realation) == relation_type_to_token_pos[entity_type]+2:
                valid_relations[prompt_type] = relation
    else:
        valid_relations = entity_type_to_relations

    # Reverse the valid_relations (key, value) -> (value, key)
    relations_to_entity_type = {key: value for (value, key) in valid_relations.items()}
    entity_type_relations = [valid_relations[key] for key in valid_relations if key.startswith(entity_type)]
    # Create prompts and answers    
    prompts_label = []
    answers_label = []
    entities_label = []

    for entity in valid_entities[known_label]:
        for relation in entity_type_relations:
            prompts_label.append(prompt_template.format(entity_name=entity, relation=relation, entity_type=entity_type))
            for query in queries:
                if query['entity'] == entity and query['prompt_type'] == relations_to_entity_type[relation]:
                    answers_label.append(' ' + query['correct_answer'].split(',')[-1].strip())
                    entities_label.append(entity)
                    break
    # Check there are the same number of prompts and answers
    assert len(prompts_label) == len(answers_label)
    prompts_label, answers_label, entities_label = shuffle_lists(prompts_label, answers_label, entities_label)

    return prompts_label, answers_label, entities_label

def load_data(model, queries, entity_type, tokenizer, known_label, prompt_template, conversation=None, relations_model_type: Literal['base', 'it']='base', split: Literal['train', 'validation', 'test'] = 'test', fixed_config: Optional[Literal['fixed_entities', 'both']] = None):

    entity_token_pos = entity_type_to_token_pos[entity_type]
    if fixed_config == 'fixed_entities' or fixed_config == 'both':
        fixed_entities_length = True
        if fixed_config == 'both':
            fixed_relation_length = True
        else:
            fixed_relation_length = False
    else:
        fixed_entities_length = False
        fixed_relation_length = False
    valid_entities = get_valid_entities(copy.deepcopy(queries), tokenizer, entity_type, entity_token_pos, split=split, fixed_length=fixed_entities_length)
    prompts_label, answers_label, entities_label = create_prompts_and_answers(tokenizer, copy.deepcopy(queries), entity_type, known_label,
                                                                              valid_entities, prompt_template, relations_model_type,
                                                                              fixed_relation_length=fixed_relation_length)
    tokenized_prompts = []
    formatted_instructions = []
    if relations_model_type == 'it':
        assert conversation is not None, "Conversation is required for IT model"
        for prompt in prompts_label:
            conversation[0]['content'] = prompt
            formatted_instructions.append(tokenizer.apply_chat_template(conversation, tokenize=False).replace(tokenizer.bos_token, '')[:-len('<end_of_turn>\n')])
    else:
        formatted_instructions = prompts_label
    tokenized_prompts = model.to_tokens(formatted_instructions)
    print('tokenized_prompts', tokenized_prompts.shape)

    pos_entities = []
    for i in range(len(tokenized_prompts)):
        entity = entities_label[i]
        # TODO: fix this
        if 'Lou ! Journal infime' in entity:
            entity = entity.replace('Lou ! Journal infime', 'Lou! Journal infime')
        slice_entity_pos = find_string_in_tokens(entity, tokenized_prompts[i], tokenizer)
        list_entity_pos = slice_to_list(slice_entity_pos)
        pos_entities.append(list_entity_pos)

    return tokenized_prompts, pos_entities, formatted_instructions, answers_label

def get_attn_to_entity_tok_mean(model, cache, entity_token_pos, n_layers, type_pattern: Literal['attn_weights', 'value_weighted']='attn_weights'):
    attn_to_entity_tok_list = []
    for layer in range(n_layers):
        if type_pattern == 'attn_weights':
            attn_patterns = cache["pattern", layer]
            attn_to_entity_toks = attn_patterns[:,:,-1]
            attn_to_entity_toks_sum = torch.zeros(attn_to_entity_toks.shape[0], attn_patterns.shape[1])
            for batch_idx in range(attn_patterns.shape[0]):
                attn_to_entity_toks_sum[batch_idx] = attn_to_entity_toks[batch_idx, :, entity_token_pos[batch_idx]].sum(dim=-1)
            attn_to_entity_tok_list.append(attn_to_entity_toks_sum)
        elif type_pattern == 'value_weighted':
            # batch query_pos key_pos head_index d_head
            weighted_values = get_value_weighted_patterns(model, cache, layer)
            # Value-weighted norms
            raw_inter_token_attribution = torch.norm(weighted_values, dim=-1, p=2)
            attn_patterns = einops.rearrange(raw_inter_token_attribution, 'batch query_pos key_pos head_index -> batch head_index query_pos key_pos')
            # weighted_values_norm -> [batch query_pos key_pos head_index]
            attn_to_entity_toks = attn_patterns[:,:,-1]
            attn_to_entity_toks_sum = torch.zeros(attn_to_entity_toks.shape[0], attn_patterns.shape[1])
            for batch_idx in range(attn_patterns.shape[0]):
                attn_to_entity_toks_sum[batch_idx] = attn_to_entity_toks[batch_idx, :, entity_token_pos[batch_idx]].sum(dim=-1)
            attn_to_entity_tok_list.append(attn_to_entity_toks_sum)
        else:
            raise ValueError(f"Invalid type_pattern: {type_pattern}")

    
    attn_to_entity_tok_stack = torch.stack(attn_to_entity_tok_list, 0)
    # [n_layers, batch_size, n_heads]
    # Rearrange the tensor to have shape [batch_size, n_layers, n_heads]
    attn_to_entity_tok_stack = einops.rearrange(attn_to_entity_tok_stack, 'layers batch heads -> batch layers heads')
        

    return attn_to_entity_tok_stack.cpu()#.numpy()


def load_steering_latents(
    entity_type: str,
    topk: int = 10,
    label: Literal['known', 'unknown'] = 'known',
    layers_range: List[int] = [5],
    random_latents: bool = False,
    scoring_method: Literal['absolute_difference', 'relative_difference'] = 'absolute_difference',
    specific_latents: List[Tuple[int, int]] = None,
    input_latent=False,
    model_alias: str = 'gemma-2-2b',
    tokens_to_cache: str = 'entity',
    evaluate_on: Literal['entities', 'prompts'] = 'entities'
    ):
    random.seed(random_seed)
    print('Loading steering latents for model:', model_alias)

    if random_latents:
        print('Loading random latents')
        
    # Read top features
    # TODO: make this dynamic if we change model
    sae_width = '16k'
    repo_id = model_alias_to_sae_repo_id[model_alias]
    if isinstance(specific_latents, list):
        layers_range = [latent[0] for latent in specific_latents]
    feats_layers = read_layer_features(model_alias, layers_range, entity_type, scoring_method, tokens_to_cache, evaluate_on)

    #print('feats_layers[13]', feats_layers[13][label].keys())

    # layer_known_label_dict = feats_layers[layer][label]
    # layer_known_label_dict[idx]['score']
    top_sae_latents = []
    
    top_sae_latents_dict = get_top_k_features(feats_layers, k=topk)
    sae_latent_dict = top_sae_latents_dict[label]

    if specific_latents is not None:
        indices = []
        all_sae_latents_dict = get_top_k_features(feats_layers, k=None)
        for layer, latent_idx in specific_latents:
            for sae_latent_idx in all_sae_latents_dict[label].keys():
                if all_sae_latents_dict[label][sae_latent_idx]['layer'] == layer and all_sae_latents_dict[label][sae_latent_idx]['latent_idx'] == latent_idx:
                    indices.append(sae_latent_idx)
        sae_latent_dict = {i: all_sae_latents_dict[label][idx] for i, idx in enumerate(indices)}

    if random_latents:
        all_sae_latents_dict = get_top_k_features(feats_layers, k=None)
        # select and index
        available_indices = list(range(len(all_sae_latents_dict[label])))

        # Remove top indices
        # for top_indices in top_sae_latents_dict[label].keys():
        #     available_indices.remove(top_indices)
        # Remove indices of latents with score > 0

        min_max_scores = json.load(open(f"./train_latents_layers_entities/absolute_difference/{model_alias.split('/')[-1]}/entity/sorted_scores_min_{label}.json"))
        for idx in all_sae_latents_dict[label].keys():
            latent_id = f"L{all_sae_latents_dict[label][idx]['layer']}F{all_sae_latents_dict[label][idx]['latent_idx']}"
            if abs(min_max_scores[latent_id]) > 0.0:
                available_indices.remove(idx)

        indices = random.sample(available_indices, topk)
        sae_latent_dict = {i: all_sae_latents_dict[label][idx] for i, idx in enumerate(indices)}
    
    for i, (ranking, latent_info) in enumerate(sae_latent_dict.items()):
        layer = latent_info['layer']
        feat_idx = latent_info['latent_idx']
        mean_act = latent_info['mean_features_acts']

        top_sae_latents.append({
            'layer': int(layer),
            'latent_idx': int(feat_idx),
            'mean_act': mean_act,
        })

        if i >= topk-1 and not specific_latents:
            break

    # group them by layer and extract the corresponding directions
    layers = set([l['layer'] for l in top_sae_latents])

    # Read SAE latents from SAEs
    latents = []
    for layer in layers:
        if model_alias == 'gemma-2b-it':
            assert layer == 13, "Layer 13 is the only layer for gemma-2b-it"
            sae_id = "gemma-2b-it-res-jb"
        elif model_alias == 'meta-llama/Llama-3.1-8B' or model_alias == 'meta-llama_Llama-3.1-8B':
            #assert layer in [14,15,16,17,18,19], "Layer 14, 15, 16, 17, 18, 19 are the only layers for meta-llama/Llama-3.1-8B"
            sae_id = f"l{layer-1}r_8x"
        elif model_alias == 'gemma-2-9b-it':
            assert layer in [10, 21, 32], "Layer 10, 21, 32 are the only layers for gemma-2-9b-it"
            sae_sparsity = layer_sparisity_widths[model_alias][layer-1][sae_width]
            sae_id = f"layer_{layer-1}/width_{sae_width}/average_l0_{str(sae_sparsity)}"
        else:
            sae_id = f"layer_{layer-1}/width_{sae_width}/average_l0_{str(layer_sparisity_widths[model_alias][layer-1][sae_width])}"
        sae = load_sae(repo_id, sae_id)
        for latent_info in top_sae_latents:
            if latent_info['layer'] == layer:
                latent_idx = latent_info['latent_idx']
                mean_act = latent_info['mean_act']
                if input_latent:
                    direction = sae.W_enc[:,latent_idx].detach()#.cpu()
                else:
                    direction = sae.W_dec[latent_idx].detach()#.cpu()
                latents.append((layer, latent_idx, mean_act, direction))
        del sae

    return latents

### Features Analysis
def get_prompt_from_ids(model_alias,input_ids, tokenizer):
    decoded_strings = tokenizer.batch_decode(input_ids)
    if 'gemma' in model_alias.lower():
        cleaned_decoded_strings = [decoded_string.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '') for decoded_string in decoded_strings]
    elif 'llama' in model_alias.lower():
        cleaned_decoded_strings = [decoded_string.replace('<|end_of_text|>', '').replace('<|begin_of_text|>', '') for decoded_string in decoded_strings]
    return cleaned_decoded_strings

def get_acts_labels_dict(model_alias, tokenizer, dataloader, queries, prompts: List[str], known_prompts_or_entities: List[str], unknown_prompts_or_entities: List[str], input_data_type: Literal['prompts', 'entities'], layers: List[int], num_samples: int=None):
    """
    Generate a dictionary containing activations and labels for each layer.

    This function categorizes the activations based on whether the entity in the prompt is known or unknown,
    and stores the activations and labels in a dictionary for each specified layer.

    Args:
        tokenizer: The tokenizer used to decode the input IDs.
        dataloader: A DataLoader object that provides batches of activations and input IDs.
        queries: A list of query dictionaries containing information about the entities and prompts.
        prompts: A list of prompt strings corresponding to the queries.
        known_prompts: A list of prompts that are known.
        unknown_prompts: A list of prompts that are unknown.
        layers: A list of layer indices for which activations and labels are to be extracted.
        num_samples: An optional integer specifying the maximum number of samples to include in the output.

    Returns:
        A dictionary where the keys are layer indices and the values are tuples containing:
            - A tensor of activations for the specified layer.
            - A tensor of labels indicating whether the entity in the prompt is known (0) or unknown (1).
    """
    eoi_string = "model\n"
    assert 0 not in layers, "Layer 0 is not a valid layer for the SAE"
    prompts = [prompt.replace('<bos>', '') if prompt.startswith('<bos>') else prompt for prompt in prompts]

    if input_data_type == 'prompts':
        prompts = set(prompts)
        known_prompts = set([prompt.replace('<bos>', '') if prompt.startswith('<bos>') else prompt for prompt in known_prompts_or_entities])
        unknown_prompts = set([prompt.replace('<bos>', '') if prompt.startswith('<bos>') else prompt for prompt in unknown_prompts_or_entities])
    elif input_data_type == 'entities':
        known_entities = set(known_prompts_or_entities)
        unknown_entities = set(unknown_prompts_or_entities)

    acts_labels_dict = {}
    labels = []
    activations_list = []
    final_prompts = []
    for _, (batch_activations, batch_input_ids) in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_cleaned_decoded_strings = get_prompt_from_ids(model_alias, batch_input_ids, tokenizer)
        for j, clean_string in enumerate(batch_cleaned_decoded_strings):
            if input_data_type == 'prompts':
                if 'gemma' in model_alias.lower():
                    clean_string = clean_string[:clean_string.find(eoi_string)+len(eoi_string)]
            if clean_string in prompts:
                if input_data_type == 'prompts':
                    if clean_string in unknown_prompts:
                        labels.append(1)
                        activations_list.append(batch_activations[j][:,0].to('cuda'))
                        final_prompts.append(clean_string)
                    elif clean_string in known_prompts:
                        labels.append(0)
                        activations_list.append(batch_activations[j][:,0].to('cuda'))
                        final_prompts.append(clean_string)
                elif input_data_type == 'entities':
                    # We check entities instead of prompts
                    entity = queries[prompts.index(clean_string)]['entity']
                    if entity in unknown_entities:
                        labels.append(1)
                        activations_list.append(batch_activations[j][:,0].to('cuda'))
                        final_prompts.append(clean_string)
                    elif entity in known_entities:
                        labels.append(0)
                        activations_list.append(batch_activations[j][:,0].to('cuda'))
                        final_prompts.append(clean_string)
        
    labels_full = torch.tensor(labels)
    if num_samples is not None:
        num_samples = min(num_samples, len(activations_full))
    for layer in layers:
        activations_full = torch.stack([activations[layer] for activations in activations_list], dim=0)
        acts_labels = {'acts': activations_full[:num_samples], 'labels': labels_full[:num_samples], 'prompts': final_prompts[:num_samples]}
        acts_labels_dict[layer] = acts_labels
    return acts_labels_dict

def get_features_layers(model_alias, acts_labels_dict, layers, sae_width, repo_id, save=True, **kwargs):
    """
    Get the features for each layer using the SAE.

    This function loads the SAE for each specified layer, computes the activations for known and unknown entities,
    and extracts the top features based on the activation differences.

    Args:
        model_alias: The alias of the model.
        entity_type: The type of entity.
        acts_labels_dict: A dictionary containing activations and labels for each layer.
        layers: A list of layer indices for which activations and labels are to be extracted.
        sae_width: The width of the SAE.
        repo_id: The repository ID of the SAE.
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
        # Load SAE
        if model_alias == 'gemma-2b-it':
            assert layer == 13, "Layer 13 is the only layer for gemma-2b-it"
            sae_id = "gemma-2b-it-res-jb"
        elif model_alias == 'meta-llama/Llama-3.1-8B' or model_alias == 'meta-llama_Llama-3.1-8B':
            #assert layer in [14,15,16,17,18,19], "Layer 14, 15, 16, 17, 18, 19 are the only layers for meta-llama/Llama-3.1-8B"
            sae_id = f"l{layer-1}r_8x"
        elif model_alias == 'gemma-2-9b-it':
            assert layer in [10, 21, 32], "Layer 10, 21, 32 are the only layers for gemma-2-9b-it"
            sae_sparsity = layer_sparisity_widths[model_alias][layer-1][sae_width]
            sae_id = f"layer_{layer-1}/width_{sae_width}/average_l0_{str(sae_sparsity)}"
        else:
            sae_sparsity = layer_sparisity_widths[model_alias][layer-1][sae_width]
            sae_id = f"layer_{layer-1}/width_{sae_width}/average_l0_{str(sae_sparsity)}"
        print(f'Loading SAE for layer {layer}')
        sae = load_sae(repo_id, sae_id)

        acts_labels = acts_labels_dict[layer]
        # Get activations
        sae_acts = sae.encode(acts_labels['acts'])

        # Get labels
        label_indices_1 = torch.where(acts_labels['labels'] == 1.0)[0]
        label_indices_0 = torch.where(acts_labels['labels'] == 0.0)[0]
        # Divide sae activations into known and unknown
        sae_acts_1 = sae_acts[label_indices_1, :].detach() # known activations
        sae_acts_0 = sae_acts[label_indices_0, :].detach() # unknown activations

        sae_acts = {'known': sae_acts_0, 'unknown': sae_acts_1}

        # We return all features
        scores_dict, freq_acts_dict, mean_features_acts = get_features(sae_acts,
                                                    metric = scoring_method,
                                                    min_activations=min_activations, # min activations for known and unknown to consider, lower than that we clamp to zero
                                                    )

        feats_per_layer[layer] = (scores_dict, freq_acts_dict, mean_features_acts)
        del sae_acts_0, sae_acts_1, sae_acts, sae
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

def get_features(
    sae_acts,
    metric: str = 'relative_difference',
    min_activations: Optional[List[float]] = None,
    eps: float = 1e-6,
):

    # Compute frequency of activations
    if sae_acts['known'].shape[0] == 0:
        # If no activation for known, we set the frequency to zero
        freq_acts_0 = torch.zeros(sae_acts['unknown'].shape[1]).to('cuda')
    else:
        freq_acts_0 = (sae_acts['known']>eps).float().mean(dim=0)
    if sae_acts['unknown'].shape[0] == 0:
        # If no activation for unknown, we set the frequency to zero
        freq_acts_1 = torch.zeros(sae_acts['unknown'].shape[1]).to('cuda')
    else:
        freq_acts_1 = (sae_acts['unknown']>eps).float().mean(dim=0)

    if metric == 'relative_difference':
        scores_0 = (freq_acts_0 - freq_acts_1) / (freq_acts_1 + eps)
        scores_1 = (freq_acts_1 - freq_acts_0) / (freq_acts_0 + eps)
    elif metric == 'absolute_difference':
        if min_activations is not None:
            mask_1 = freq_acts_1 < min_activations[0]
            mask_0 = freq_acts_0 < min_activations[1]

            scores_0 = ((freq_acts_0 - freq_acts_1) * mask_1)#.abs()
            scores_1 = ((freq_acts_1 - freq_acts_0) * mask_0)#.abs()

        else:
            scores_0 = (freq_acts_0 - freq_acts_1)
            scores_1 = (freq_acts_1 - freq_acts_0)

    elif metric == 'absolute_mean_difference':
        scores_0 = sae_acts['known'].mean(dim=0) - sae_acts['unknown'].mean(dim=0)
        scores_1 = sae_acts['unknown'].mean(dim=0) - sae_acts['known'].mean(dim=0)
    elif metric == 'relative_mean_difference':
        scores_0 = (sae_acts['known'].mean(dim=0) - sae_acts['unknown'].mean(dim=0)) / (sae_acts['unknown'].mean(dim=0) + eps)
        scores_1 = (sae_acts['unknown'].mean(dim=0) - sae_acts['known'].mean(dim=0)) / (sae_acts['known'].mean(dim=0) + eps)
    elif metric == 't_test':
        scores_0 = []
        scores_1 = []
        for i in range(0, sae_acts['known'].shape[1]):
            if sae_acts['known'][:,i].sum() == 0 and sae_acts['unknown'][:,i].sum() == 0:
                scores_0.append(0.0)
                scores_1.append(0.0)
            else:
                scores_0.append(stats.ttest_ind(sae_acts['known'][:,i].cpu().numpy(), sae_acts['unknown'][:,i].cpu().numpy(), axis=0, equal_var=False).statistic)
                scores_1.append(stats.ttest_ind(sae_acts['unknown'][:,i].cpu().numpy(), sae_acts['known'][:,i].cpu().numpy(), axis=0, equal_var=False).statistic)
        scores_0 = torch.tensor(scores_0)
        scores_1 = torch.tensor(scores_1)
        scores_0 = torch.nan_to_num(scores_0, nan=0.0)
        scores_1 = torch.nan_to_num(scores_1, nan=0.0)
    else:
        raise ValueError(f"Invalid metric: {metric}")

    latents_ids_dict = {}
    scores_dict = {}
    freq_acts_dict = {}
    mean_features_acts = {}
    for known_label, scores in zip(['known', 'unknown'], [scores_0, scores_1]):
        scores_dict[known_label] = scores.tolist()
        # Frequency of activation in known and unknown entity prompts for each ordered latent ranking (known and unknown)
        freq_acts_dict[known_label] = (freq_acts_0.cpu().tolist(), freq_acts_1.cpu().tolist())
        mean_features_acts[known_label] = sae_acts[known_label].mean(0).detach().cpu().tolist()

    return scores_dict, freq_acts_dict, mean_features_acts

def format_layer_features(feats_per_layer, filename_prefix, save=False):
    # Get dictionary of {'L2F3214' : [x, y]} with top k known and unknown features
        
    final_feats_dict = {}
    for layer in feats_per_layer.keys():
        final_feats_layer_dict = {}
        for known_label in ['known', 'unknown']:
            final_feats_layer_dict[known_label] = {}
            full_top_feats = []
            scores_dict, freq_acts_dict, mean_features_acts_dict = feats_per_layer[layer]
            scores, freq_acts, mean_features_acts = scores_dict[known_label], freq_acts_dict[known_label], mean_features_acts_dict[known_label]
            full_top_feats.extend(list(range(0,len(scores))))
            
            ordered_indices = np.argsort(np.array(scores))[::-1]

            # Get features, layers, scores, and activation frequencies of top k features
            top_k_feats = np.array(full_top_feats)[ordered_indices]

            for i, idx in enumerate(ordered_indices):
                final_feats_layer_dict[known_label][str(i)] = {'layer': layer, 'latent_idx': idx,
                                                'score': scores[idx],
                                                'freq_acts_known': freq_acts[0][idx],
                                                'freq_acts_unknown': freq_acts[1][idx],
                                                'mean_features_acts': mean_features_acts[idx]}
        if save == True:
            output_file = f'{filename_prefix}_L_{str(layer)}.json'
            with open(output_file, 'w') as f:
                json.dump(final_feats_layer_dict, f, indent=4, default=float)

        final_feats_dict[layer] = final_feats_layer_dict


    return final_feats_dict

def read_layer_features(model_alias, layers, entity_type, scoring_method, tokens_to_cache, evaluate_on):
    """
    Read the layer features from JSON files.
    """
    folder = f'./train_latents_layers_{evaluate_on}/{scoring_method}/{model_alias.split("/")[-1]}/{tokens_to_cache}'
    filename_prefix = f'{folder}/{entity_type}'

    feats_layers = {}
    for layer in layers:
        filename = f'{filename_prefix}_L_{str(layer)}.json'
        with open(filename, 'r') as f:
            layer_features_dict = json.load(f)
        feats_layers[layer] = layer_features_dict
    return feats_layers


def get_top_k_features(feats_layers, k=10):
    """
    Get dictionary with {'L2F3214' : [x, y]} with top k known and unknown features
    """
    final_feats_dict = {}
    for known_label in ['known', 'unknown']:
        final_feats_dict[known_label] = {}

        full_top_mean_features_acts = []
        full_top_freq_acts_0 = []
        full_top_freq_acts_1 = []
        full_layers_list = []
        full_scores = []
        full_latent_ids = []
        for layer in feats_layers.keys():
            layer_known_label_dict = feats_layers[layer][known_label]
            for idx in layer_known_label_dict.keys():
                full_scores.append(layer_known_label_dict[idx]['score'])
                full_layers_list.append(layer)
                full_latent_ids.append(layer_known_label_dict[idx]['latent_idx'])
                full_top_freq_acts_0.append(layer_known_label_dict[idx]['freq_acts_known'])
                full_top_freq_acts_1.append(layer_known_label_dict[idx]['freq_acts_unknown'])
                full_top_mean_features_acts.append(layer_known_label_dict[idx]['mean_features_acts'])

        # Join scores across all layers
        full_scores_array = np.array(full_scores)
        
        # Get top k features indices in joint scores array (full_top_scores_array)
        if k is not None:
            top_k_indices = np.argsort(full_scores_array)[-k:][::-1]
        else:
            top_k_indices = np.argsort(full_scores_array)[::-1]

        for i, idx in enumerate(top_k_indices):
            final_feats_dict[known_label][i] = {'layer': full_layers_list[idx], 'latent_idx': full_latent_ids[idx],
                                            'score': full_scores_array[idx],
                                            'freq_acts_known': full_top_freq_acts_0[idx], 'freq_acts_unknown': full_top_freq_acts_1[idx],
                                            'mean_features_acts': full_top_mean_features_acts[idx]}
    return final_feats_dict

def plot_top_k_features(final_feats_dict, entity_type):
    # final_feats_dict -> {'L2F3214' : [x, y] ...} with top k known and unknown features

    plt.figure(figsize=(7, 7))

    colors = {'known': html_colors['green_drawio'], 'unknown': html_colors['red_drawio']}

    for label in ['known', 'unknown']:
        x = []
        y = []
        for feature_id in final_feats_dict[label].keys():
            x.append(final_feats_dict[label][feature_id]['freq_acts_known']*100)
            y.append(final_feats_dict[label][feature_id]['freq_acts_unknown']*100)
            # x.append(final_feats_dict[label][feature_id][0]*100)
            # y.append(final_feats_dict[label][feature_id][1]*100)

            
        plt.scatter(x, y, color=colors[label], alpha=0.6)        
        # Add a diagonal line for reference
        plt.plot([0, 105], [0, 105], color='grey', linestyle='--', alpha=0.25)

        texts = []
        for feature_id in final_feats_dict[label].keys():
            text_feature_id = f'L{final_feats_dict[label][feature_id]["layer"]}F{final_feats_dict[label][feature_id]["latent_idx"]}'
            texts.append(plt.text(final_feats_dict[label][feature_id]['freq_acts_known']*100, final_feats_dict[label][feature_id]['freq_acts_unknown']*100, text_feature_id))
            
        adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='grey', lw=0.5))

    plt.xlim(0, 105)
    plt.ylim(0, 105)
    # Add labels and title
    plt.xlabel('Activation Frequency Known Entities (%)')
    plt.ylabel('Activation Frequency Unknown Entities (%)')
    plt.title(f'Feature Activation Frequencies: Known vs Unknown, Entity: {entity_type}')


def plot_all_features(final_feats_dict, train_feats_dict, entity_type, k=10, labels=True):
    from matplotlib import patches
    # final_feats_dict -> {'L2F3214' : [x, y] ...} with top k known and unknown features

    plt.figure(figsize=(4.5, 4.5), dpi=500)

    colors = {'known': html_colors['green_drawio'], 'unknown': html_colors['dark_red_drawio']}
    texts = []
    for label in ['known', 'unknown']:
        layer_feat_top_k_training = [(train_feats_dict[label][i]['layer'], train_feats_dict[label][i]['latent_idx']) for i in list(train_feats_dict[label].keys())[:k]]
        x = []
        y = []
        x_top = []
        y_top = []
        for i, feature_id in enumerate(final_feats_dict[label].keys()):
            
            x_ = final_feats_dict[label][feature_id]['freq_acts_known']*100
            y_ = final_feats_dict[label][feature_id]['freq_acts_unknown']*100

            
            #if (final_feats_dict[label][feature_id]['layer'], final_feats_dict[label][feature_id]['latent_idx']) in layer_feat_top_k_training and i < k:
            if (final_feats_dict[label][feature_id]['layer'], final_feats_dict[label][feature_id]['latent_idx']) in layer_feat_top_k_training:
                x_top.append(x_)
                y_top.append(y_)
                text_feature_id = f'L{final_feats_dict[label][feature_id]["layer"]}F{final_feats_dict[label][feature_id]["latent_idx"]}'
                if labels:
                    texts.append(plt.text(x_, y_, text_feature_id))
            else:
                x.append(x_)
                y.append(y_)

        plt.scatter(x, y, color='grey', alpha=0.02)
        plt.scatter(x_top, y_top, color=colors[label], alpha=0.6)

        
        # for i, feature_id in enumerate(list(final_feats_dict[label].keys())[:k]):
        #     if (final_feats_dict[label][feature_id]['layer'], final_feats_dict[label][feature_id]['latent_idx']) in layer_feat_top_k_training:
        #         text_feature_id = f'L{final_feats_dict[label][feature_id]["layer"]}F{final_feats_dict[label][feature_id]["latent_idx"]}'
        #         texts.append(plt.text(final_feats_dict[label][feature_id]['freq_acts_known']*100, final_feats_dict[label][feature_id]['freq_acts_unknown']*100, text_feature_id))
    if labels:
        adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='grey', lw=0.5))
    
    # Add a diagonal line for reference
    plt.plot([0, 105], [0, 105], color='grey', linestyle='--', alpha=0.25)
    
    plt.xlim(0, 105)
    plt.ylim(0, 105)
    # Add labels and title
    plt.xlabel('Activation Frequency Known Entities (%)', fontsize=10)
    plt.ylabel('Activation Frequency Unknown Entities (%)', fontsize=10)
    plt.title(f'Activation Frequencies of SAE Latents: {entity_type.capitalize()}', fontsize=10)
    return plt

## Plot functions
def plot_heads_heatmaps(mean_attn, title):
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=mean_attn,
        x=[f'Head {i}' for i in range(mean_attn.shape[1])],
        y=[f'Layer {i}' for i in range(mean_attn.shape[0])],
        colorscale='Blues',
        # zmin=0,
        # zmax=1
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Heads',
        yaxis_title='Layers',
        width=600,
        height=400
    )

    # Show the plot
    fig.show()
    return fig

def plot_heads_boxplot_entity_types(attn_heads_A, attn_heads_B, A_label, B_label, heads, head_colors,title):
    # Prepare data for known and unknown for each head
    fig = go.Figure()

    # Define colors for each head
    #head_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    # Blue, Orange, Green, Red
    # Prepare data for each entity type
    entity_types = list(attn_heads_A.keys())
    num_entity_types = len(entity_types)

    # Define a color palette for entity types
    #entity_type_colors = px.colors.qualitative.Set3[:num_entity_types]
    
    for i, (layer, head) in enumerate(heads):
        for j, entity_type in enumerate(entity_types):
            known_data = attn_heads_A[entity_type][:, layer, head].cpu().numpy()
            unknown_data = attn_heads_B[entity_type][:, layer, head].cpu().numpy()
            
            # Calculate position for each box
            pos = i * (num_entity_types * 2 + 1) + j * 2
            
            # Add traces for known and unknown
            fig.add_trace(go.Box(
                y=known_data,
                name=f'{A_label} {entity_type} L{layer}H{head}',
                marker_color=head_colors[0],
                offsetgroup=j,
                x0=pos
            ))
            fig.add_trace(go.Box(
                y=unknown_data,
                name=f'{B_label} {entity_type} L{layer}H{head}',
                marker_color=head_colors[1],
                offsetgroup=j,
                x0=pos + 1
            ))
    # Create custom x-axis ticks and labels
    xticks = []
    xticklabels = []
    for i, (layer, head) in enumerate(heads):
        for j, entity_type in enumerate(entity_types):
            pos = i * (num_entity_types * 2 + 1) + j * 2 + 0.5
            xticks.append(pos)
            xticklabels.append(f'L{layer}H{head}\n{entity_type}')

    # Update x-axis
    fig.update_xaxes(
        tickmode='array',
        tickvals=xticks,
        ticklabelposition='outside bottom',
        ticktext=xticklabels,
        title='Layers, Heads, and Entity Types'
    )
    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Attention Score',
        boxmode='group',
        boxgroupgap=0.1,
        boxgap=0.1
    )
    # Remove the legend
    # Create a custom legend with only two items
    fig.update_layout(
        showlegend=True,
        legend=dict(
            itemsizing='constant',
            traceorder='normal',
            font=dict(size=12),
            title_font_family='Arial',
            title_font_size=14,
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.5)',
        )
    )

    # Add two custom legend items
    fig.add_trace(go.Box(
        y=[None],
        name=A_label,
        marker_color=head_colors[0],
        showlegend=True,
        visible=True
    ))
    fig.add_trace(go.Box(
        y=[None],
        name=B_label,
        marker_color=head_colors[1],
        showlegend=True,
        visible=True
    ))

    fig.update_layout(
    title={
        'text': title,
        'y': 0.825,  # Moves the title closer to the plot (default is 0.9)
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    })

    # Hide legend for all other traces
    for trace in fig.data[:-2]:
        trace.showlegend = False
    #fig.update_layout(showlegend=False)
    fig = paper_plot(fig, tickangle=20)

    # Show the plot
    fig.show()

    return fig


import plotly.graph_objects as go
import numpy as np

def plot_heads_scatter_entity_types(attn_heads_A, attn_heads_B, A_label, B_label, heads, head_colors, title):
    fig = go.Figure()

    # Add the diagonal reference line
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            color="lightgrey",
            width=2,
            dash="dash",  # You can change this to "solid" if you prefer
        ),
        xref="paper",
        yref="paper"
    )

    entity_types = list(attn_heads_A.keys())
    num_entity_types = len(entity_types)

    # Define shapes for entity types
    entity_shapes = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'star']

    for i, (layer, head) in enumerate(heads):
        head_color = head_colors[i % len(head_colors)]
        for j, entity_type in enumerate(entity_types):
            known_data = attn_heads_A[entity_type][:, layer, head].cpu().numpy()
            unknown_data = attn_heads_B[entity_type][:, layer, head].cpu().numpy()
            
            # Add trace for this head and entity type
            fig.add_trace(go.Scatter(
                x=known_data,
                y=unknown_data,
                mode='markers',
                name=f'L{layer}H{head} {entity_type.capitalize()}',
                marker=dict(
                    color=head_color,
                    symbol=entity_shapes[j % len(entity_shapes)],
                    size=8,
                    line=dict(width=1, color='black')
                ),
                showlegend=False
            ))

    # Add legend items for attention heads
    for i, (layer, head) in enumerate(heads):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=head_colors[i % len(head_colors)]),
            name=f'L{layer}H{head}',
            showlegend=True
        ))

    # Add legend items for entity types
    for j, entity_type in enumerate(entity_types):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, symbol=entity_shapes[j % len(entity_shapes)], color='black'),
            name=entity_type,
            showlegend=True
        ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=f'Attention Score ({A_label})',
        yaxis_title=f'Attention Score ({B_label})',
        xaxis=dict(range=[0, 1]),  # Set x-axis limit to 1
        yaxis=dict(range=[0, 1]),
        showlegend=True,
        legend=dict(
            itemsizing='constant',
            traceorder='normal',
            font=dict(size=12),
            title_font_family='Arial',
            title_font_size=14,
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.5)',
        )
    )

    # y=0.665,
    # xanchor='left',
    # x=0.8,
    

    fig.update_layout(
    title={
        'text': title,
        'y': 0.825,  # Moves the title closer to the plot (default is 0.9)
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)

    # Apply paper plot style (assuming this function exists)
    fig = paper_plot(fig, tickangle=0)

    # Show the plot
    fig.show()

    return fig

def plot_heads_boxplot(attn_heads_A, attn_heads_B, A_label, B_label, heads, title):
    # Prepare data for known and unknown for each head
    fig = go.Figure()

    # Define colors for each head
    #head_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    head_colors = [html_colors['green_drawio'], html_colors['red_drawio']]  # Blue, Orange, Green, Red

    for i, (layer, head) in enumerate(heads):
        known_data = attn_heads_A[:, layer, head].cpu().numpy()
        unknown_data = attn_heads_B[:, layer, head].cpu().numpy()

        # Add traces for known and unknown
        fig.add_trace(go.Box(y=known_data, name=f'{A_label} L{layer}H{head}', marker_color=head_colors[0]))
        fig.add_trace(go.Box(y=unknown_data, name=f'{B_label} L{layer}H{head}', marker_color=head_colors[1]))

    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Attention Score',
        boxmode='group',
        boxgroupgap=0.1,
        boxgap=0.1
    )
    # Remove the legend
    fig.update_layout(showlegend=False)
    fig = paper_plot(fig)

    # Show the plot
    fig.show()


## Activation Patching and DLA functions
def get_logit_diff(logits, answer_token_indices, mean=True):
    answer_token_indices = answer_token_indices.to(logits.device)
    if len(logits.shape)==3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    if mean:
        return (correct_logits - incorrect_logits).mean()
    else:
        return (correct_logits - incorrect_logits)

def residual_stack_to_logit_diff(
    model: HookedTransformer,
    residual_stack: Float[torch.Tensor, "components batch d_model"],
    cache: ActivationCache,
    logit_diff_directions : Float[torch.Tensor, "batch d_model"],
    mean : bool = True
    ) -> float:
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    res_to_logit_diff = einsum(
        "... batch d_model, batch d_model -> batch ...",
        scaled_residual_stack * model.ln_final.w,
        logit_diff_directions,
    )
    if mean==True:
        return res_to_logit_diff.mean(0)
    else:
        return res_to_logit_diff

def get_logit_diffs_res_stream(model, base_tokens, src_tokens, answer_token_indices, analysis_type = Literal['accum_res', 'per_layer_res', 'attn_heads'], batch_size=8):
    layer_logit_diffs_list = []
    for i in range(0, len(base_tokens), batch_size):
        answer_token_indices_batch = answer_token_indices[i:i+batch_size]

        base_logits, base_cache = model.run_with_cache(base_tokens[i:i+batch_size])
        #src_logits, src_cache = model.run_with_cache(src_tokens[i:i+batch_size])

        # base_logit_diff = get_logit_diff(base_logits, answer_token_indices_batch, mean=False)
        # src_logit_diff = get_logit_diff(src_logits, answer_token_indices_batch, mean=False)

        answer_residual_directions = model.tokens_to_residual_directions(answer_token_indices_batch)

        # Difference of unembedding vectors
        logit_diff_directions = (
            answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
        )
        if analysis_type == 'accum_res':
            tensor_to_get_logit_diff, labels = base_cache.accumulated_resid(
                layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
            )
        elif analysis_type == 'per_layer_res':
            tensor_to_get_logit_diff, labels = base_cache.decompose_resid(
                layer=-1, pos_slice=-1, return_labels=True
                )
        elif analysis_type == 'attn_heads':
            heads_otuput_list = []
            labels = [f'L{layer}H{head}' for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
            for layer in range(model.cfg.n_layers):
                heads_outputs = base_cache[f'blocks.{layer}.attn.hook_result'][:,-1]
                heads_otuput_list.append(heads_outputs)
            tensor_to_get_logit_diff = torch.stack(heads_otuput_list, dim=1)
            tensor_to_get_logit_diff = einops.rearrange(tensor_to_get_logit_diff, 'batch n_layers n_heads d_model -> (n_layers n_heads) batch d_model')

        layer_logit_diffs = residual_stack_to_logit_diff(model, tensor_to_get_logit_diff, base_cache,
                                                                logit_diff_directions, mean=False)
        # batch, n_components
        del base_cache

        layer_logit_diffs_list.append(layer_logit_diffs)
        torch.cuda.empty_cache()
        gc.collect()
    
    return torch.cat(layer_logit_diffs_list, dim=0), labels


def compute_metric(model, base_tokens, src_tokens, answer_token_indices, batch_size=16):
    # Get baseline logit diffs over dataset to define metric
    base_logit_diffs_list = []
    src_logit_diffs_list = []
    for i in range(0, len(base_tokens), batch_size):
        answer_token_indices_batch = answer_token_indices[i:i+batch_size]

        base_logits, base_cache = model.run_with_cache(base_tokens[i:i+batch_size])
        src_logits, src_cache = model.run_with_cache(src_tokens[i:i+batch_size])
        del base_cache, src_cache

        base_logit_diff = get_logit_diff(base_logits.detach(), answer_token_indices_batch, mean=False)
        src_logit_diff = get_logit_diff(src_logits.detach(), answer_token_indices_batch, mean=False)
        del base_logits, src_logits

        base_logit_diffs_list.append(base_logit_diff)
        src_logit_diffs_list.append(src_logit_diff)

        torch.cuda.empty_cache()
        gc.collect()

    # Compute logit diff baselines
    CLEAN_BASELINE = torch.cat(base_logit_diffs_list, dim=0).mean(0)
    CORRUPTED_BASELINE = torch.cat(src_logit_diffs_list, dim=0).mean(0)

    # Define metric with baselines
    def metric(logits, answer_token_indices):
        answer_token_indices = answer_token_indices.to(logits.device)
        return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (CLEAN_BASELINE  - CORRUPTED_BASELINE)

    return metric


def compute_patching(model, base_tokens, src_tokens, patching_type:Literal['resid_streams', 'heads_all_pos', 'heads_last_pos', 'full'], answer_token_indices, metric, batch_size=16):
    # Compute activation patching across dataset

    # resid_streams
    # heads_all_pos : attn heads all positions at the same time
    # heads_last_pos: attn heads last position
    # full: (resid streams, attn block outs and mlp outs)

    list_resid_pre_act_patch_results = []
    for i in range(0, len(base_tokens), batch_size):

        base_logits, base_cache = model.run_with_cache(base_tokens[i:i+batch_size])
        del base_logits
        answer_token_indices_batch = answer_token_indices[i:i+batch_size]
        src_tokens_batch = src_tokens[i:i+batch_size]

        metric_fn = partial(metric, answer_token_indices=answer_token_indices_batch)

        if patching_type=='resid_streams':
            # resid_pre_act_patch_results -> [n_layers, pos]
            patch_results = patching.get_act_patch_resid_pre(model, src_tokens_batch, base_cache, metric_fn)
        elif patching_type=='heads_all_pos':
            patch_results = patching.get_act_patch_attn_head_out_all_pos(model, src_tokens_batch, base_cache, metric_fn)
        elif patching_type=='heads_last_pos':
            # Activation patching per position
            attn_head_out_per_pos_patch_results = patching.get_act_patch_attn_head_out_by_pos(model, src_tokens_batch, base_cache, metric_fn)
            # Select last position
            patch_results = attn_head_out_per_pos_patch_results[:,-1]
        elif patching_type=='full':
            patch_results = patching.get_act_patch_block_every(model, src_tokens_batch, base_cache, metric_fn)

        del base_cache
        torch.cuda.empty_cache()

        list_resid_pre_act_patch_results.append(patch_results)
        del patch_results

    total_resid_pre_act_patch_results = torch.stack(list_resid_pre_act_patch_results, 0).mean(0)

    return total_resid_pre_act_patch_results

def get_value_weighted_patterns(model, cache, layer):
    # We compute attention heads weighted value vectors a_{i,j} x_j W_V
    pattern = cache[f'blocks.{layer}.attn.hook_pattern']
    v = cache[f'blocks.{layer}.attn.hook_v']
    
    # TODO: if using grouped query attention
    if model.cfg.n_heads != v.shape[2]:
        repeat_kv_heads = model.cfg.n_heads // model.cfg.n_key_value_heads
        v = torch.repeat_interleave(v, dim=2, repeats=repeat_kv_heads)

    weighted_values = einsum(
                "batch key_pos head_index d_head, \
                batch head_index query_pos key_pos -> \
                batch query_pos key_pos head_index d_head",
                v,
                pattern,
            )# [batch, query_pos, key_pos, head_index, d_head]
    return weighted_values

def visualize_attention_patterns(
    model,
    type_pattern: Literal['attn_weights', 'value_weighted', 'output_value_weighted', 'distance_based'],
    heads: Union[List[int], int, Float[torch.Tensor, "heads"]],
    local_cache: ActivationCache,
    #local_tokens: torch.Tensor,
    title: Optional[str] = "",
    max_width: Optional[int] = 700,
    html: Optional[bool] = True
):
    # If a single head is given, convert to a list
    if isinstance(heads, int):
        heads = [heads]

    # Create the plotting data
    labels: List[str] = []
    patterns: List[Float[torch.Tensor, "dest_pos src_pos"]] = []

    # Assume we have a single batch item
    #batch_index = 0

    for head in heads:
        # Set the label
        layer = head // model.cfg.n_heads
        head_index = head % model.cfg.n_heads
        labels.append(f"L{layer}H{head_index}")

        # Get the attention patterns for the head
        # Attention patterns have shape [batch, head_index, query_pos, key_pos]
        if type_pattern == 'attn_weights':
            updated_pattern = local_cache["attn", layer][:, head_index]
        else:
            weighted_values = get_value_weighted_patterns(model, local_cache, layer)
            weighted_values = weighted_values.float()

            if type_pattern == 'value_weighted':
                # Value-weighted norms
                raw_inter_token_attribution = torch.norm(weighted_values, dim=-1, p=2)
                # weighted_values_norm -> [batch query_pos key_pos head_index]

            elif type_pattern == 'output_value_weighted' or type_pattern == 'distance_based':
                # We decompose attention heads further by computing a_{i,j} x_j W_OV
                output_weighted_values = einsum(
                        "batch query_pos key_pos head_index d_head, \
                            head_index d_head d_model -> \
                            batch query_pos key_pos head_index d_model",
                        weighted_values,
                        model.W_O[layer],
                    )

                # Check sum decomposition is equivalent to cached values
                output_heads = output_weighted_values.sum(2)
                output_attention = output_heads.sum(-2) + model.b_O[layer]
                # assert torch.dist(output_attention, local_cache[f'blocks.{layer}.hook_attn_out']).item() < 1e-3 * local_cache[f'blocks.{layer}.hook_attn_out'].numel()

                if type_pattern == 'output_value_weighted':
                    # Output-value-weighted norms
                    # weighted_values_norm -> [batch query_pos key_pos head_index]
                    raw_inter_token_attribution = torch.norm(output_weighted_values, dim=-1, p=1)

                elif type_pattern == 'distance_based':
                    # Distance-based
                    EPS = 1e-5
                    # distance -> [batch query_pos key_pos head_index]
                    distance = -F.pairwise_distance(output_weighted_values, output_heads.unsqueeze(2),p=2)
                    # head_output_norm -> [batch query_pos head_index]
                    head_output_norm = torch.norm(output_heads, p=2, dim=-1)
                    raw_inter_token_attribution = (distance + head_output_norm.unsqueeze(2)).clip(min=EPS)
                    

            # Normalize over key_pos
            #inter_token_attribution = raw_inter_token_attribution / raw_inter_token_attribution.sum(dim=-2, keepdim=True)
            inter_token_attribution = raw_inter_token_attribution
            updated_pattern = inter_token_attribution[:, :, :, head_index]
            #print(updated_pattern.shape)
            
        patterns.append(updated_pattern)

    # Combine the patterns into a single tensor
    patterns: Float[torch.Tensor, "batch head_index dest_pos src_pos"] = torch.stack(
        patterns, dim=1
    )

    if html:
        pass
        # # Convert the tokens to strings (for the axis labels)
        # str_tokens = model.to_str_tokens(local_tokens)
        # # Circuitsvis Plot (note we get the code version so we can concatenate with the title)
        # plot = attention_heads(
        #     attention=patterns, tokens=str_tokens, attention_head_names=labels
        # ).show_code()
        # # Display the title
        # title_html = f"<h2>{title}</h2><br/>"
        # # Return the visualisation as raw code
        # return f"<div style='max-width: {str(max_width)}px;'>{title_html + plot}</div>"
    else:
        return patterns


def bar_latents_effect_plot(logit_diffs_results:List[Tensor], xticks_labels:List[str], title:str):
    # Custom colors
    colors = [html_colors['blue_matplotlib'], html_colors['orange_matplotlib'], html_colors['brown_D3']]    
    logit_diff_results = [logit_diff_result.cpu().numpy() for logit_diff_result in logit_diffs_results]

    means = [np.mean(logit_diff_result) for logit_diff_result in logit_diff_results]
    errors = [np.std(logit_diff_result) for logit_diff_result in logit_diff_results]

    # Prepare data for plotting
    data = {
        'Condition': xticks_labels,
        'Mean': means,
        'Error': errors
    }


    # Create the bar plot with error bars
    fig = px.bar(
        data_frame=data,
        x='Condition',
        y='Mean',
        color='Condition',
        error_y='Error',
        color_discrete_sequence=colors,
        title=title
    )

    # Make x ticks bigger and rotate them
    fig.update_xaxes(tickfont=dict(size=12))
    fig.update_layout(
        xaxis_title=dict(text="", font=dict(size=16)),
        yaxis_title=dict(text="Logit Difference (Yes - No)", font=dict(size=16))
    )
    fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = [-0.1, 0.7, 1.7],  # Adjust these values to move labels left
        #tickfont = dict(size=10)
    )
)
    # Remove legend
    fig.update_layout(showlegend=False)
    fig = paper_plot(fig, tickangle=10)
    fig.show()

    return fig


def load_latents(model_alias, top_latents, filter_with_pile=False, **kwargs):
    # Load steering latents
    # Read the sorted scores for unknown entities
    if filter_with_pile == True:
        with open(f'./train_latents_layers_entities/absolute_difference/{model_alias}/entity/pile_filtered_scores_min_known.json', 'r') as f:
            sorted_scores_known = json.load(f)
        with open(f'./train_latents_layers_entities/absolute_difference/{model_alias}/entity/pile_filtered_scores_min_unknown.json', 'r') as f:
            sorted_scores_unknown = json.load(f)
    else:
        with open(f'./train_latents_layers_entities/absolute_difference/{model_alias}/entity/sorted_scores_min_known.json', 'r') as f:
            sorted_scores_known = json.load(f)
        with open(f'./train_latents_layers_entities/absolute_difference/{model_alias}/entity/sorted_scores_min_unknown.json', 'r') as f:
            sorted_scores_unknown = json.load(f)
    # Known latent
    known_latent_ = list(sorted_scores_known.keys())[top_latents['known']]
    layer_known = int(known_latent_[1:known_latent_.find('F')])
    head_known = int(known_latent_[known_latent_.find('F')+1:-2])
    known_latent_id = [(layer_known, head_known)]
    # Unknown latent
    unknown_latent_ = list(sorted_scores_unknown.keys())[top_latents['unknown']]
    layer_unknown = int(unknown_latent_[1:unknown_latent_.find('F')])
    head_unknown = int(unknown_latent_[unknown_latent_.find('F')+1:-2])
    unknown_latent_id = [(layer_unknown, head_unknown)]

    known_latent: List[Tuple[int, float, Tensor]] = load_steering_latents('movie', label='unknown', topk=1,
                                                                            #layers_range=[known_latent[0]],
                                                                            specific_latents=known_latent_id,
                                                                            model_alias=model_alias,
                                                                            random_latents=False)

    unknown_latent: List[Tuple[int, float, Tensor]] = load_steering_latents('movie', label='unknown', topk=1,
                                                                            #layers_range=[unknown_latent[0]],
                                                                            specific_latents=unknown_latent_id,
                                                                            model_alias=model_alias,
                                                                            random_latents=False)

    random_latents_known: List[Tuple[int, float, Tensor]] = load_steering_latents('movie', label='known', topk=kwargs['random_n_latents'],
                                                                              layers_range=[layer_known],
                                                                              model_alias=model_alias,
                                                                              random_latents=True)
    
    random_latents_unknown: List[Tuple[int, float, Tensor]] = load_steering_latents('movie', label='unknown', topk=kwargs['random_n_latents'],
                                                                              layers_range=[layer_unknown],
                                                                              model_alias=model_alias,
                                                                              random_latents=True)
    
    return known_latent, unknown_latent, random_latents_known, random_latents_unknown