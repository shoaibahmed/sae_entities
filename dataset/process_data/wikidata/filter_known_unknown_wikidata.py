# %%
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

# %%

import sys
sys.path.append("../../..")

# %%
import torch
import pickle
import json
import pandas as pd
import os
from typing import List, Union, Optional, Dict, Any
from tqdm import tqdm
from collections import defaultdict
import random
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import random
import plotly.express as px

from utils.generation_utils import is_generation_refusal
from dataset.load_entity_prompts import load_entity_prompts, load_entity_scores
from check_correctness_wikidata import compute_is_known, check_name_correctness, string_match_in_list_genres, number_match, geo_location_match, number_match_in_list

torch.set_grad_enabled(False)

# %%

def gemma_filter_city_is_known(entity_queries: Dict[str, Dict[str, Any]]) -> str:

    is_known_generations = [entity_queries['city_is_known']['generation']] + entity_queries['city_is_known']['extra_data']['multiple_generations']
    is_known_all_refusal = all([is_generation_refusal(generation) for generation in is_known_generations])

    knowledge_conditions = [ 
        # important to compare against 1, not to sum directly the value since refusal=2
        entity_queries['city_country']['score'] == 1,
        entity_queries['city_location']['score'] == 1,
        entity_queries['city_population']['score'] == 1,
        entity_queries['city_elevation']['score'] == 1,
    ]

    if is_known_all_refusal == True and sum(knowledge_conditions) == 0:
        return 'unknown'
    if is_known_all_refusal == False and sum(knowledge_conditions) > 1:
        return 'known'
    return 'other'

def gemma_filter_movie_is_known(entity_queries: Dict[str, Dict[str, Any]]) -> str:
 
    is_known_generations = [entity_queries['movie_is_known']['generation']]
    is_known_all_refusal = all([is_generation_refusal(generation) for generation in is_known_generations])

    release_year = int(entity_queries['movie_release_year']['attribute'])
    release_year_generations = [entity_queries['movie_release_year']['generation']] + entity_queries['movie_release_year']['extra_data']['multiple_generations']
    release_year_confidence = sum([
        number_match(generation, release_year, tolerance=2) for generation in release_year_generations
    ])

    knowledge_conditions = [
        release_year_confidence >= 5,
        entity_queries['movie_cast']['score'] == 1,
        entity_queries['movie_directors']['score'] == 1,
        entity_queries['movie_screenwriters']['score'] == 1,
    ]

    if sum(knowledge_conditions) >= 2:
        return 'known'
    if is_known_all_refusal == True and sum(knowledge_conditions) == 0:
        return 'unknown'
    return 'other'


def gemma_filter_player_is_known(entity_queries: Dict[str, Dict[str, Any]]) -> str:
    pass

def gemma_filter_song_is_known(entity_queries: Dict[str, Dict[str, Any]]) -> str:
    is_known_generations = [entity_queries['song_is_known']['generation']]
    is_known_all_refusal = all([is_generation_refusal(generation) for generation in is_known_generations])

    knowledge_conditions = [
        entity_queries['song_album']['score'] == 1,
        entity_queries['song_publication_year']['score'] == 1,
        entity_queries['song_genres']['score'] == 1,
        entity_queries['song_performers']['score'] == 1
    ]

    if sum(knowledge_conditions) >= 2:
        return 'known'
    if is_known_all_refusal == True and sum(knowledge_conditions) == 0:
        return 'unknown'
    return 'other'

def llama3_filter_city_is_known(entity_queries: Dict[str, Dict[str, Any]]) -> str:

    generations = [entity_queries['city_is_known']['generation']] + entity_queries['city_is_known']['extra_data']['multiple_generations']
    generations += [entity_queries['city_country']['generation']] + entity_queries['city_country']['extra_data']['multiple_generations']
    generations += [entity_queries['city_location']['generation']] + entity_queries['city_location']['extra_data']['multiple_generations']
    generations += [entity_queries['city_population']['generation']] + entity_queries['city_population']['extra_data']['multiple_generations']
    generations += [entity_queries['city_elevation']['generation']] + entity_queries['city_elevation']['extra_data']['multiple_generations']

    refusal = any(
        is_generation_refusal(generation) for generation in generations
    )

    knowledge_conditions = [
        entity_queries['city_country']['score'] == 1,
        entity_queries['city_location']['score'] == 1,
        entity_queries['city_population']['score'] == 1,
        entity_queries['city_elevation']['score'] == 1,
    ]

    if refusal == False and sum(knowledge_conditions) >= len(knowledge_conditions) - 1:
        return 'known'
    elif sum(knowledge_conditions) == 0:
        return 'unknown'
    return 'other'

def llama3_filter_movie_is_known(entity_queries: Dict[str, Dict[str, Any]]) -> str:
    is_known_generations = [entity_queries['movie_is_known']['generation']]
    is_known_all_refusal = all([is_generation_refusal(generation) for generation in is_known_generations])

    release_year = int(entity_queries['movie_release_year']['attribute'])
    release_year_generations = [entity_queries['movie_release_year']['generation']] + entity_queries['movie_release_year']['extra_data']['multiple_generations']
    release_year_confidence = sum([
        number_match(generation, release_year, tolerance=2) for generation in release_year_generations
    ])

    knowledge_conditions = [
        release_year_confidence >= 5,
        entity_queries['movie_genres']['score'] == 1,
        entity_queries['movie_directors']['score'] == 1,
    ]

    if is_known_all_refusal == False and sum(knowledge_conditions) == len(knowledge_conditions):
        return 'known'
    if sum(knowledge_conditions) == 0:
        return 'unknown'
    return 'other'

def llama3_filter_player_is_known(entity_data: dict) -> str:

    generations = [entity_data['player_is_known']['generation']] + entity_data['player_is_known']['extra_data']['multiple_generations']
    generations += [entity_data['player_place_birth']['generation']] + entity_data['player_place_birth']['extra_data']['multiple_generations']
    generations += [entity_data['player_date_birth']['generation']] + entity_data['player_date_birth']['extra_data']['multiple_generations']
    generations += [entity_data['player_teams_list']['generation']] + entity_data['player_teams_list']['extra_data']['multiple_generations']

    refusal = any(
        is_generation_refusal(generation) for generation in generations
    )

    knowledge_conditions = [
        entity_data['player_place_birth']['score'] == 1,
        entity_data['player_date_birth']['score'] == 1,
        entity_data['player_teams_list']['score'] == 1,
    ]

    if sum(knowledge_conditions) >= len(knowledge_conditions) - 1:
        return 'known'
    elif sum(knowledge_conditions) == 0:
        return 'unknown'
    return 'other'

def llama3_filter_song_is_known(entity_queries: Dict[str, Dict[str, Any]]) -> str:
    is_known_generations = [entity_queries['song_is_known']['generation']]
    is_known_all_refusal = all([is_generation_refusal(generation) for generation in is_known_generations])

    knowledge_conditions = [
        entity_queries['song_album']['score'] == 1,
        entity_queries['song_publication_year']['score'] == 1,
        entity_queries['song_genres']['score'] == 1,
        entity_queries['song_performers']['score'] == 1
    ]

    if sum(knowledge_conditions) >= 2:
        return 'known'
    if is_known_all_refusal == True and sum(knowledge_conditions) == 0:
        return 'unknown'
    return 'other'
    

# %%

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_alias', type=str, default='gemma-2b-it', help='Model alias')
parser.add_argument('--plot', action='store_true', help='Generate plots')
args = parser.parse_args()

model_alias = args.model_alias
plot = args.plot
entity_knowledge_path = f"../../processed/entity_prompts/{model_alias}/entity_knowledge.json"

wikidata_prompt_types = {
    "player": ["player_is_known", "player_place_birth", "player_date_birth", "player_teams_list"],
    "movie": ["movie_is_known", "movie_directors", "movie_cast", "movie_screenwriters", "movie_genres", "movie_release_year", "movie_durations"],
    "city": ["city_is_known", "city_country", "city_location", "city_population", "city_elevation"],
    "song": ["song_is_known", "song_album", "song_publication_year", "song_genres", "song_performers"],
}

filtering_functions = {
    "player": llama3_filter_player_is_known,
    "movie": llama3_filter_movie_is_known,
    "city": llama3_filter_city_is_known,
    "song": llama3_filter_song_is_known,
}

# %%

if os.path.exists(entity_knowledge_path):
    with open(entity_knowledge_path, "r") as file:
        entity_knowledge = json.load(file)
else:
    entity_knowledge = {}

len_dict = {}

for entity_type in tqdm(wikidata_prompt_types.keys()):

    knowledge_filtering_fn = filtering_functions[entity_type]

    if entity_type not in entity_knowledge:
        entity_knowledge[entity_type] = {
            "known": [],
            "unknown": [],
            "fake": [],
            "other": [],
        }

    entity_queries = {}

    for prompt_type in wikidata_prompt_types[entity_type]:
        queries = load_entity_prompts(entity_type, prompt_type, model_alias, factual_type=None, prompts_only=False)

        for query in queries:
            entity = query['entity']
            
            if entity not in entity_queries:
                entity_queries[entity] = {}

            entity_queries[entity][prompt_type] = query

        # FOR LATER:
        # entity_queries[entity][prompt_type]['score'] = ...
        # entity_queries[entity][prompt_type]['extra_data']['refusal_score'] = ...
        # entity_queries[entity][prompt_type]['extra_data']['num_correct_generations'] = ...

    entities = list(entity_queries.keys())

    labels = [
        knowledge_filtering_fn(entity_queries[entity])
        for entity in entities
    ]

    len_dict[entity_type] = {label: sum([l == label for l in labels]) for label in ['known', 'unknown', 'other']}

    for entity, label in zip(entities, labels):
        entity_knowledge[entity_type][label].append(entity)

    entity_knowledge[entity_type] = {
        label: list(set(entity_knowledge[entity_type][label]))
        for label in ['known', 'unknown', 'other']
    }

print("Length dict:")
print(len_dict)

with open(entity_knowledge_path, "w") as file:
    json.dump(entity_knowledge, file, indent=4)

# %%

if plot: 

    # Plot stats
    df_k_u = pd.DataFrame(len_dict).T

    df_k_u.reset_index(inplace=True)
    df_k_u.rename(columns={'index': 'label'}, inplace=True)

    fig = px.bar(df_k_u, x="label",
                y=['known', 'unknown', 'other'], 
                title=f"Number of known and unknown entities ({model_alias})")
    fig.show()

    # Save the figure as a PNG file in the local folder
    fig.write_image(f"./entity_knowledge_distribution_{model_alias}.png")

    print(f"Figure saved as 'entity_knowledge_distribution_{model_alias}.png' in the local folder.")

# %%