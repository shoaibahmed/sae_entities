import os
import json
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import random
from jaxtyping import Float, Int
import torch
from torch import Tensor

dataset_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "processed", "entity_prompts")
dataset_dir_path = "/root/mats_hallucinations/dataset/processed/entity_prompts"
VALID_ENTITY_TYPES = ["book", "movie", "city", "player", "song"]

# we slice the prompts so that they all end with question mark
def process_prompt_fn(prompt, token_str="?"):
    assert token_str in prompt
    return prompt[:prompt.index(token_str)+len(token_str)]

# LEGACY_VALID_PROMPT_TYPES = {
#     "book": ["is_known", "attribute_creator"],
#     "movie": ["is_known", "attribute_director"],
#     "city": ["is_known", "attribute_state_name"],
#     "player": ["is_known", "attribute_birthplace", "attribute_birthday"],
#     "song": ["is_known", "attribute_artist_name"]
# }

VALID_PROMPT_TYPES = {
     "movie": ["movie_is_known", "movie_directors", "movie_cast",  "movie_screenwriters", "movie_genres", "movie_release_year", "movie_durations"],
    "city": ["city_is_known", "city_country", "city_location", "city_population", "city_elevation"],
    "player": ["player_is_known", "player_place_birth", "player_date_birth", "player_teams_list", "player_weight", "player_height"],
     "song": ["song_is_known", "song_album", "song_publication_year", "song_genres", "song_performers"]
}


def load_entity_scores(entity_type: str, prompt_type: str, model_alias: str):
    assert entity_type in VALID_ENTITY_TYPES

    file_name = f"{entity_type}.json"

    file_path = os.path.join(dataset_dir_path, model_alias, file_name)

    if os.path.exists(file_path) == False:
        print(f"WARNING: file path {file_path} doesn't exist")
        return []

    with open(file_path, "r") as f:
        #print(f"Loading path {file_path}")
        prompts = json.load(f)

    # we filter out queries based on the type (either attribute type or is known)
    prompts = [p for p in prompts if p["prompt_type"] == prompt_type]

    scores = [p["score"] for p in prompts]

    return scores

def load_entity_prompts(entity_type: str, prompt_type: str, model_alias: str, factual_type: str=None, prompts_only: bool=False):
    assert entity_type in VALID_ENTITY_TYPES

    if prompt_type.split("_")[0] == entity_type:
        # prompt type might include the entity type (e.g 'movie_directors' or 'song_album')
        file_name = f"{prompt_type}.json"
    else:
        file_name = f"{entity_type}_{prompt_type}.json"

    file_path = os.path.join(dataset_dir_path, model_alias, file_name)

    if os.path.exists(file_path) == False:
        print(f"WARNING: file path {file_path} doesn't exist")
        return []

    with open(file_path, "r") as f:
        #print(f"Loading path {file_path}")
        prompts = json.load(f)

    # we filter out queries based on the type (either attribute type or is known)
    prompts = [p for p in prompts if p["prompt_type"] == prompt_type]

    # we filter out queries based on their factuality
    if factual_type is not None:
        prompts = [p for p in prompts if p["score"] == factual_type]

    # we return only a list of prompts (i.e. a list of strings)
    if prompts_only:
        return [p["prompt"] for p in prompts]

    return prompts

def load_prompts_dict(model_alias: str, factual_type: str=None, prompts_only: bool=False):
    prompts_dict = {}

    for entity_type in VALID_ENTITY_TYPES:
        for prompt_type in VALID_PROMPT_TYPES[entity_type]:
            prompts_dict[f"{entity_type}_{prompt_type}"] = load_entity_prompts(entity_type, prompt_type, model_alias, factual_type, prompts_only)

    return prompts_dict

def legacy_get_known_unknown(model_alias, plot=False):
    """
        Calculates the number of known and unknown entities for each entity type and prompt type.
        
        Returns:
            None
    """
    known_prompts_dict = {}
    unknown_prompts_dict = {}
    for entity_type in tqdm(VALID_PROMPT_TYPES.keys()):
        attributes_entity = VALID_PROMPT_TYPES[entity_type][1:]
        scores_dict = {}
        for prompt_type in attributes_entity:
            scores_dict[prompt_type] = load_entity_scores(entity_type, prompt_type, model_alias)
        known_prompts_dict[entity_type] = {}
        unknown_prompts_dict[entity_type] = {}
        results_list = []
        for i in range(len(scores_dict[prompt_type])):
            score_i = 0
            for prompt_type in scores_dict.keys():
                score_ = scores_dict[prompt_type][i]
                if score_ != 2:
                    # Don't add if refusal (2)
                    score_i += score_
            results_list.append(score_i)

        known = [j for j, res in enumerate(results_list) if res > 2]#[:1000]
        unknown = [j for j, res in enumerate(results_list) if res == 0]#[:1000]
        for prompt_type in scores_dict.keys():
            prompts = load_entity_prompts(entity_type, prompt_type, model_alias)
            known_prompts = [prompts[j] for j in known]
            unknown_prompts = [prompts[j] for j in unknown]
            known_prompts_dict[entity_type][prompt_type] = known_prompts
            unknown_prompts_dict[entity_type][prompt_type] = unknown_prompts

    len_dict = {}
    for entity_type in VALID_PROMPT_TYPES.keys():
        len_dict[entity_type]= {}
        len_dict[entity_type]['known'] = len(known_prompts_dict[entity_type][VALID_PROMPT_TYPES[entity_type][1]])
        len_dict[entity_type]['unknown'] = len(unknown_prompts_dict[entity_type][VALID_PROMPT_TYPES[entity_type][1]])

    if plot:
        # Plot stats
        df_k_u = pd.DataFrame(len_dict).T

        df_k_u.reset_index(inplace=True)
        df_k_u.rename(columns={'index': 'label'}, inplace=True)

        fig = px.bar(df_k_u, x="label",
                    y=['known', 'unknown'], 
                    title=f"Number of known and unknown entities ({model_alias})")
        fig.show()
    return known_prompts_dict, unknown_prompts_dict

def get_known_unknown_data(model_alias, plot=False):
    """
        Calculates the number of known and unknown entities for each entity type and prompt type.
        
        Returns:
            None
    """
    entity_knowledge_path = f"/root/mats_hallucinations/dataset/processed/entity_prompts/{model_alias}/entity_knowledge.json"
    # Read the JSON file
    with open(entity_knowledge_path, 'r') as file:
        data = json.load(file)

    known_prompts_dict = {}
    unknown_prompts_dict = {}
    for entity_type in tqdm(VALID_PROMPT_TYPES.keys()):
        if "gemma" in model_alias:
            if entity_type== 'player':
                continue
        known_prompts_dict[entity_type] = {}
        unknown_prompts_dict[entity_type] = {}
        try:
            attributes_entity = VALID_PROMPT_TYPES[entity_type][1:]
            known_enitites = data[entity_type]['known']
            unk_entities = data[entity_type]['unknown']
        except:
            print(f"No entity type '{entity_type}'")
            continue
        for prompt_type in attributes_entity:
            prompts = load_entity_prompts(entity_type, prompt_type, model_alias)
            known_prompts = [prompt for prompt in prompts if prompt['entity'] in known_enitites]
            unknown_prompts = [prompt for prompt in prompts if prompt['entity'] in unk_entities]
            known_prompts_dict[entity_type][prompt_type] = known_prompts
            unknown_prompts_dict[entity_type][prompt_type] = unknown_prompts

    return known_prompts_dict, unknown_prompts_dict

def get_k_u_prompts(known_prompts_dict, unknown_prompts_dict,
                 mean_diff_prompt_types, process_prompt:bool=False,
                 slice_str:str="?",
                 N:int=500):
    
    prompts = {}
    for entity_type, prompt_type in tqdm(mean_diff_prompt_types):
        print(entity_type, prompt_type)
        prompts[f'{prompt_type}'] = {}

        # Load prompts
        known_data = known_prompts_dict[entity_type][prompt_type]
        unk_data = unknown_prompts_dict[entity_type][prompt_type]
        known_prompts = [prompt['prompt'] for prompt in known_data]
        unk_prompts = [prompt['prompt'] for prompt in unk_data]
        if process_prompt:
            # Slice prompts if necessary
            known_prompts = [process_prompt_fn(prompt, slice_str) for prompt in known_prompts]
            unk_prompts = [process_prompt_fn(prompt, slice_str) for prompt in unk_prompts]
        
        # Get N prompts and shuffle (fornce to take same amount of known and unknown)
        n_prompts = min(N, len(known_prompts), len(unk_prompts))
        known_prompts = random.sample(known_prompts, n_prompts)
        unk_prompts = random.sample(unk_prompts, n_prompts)
        prompts[f'{prompt_type}']['known'] = known_prompts
        prompts[f'{prompt_type}']['unknown'] = unk_prompts

    return prompts