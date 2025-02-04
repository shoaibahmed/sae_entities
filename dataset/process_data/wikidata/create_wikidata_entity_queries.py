# %%
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

# %%

import sys
sys.path.append("../../../")
sys.path.append("../../")
sys.path.append("../")
# sys.path.append(".")
sys.path.append("/root/mats_hallucinations/dataset/process_data")
# %%
import torch
import pickle
import json
import pandas as pd
import os
from typing import List, Union, Optional, Callable, Dict, Any
from tqdm import tqdm
from collections import defaultdict
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import random
import argparse
from dataclasses import dataclass
from functools import partial
from copy import deepcopy

from utils.plotly_utils import plot_bar_chart
from utils.generation_utils import get_batch_completion_fn, is_generation_refusal
from utils.utils import model_is_chat_model

from .check_correctness_wikidata import compute_is_known, check_name_correctness, string_match, string_match_in_list, string_match_in_list_genres, number_match, geo_location_match, number_match_in_list

torch.set_grad_enabled(False)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# %%
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

WIKIDATA_QUERIES_FREE = {
    "player_place_birth":{
        "messages": [
            {"role": "user", "content": "What is the place of birth for the player '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    },
    "player_date_birth": {
        "messages": [
            {"role": "user", "content": "What is the year of birth for the player '{entity}'?"},
        ],
        "correctness_function": partial(number_match, tolerance=0),
    },
    "player_teams_list":{
        "messages": [
            {"role": "user", "content": "What team (name at least one) signed the player '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    },
    "player_weight":{
        "messages": [
            {"role": "user", "content": "What is the weight, in kg, of the player '{entity}'?"},
        ],
        "correctness_function": partial(number_match, tolerance=5),
    },
    "player_height":{
        "messages": [
            {"role": "user", "content": "What is the height, in cm, of the player '{entity}'?"},
        ],
        "correctness_function": partial(number_match, tolerance=3),
    },
    "movie_directors":{
        "messages": [
            {"role": "user", "content": "Who is the director of the movie '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    },
    "movie_cast": {
        "messages": [
            {"role": "user", "content": "What is the name of an actor starring in the movie '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    },
    "movie_release_year":{
        "messages": [
            {"role": "user", "content": "What was the release year for the movie '{entity}'?"},
        ],
        "correctness_function": partial(number_match, tolerance=2),
    },
    "movie_screenwriters":{
        "messages": [
            {"role": "user", "content": "Who is the screenwriter of the movie '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    },
    "movie_genres":{
        "messages": [
            {"role": "user", "content": "What genre label best describes the movie '{entity}'?"},
        ],
        "correctness_function": string_match_in_list_genres,
    },
    "movie_durations":{
        "messages": [
            {"role": "user", "content": "What is the duration, in minutes, of the movie '{entity}'?"},
        ],
        "correctness_function": partial(number_match_in_list, tolerance=3),
    },
    "city_country":{
        "messages": [
            {"role": "user", "content": "What country contains the city of '{entity}'?"},
        ],
        "correctness_function": string_match,
    },
    "city_location":{
        "messages": [
            {"role": "user", "content": "What geographic coordinates correspond to the city of '{entity}'?"},
        ],
        "correctness_function": geo_location_match,
    },
    "city_population":{
        "messages": [
            {"role": "user", "content": "How large, in number, is the population of the city '{entity}'?"},
        ],
        "correctness_function": partial(number_match, tolerance=0, pct=0.1),
    },
    "city_elevation":{
        "messages": [
            {"role": "user", "content": "What is the altitude, in meters, of the city '{entity}'?"},
        ],
        "correctness_function": partial(number_match, tolerance=20),
    },
    "song_album": {
        "messages": [
            {"role": "user", "content": "What is the album of the song '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    },
    "song_publication_year": {
        "messages": [
            {"role": "user", "content": "What is the publication year of the song '{entity}'?"},
        ],
        "correctness_function": partial(number_match, tolerance=0),
    },
    "song_genres": {
        "messages": [
            {"role": "user", "content": "What is the genre of the song '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    },
    "song_performers": {
        "messages": [
            {"role": "user", "content": "Who is the performer of the song '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    }
}

WIKIDATA_QUERIES = {
    "player_is_known" : {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Do you know the player '{entity}'?"},
            {"role": "assistant", "content": "" }
        ],
        "correctness_function": string_match,
    },
    "player_place_birth":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the place of birth for the player '{entity}'?"},
            {"role": "assistant", "content": "City of birth=" }# The player '{entity}' was born in the city of
        ],
        "correctness_function": check_name_correctness,
    },
    "player_date_birth": {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the year of birth for the player '{entity}'?"},
            {"role": "assistant", "content": "Year of birth=" }#The player '{entity}' was born in the year
        ],
        "correctness_function": partial(number_match, tolerance=5),
    },
    "player_teams_list":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What team (name at least one) signed the player '{entity}'?"},
            {"role": "assistant", "content": "Team=" }
        ],
        "correctness_function": check_name_correctness,
    },
    "player_weight":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the weight, in kg, of the player '{entity}'?"},
            {"role": "assistant", "content": "Weight=" }
        ],
        "correctness_function": partial(number_match, tolerance=5),
    },
    "player_height":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the height, in cm, of the player '{entity}'?"},
            {"role": "assistant", "content": "Height=" }
        ],
        "correctness_function": partial(number_match, tolerance=3),
    },
    "movie_is_known":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Do you know the movie '{entity}'?"},
            {"role": "assistant", "content": "" }
        ],
        "correctness_function": string_match,
    },
    "movie_directors":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Who is the director of the movie '{entity}'?"},
            {"role": "assistant", "content": "Director=" }
        ],
        "correctness_function": check_name_correctness,
    },
    'movie_cast': {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the name of an actor starring in the movie '{entity}'?"},
            {"role": "assistant", "content": "Actor=" } 
        ],
        "correctness_function": check_name_correctness,
    },
    "movie_release_year":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What was the release year for the movie '{entity}'?"},
            {"role": "assistant", "content": "Release year=" }
        ],
        "correctness_function": partial(number_match, tolerance=2),
    },
    "movie_screenwriters":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Who is the screenwriter of the movie '{entity}'?"},
            {"role": "assistant", "content": "Screenwriter=" }
        ],
        "correctness_function": check_name_correctness,
    },
    "movie_genres":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What genre label best describes the movie '{entity}'?"},
            {"role": "assistant", "content": "Genre=" }
        ],
        "correctness_function": string_match_in_list_genres,
    },
    "movie_durations":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the duration, in minutes, of the movie '{entity}'?"},
            {"role": "assistant", "content": "Duration(minutes)=" }
        ],
        "correctness_function": partial(number_match_in_list, tolerance=3),
    },
    "city_is_known":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Do you know the city '{entity}'?"},
            {"role": "assistant", "content": "" }
        ],
        "correctness_function": string_match,
    },
    "city_country":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What country contains the city of '{entity}'?"},
            {"role": "assistant", "content": "Country (official name)=" }
        ],
        "correctness_function": check_name_correctness,
    },
    "city_location":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What geographic coordinates correspond to the city of '{entity}'?"},
            {"role": "assistant", "content": "Latitude=" }
        ],
        "correctness_function": geo_location_match,
    },
    "city_population":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "How large, in number, is the population of the city '{entity}'?"},
            {"role": "assistant", "content": "Population=" }
        ],
        "correctness_function": partial(number_match, tolerance=0, pct=0.1),
    },
    "city_elevation":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the altitude, in meters, of the city '{entity}'?"},
            {"role": "assistant", "content": "Altitude=" }
        ],
        "correctness_function": partial(number_match, tolerance=20),
    },
    "song_is_known": {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Do you know the song '{entity}'?"},
            {"role": "assistant", "content": "" }
        ],
        "correctness_function": string_match,
    },
    "song_album": {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the album of the song '{entity}'?"},
            {"role": "assistant", "content": "Album=" }
        ],
        "correctness_function": check_name_correctness,
    },
    "song_publication_year": {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the publication year of the song '{entity}'?"},
            {"role": "assistant", "content": "Publication year=" }
        ],
        "correctness_function": partial(number_match, tolerance=0),
    },
    "song_genres": {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the genre of the song '{entity}'?"},
            {"role": "assistant", "content": "Genre=" }
        ],
        "correctness_function": check_name_correctness,
    },
    "song_performers": {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Who is the performer of the song '{entity}'?"},
            {"role": "assistant", "content": "Performer=" }
        ],
        "correctness_function": check_name_correctness,
    }
}

WIKIDATA_QUERIES_BASE = {
    "player_is_known" : {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Do you know the player '{entity}'?"},
            {"role": "assistant", "content": "" }
        ],
        "correctness_function": string_match,
    },
    "player_place_birth":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the place of birth for the player '{entity}'?"},
            {"role": "assistant", "content": "The city of birth for the player '{entity}' is" }# The player '{entity}' was born in the city of
        ],
        "correctness_function": check_name_correctness,
    },
    "player_date_birth": {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the year of birth for the player '{entity}'?"},
            {"role": "assistant", "content": "The year of birth for the player '{entity}' is" }#The player '{entity}' was born in the year
        ],
        "correctness_function": partial(number_match, tolerance=5),
    },
    "player_teams_list":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What team (name at least one) signed the player '{entity}'?"},
            {"role": "assistant", "content": "The team that signed the player '{entity}' is" }
        ],
        "correctness_function": check_name_correctness,
    },
    "player_weight":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the weight, in kg, of the player '{entity}'?"},
            {"role": "assistant", "content": "The weight of the player '{entity}' is" }
        ],
        "correctness_function": partial(number_match, tolerance=5),
    },
    "player_height":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the height, in cm, of the player '{entity}'?"},
            {"role": "assistant", "content": "The height of the player '{entity}' is" }
        ],
        "correctness_function": partial(number_match, tolerance=3),
    },
    "movie_is_known":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Do you know the movie '{entity}'?"},
            {"role": "assistant", "content": "" }
        ],
        "correctness_function": string_match,
    },
    "movie_directors":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Who is the director of the movie '{entity}'?"},
            {"role": "assistant", "content": "The director of the movie '{entity}' is" }
        ],
        "correctness_function": check_name_correctness,
    },
    'movie_cast': {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the name of an actor starring in the movie '{entity}'?"},
            {"role": "assistant", "content": "The actor starring in the movie '{entity}' is" } 
        ],
        "correctness_function": check_name_correctness,
    },
    "movie_release_year":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What was the release year for the movie '{entity}'?"},
            {"role": "assistant", "content": "The release year for the movie '{entity}' is" }
        ],
        "correctness_function": partial(number_match, tolerance=2),
    },
    "movie_screenwriters":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Who is the screenwriter of the movie '{entity}'?"},
            {"role": "assistant", "content": "The screenwriter of the movie '{entity}' is" }
        ],
        "correctness_function": check_name_correctness,
    },
    "movie_genres":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What genre label best describes the movie '{entity}'?"},
            {"role": "assistant", "content": "The genre label that best describes the movie '{entity}' is" }
        ],
        "correctness_function": string_match_in_list_genres,
    },
    "movie_durations":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the duration, in minutes, of the movie '{entity}'?"},
            {"role": "assistant", "content": "The duration of the movie '{entity}' is" }
        ],
        "correctness_function": partial(number_match_in_list, tolerance=3),
    },
    "city_is_known":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Do you know the city '{entity}'?"},
            {"role": "assistant", "content": "" }
        ],
        "correctness_function": string_match,
    },
    "city_country":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What country contains the city of '{entity}'?"},
            {"role": "assistant", "content": "The country that contains the city '{entity}' is" }
        ],
        "correctness_function": check_name_correctness,
    },
    "city_location":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What geographic coordinates correspond to the city of '{entity}'?"},
            {"role": "assistant", "content": "The geographic coordinates that correspond to the city '{entity}' are" }
        ],
        "correctness_function": geo_location_match,
    },
    "city_population":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "How large, in number, is the population of the city '{entity}'?"},
            {"role": "assistant", "content": "The population of the city '{entity}' is" }
        ],
        "correctness_function": partial(number_match, tolerance=0, pct=0.1),
    },
    "city_elevation":{
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the altitude, in meters, of the city '{entity}'?"},
            {"role": "assistant", "content": "The altitude of the city '{entity}' is" }
        ],
        "correctness_function": partial(number_match, tolerance=20),
    },
    "song_is_known": {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Do you know the song '{entity}'?"},
            {"role": "assistant", "content": "" }
        ],
        "correctness_function": string_match,
    },
    "song_album": {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the album of the song '{entity}'?"},
            {"role": "assistant", "content": "The album of the song '{entity}' is" }
        ],
        "correctness_function": check_name_correctness,
    },
    "song_publication_year": {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the publication year of the song '{entity}'?"},
            {"role": "assistant", "content": "The publication year for the song '{entity}' is" }
        ],
        "correctness_function": partial(number_match, tolerance=0),
    },
    "song_genres": {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the genre of the song '{entity}'?"},
            {"role": "assistant", "content": "The genre label that best describes the song '{entity}' is" }
        ],
        "correctness_function": check_name_correctness,
    },
    "song_performers": {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Who is the performer of the song '{entity}'?"},
            {"role": "assistant", "content": "The performer of the song '{entity}' is" }
        ],
        "correctness_function": check_name_correctness,
    }
}

prompt_types_to_include = {
    'player': ['player_place_birth', 'player_date_birth', 'player_teams_list'],
    'movie': ['movie_directors', 'movie_cast', 'movie_screenwriters', 'movie_genres', 'movie_release_year', 'movie_durations'],
    'city': ['city_country', 'city_location', 'city_population', 'city_elevation'],  
    "song": ['song_album', 'song_publication_year', 'song_genres', 'song_performers']
    #'city': ['city_country'],
}

# %%

def merge_json_files(folder_path):
    merged_data = []
    
    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # Read and parse each JSON file
            with open(file_path, 'r') as file:
                try:
                    file_data = json.load(file)
                    
                    # Ensure file_data is a list
                    if isinstance(file_data, list):
                        # Extend the merged_data list with the file's data
                        merged_data.extend(file_data)
                    else:
                        print(f"Warning: {filename} does not contain a list of dictionaries. Skipping.")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {filename}")
    
    return merged_data

def format_prompt(tokenizer, model_alias, entity_data: Dict[str, str], prompt_type: str, QUERIES_TEMPLATE=WIKIDATA_QUERIES):
    conversation = deepcopy(QUERIES_TEMPLATE[prompt_type]['messages'])
    if model_is_chat_model[model_alias]:
        # Format entity data for each message in the conversation
        for message in conversation:
            message['content'] = message['content'].format(**entity_data)
        # Remove empty messages and remove the system message if the model is gemma
        conversation = [m for m in conversation if len(m['content']) > 0 and not (m['role'] == 'system' and 'gemma' in model_alias)]
        # Add the generation prompt if the last message is a user message
        if conversation[-1]['role'] == 'assistant':
            append_str = conversation[-1]['content']
            conversation = conversation[:-1]
        else:
            append_str = ""
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        prompt += append_str
    else:
        prompt = conversation[1]['content'].format(**entity_data)
        prompt += " " + conversation[2]['content'].format(**entity_data)
        conversation = None
    return prompt, conversation

# %%

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_engine', type=str, default='vllm') # ['hf', 'vllm', 'api']
    parser.add_argument('--model_path', type=str, default='google/gemma-2-9b-it')
    parser.add_argument('--free_generation', type=bool, default=True, help='Enable free generation mode')
    parser.add_argument('--max_new_tokens', type=int, default=32, help='Maximum number of new tokens to generate')
    parser.add_argument('--max_num_queries', type=int, default=None, help='Maximum number of queries to generate')
    parser.add_argument('--num_sampled_generations', type=int, default=10, help='Number of sampled generations to run')
    return parser.parse_args()

# %%

def run_query_generations(batch_completion_fn: Callable, queries: List[Dict[str, Any]], max_new_tokens: int=32, num_sampled_generations: int=10):

    if len(queries) == 0:
        return []

    print(f"Running generations for {len(queries)} queries.")

    prompts = [q['prompt'] for q in queries]
    labels = [q['correct_answer'] for q in queries]
    print('Prompts, Labels:', random.sample(list(zip(prompts, labels)), 5))

    greedy_llm_outputs = batch_completion_fn(prompts=prompts, n=1, temperature=0.0, max_tokens=max_new_tokens)
    generations = [output['choices'][0]['message']['content'] for output in greedy_llm_outputs]
    print('Generations:', random.sample(generations, 5))

    if num_sampled_generations > 0:
        sampled_llm_outputs = batch_completion_fn(prompts=prompts, n=num_sampled_generations, temperature=1.0, max_tokens=max_new_tokens)
        multiple_generations = [[choice['message']['content'] for choice in outputs['choices']] for outputs in sampled_llm_outputs]
    else:
        multiple_generations = [[] for _ in range(len(queries))]

    for query_idx, query in enumerate(queries):
        query['greedy_completion'] = generations[query_idx]
        query['sampled_completions'] = multiple_generations[query_idx]

    return queries

def get_label(completion: str, correct_answer: str, check_correctness_fn: Callable):
    if check_correctness_fn(completion, correct_answer)==True:
            label = 'correct'
    else:
        if is_generation_refusal(completion)==True:
            label = 'refusal'
        else:
            label = 'error'
    return label

def post_process_queries(queries: List[Dict[str, Any]], check_correctness_fn: Callable):
    for query in queries:
        query['string_matching_greedy_label'] = get_label(query['greedy_completion'], query['correct_answer'], check_correctness_fn)
        query['label'] = query['string_matching_greedy_label']

        for completion in query['sampled_completions']:
            label = get_label(completion, query['correct_answer'], check_correctness_fn)
            query['string_matching_sampled_labels'].append(label)

    return queries

def main(batch_completion_fn: Callable, tokenizer: AutoTokenizer, model_alias: str, free_generation: bool = True, max_new_tokens: int = 32, max_num_queries: int = None, num_sampled_generations: int=10):

    if max_num_queries is not None:
        print(f"Max number of queries: {max_num_queries}")
    print(f"Model alias: {model_alias}")
    print(f"Max new tokens: {max_new_tokens}")

    raw_entities_directory = "../../processed/entities"

    processed_queries_directory = f"../../processed/entity_prompts/{model_alias}"

    os.makedirs(processed_queries_directory, exist_ok=True)

    statistics = {}

    for entity_type in prompt_types_to_include.keys():
        
        print(f"Creating queries and running generations for entity_type={entity_type}...")
        print(os.path.join(raw_entities_directory, f"{entity_type}.json"))
        items = json.load(open(os.path.join(raw_entities_directory, f"{entity_type}.json"), 'r'))

        if max_num_queries is not None:
            items = items[:max_num_queries]

        queries_by_prompt_type = defaultdict(list)

        if model_is_chat_model[model_alias]:
            if free_generation:
                QUERIES_TEMPLATE = WIKIDATA_QUERIES_FREE
            else:
                QUERIES_TEMPLATE = WIKIDATA_QUERIES
        else:
            QUERIES_TEMPLATE = WIKIDATA_QUERIES_BASE

        for item in items:
            assert item['entity_type'] == entity_type

            entity = item['entity']
            entity_data = {"entity": entity, "entity_type": entity_type}

            for attribute_obj in item['attributes']:
                entity_data[attribute_obj['attribute_type']] = attribute_obj['attribute_value']

            is_known_prompt_type = f"{entity_type}_is_known"

            if is_known_prompt_type in prompt_types_to_include[entity_type]:
                

                prompt, conversation = format_prompt(tokenizer, model_alias, entity_data=entity_data, prompt_type=is_known_prompt_type, QUERIES_TEMPLATE=QUERIES_TEMPLATE)
                queries_by_prompt_type[is_known_prompt_type].append({
                    "entity": entity,
                    "entity_type": entity_type,
                    "prompt_type": is_known_prompt_type,
                    "correct_answer": "",
                    "prompt": prompt,
                    "question": conversation[0]['content'] if conversation is not None else None,
                    "greedy_completion": None,
                    "string_matching_sampled_labels": [],
                    "label": None,
                })

            for attribute_obj in item['attributes']:

                attribute_type = attribute_obj['attribute_type']
                print(attribute_type)
                attribute = attribute_obj['attribute_value']

                prompt_type = f"{entity_type}_{attribute_type}"

                if prompt_type not in prompt_types_to_include[entity_type]:
                    continue

                prompt, conversation = format_prompt(tokenizer, model_alias, entity_data=entity_data, prompt_type=prompt_type, QUERIES_TEMPLATE=QUERIES_TEMPLATE)

                queries_by_prompt_type[prompt_type].append({
                    "entity": entity,
                    "entity_type": entity_type,
                    "prompt_type": prompt_type,
                    "correct_answer": attribute,
                    "prompt": prompt,
                    "question": conversation[0]['content'] if conversation is not None else None,
                    "greedy_completion": None,
                    "string_matching_sampled_labels": [],
                    "label": None,
                })

        prompt_types = list(queries_by_prompt_type.keys())

        for prompt_type in prompt_types:

            queries_with_generations = []
            if model_is_chat_model[model_alias]:
                if free_generation:
                    file_path = os.path.join(processed_queries_directory, f"free_generation/{prompt_type}.json")
                else:
                    file_path = os.path.join(processed_queries_directory, f"{prompt_type}.json")
            else:
                file_path = os.path.join(processed_queries_directory, f"{prompt_type}.json")
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Create a set of existing entity names for faster lookup
            entities_with_generations = set(q['entity'] for q in queries_with_generations)

            # Filter out queries that are already in the file
            queries_without_generations = [
                q for q in queries_by_prompt_type[prompt_type]
                if q['entity'] not in entities_with_generations
            ]

            print(f"Filtered out {len(queries_by_prompt_type[prompt_type]) - len(queries_without_generations)} existing queries.")

            queries_with_generations += run_query_generations(batch_completion_fn, queries_without_generations, max_new_tokens, num_sampled_generations=num_sampled_generations)

            if model_is_chat_model[model_alias]:
                check_correctness_fn = WIKIDATA_QUERIES[prompt_type]['correctness_function']
            else:
                check_correctness_fn = WIKIDATA_QUERIES_BASE[prompt_type]['correctness_function']

            processed_queries = post_process_queries(queries_with_generations, check_correctness_fn)

            labels = [q['label'] for q in processed_queries]
            # these info is for greedy completion
            n_corrects = sum([q['label'] == 'correct' for q in processed_queries])
            n_errors = sum([q['label'] == 'error' for q in processed_queries])
            n_refusals = sum([q['label'] == 'refusal' for q in processed_queries])

            average_score = n_corrects / len(labels)

            print(f"Entity type {entity_type}, Prompt Type {prompt_type}, Avg Accuracy {average_score*100:.2f}%")

            statistics[prompt_type] = {
                'correct': n_corrects,
                'error': n_errors,
                'refusal': n_refusals,
            }

            # Save the updated list of queries
            print('file_path', file_path)
            with open(file_path, 'w') as f:
                json.dump(processed_queries, f, indent=4)

            print(f"Updated {file_path} with {len(processed_queries)} new queries. Total queries: {len(processed_queries)}")

    print(statistics)     
    plot_bar_chart(statistics, ['correct', 'error', 'refusal'], title=f"{model_alias}", xaxis_title="Prompt type", yaxis_title="Count", legend_title="Result", save_path=f"./{model_alias}_free_{free_generation}_stats_wikidata.png")

# %%

if __name__ == "__main__":

    args = parse_args()

    # TODO: Remove this
    batch_completion_fn = get_batch_completion_fn(
        model_engine=args.model_engine,
        model_path=args.model_path,
    )
    #batch_completion_fn = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'

    model_alias = args.model_path.split('/')[-1]

    main(
        batch_completion_fn=batch_completion_fn, 
        tokenizer=tokenizer, 
        model_alias=model_alias,
        free_generation=args.free_generation,
        max_new_tokens=args.max_new_tokens, 
        max_num_queries=args.max_num_queries,
        num_sampled_generations=args.num_sampled_generations,
    )

# # %%

# batch_completion_fn = get_batch_completion_fn(
#     model="google/gemma-2b-it",
#     use_vllm=True,
# )

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", trust_remote_code=True)
# tokenizer.padding_side = 'left'

# %%

# main(
#     batch_completion_fn=batch_completion_fn,
#     tokenizer=tokenizer,
#     model_alias="gemma-2b-it",
#     max_new_tokens=32,
# )

# %%