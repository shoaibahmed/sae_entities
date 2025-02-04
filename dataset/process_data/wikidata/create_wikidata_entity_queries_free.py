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
# sys.path.append("../")
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
from utils.generation_utils import get_batch_completion_fn
from wikidata.check_correctness_wikidata import check_name_correctness, string_match, string_match_in_list, string_match_in_list_genres, number_match, geo_location_match, number_match_in_list
from wikidata.create_wikidata_entity_queries import format_prompt, parse_args, run_query_generations, post_process_queries, merge_json_files

torch.set_grad_enabled(False)

# %%

WIKIDATA_QUERIES = {
    "player_place_birth_free":{
        "messages": [
            {"role": "user", "content": "What is the place of birth for the player '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    },
    "player_date_birth_free": {
        "messages": [
            {"role": "user", "content": "What is the year of birth for the player '{entity}'?"},
        ],
        "correctness_function": partial(number_match, tolerance=0),
    },
    "player_teams_list_free":{
        "messages": [
            {"role": "user", "content": "What team (name at least one) signed the player '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    },
    "player_weight_free":{
        "messages": [
            {"role": "user", "content": "What is the weight, in kg, of the player '{entity}'?"},
        ],
        "correctness_function": partial(number_match, tolerance=5),
    },
    "player_height_free":{
        "messages": [
            {"role": "user", "content": "What is the height, in cm, of the player '{entity}'?"},
        ],
        "correctness_function": partial(number_match, tolerance=3),
    },
    "movie_directors_free":{
        "messages": [
            {"role": "user", "content": "Who is the director of the movie '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    },
    "movie_cast_free": {
        "messages": [
            {"role": "user", "content": "What is the name of an actor starring in the movie '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    },
    "movie_release_year_free":{
        "messages": [
            {"role": "user", "content": "What was the release year for the movie '{entity}'?"},
        ],
        "correctness_function": partial(number_match, tolerance=2),
    },
    "movie_screenwriters_free":{
        "messages": [
            {"role": "user", "content": "Who is the screenwriter of the movie '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    },
    "movie_genres_free":{
        "messages": [
            {"role": "user", "content": "What genre label best describes the movie '{entity}'?"},
        ],
        "correctness_function": string_match_in_list_genres,
    },
    "movie_durations_free":{
        "messages": [
            {"role": "user", "content": "What is the duration, in minutes, of the movie '{entity}'?"},
        ],
        "correctness_function": partial(number_match_in_list, tolerance=3),
    },
    "city_country_free":{
        "messages": [
            {"role": "user", "content": "What country contains the city of '{entity}'?"},
        ],
        "correctness_function": string_match,
    },
    "city_location_free":{
        "messages": [
            {"role": "user", "content": "What geographic coordinates correspond to the city of '{entity}'?"},
        ],
        "correctness_function": geo_location_match,
    },
    "city_population_free":{
        "messages": [
            {"role": "user", "content": "How large, in number, is the population of the city '{entity}'?"},
        ],
        "correctness_function": partial(number_match, tolerance=0, pct=0.1),
    },
    "city_elevation_free":{
        "messages": [
            {"role": "user", "content": "What is the altitude, in meters, of the city '{entity}'?"},
        ],
        "correctness_function": partial(number_match, tolerance=20),
    },
    "song_album_free": {
        "messages": [
            {"role": "user", "content": "What is the album of the song '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    },
    "song_publication_year_free": {
        "messages": [
            {"role": "user", "content": "What is the publication year of the song '{entity}'?"},
        ],
        "correctness_function": partial(number_match, tolerance=0),
    },
    "song_genres_free": {
        "messages": [
            {"role": "user", "content": "What is the genre of the song '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    },
    "song_performers_free": {
        "messages": [
            {"role": "user", "content": "Who is the performer of the song '{entity}'?"},
        ],
        "correctness_function": check_name_correctness,
    }
}

prompt_types_to_include = {
    # 'player': ['player_place_birth_free', 'player_date_birth_free'],
    # 'movie': ['movie_directors_free', 'movie_release_year_free'],
    # 'city': ['city_location_free', 'city_population_free'],
    # 'song': ['song_perfomers_free', 'song_publication_year_free']
    'player': ['player_place_birth_free', 'player_date_birth_free', 'player_teams_list_free'],
    'movie': ['movie_directors_free', 'movie_cast_free', 'movie_screenwriters_free', 'movie_genres_free', 'movie_release_year_free', 'movie_durations_free'],
    'city': ['city_country_free', 'city_location_free', 'city_population_free', 'city_elevation_free'],
    "song": ['song_album_free', 'song_publication_year_free', 'song_genres_free', 'song_performers_free']
}

# %%

def main(batch_completion_fn: Callable, tokenizer: AutoTokenizer, model_alias: str, max_new_tokens: int = 32, max_num_queries: int = None, num_sampled_generations: int=10):

    if max_num_queries is not None:
        print(f"Max number of queries: {max_num_queries}")
    print(f"Model alias: {model_alias}")
    print(f"Max new tokens: {max_new_tokens}")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Current directory: {os.getcwd()}")
    raw_entities_directory = "../../processed/entities"
    processed_queries_directory = f"../../processed/entity_prompts/{model_alias}"

    statistics = {}

    for entity_type in prompt_types_to_include.keys():
        
        print(f"Creating queries and running generations for entity_type={entity_type}...")

        items = json.load(open(os.path.join(raw_entities_directory, f"{entity_type}.json"), 'r'))

        if max_num_queries is not None:
            items = items[:max_num_queries]

        queries_by_prompt_type = defaultdict(list)

        for item in items:
            assert item['entity_type'] == entity_type

            entity = item['entity']
            entity_data = {"entity": entity, "entity_type": entity_type}

            for attribute_obj in item['attributes']:
                entity_data[attribute_obj['attribute_type']] = attribute_obj['attribute_value']

            is_known_prompt_type = f"{entity_type}_is_known"

            if is_known_prompt_type in prompt_types_to_include[entity_type]:

                prompt, conversation = format_prompt(tokenizer, model_alias, entity_data=entity_data, prompt_type=is_known_prompt_type, QUERIES_TEMPLATE=WIKIDATA_QUERIES)
                
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
                attribute = attribute_obj['attribute_value']

                # IMPORTANT CHANGE: Add _free to the prompt type
                prompt_type = f"{entity_type}_{attribute_type}_free"

                if prompt_type not in prompt_types_to_include[entity_type]:
                    continue

                prompt, conversation = format_prompt(tokenizer, model_alias, entity_data=entity_data, prompt_type=prompt_type, QUERIES_TEMPLATE=WIKIDATA_QUERIES)

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

            # Load existing queries if the file exists
            queries_with_generations = []

            file_path = os.path.join(processed_queries_directory, f"{prompt_type}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    queries_with_generations = json.load(f)

            # Create a set of existing entity names for faster lookup
            entities_with_generations = set(q['entity'] for q in queries_with_generations)

            # Filter out queries that are already in the file
            queries_without_generations = [
                q for q in queries_by_prompt_type[prompt_type]
                if q['entity'] not in entities_with_generations
            ]

            print(f"Filtered out {len(queries_by_prompt_type[prompt_type]) - len(queries_without_generations)} existing queries.")

            queries_with_generations += run_query_generations(batch_completion_fn, queries_without_generations, max_new_tokens, num_sampled_generations=num_sampled_generations)

            check_correctness_fn = WIKIDATA_QUERIES[prompt_type]['correctness_function']

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
            if not os.path.exists(processed_queries_directory):
                os.makedirs(processed_queries_directory)
            with open(file_path, 'w') as f:
                json.dump(processed_queries, f, indent=4)

            print(f"Updated {file_path} with {len(processed_queries)} new queries. Total queries: {len(processed_queries)}")
                
    plot_bar_chart(statistics, ['correct', 'error', 'refusal'], title=f"{model_alias}", xaxis_title="Prompt type", yaxis_title="Count", legend_title="Result", save_path=f"./{model_alias}_stats_wikidata_free.png")

# %%

if __name__ == "__main__":

    args = parse_args()

    batch_completion_fn = get_batch_completion_fn(
        model_engine=args.model_engine,
        model_path=args.model_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'

    model_alias = args.model_path.split('/')[-1]

    main(
        batch_completion_fn=batch_completion_fn, 
        tokenizer=tokenizer, 
        model_alias=model_alias, 
        max_new_tokens=args.max_new_tokens, 
        max_num_queries=args.max_num_queries,
        num_sampled_generations=args.num_sampled_generations
    )

# # %%

# batch_completion_fn = get_batch_completion_fn(
#     model_engine='vllm',
#     model_path="google/gemma-2-2b-it",
# )

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", trust_remote_code=True)
# tokenizer.padding_side = 'left'

# # %%

# main(
#     batch_completion_fn=batch_completion_fn,
#     tokenizer=tokenizer,
#     model_alias="gemma-2-2b-it",
#     max_new_tokens=32,
#     max_num_queries=1024,
# )

# %%