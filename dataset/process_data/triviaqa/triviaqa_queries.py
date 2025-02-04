# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

# %%
import sys
sys.path.append("../../../")

# %%
import os
import datasets
from datasets import Dataset, load_dataset, concatenate_datasets
from typing import Callable, List, Any, Dict
from thefuzz import fuzz
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass, field
import glob
import json
from utils.generation_utils import is_generation_refusal, get_batch_completion_fn

# %%
@dataclass
class TriviaQAQuery:
    question_id: str
    question: str
    correct_answer: List[str]
    split: str
    prompt: str

    # Optional attributes
    sampled_completions: List[str] = field(default_factory=list)
    greedy_completion: str = ''
    llm_greedy_label: str = ''
    llm_sampled_labels: List[str] = field(default_factory=list)
    string_matching_greedy_label: str = ''
    string_matching_sampled_labels: List[str] = field(default_factory=list)
    greedy_agreement_label: int = -1
    label: str = ''

    def to_dict(self):
        return {
            'question_id': self.question_id,
            'question': self.question,
            'correct_answer': self.correct_answer,
            'split': self.split,
            'prompt': self.prompt,
            'sampled_completions': self.sampled_completions,
            'greedy_completion': self.greedy_completion,
            'llm_greedy_label': self.llm_greedy_label,
            'llm_sampled_labels': self.llm_sampled_labels,
            'string_matching_greedy_label': self.string_matching_greedy_label,
            'string_matching_sampled_labels': self.string_matching_sampled_labels,
            'greedy_agreement_label': self.greedy_agreement_label,
            'label': self.label,
        }

# def check_matching_dataset_features(existing_dataset, new_dataset):
#     existing_features = set(existing_dataset.features.keys())
#     new_features = set(new_dataset.features.keys())
    
#     if existing_features != new_features:
#         missing_in_existing = new_features - existing_features
#         missing_in_new = existing_features - new_features
        
#         error_message = "Features of the dataset do not match the existing dataset on disk.\n"
#         if missing_in_existing:
#             error_message += f"Features missing in existing dataset: {', '.join(missing_in_existing)}\n"
#         if missing_in_new:
#             error_message += f"Features missing in new dataset: {', '.join(missing_in_new)}\n"
        
#         raise AssertionError(error_message)
    
#     # If keys match, check if the feature types match
#     for feature in existing_features:
#         if type(existing_dataset.features[feature]) != type(new_dataset.features[feature]):
#             raise AssertionError(f"Feature type mismatch for '{feature}': "
#                                     f"existing {type(existing_dataset.features[feature])} != "
#                                     f"new {type(new_dataset.features[feature])}")

def load_and_format_triviaqa_queries_from_hub(format_instruction_chat_fn: Callable, local_dataset_path: str, split: str = "train") -> Dataset:
    dataset = load_dataset("trivia_qa", 'rc', split=split)

    # First duplicate elements are filtered
    unique_questions = set()

    def is_duplicate(example):
        if example['question'] in unique_questions:
            return False
        unique_questions.add(example['question'])
        return True

    # Filter the dataset efficiently using the is_duplicate function
    dataset = dataset.filter(is_duplicate, num_proc=1) # IMPORTANT: Use num_proc = 1 to avoid race conditions

    def add_fields(example):
        example['split'] = split
        example['prompt'] = format_instruction_chat_fn(example['question'])
        example['correct_answer'] = example['answer']['aliases']
        assert isinstance(example['correct_answer'], list) and all(isinstance(item, str) for item in example['correct_answer']), f"Correct answer is not a list of strings: {example['correct_answer']}"
        return example

    # Apply the function to all items in the dataset
    dataset = dataset.map(add_fields)

    # Keep only the specified fields
    required_columns = ['question_id', 'question', 'correct_answer', 'split', 'prompt']
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in required_columns])

    def add_triviaqa_query_attributes(example):
        example = TriviaQAQuery(**example).to_dict()
        return example

    dataset = dataset.map(add_triviaqa_query_attributes)

    # # Verify that the features are correct by checking the features of the dataset that is already on the disk
    # # Concatenate and save both datasets
    # if os.path.exists(local_dataset_path):
    #     existing_dataset = datasets.load_from_disk(local_dataset_path)
    #     check_matching_dataset_features(existing_dataset, dataset)

    return dataset

necessary_query_attributes_ = ["greedy_completion", "sampled_completions", "label", "string_matching_greedy_label", "string_matching_sampled_labels"]

# check if the query has already been processed given its attributes
def is_query_processed(query: Dict[str, Any], necessary_attributes: List[str] = necessary_query_attributes_):
    for attribute in necessary_attributes:
        if attribute not in query:
            return False
        if len(query[attribute]) == 0:
            return False
    return True

# filter out queries that have already been processed and have already been saved
def filter_processed_queries(new_queries: Dataset, local_dataset_path: str):
    # processed_queries = datasets.load_from_disk(local_dataset_path)
    # return new_queries.filter(lambda x: not is_query_processed(x))
    return new_queries

def string_matching_classify_response_fn(prompt: str, generation: str, correct_answers: List[str]):
    if is_generation_refusal(generation):
        return 'refusal'

    generation = generation.lower().replace(",", " ").replace(".", " ").replace("-", " ")

    any_correct = False
    for correct_answer in correct_answers:
        correct_answer = correct_answer.lower()
        any_correct = any_correct or fuzz.token_set_ratio(generation, correct_answer) > 90

    if any_correct:
        return 'correct'
    return 'error'


def process_triviaqa_queries(batch_completion_fn: Callable, queries: Dataset, local_dataset_path: str, max_tokens: int = 32, batch_size: int = 512, string_matching_labeling_fn: Callable = string_matching_classify_response_fn, necessary_attributes: List[str] = necessary_query_attributes_) -> None:

    assert isinstance(queries, Dataset), "queries must be a HuggingFacedatasets.Dataset object"
    
    initial_queries_length = queries.num_rows

    # Filter out processed queries
    queries = filter_processed_queries(queries, local_dataset_path)

    print(f"Filtered out {initial_queries_length - queries.num_rows} queries that have already been processed.")
    
    if queries.num_rows == 0:
        print("No new queries to process.")
        return

    # TODO: Remove this
    #start_batch_idx = 58 * batch_size
    start_batch_idx = 0

    # for i in tqdm(range(0, len(queries), batch_size), desc="Processing query batches"):
    for i in tqdm(range(start_batch_idx, len(queries), batch_size), desc="Processing query batches"):
        # a HF Arrow dataset has to be sliced like this
        batch: Dict[str, List[Any]] = queries[i:i+batch_size] # IMPORTANT: This is a dictionary of lists, not a list of dictionaries
        batch_prompts: List[str] = batch['prompt']
        batch_questions: List[str] = batch['question']
        batch_correct_answers: List[List[str]] = batch['correct_answer']
        
        # Greedy completions
        greedy_outputs = batch_completion_fn(prompts=batch_prompts, n=1, temperature=0.0, max_tokens=max_tokens)
        batch_greedy_completions: List[str] = [output['choices'][0]['message']['content'] for output in greedy_outputs]
        
        # Sampled completions
        sampled_outputs = batch_completion_fn(prompts=batch_prompts, n=10, temperature=1.0, max_tokens=max_tokens)
        batch_sampled_completions: List[List[str]] = [[choice['message']['content'] for choice in output['choices']] for output in sampled_outputs]

        string_matching_greedy_labels = []
        string_matching_sampled_labels = []

        for question, greedy_completion, prompt_sampled_completions, correct_answers in zip(batch_questions, batch_greedy_completions, batch_sampled_completions, batch_correct_answers):
            string_matching_greedy_labels.append(string_matching_labeling_fn(question, greedy_completion, correct_answers))
            string_matching_sampled_labels.append([string_matching_labeling_fn(question, completion, correct_answers) for completion in prompt_sampled_completions])

        batch_processed_queries: List[Dict[str, Any]] = []

        for query_idx in range(len(batch['prompt'])):
            # Create a dictionary for the current query
            query = {key: batch[key][query_idx] for key in batch.keys()}

            # Update query with new information
            processed_query = {
                **query,
                "greedy_completion": batch_greedy_completions[query_idx],
                "sampled_completions": batch_sampled_completions[query_idx],
                "string_matching_greedy_label": string_matching_greedy_labels[query_idx],
                "string_matching_sampled_labels": string_matching_sampled_labels[query_idx],
                "label": string_matching_greedy_labels[query_idx] # At the moment we simply take the greedy label as the final label
            }
            batch_processed_queries.append(processed_query)

        # Convert DataFrame to an Arrow dataset
        #processed_dataset = Dataset.from_pandas(pd.DataFrame(batch_processed_queries))

        print(f"Saving batch {i//batch_size} of {len(queries)//batch_size}")
        #processed_dataset.to_json(f"{local_dataset_path}/batch_{i//batch_size}.jsonl")
        with open(f"{local_dataset_path}/batch_{i//batch_size}.json", "w") as f:
            json.dump(batch_processed_queries, f)

    # save_processed_triviaqa_queries(save_path=local_dataset_path, file_pattern=f"{local_dataset_path}/batch_*.jsonl")

# def save_processed_triviaqa_queries(save_path: str, file_pattern: str):

#     # Find all files matching the pattern
#     jsonl_files = glob.glob(file_pattern)

#     # Load each file as a dataset and store in a list
#     datasets_list = []

#     for file in jsonl_files:
#         dataset = load_dataset("json", data_files=file)
#         datasets_list.append(dataset["train"])  # Assuming each file creates a dataset with a "train" split

#     # Count unique questions in each individual dataset
#     individual_unique_counts = []
#     for dataset in datasets_list:
#         unique_questions = set(dataset['question'])
#         individual_unique_counts.append(len(unique_questions))

#     # Concatenate all datasets
#     if datasets_list:
#         combined_dataset = concatenate_datasets(datasets_list)
#         print(f"Combined dataset size: {len(combined_dataset)}")
#     else:
#         print("No matching files found.")
#         combined_dataset = None

#     # Sum of unique questions across individual datasets
#     sum_individual_unique = sum(individual_unique_counts)

#     # Count unique questions in the combined dataset
#     combined_unique_questions = set(combined_dataset['question'])
#     combined_unique_count = len(combined_unique_questions)

#     print(f"Number of unique questions in combined dataset: {combined_unique_count}")
#     print(f"Sum of unique questions across individual datasets: {sum_individual_unique}")

#     if combined_unique_count < sum_individual_unique:
#         print("There are duplicate questions across different datasets.")
#     elif combined_unique_count == sum_individual_unique:
#         print("There are no duplicate questions across different datasets.")
#     else:
#         print("Warning: The combined dataset has more unique questions than the sum of individual datasets. This shouldn't happen and might indicate an error in the data processing.")

#     combined_dataset.save_to_disk(save_path)

# %%

def main():

    model_path = "google/gemma-2b-it"
    model_alias = "gemma-2b-it"
    local_dataset_path = f"/root/entity_recognition/dataset/processed/triviaqa/{model_alias}"
    os.makedirs(local_dataset_path, exist_ok=True)
    max_tokens = 48
    
    if 'gemma' in model_alias:
        from utils.hf_models.gemma_model import format_instruction_gemma_chat
        format_instruction_chat_fn = format_instruction_gemma_chat
    elif 'llama-3' in model_alias:
        from utils.hf_models.llama3_model import format_instruction_llama3_chat
        format_instruction_chat_fn = format_instruction_llama3_chat
    else:
        raise ValueError(f"Model alias {model_alias} not supported.")

    dataset = load_and_format_triviaqa_queries_from_hub(format_instruction_chat_fn, local_dataset_path, split='validation')

    batch_completion_fn = get_batch_completion_fn(
        model_engine="vllm",
        model_path=model_path,
    )

    process_triviaqa_queries(
        batch_completion_fn=batch_completion_fn,
        queries=dataset,
        local_dataset_path=local_dataset_path,
        max_tokens=max_tokens,
    )

# %%

if __name__ == '__main__':
    main()