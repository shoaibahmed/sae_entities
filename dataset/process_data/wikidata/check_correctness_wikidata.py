# %%
import re
from thefuzz import fuzz

def check_refusal(generation, refusal_substrings = ["unable", "no,", "sorry"]):
    generation = generation.lower().strip()
    for refusal_sub_string in refusal_substrings:
        if refusal_sub_string in generation:
            return 2
    return 1

def compute_is_known(generation, refusal_substrings = ["unable", "no,", "sorry"]):
    generation = generation.lower().strip()
    for refusal_sub_string in refusal_substrings:
        if refusal_sub_string in generation:
            return 2
    return 1

def check_name_correctness(generation, correct_answer):
    generation = generation.lower().replace(",", " ").replace(".", " ").replace("-", " ")
    correct_answer = correct_answer.lower()
    if correct_answer == "united states of america":
        correct_answer = "united states of america, united states, us, usa"
    elif correct_answer == "people's republic of china":
        correct_answer = "people's republic of china, china"
        
    any_correct = False

    for name in correct_answer.split(","):
        any_correct = any_correct or fuzz.token_set_ratio(name, generation) > 90

    return any_correct

# %%


def string_match(generation, correct_answer):
    generation = generation.lower().strip()
    correct_answer = correct_answer.lower()
    return correct_answer.lower() in generation.lower()

def string_match_in_list(generation, correct_answer):
    generation = generation.lower().strip()
    correct_answer = correct_answer.lower()
    correct_answer_list = correct_answer.split(',')
    for correct_answer_ in correct_answer_list:
        if correct_answer_.lower() in generation.lower():
            return 1
    return 0

def number_match_in_list(generation, correct_answer, tolerance:int=5):
    generation = generation.lower().strip()
    correct_answer = correct_answer.lower()
    correct_answer_list = correct_answer.split(',')
    numbers_found = re.findall(r'\d+', generation)
    if len(numbers_found) == 0:
        return 0
    else:
        for correct_answer_ in correct_answer_list:
            for number in numbers_found:
                if abs(int(float(number)) - int(float(correct_answer_))) <= tolerance:
                    return 1
    return 0

def string_match_in_list_genres(generation, correct_answer):
    generation = generation.lower().strip()
    correct_answer = correct_answer.lower()
    correct_answer_list = correct_answer.split(',')
    correct_answer_list = [movie_genre.replace(' film','') for movie_genre in correct_answer_list]
    for correct_answer_ in correct_answer_list:
        if correct_answer_ in generation:
            return 1
    return 0

def number_match(generation, correct_answer, tolerance:int=5, pct=None):
    generation = generation.lower().strip()
    # Remove comma separator of numbers
    generation = generation.replace(',', '')
    numbers_found = re.findall(r'\d+', generation)
    if len(numbers_found) == 0:
        return 0
    else:
        for number in numbers_found:
            if pct==None:
                if abs(int(float(number)) - int(float(correct_answer))) <= tolerance:
                    return 1
            else:
                if abs(int(float(number)) - int(float(correct_answer))) <= pct*int(float(correct_answer)):
                    return 1
    return 0

def geo_location_match(generation, correct_answer, tolerance:int=1):
    """
    Checks if the geographical location mentioned in the model response matches the correct answer.

    Args:
        generation (str): The model response.
        correct_answer (str): The correct geographical location.

    Returns:
        bool: 1 if the geographical location in the assistant's response matches the correct answer, 0 otherwise.
    """
    longitude, latitude = correct_answer.replace('Point(', '').replace(')','').split(' ')
    longitude, latitude = abs(int(float(longitude))), abs(int(float(latitude)))
    generation = generation.lower()
    numbers_found = re.findall(r'-?\d+\.?\d*', generation)
    numbers_found = [abs(int(float(number))) for number in numbers_found]
    if len(numbers_found) == 0:
        return 0
    else:
        counter_correct = 0
        for number in numbers_found:
            for correct_ans in [longitude, latitude]:
                if abs(int(number) - int(correct_ans)) <= tolerance:
                    counter_correct += 1
            if counter_correct==2:
                return 1
    return 0
