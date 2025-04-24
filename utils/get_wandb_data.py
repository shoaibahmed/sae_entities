#!/bin/python

import os
import re
import wandb
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


timestamp = "24_04_25"
pickle_output_file = f"sae_entities_data_{timestamp}.pkl"
if not os.path.exists(pickle_output_file):
    api = wandb.Api()
    project_name = "knowledge_awareness"

    # Fetch all runs in the specified project
    runs = api.runs(path=project_name)
    print("Total runs in the project:", len(runs))

    # Iterate through each run and retrieve the data
    summary_list, config_list, name_list = [], [], []
    data_list = []
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
        name_list.append(run.name)

    output_dict = dict(summary_list=summary_list, config_list=config_list, name_list=name_list)
    with open(pickle_output_file, "wb") as f:
        pickle.dump(output_dict, f)

# Reload the file
print("Loading data from file:", pickle_output_file)
with open(pickle_output_file, "rb") as f:
    output_dict = pickle.load(f)
print("Loaded keys:", output_dict.keys())
name_list = output_dict["name_list"]
summary_list = output_dict["summary_list"]
print("Model names:", name_list)
layer_key_list = sorted(set([int(x.split("/")[0].replace("layer_", "")) for x in summary_list[0].keys() if x.startswith("layer_")]))
print("Layer key list:", layer_key_list)

output_file_format = "png"
plots_output_dir = f"knowledge_awareness_evals_{timestamp}/"
if not os.path.exists(plots_output_dir):
    os.mkdir(plots_output_dir)
    print("Plots output directory created:", plots_output_dir)

# Plot the predictions
fontsize = 18
dataset_list = ["test_player", "eval_movie", "eval_song", "eval_city"]
for idx, full_name in enumerate(name_list):
    print("!! Run name:", full_name)
    plt.figure(figsize=(6, 4))

    for dataset in dataset_list:
        ds_list = []
        for layer in layer_key_list:
            acc_key = f"layer_{layer}/probe/{dataset}/acc"
            assert acc_key in summary_list[idx], f"{acc_key} not found in {summary_list[idx].keys()}"
            acc = summary_list[idx][acc_key]
            ds_list.append(acc)
        plt.plot(layer_key_list, ds_list, label=dataset, linewidth=4)

    plt.ylim(-5, 105)
    plt.xlabel('Layer', fontsize=fontsize)
    plt.ylabel('Accuracy (%)', fontsize=fontsize)
    plt.legend(loc="lower right", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_dir, f"{full_name}.{output_file_format}"), dpi=300, bbox_inches="tight")
    plt.close()

print("Plotting finished")
