# Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models

Paper: https://arxiv.org/abs/2411.14257
## Setup
Setup a virtual environment and install all requirements (this will ask for your HuggingFace token):
```bash
git clone https://github.com/javiferran/sae_entities.git
cd sae_entities
source setup.sh
```

For installing [SAE-Lens](https://github.com/jbloomAus/SAELens/tree/main):
```bash
pip install sae-lens
```

## Codebase structure
The `/dataset` folder contains the necessary code to create the dataset and run the model generations. It also includes the generations at `/dataset/processed`.

The `/mech_interp` folder contains the code to perform the analysis of the SAE latents.

## Get Activations
To cache residual stream activations on entity tokens, for instance of Gemma 2 2B run:
```bash
cd sae_entities
python -m utils.activation_cache --model_alias gemma-2-2b --tokens_to_cache entity --batch_size 128 --entity_type_and_entity_name_format
```

To ensure specificity to entity tokens, we exclude latents that activate frequently (>2%) on random
tokens sampled from the Pile dataset. So, for extracting activations of random tokens of the Pile dataset, run:
```bash
cd sae_entities
python -m utils.activation_cache --model_alias gemma-2-2b --tokens_to_cache random --batch_size 128 --dataset pile
```

### (Optional) Run the model generations
Model generations can be found in `/dataset/processed`. However, in case you want to get the generations yourself, you can run the following command (this is for the wikidata dataset):
```bash
cd sae_entities
python -m dataset.process_data.wikidata.create_wikidata_entity_queries --model_path gemma-2-2b --free_generation False
```

## [New] Direct Hidden Rep Analysis
In `mech_interp/hidden_analysis.py` we run the metrics to find the most relevant hiddens.

## SAE Latent Analysis
In `mech_interp/feature_analysis.py` we compute the SAE latent scores for all layers as well as run metrics to find the most relevant latents.

## Uncertainty Latents
Generate and cache the activations for `model` token, appearing at the end of instruction tokens (only for Gemma models):
```bash
python -m utils.activation_cache --model_alias gemma-2b-it --tokens_to_cache model --batch_size 128
```

## Refusal Analysis
Models refuse to answer questions about unknown entities. We can therefore analyze the effect of steering the model towards an unknown entity latent by measuring the model's refusal rate.
In `mech_interp/feature_analysis.py` we evaluate the impact of this steering.

## Citation
If you find this work useful, please consider citing:

```bibtex
@inproceedings{
ferrando2025iknowentityknowledge,
title={Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models},
author={Javier Ferrando and Oscar Obeso and Senthooran Rajamanoharan and Neel Nanda},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=WCRQFlji2q}
}
```
