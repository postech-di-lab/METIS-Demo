# METIS Demo
This repository contains a demo for multimodal recommendation, which is a part of [METIS](https://github.com/postech-di-lab/METIS).

## Environements
### Pip
`pip install -r requirements.txt`
### Conda
`conda env create -f environment.yml`

## Usage
> Note: These instructions are not fully documented yet.
### Preprocess
> Note: Preprocess requires a few csv files.\
> The specification is not fully finalized.

First, run `preprocessing.py`:
```sh
python preprocessing.py  input.csv output_dir
```
Next, run `preprocessed_dataset.py`:
```sh
python preprocessed_dataset --split_csv_file input.csv output_dir
```

Instead, you can do the things at once:
```sh
python preprocessed_dataset output_dir
```

### Train & Evaluation
```sh
python train.py --dataset output_dir model_save_dir
```
### Inference
```sh
python inference.py --dataset output_dir path_to_model
```
### Visualization
> 🚧 Working in process.