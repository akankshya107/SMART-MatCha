# SMART-MatCha

## SMART Dataset
Download the dataset using:

```bash
wget https://zenodo.org/records/7775984/files/SMART101-release-v1.zip
unzip SMART101-release-v1.zip
```

The data_aug.zip is also unzipped at the same folder. This contains the folders with augmented puzzles.

## Zero-Shot Model Baselines

The Model_baselines.ipynb file can be run to evaluate each of the models on the test data. These are the zero-shot generalization benchmarks.

## MatCha finetuning

The flags ty ["counting", "math", "logic", "path", "algebra", "spatial", "pattern", "measure", "order", "all"] and aug ["aug", "noaug"] control the puzzle types and augmentation used while training.

```
pip install transformers
pip install peft
pip install bitsandbytes==0.41.3 accelerate==0.25.0
python3 MatCha_baselines.py
```

The test code is at MatCha_test.py evaluates all provided checkpoints.

## Data Augmentation
Generate augmented puzzles using puzzles_10.py, puzzles_12.py, and puzzles_28.py
