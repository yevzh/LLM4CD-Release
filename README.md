# LLM4CD: Leveraging Large Language Models for Open-World Knowledge Augmented Cognitive Diagnosis
## Introduction

This repository contains the PyTorch implementation of the paper LLM4CD: Leveraging Large Language Models for Open-World Knowledge Augmented Cognitive Diagnosis.

## Requirements

- numpy
- transformers
- torch
- dgl
- pickle

## Quick Start

To get started, run the following command in your terminal:

```bash
python main.py --dataset ass09 --llm chatglm2 --num_epochs 100 --lr 1e-4 --model lamdacd --batch_size 4000 --encoder graph
```

## Dataset

The public datasets we used are ASSIST09, ASSIST12 and Junyi. For data processing, please refer to `data/ass09/log_data.json`.

To divide data into train_set, val_set and test_set, run the following command:

```bash
python divide_data.py
```

To get the graph information, please refer to `data/ass09/graph`

```bash
python construct_concept_map.py
python process_edge.py
```

## Text generation

The exercise information and concept information are available at `text/ass09`, please refer to `id_skill_desc_dict.json`, `id_skillname_dict.json` and the augmented `question.json`.

To construct cognitive text for students, please refer to `text/ass09/prompt_generator.py`:

```bash
python prompt_generator.py
```

After obtain the text from exercise and student sides, encode the generated text into semantic embeddings:

```bash
python encoder.py --model_name chatglm2
```

