# LoRA Deep vs Diverse Fine-Tuning on Qwen2.5-Coder

This repository presents a comparative study of **LoRA-based fine-tuning** on the  
**Qwen/Qwen2.5-Coder-1.5B-Instruct** model using two different instruction-style datasets:
**Deep Instruction** and **Diverse Instruction**.

The objective is to analyze how dataset characteristics affect code generation performance
and to identify the best-performing checkpoint using a standardized benchmark.

---

## Project Overview

- **Base Model:** Qwen/Qwen2.5-Coder-1.5B-Instruct  
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)  
- **Training Variants:**
  - `deep_instruction`
  - `diverse_instruction`
- **Evaluation Benchmark:** LiveCodeBench  
- **Evaluation Subset:** AtCoder – Easy (41 problems)  
- **Evaluation Metric:** Pass@1  

Both models are fine-tuned from the same base model using **identical training
hyperparameters** and evaluated under exactly the same benchmark conditions to ensure
a fair comparison.

---

## Motivation

Instruction-tuned code models can be sensitive to the **structure** and **diversity**
of training data. This project investigates:

- Whether deep, step-by-step instruction-style data improves reasoning accuracy
- Whether diverse instruction formulations improve generalization
- Which training strategy yields better real-world coding performance

---

## Repository Structure

```text
CodeGen/
├── models/
│   ├── deep_instruction/
│   │   └── checkpoints/
│   │       ├── checkpoint-step-100-epoch-1/
│   │       ├── checkpoint-step-200-epoch-1/
│   │       └── ...
│   ├── diverse_instruction/
│   │   └── checkpoints/
│   │       ├── checkpoint-step-100-epoch-1/
│   │       ├── checkpoint-step-200-epoch-1/
│   │       └── ...
├── livecodebench_eval.py        # LiveCodeBench evaluation script
├── run_all_evaluations.py
├── results/
│   └── livecodebench/
│       ├── detailed/
│       ├── evaluations/
│       ├── generations/
│       └── summary.json
├── requirements.txt
├── README.md
└── REPORT.md

## Methodology

### Base Model

- **Model:** Qwen/Qwen2.5-Coder-1.5B-Instruct  
- **Architecture:** Transformer-based causal language model  
- **Pre-training:** Large-scale code and instruction datasets  

The base model remains **frozen** during fine-tuning.

---

## Fine-Tuning Strategy

LoRA adapters are applied to the attention layers of the base model.  
All hyperparameters—including **learning rate, batch size, number of epochs,
and LoRA rank**—are kept **identical across both variants** to ensure
experimental fairness.

---

## Dataset Splits

Each dataset is split into:

- **Train**
- **Validation**
- **Test**

Validation loss is monitored during training; however, **checkpoint selection
is based on benchmark performance**, not training loss alone.

---

## Evaluation Setup

- **Benchmark:** LiveCodeBench  
- **Platform:** AtCoder  
- **Difficulty:** Easy  
- **Number of Problems:** 41  

This subset is selected to match the evaluation guidelines provided in the
project specification.

---

## Metric

### Pass@1

The percentage of problems for which the model generates a **fully correct
solution on the first attempt**.

This metric reflects real-world usage, where only a single generated solution
is typically evaluated.

---

## Checkpoint Evaluation

Multiple checkpoints are produced during training.  
Each checkpoint is evaluated **independently** on the same benchmark set.  
The best checkpoint is selected as the one with the **highest Pass@1 score**.

---

## Results

The models were evaluated on **41 AtCoder Easy problems** using the Pass@1 metric.

| Model               | Best Checkpoint              | Pass@1 (%) | Solved |
|---------------------|------------------------------|------------|--------|
| deep_instruction    | checkpoint-step-800-epoch-3  | 36.6       | 15 / 41 |
| diverse_instruction | checkpoint-step-852-epoch-3  | 29.3       | 12 / 41 |

Overall, the **deep_instruction** model consistently outperformed the
**diverse_instruction** model under identical evaluation settings.
