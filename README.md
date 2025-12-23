# LoRA Deep vs Diverse Fine-Tuning on Qwen2.5-Coder

This repository presents a comparative study of LoRA-based fine-tuning on the  
**Qwen/Qwen2.5-Coder-1.5B-Instruct** model using two different instruction-style datasets:
**Deep Instruction** and **Diverse Instruction**.

The objective is to analyze how dataset characteristics affect code generation
performance and to identify the best-performing checkpoint using a standardized benchmark.

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
hyperparameters** and evaluated under exactly the same benchmark conditions to
ensure a fair comparison.

---

## Motivation

Instruction-tuned code models can be sensitive to the **structure** and
**diversity** of training data. This project investigates:

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

