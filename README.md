LoRA Deep vs Diverse Fine-Tuning on Qwen2.5-Coder

This repository contains a comparative study of LoRA-based fine-tuning on the Qwen/Qwen2.5-Coder-1.5B-Instruct model using two different instruction-style datasets: Deep Instruction and Diverse Instruction.
The goal is to analyze how dataset characteristics affect code generation performance and to identify the best-performing checkpoint using a standardized benchmark.


Project Overview

Base Model: Qwen/Qwen2.5-Coder-1.5B-Instruct

Fine-tuning Method: LoRA (Low-Rank Adaptation)

Training Variants:

deep_instruction

diverse_instruction

Evaluation Benchmark: LiveCodeBench

Evaluation Subset: AtCoder – Easy (41 problems)

Evaluation Metric: Pass@1

Both models are fine-tuned from the same base model, using identical training hyperparameters, and evaluated under exactly the same benchmark conditions to ensure a fair comparison.
