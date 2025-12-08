import os
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_SEQ_LEN = 2048


def load_base_model_4bit():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
    )
    return model, tokenizer


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["solution"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )


def main():
    # Lora.ipynb 47. hücre
    deep_ckpts = sorted(
        [d for d in os.listdir("qwen2.5-deep-lora-ckpt100") if d.startswith("checkpoint")]
    )
    diverse_ckpts = sorted(
        [d for d in os.listdir("qwen2.5-diverse-lora-ckpt100") if d.startswith("checkpoint")]
    )

    print("Deep checkpoints:", deep_ckpts)
    print("Diverse checkpoints:", diverse_ckpts)

    # Tokenizer + collator
    _, tokenizer = load_base_model_4bit()
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # DEEP test set (Lora.ipynb 50. hücre)
    deep_full = load_dataset("Naholav/CodeGen-Deep-5K")["train"]
    deep_test = deep_full.select(range(0, 100))

    tokenized_test_deep = deep_test.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=deep_test.column_names,
    )

    # DIVERSE test set (Lora.ipynb 54. hücre)
    diverse_full = load_dataset("Naholav/CodeGen-Diverse-5K")["train"]
    diverse_test = diverse_full.select(range(0, 100))

    tokenized_test_diverse = diverse_test.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=diverse_test.column_names,
    )

    # DEEP evaluation (Lora.ipynb 52 + 53)
    results_deep = {}

    for ckpt in deep_ckpts:
        print(f"\n[DEEP] Değerlendiriliyor: {ckpt}")

        base_model, _ = load_base_model_4bit()

        model = PeftModel.from_pretrained(
            base_model,
            f"qwen2.5-deep-lora-ckpt100/{ckpt}",
        )

        eval_args = TrainingArguments(
            output_dir=f"eval-deep-{ckpt}",
            per_device_eval_batch_size=1,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=tokenized_test_deep,
            data_collator=data_collator,
        )

        eval_res = trainer.evaluate()
        results_deep[ckpt] = float(eval_res["eval_loss"])

    best_deep = min(results_deep, key=results_deep.get)
    print("Deep için en iyi checkpoint:", best_deep, "loss:", results_deep[best_deep])

    # DIVERSE evaluation (Lora.ipynb 56 + 57)
    results_diverse = {}

    for ckpt in diverse_ckpts:
        print(f"\n[DIVERSE] Değerlendiriliyor: {ckpt}")

        base_model, _ = load_base_model_4bit()

        model = PeftModel.from_pretrained(
            base_model,
            f"qwen2.5-diverse-lora-ckpt100/{ckpt}",
        )

        eval_args = TrainingArguments(
            output_dir=f"eval-diverse-{ckpt}",
            per_device_eval_batch_size=1,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=tokenized_test_diverse,
            data_collator=data_collator,
        )

        eval_res = trainer.evaluate()
        results_diverse[ckpt] = float(eval_res["eval_loss"])

    best_diverse = min(results_diverse, key=results_diverse.get)
    print("Diverse için en iyi checkpoint:", best_diverse, "loss:", results_diverse[best_diverse])

    # Özet JSON (Lora.ipynb 58. hücre)
    summary = {
        "deep": {
            "all_losses": results_deep,
            "best_checkpoint": best_deep,
            "best_loss": results_deep[best_deep],
        },
        "diverse": {
            "all_losses": results_diverse,
            "best_checkpoint": best_diverse,
            "best_loss": results_diverse[best_diverse],
        },
    }

    with open("checkpoint_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
