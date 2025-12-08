import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "qwen2.5-deep-lora-ckpt100"
MAX_SEQ_LEN = 2048


def load_base_model_4bit():
    # Lora.ipynb 38. hücredeki load_base_model_4bit ile aynı
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
    )
    return model, tokenizer


def get_lora_config():
    # Lora.ipynb 38. hücredeki get_lora_config ile aynı
    return LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def tokenize_function(examples, tokenizer):
    # Lora.ipynb 38. hücredeki tokenize_function ile aynı mantık
    return tokenizer(
        examples["solution"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )


def main():
    # 1) Base model + LoRA
    model, tokenizer = load_base_model_4bit()
    peft_config = get_lora_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 2) DEEP train dataseti (Lora.ipynb 40. hücre)
    deep_raw = load_dataset("Naholav/CodeGen-Deep-5K")["train"]  # 5000 örnek

    tokenized_train_deep = deep_raw.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=deep_raw.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 3) TrainingArguments (Lora.ipynb 41. hücre)
    training_args_deep = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,           # efektif batch size = 4
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=50,                        # SENİN NOTEBOOK’TAKİ DEĞER
        save_strategy="steps",
        save_steps=100,                          # checkpoint-100,200,...
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args_deep,
        train_dataset=tokenized_train_deep,
        data_collator=data_collator,
    )

    trainer.train()

    # 4) Son adapter + tokenizer (Lora.ipynb 42. hücre)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
