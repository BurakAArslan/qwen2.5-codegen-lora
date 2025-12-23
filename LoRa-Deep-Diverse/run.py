import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

def resolve_adapter_path(which: str) -> str:
    """
    which:
      - "deep" / "diverse"  -> models/deep_lora veya models/diverse_lora
      - doğrudan bir yol     -> models/deep_lora/checkpoint-200 gibi
    """
    if which.lower() == "deep":
        return os.path.join("models", "deep_lora")
    if which.lower() == "diverse":
        return os.path.join("models", "diverse_lora")
    return which  # direkt path

def build_prompt(problem_text: str) -> str:
    """
    Senin eğitim yaklaşımın (solution-only) ile uyumlu: sadece Python kodu istiyoruz.
    """
    return (
        "You are a coding assistant.\n"
        "Solve the programming problem.\n"
        "Return ONLY the final Python solution code.\n"
        "No explanation. No markdown.\n\n"
        f"{problem_text.strip()}\n"
    )

def load_model_and_tokenizer(adapter_path: str, dtype: str = "auto"):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dtype seçimi
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Base model (stabil yükleme: device_map=None + low_cpu_mem_usage=False)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return tokenizer, model, device

@torch.inference_mode()
def generate_code(tokenizer, model, device, prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    full = tokenizer.decode(out[0], skip_special_tokens=True)

    # prompt'u da decode ettiği için prompt sonrası kısmı al
    if full.startswith(prompt):
        full = full[len(prompt):]
    return full.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter",
        default="diverse",
        help='deep | diverse | veya adapter path (örn: models\\deep_lora\\checkpoint-200)'
    )
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)  # 0.0 = deterministik
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    args = ap.parse_args()

    adapter_path = resolve_adapter_path(args.adapter)

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path bulunamadı: {adapter_path}")

    # Basit doğrulama: adapter dosyaları var mı?
    cfg = os.path.join(adapter_path, "adapter_config.json")
    wts1 = os.path.join(adapter_path, "adapter_model.safetensors")
    wts2 = os.path.join(adapter_path, "adapter_model.bin")

    if not os.path.exists(cfg):
        raise FileNotFoundError(f"adapter_config.json yok: {cfg}")
    if not (os.path.exists(wts1) or os.path.exists(wts2)):
        raise FileNotFoundError(f"adapter_model.safetensors veya adapter_model.bin yok: {adapter_path}")

    print("==============================================")
    print(f"Base Model : {BASE_MODEL}")
    print(f"Adapter    : {adapter_path}")
    print("Loading model...")
    print("==============================================")

    tokenizer, model, device = load_model_and_tokenizer(adapter_path, dtype=args.dtype)

    print(f"✅ Hazır ({device}).")
    print("Çıkmak için: exit")
    print("Not: Çok satırlı soru girmek istersen bitince boş satır bırak.\n")

    while True:
        print("Problem / Soru:")
        lines = []
        while True:
            line = input()
            if line.strip().lower() == "exit":
                return
            if line.strip() == "":
                break
            lines.append(line)

        problem = "\n".join(lines).strip()
        if not problem:
            continue

        prompt = build_prompt(problem)
        answer = generate_code(
            tokenizer, model, device,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )

        print("\n--- MODEL OUTPUT (Python code) ---")
        print(answer)
        print("---------------------------------\n")

if __name__ == "__main__":
    main()
