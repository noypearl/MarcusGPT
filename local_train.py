# train_marcus_lora_local.py
#
# Sanity run on Mac (CPU or MPS), no bitsandbytes, small subset of data.
# This is for debugging the flow, not for the final quality model.

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import torch

# 🔧 CONFIG: adjust to the exact LLaMA 3 3B Instruct model you have on HF
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

TRAIN_PATH = "dataset_train.jsonl"
VAL_PATH   = "dataset_val.jsonl"
OUTPUT_DIR = "marcus-llama3-3b-lora-local"

# How many samples to use for the sanity test (keep small!)
MAX_TRAIN_SAMPLES = 200
MAX_VAL_SAMPLES   = 50


def format_example(example, tokenizer):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    # Pick device: MPS if available, else CPU
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # 1. Load datasets
    train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
    val_ds   = load_dataset("json", data_files=VAL_PATH, split="train")

    print("Full train samples:", len(train_ds))
    print("Full val samples:", len(val_ds))

    # Subsample for sanity run
    if len(train_ds) > MAX_TRAIN_SAMPLES:
        train_ds = train_ds.select(range(MAX_TRAIN_SAMPLES))
    if len(val_ds) > MAX_VAL_SAMPLES:
        val_ds = val_ds.select(range(MAX_VAL_SAMPLES))

    print("Sanity train samples:", len(train_ds))
    print("Sanity val samples:", len(val_ds))

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Format datasets
    train_ds = train_ds.map(lambda ex: format_example(ex, tokenizer))
    val_ds   = val_ds.map(lambda ex: format_example(ex, tokenizer))

    print("\nFormatted example text (first 400 chars of first sample):")
    print(train_ds[0]["text"][:400])

    # 4. Load base model (full precision / fp16 on MPS)
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    )

    if device == "mps":
        model.to("mps")
    else:
        model.to("cpu")

    # 5. LoRA config
    lora_config = LoraConfig(
        r=8,                   # smaller rank for sanity run
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    # 6. Training arguments (1 epoch, small batch)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=20,
        save_steps=1000,  # probably won't reach this in small run
        save_total_limit=1,
        bf16=(device == "mps"),  # use bf16 on MPS if supported
        fp16=False,
        report_to="none",
    )

    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=256,  # keep small for sanity
    )

    # 8. Train (sanity)
    trainer.train()

    # 9. Save adapter + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nSanity training complete. Local LoRA adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
