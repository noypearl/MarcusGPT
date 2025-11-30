# train_marcus_lora.py
#
# Fine-tune LLaMA 3 3B Instruct with LoRA on your Marcus dataset.
#
# Requirements (install once):
#   pip install -U "transformers>=4.43.0" "datasets" "accelerate" "peft" "trl" "bitsandbytes"
#
# Make sure you have:
#   - dataset_train.jsonl
#   - dataset_val.jsonl
# in the same directory as this script.
#
# Each line in those files should look like:
#   {"messages": [
#       {"role": "system", "content": "You are Marcus ..."},
#       {"role": "user", "content": "some user input"},
#       {"role": "assistant", "content": "Marcus-style reply"}
#   ]}

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# 🔧 CONFIG: update MODEL_NAME to the exact LLaMA 3 3B Instruct ID you have access to on Hugging Face.
# Examples (check on HF):
#   "meta-llama/Llama-3.2-3B-Instruct"  or  "meta-llama/Llama-3.1-3B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

TRAIN_PATH = "dataset_train.jsonl"
VAL_PATH   = "dataset_val.jsonl"
OUTPUT_DIR = "marcus-llama3-3b-lora"


def format_example(example, tokenizer):
    """
    Turn a {"messages": [...]} example into a single prompt string
    using the model's chat template.
    """
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # we include assistant output as part of training
    )
    return {"text": text}


def main():
    # 1. Load datasets
    train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
    val_ds   = load_dataset("json", data_files=VAL_PATH, split="train")

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    # 2. Load tokenizer
    # Make sure you've run `huggingface-cli login` so this can download the model.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Some LLaMA models don't define a pad token; using eos as pad is common practice.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Format datasets with chat template
    train_ds = train_ds.map(lambda ex: format_example(ex, tokenizer))
    val_ds   = val_ds.map(lambda ex: format_example(ex, tokenizer))

    print("\nFormatted example text (first 400 chars of first sample):")
    print(train_ds[0]["text"][:400])

    # 4. Quantization config for QLoRA (4-bit base model)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",  # if your GPU doesn't support bfloat16, you can switch to float16
    )

    # 5. Load base model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",  # spread layers across available devices (usually your GPU)
    )

    # 6. LoRA config: where and how big the adapters are
    lora_config = LoraConfig(
        r=16,                  # rank (adapter size) - smaller = lighter, bigger = more capacity
        lora_alpha=32,         # scaling factor
        lora_dropout=0.05,     # small dropout for regularization
        bias="none",
        task_type="CAUSAL_LM", # language modeling
        target_modules=[       # which submodules get LoRA adapters
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    # 7. Training hyperparameters
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch = 2 * 4 = 8
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        # If your GPU doesn't support bf16, set bf16=False and fp16=True instead:
        bf16=True,
        fp16=False,
        report_to="none",
    )

    # 8. SFTTrainer: glue data, model, tokenizer, and LoRA together
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        tokenizer=tokenizer,
        dataset_text_field="text",   # the field we're training on after formatting
        max_seq_length=512,          # truncate examples longer than this
    )

    # 9. Run training 🚀
    trainer.train()

    # 10. Save LoRA adapter + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nTraining complete. LoRA adapter + tokenizer saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
