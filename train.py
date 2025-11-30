from huggingface_hub import login
from google.colab import userdata
from datasets import load_dataset
from transformers import (
   AutoTokenizer,
   AutoModelForCausalLM,
   BitsAndBytesConfig,
   TrainingArguments,
   Trainer,
   default_data_collator,
)
from peft import LoraConfig, get_peft_model

HF_TOKEN = userdata.get('HF_READ') # Get the Hugging Face token from secrets
if HF_TOKEN:
   login(HF_TOKEN)
   print("Successfully logged in to Hugging Face!")
else:
   print("Token is not set. Please save the token in Colab secrets.")

# ======================
# CONFIG
# ======================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"   # you already requested access to this
TRAIN_PATH = "dataset_train.jsonl"
VAL_PATH   = "dataset_val.jsonl"
OUTPUT_DIR = "marcus-llama3-3b-lora-colab"

MAX_SEQ_LEN = 512  # you can lower to 256 if you hit VRAM issues

def format_example(example, tokenizer):
   """
   Turn {"messages": [...]} into a single chat-formatted text string
   using the model's chat template.
   """
   messages = example["messages"]
   text = tokenizer.apply_chat_template(
       messages,
       tokenize=False,
       add_generation_prompt=False,  # include assistant turn as target
   )
   return {"text": text}

# ======================
# 1. Load datasets
# ======================

train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
val_ds   = load_dataset("json", data_files=VAL_PATH, split="train")

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))

# ======================
# 2. Tokenizer
# ======================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
   tokenizer.pad_token = tokenizer.eos_token

# ======================
# 3. Apply chat template → "text"
# ======================

train_ds = train_ds.map(lambda ex: format_example(ex, tokenizer))
val_ds   = val_ds.map(lambda ex: format_example(ex, tokenizer))

print("\nFormatted example text (first 400 chars of first train sample):")
print(train_ds[0]["text"][:400])

# ======================
# 4. Tokenize "text" → input_ids / attention_mask / labels
# ======================

def tokenize_function(batch):
   out = tokenizer(
       batch["text"],
       truncation=True,
       max_length=MAX_SEQ_LEN,
       padding="max_length",
   )
   # For causal LM: labels = input_ids (model learns to predict next token)
   out["labels"] = out["input_ids"].copy()
   return out

train_tok = train_ds.map(
   tokenize_function,
   batched=True,
   remove_columns=train_ds.column_names,  # keep only tokenized fields
)

val_tok = val_ds.map(
   tokenize_function,
   batched=True,
   remove_columns=val_ds.column_names,
)

print("\nTokenized keys:", train_tok.column_names)
print("Example tokenized input length:", len(train_tok[0]["input_ids"]))

# ======================
# 5. QLoRA 4-bit config (GPU)
# ======================

bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype="float16",  # safer for Colab T4s
)

# ======================
# 6. Load base model in 4-bit on GPU
# ======================

model = AutoModelForCausalLM.from_pretrained(
   MODEL_NAME,
   quantization_config=bnb_config,
   device_map="auto",
)

# ======================
# 7. Attach LoRA adapters
# ======================

lora_config = LoraConfig(
   r=16,
   lora_alpha=32,
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

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # just to see trainable param count

# ======================
# 8. Training arguments
# ======================

training_args = TrainingArguments(
   output_dir=OUTPUT_DIR,
   per_device_train_batch_size=2,
   per_device_eval_batch_size=2,
   gradient_accumulation_steps=4,   # effective batch = 8
   learning_rate=2e-4,
   num_train_epochs=3,
   logging_steps=20,
   save_steps=200,
   save_total_limit=2,
   fp16=True,       # use fp16 on most Colab GPUs
   bf16=False,
   report_to="none",
)

# ======================
# 9. Trainer
# ======================

trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=train_tok,
   eval_dataset=val_tok,          # not doing mid-training eval; used if you call trainer.evaluate()
   tokenizer=tokenizer,
   data_collator=default_data_collator,
)

trainer.train()

# ======================
# 10. Save LoRA adapter + tokenizer
# ======================

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nTraining complete. LoRA adapter + tokenizer saved to: {OUTPUT_DIR}")
