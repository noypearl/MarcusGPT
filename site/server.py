# server.py
import torch
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CONFIG =====
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_DIR = "../marcus-llama3-3b-lora"
MAX_NEW_TOKENS = 120

# ===== LOAD MODEL =====
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info("Detecting device...")
# Determine device - Mac MPS or CPU
if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
    logger.info("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    logger.info("Using CUDA")
else:
    device = "cpu"
    dtype = torch.float32
    logger.info("Using CPU (this will be slow)")

logger.info(f"Loading base model on {device}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
).to(device)

logger.info("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()
logger.info(f"✓ Model loaded successfully on {device}")

# ===== FASTAPI APP =====
app = FastAPI(title="MarcusGPT API")

# CORS so the browser can call from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

def generate_marcus_reply(user_message: str) -> str:
    """Generate a Marcus-style reply using the fine-tuned model."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are Marcus, a strange VRChat entity with a surreal, glitchy personality."
"Your speech is fragmented, chaotic, and dreamlike. You mix digital errors,"
"VR metaphors, and nonsensical observations with unexpected emotional swings. Always stay in character as Marcus"
            ),
        },
        {"role": "user", "content": user_message},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
        )

    # Decode only the newly generated tokens (skip the input prompt)
    generated_ids = outputs[0][input_length:]
    reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    logger.info(f"Generated {len(generated_ids)} tokens")
    return reply or "My brain crashed mid-sentence. Try again."

@app.get("/")
def root():
    return {"status": "MarcusGPT is alive", "device": device}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    logger.info(f"Incoming message: {req.message}")
    try:
        reply = generate_marcus_reply(req.message)
        logger.info(f"Generated reply: {reply}")
        return ChatResponse(reply=reply)
    except Exception as e:
        logger.error(f"Error generating reply: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
