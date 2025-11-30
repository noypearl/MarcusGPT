# MarcusGPT – VRChat Worm Energy Edition

A full-stack AI chatbot powered by a fine-tuned LLaMA 3.2 3B model trained on Marcus quotes from VRChat.

## 🎮 Features

- Fine-tuned LLaMA model with LoRA adapters
- FastAPI backend with GPU/MPS/CPU support
- VRChat-styled glitchy UI
- Local-first, no cloud dependencies

## 📁 Project Structure

```
MarcusGPT/
├── site/
│   ├── server.py         # FastAPI backend
│   ├── index.html        # Frontend UI
│   ├── style.css         # VRChat styling
│   └── script.js         # Frontend logic
├── marcus-llama3-3b-lora/  # LoRA adapter weights
├── dataset_train.jsonl   # Training data
├── dataset_val.jsonl     # Validation data
└── requirements.txt      # Python dependencies
```

## 🚀 Setup

### 1. Install Dependencies

```bash
# Activate your virtual environment
source marcus/bin/activate

# Install requirements
pip install -r requirements.txt

# Install FastAPI and uvicorn if not already installed
pip install fastapi uvicorn[standard]
```

### 2. Start the Backend

```bash
cd site
python3 server.py
```

You should see:
```
INFO:     Loading tokenizer...
INFO:     Detecting device...
INFO:     Using MPS (Apple Silicon GPU)
INFO:     Loading base model on mps...
INFO:     Attaching LoRA adapter...
INFO:     ✓ Model loaded successfully on mps
INFO:     Starting server on http://localhost:8000
```

### 3. Start the Frontend

In a new terminal:

```bash
cd site
python3 -m http.server 8080
```

### 4. Open in Browser

Navigate to: **http://localhost:8080**

## 🐛 Troubleshooting

### Backend won't start

- **Error: `'NoneType' object has no attribute 'cadam32bit_grad_fp32'`**
  - This is a bitsandbytes issue. The fixed server.py removes 4-bit quantization and uses MPS/CPU directly.

- **Model loading is slow**
  - First load downloads ~6GB model from HuggingFace
  - Subsequent loads are faster (cached)

### Frontend can't connect

- Check that backend is running on port 8000: `curl http://localhost:8000`
- Check browser console (F12) for CORS errors
- Ensure you're accessing frontend via `http://localhost:8080`, not `file://`

### Response is slow

- On CPU: Expect 30-60s per response
- On MPS (Apple Silicon): ~5-15s per response
- On CUDA: ~2-5s per response

## 🎨 Customization

### Change Marcus personality

Edit the system prompt in `server.py`:

```python
"content": "Your custom Marcus personality here..."
```

### Adjust response length

In `server.py`, change `MAX_NEW_TOKENS`:

```python
MAX_NEW_TOKENS = 200  # Default: 120
```

### Modify UI colors

Edit `style.css` - main colors are:
- Orange: `#ff8c3c`
- Blue: `#3b82f6`
- Dark: `#0a0404`

## 📝 API Endpoints

### `POST /chat`

**Request:**
```json
{
  "message": "Hello Marcus!"
}
```

**Response:**
```json
{
  "reply": "yo the worms are vibrating again in sector 7... u feel that?"
}
```

### `GET /`

Health check endpoint.

## ⚙️ Technical Details

- **Base Model:** meta-llama/Llama-3.2-3B-Instruct
- **Fine-tuning:** LoRA (r=16, alpha=32)
- **Training:** 153 steps on Marcus quotes dataset
- **Inference:** PyTorch with HuggingFace Transformers
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Vanilla HTML/CSS/JS

## 📄 License

For personal/educational use.

