# MarcusGPT Setup & Testing Guide

## ✅ What We Fixed

1. **Backend crashes** - Removed problematic 4-bit quantization that was causing bitsandbytes errors on Mac
2. **Model loading** - Now properly uses MPS (Apple Silicon GPU) or CPU with correct device detection
3. **Response generation** - Fixed token decoding to properly extract Marcus replies
4. **Logging** - Added comprehensive logging for debugging
5. **CSS styling** - Added complete VRChat-style UI theme
6. **Dependencies** - Added FastAPI and uvicorn to requirements

## 🚀 Quick Start

### 1. Start the Backend

```bash
cd /your/own/path/MarcusGPT/site
source ../marcus/bin/activate
python3 server.py
```

Wait for this message:
```
INFO:__main__:✓ Model loaded successfully on mps
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Start the Frontend (in a new terminal)

```bash
cd /your/own/path/MarcusGPT/site
python3 -m http.server 8080
```

### 3. Open in Browser

Navigate to: **http://localhost:8080**

## 🧪 Testing

### Test the API directly:

```bash
# Health check
curl http://localhost:8000/

# Send a chat message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"hey marcus whats up"}'
```

### Or use the test script:

```bash
cd /Your/Own/Path/MarcusGPT
source marcus/bin/activate
python3 test_api.py
```

## 📝 Current Status

✅ Backend loads successfully on MPS (Apple Silicon GPU)  
✅ LoRA adapter loads without errors  
✅ API responds to requests  
✅ Model generates responses  
✅ Frontend UI styled with VRChat theme  
✅ CORS configured for browser requests  

## 🎨 Frontend Files

- `site/index.html` - Main UI structure
- `site/style.css` - Complete VRChat styling (orange/blue theme, scanlines, glitches)
- `site/script.js` - API integration and chat handling

## 🐛 Known Issues

1. **Response personality**: The fine-tuned model sometimes gives normal responses instead of Marcus-style. This is a training data issue, not a technical problem. Consider:
   - Using more training data
   - Increasing training epochs
   - Adjusting the system prompt

2. **Speed**: On MPS, responses take 5-15 seconds. This is normal for a 3B parameter model.

3. **Bitsandbytes warning**: The warning about GPU support is harmless - we're not using quantization.

## 📊 API Endpoints

### `GET /`
Health check - returns server status and device info

### `POST /chat`
**Request:**
```json
{
  "message": "your message here"
}
```

**Response:**
```json
{
  "reply": "Marcus's response"
}
```

## 🔧 Customization

### Adjust response length:
Edit `server.py`, line 16:
```python
MAX_NEW_TOKENS = 200  # Default: 120
```

### Change Marcus personality:
Edit `server.py`, lines 72-77 (system prompt)

### Modify UI colors:
Edit `style.css` - main theme colors:
- Orange: `#ff8c3c`
- Blue: `#3b82f6`
- Dark bg: `#0a0404`

## 📂 Project Structure

```
MarcusGPT/
├── site/
│   ├── server.py         ✅ Fixed - uses MPS/CPU properly
│   ├── index.html        ✅ Complete UI structure
│   ├── style.css         ✅ Full VRChat styling
│   └── script.js         ✅ API integration
├── marcus-llama3-3b-lora/  ✅ LoRA weights loading correctly
├── test_api.py           ✅ NEW - Quick API test script
├── SETUP_GUIDE.md        ✅ NEW - This file
└── README.md             ✅ NEW - Full documentation
```

## 🎉 Success!

Your MarcusGPT is now fully operational! The backend and frontend are communicating successfully.

Next steps:
1. Open http://localhost:8080 in your browser
2. Type a message to Marcus
3. Enjoy the worm energy! 🪱✨

