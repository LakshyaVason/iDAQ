# iDAQ Diagnostics - Setup & Troubleshooting Guide
## Complete Setup Instructions

### Step 1: Install Ollama

**For Windows:**
Download from https://ollama.com and run the installer.

### Step 2: Start Ollama Service

```bash
# Start the Ollama server (keep this running in a terminal)
ollama serve
```

**You should see:** `Ollama is running on http://127.0.0.1:11434`

### Step 3: Pull the Required Model

```bash
# In a NEW terminal (while ollama serve is running)
ollama pull llama3.2:1b
```

This downloads the 1B parameter Llama model (~1GB).

**Alternative models if you have more resources:**
```bash
ollama pull llama3.2:3b  # 3B parameters, better quality
ollama pull mistral      # Mistral 7B, best quality
```

If you use a different model, update `MODEL_NAME` in `local_llama_agent.py`:
```python
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2:1b")  # Change this
```

### Step 4: Verify Ollama is Working

```bash
# Test the model
ollama run llama3.2:1b "Hello, how are you?"

# Or test via API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:1b",
  "prompt": "Hello!",
  "stream": false
}'
```

### Step 5: Install Python Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 6: Replace Files with Fixed Versions

1. Replace `main.py` with the fixed version (includes CORS and error handling)
2. Replace `templates/user.html` with the fixed version (includes complete JavaScript)

### Step 7: Start the Web Application

```bash
# Make sure Ollama is STILL running in another terminal!
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 8: Access the Application

Open your browser to: **http://localhost:8000**

---

## Testing the Setup

### Test 1: Check Ollama Connection
```bash
# Should return Ollama is running
curl http://localhost:11434
```

### Test 2: Check FastAPI Server
```bash
# Should return sensor data
curl http://localhost:8000/sensor-data
```

### Test 3: Test Chat Endpoint
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

**Expected response:**
```json
{"response": "Hello! How can I help you today?"}
```

---

## Common Issues & Solutions

### Issue 1: "Connection refused" on port 11434
**Cause:** Ollama is not running.
**Solution:**
```bash
# Start Ollama in a separate terminal
ollama serve
```

### Issue 2: "Model not found" error
**Cause:** Model hasn't been downloaded.
**Solution:**
```bash
ollama pull llama3.2:1b
```

### Issue 3: Chat button does nothing
**Cause:** Old user.html without JavaScript.
**Solution:** Replace `templates/user.html` with the fixed version.

### Issue 4: CORS errors in browser console
**Cause:** Missing CORS configuration.
**Solution:** Replace `main.py` with the fixed version.

### Issue 5: "Failed to establish connection" error
**Cause:** Firewall or wrong port.
**Solution:**
```bash
# Check if Ollama is listening
netstat -an | grep 11434  # Linux/Mac
netstat -an | findstr 11434  # Windows

# Try accessing directly
curl http://127.0.0.1:11434
```

### Issue 6: Slow responses
**Cause:** Large model or CPU-only inference.
**Solution:**
- Use smaller model: `ollama pull llama3.2:1b`
- Check GPU usage: `nvidia-smi` (if you have GPU)
- Reduce max_tokens in `local_llama_agent.py`

### Issue 7: RAG says "No vector store found"
**Cause:** No PDF has been uploaded yet.
**Solution:**
1. Login as admin (username: `pqlab`, password: `PQ2025!`)
2. Go to Admin Dashboard
3. Upload a PDF datasheet
4. Wait for ingestion to complete

---

## Architecture Overview

```
┌─────────────┐          ┌──────────────┐
│   Browser   │◄────────►│  FastAPI     │
│  (user.html)│          │  (main.py)   │
└─────────────┘          └──────┬───────┘
                                │
                                ├──► local_llama_agent.py
                                │        │
                         ┌──────┴───────┼────────────┐
                         │              │            │
                         ▼              ▼            ▼
                  ┌─────────┐    ┌──────────┐  ┌────────┐
                  │ Ollama  │    │ sklearn  │  │ FAISS  │
                  │(LLM API)│    │(ML Model)│  │(Vector │
                  │Port 11434│   └──────────┘  │ Store) │
                  └─────────┘                   └────────┘
```

---

## Running Everything Together

**Terminal 1 - Ollama:**
```bash
ollama serve
```

**Terminal 2 - FastAPI:**
```bash
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Browser:**
```
http://localhost:8000
```

---

## Verification Checklist

✅ **Ollama installed:** `ollama --version`  
✅ **Ollama running:** `curl http://localhost:11434`  
✅ **Model downloaded:** `ollama list` shows `llama3.2:1b`  
✅ **Python deps installed:** `pip list | grep fastapi`  
✅ **FastAPI running:** `curl http://localhost:8000/sensor-data`  
✅ **Chat works:** Test in browser at `http://localhost:8000/user`  

---

## Next Steps

1. **Upload training data** (Admin Dashboard):
   - normal.csv for anomaly detection
   - fault.csv for fault classification
   - PDF datasheets for RAG

2. **Train models** (Admin Dashboard):
   - Click "Train Classifier and Baseline"

3. **Test diagnostics** (User Dashboard):
   - Ask: "What are the current sensor readings?"
   - Ask: "Diagnose high temperature issue"
   - Use RAG chat after uploading PDFs

---

## Environment Variables (Optional)

Create a `.env` file:
```bash
OLLAMA_HOST=http://127.0.0.1:11434
MODEL_NAME=llama3.2:1b
```

---

## Performance Tips

1. **Use GPU if available:** Ollama automatically uses GPU
2. **Reduce model size:** Use `llama3.2:1b` instead of larger models
3. **Limit context:** Keep chat messages concise
4. **Batch operations:** Train models once, not repeatedly

---

## Security Notes

⚠️ **This is a development setup. For production:**
- Change default admin password
- Use proper authentication (OAuth, JWT)
- Enable HTTPS
- Restrict CORS origins
- Add rate limiting
- Secure API endpoints

---

## Support

If you still have issues after following this guide:

1. Check the terminal output for error messages
2. Look at browser console (F12) for JavaScript errors
3. Verify all services are running: `ps aux | grep ollama`
4. Test each component individually using the curl commands above