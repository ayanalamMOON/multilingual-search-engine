# HuggingFace RAG Setup Guide

## Quick Setup

### 1. Get Your Free HuggingFace API Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "multilingual-search-rag")
4. Select "Read" permission
5. Click "Generate token"
6. Copy the token

### 2. Add Token to Your .env File

Open or create `.env` file in your project root and add:

```
HUGGINGFACEHUB_API_TOKEN=hf_YourTokenHere
```

### 3. Test RAG

```bash
python example_rag.py
```

## What Changed

✅ **Removed conflicting packages**: Removed langchain-openai and OpenAI dependency
✅ **Kept LangChain core**: Using langchain==0.2.6 with langchain-huggingface==0.0.3
✅ **Custom HuggingFace integration**: Created HuggingFaceLLM wrapper that works with LangChain
✅ **Free model**: Using Llama-3.2-1B-Instruct via HuggingFace Router API
✅ **New endpoint**: Updated to https://router.huggingface.co/v1/chat/completions

## How It Works

1. **Vector Search**: FAISS finds relevant poems/lyrics
2. **LangChain**: Structures the prompts and manages retrieval
3. **HuggingFace Router API**: Generates AI responses using Llama-3.2-1B model (free)
4. **No OpenAI required**: Completely free with HuggingFace token

## API Endpoints

### Search (works without token)
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "love songs", "top_k": 5}'
```

### RAG Summary (requires token)
```bash
curl -X POST http://localhost:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "love songs", "mode": "summary", "top_k": 3}'
```

### RAG Recommendation (requires token)
```bash
curl -X POST http://localhost:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "romantic poetry", "mode": "recommendation", "top_k": 3}'
```

### RAG Chat (requires token)
```bash
curl -X POST http://localhost:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "love poems",
    "mode": "chat",
    "top_k": 3,
    "user_message": "What themes do these poems explore?"
  }'
```

## Note on Model Performance

✅ **Works out of the box**: The new HuggingFace Router API endpoint works immediately
🚀 **Fast responses**: Llama-3.2-1B is optimized for quick inference
💡 **Customizable**: You can change the model in `rag.py` to any supported HuggingFace model

## Troubleshooting

### "Model is loading"
- Wait 30-60 seconds for the model to warm up
- Try the request again

### "Rate limit exceeded"
- Free tier has limits on requests per hour
- Wait a few minutes and try again
- Consider upgrading HuggingFace account for higher limits

### RAG not available
- Check .env file has HUGGINGFACEHUB_API_TOKEN
- Restart the FastAPI server after adding token
- Verify token is valid at https://huggingface.co/settings/tokens
