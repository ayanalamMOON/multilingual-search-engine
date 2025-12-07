#!/bin/bash
# Render build script for backend service

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "🤖 Pre-downloading sentence transformer model..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

echo "✅ Build complete!"
