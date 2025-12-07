# Multilingual Song & Poem Discovery Engine

A modern semantic search engine for discovering songs and poems across Hindi, Hinglish, and English using state-of-the-art multilingual embeddings and vector similarity.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Screenshot

![Application Screenshot](./assets/Screenshot%202025-12-08%20004810.png)

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Using Weaviate](#using-weaviate-optional)
- [Frontend Features](#frontend-features)
- [Data Sources](#data-sources)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contributing](#contributing)

## Features

- **Trilingual Support**: Seamlessly search across Hindi (Devanagari), Hinglish (romanized Hindi), and English
- **Semantic Search**: Powered by multilingual sentence transformers for deep contextual understanding
- **High Performance**: FAISS vector indexing for lightning-fast similarity search
- **Modern UI**: Beautiful React interface with real-time search and elegant design
- **Flexible Backend**: Support for both FAISS (local) and Weaviate (distributed) vector databases
- **Rich Metadata**: Search results include language detection, similarity scores, and transliterations

## Architecture

This project leverages a powerful tech stack:

- **Backend**: FastAPI for high-performance async API endpoints
- **Embeddings**: \`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\` (500MB model)
- **Vector Store**: FAISS for local indexing with cosine similarity; Weaviate optional for production
- **Frontend**: React 18 + Vite with modern CSS animations and Lucide icons
- **Data Sources**: Curated Hugging Face datasets (~1.1k Hindi poems, ~20k+ English lyrics)
- **Document Processing**: LangChain for intelligent text chunking and retrieval

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm
- Git Bash or compatible shell (Windows users)

### Installation

**Step 1: Clone the repository**

```bash
git clone <repository-url>
cd GenAI_project
```

**Step 2: Set up Python environment**

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
# source .venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

**Step 3: Configure environment variables**

Create a `.env` file in the project root (see [Configuration](#configuration) section)

### Running the Application

**Option 1: CLI Search (Quick Test)**

```bash
# Activate virtual environment
source .venv/Scripts/activate

# Hindi (Devanagari) search
python app.py --rebuild --limit 50 --query "प्रेम गीत" --top_k 5

# Hinglish (romanized Hindi) search
python app.py --query "prem geet" --top_k 5

# English search
python app.py --query "heartbreak love song" --top_k 5

# Force Hindi-only results
python app.py --query "prem geet" --lang hi --top_k 5
```

**Note:** First run will download the model (~500MB) and datasets. Use `--rebuild` to recreate the index after configuration changes.

**Option 2: Full Web Application**

Terminal 1 - Start Backend API:

```bash
# From project root
source .venv/Scripts/activate
uvicorn api:app --reload --port 8000
```

Terminal 2 - Start Frontend Dev Server:

```bash
# From project root
cd webui
npm install  # First time only
node ./node_modules/vite/bin/vite.js dev --host --port 5173
```

Access the application at: http://localhost:5173

API documentation available at: http://localhost:8000/docs

### Production Build

```bash
cd webui
node ./node_modules/vite/bin/vite.js build
# Serve the dist/ folder with your preferred static server
```

**Windows Users:** Due to the `&` character in `Stats&AI`, use the direct Vite binary invocation shown above instead of `npm run` commands in CMD.

## API Reference

### Endpoints

#### Health Check

```http
GET /api/health
```

Response:

```json
{
  "status": "healthy",
  "backend": "faiss",
  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}
```

#### Search

```http
POST /api/search
Content-Type: application/json
```

Request body:

```json
{
  "query": "prem geet",
  "top_k": 5,
  "lang": "auto",
  "include_english": true
}
```

Parameters:

- `query` (string, required): Search query in Hindi/Hinglish/English
- `top_k` (integer, optional): Number of results to return (default: 5, max: 20)
- `lang` (string, optional): Language routing - `"auto"`, `"hi"`, `"en"`, or `"both"` (default: `"auto"`)
- `include_english` (boolean, optional): Include English corpus in results (default: `true`)

Response:

```json
{
  "results": [
    {
      "id": "unique-id",
      "text": "Original text content",
      "hinglish": "Transliteration (if Hindi)",
      "title": "Song/Poem title",
      "poet": "Artist name",
      "language": "hi",
      "score": 0.923,
      "period": "Modern"
    }
  ],
  "backend": "faiss",
  "counts": {
    "hi": 1142,
    "en": 23456
  }
}
```

## Configuration

Create a `.env` file in the project root:

```env
# Hugging Face (optional, for higher rate limits)
HUGGINGFACE_TOKEN=your_token_here

# Dataset Configuration
DATASET_ID=Sourabh2/Hindi_Poems
EN_DATASET_ID=Santarabantoosoo/hf_song_lyrics_with_names,Annanay/aml_song_lyrics_balanced,sheacon/song_lyrics
DATASET_LIMIT=                  # Optional: cap rows per dataset for faster testing

# Model Configuration
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Vector Store Configuration
VECTOR_BACKEND=faiss            # Options: "faiss" or "weaviate"
FAISS_INDEX_PATH=artifacts/faiss_index
INCLUDE_ENGLISH=1               # Set to 0 to skip English corpus entirely

# Weaviate Configuration (optional)
WEAVIATE_URL=                   # e.g., http://localhost:8080
WEAVIATE_API_KEY=               # Required for cloud instances
WEAVIATE_PERSIST_PATH=.weaviate # Local persistence directory
```

### Configuration Options

| Variable            | Description                                                 | Default                                 |
| ------------------- | ----------------------------------------------------------- | --------------------------------------- |
| `HUGGINGFACE_TOKEN` | Personal access token for Hugging Face (avoids rate limits) | None                                    |
| `DATASET_ID`        | Hindi poems dataset(s), comma/newline separated             | `Sourabh2/Hindi_Poems`                  |
| `EN_DATASET_ID`     | English lyrics dataset(s), comma/newline separated          | Multiple datasets                       |
| `EMBED_MODEL`       | Sentence transformer model for embeddings                   | `paraphrase-multilingual-MiniLM-L12-v2` |
| `VECTOR_BACKEND`    | Vector database to use                                      | `faiss`                                 |
| `FAISS_INDEX_PATH`  | Path to store FAISS index                                   | `artifacts/faiss_index`                 |
| `INCLUDE_ENGLISH`   | Whether to index English corpus                             | `1` (yes)                               |
| `DATASET_LIMIT`     | Max rows per dataset (for testing)                          | None (all)                              |
| `WEAVIATE_URL`      | Weaviate instance URL                                       | None                                    |
| `WEAVIATE_API_KEY`  | API key for Weaviate cloud                                  | None                                    |

### Runtime Language Control

You can override language behavior at query time:

- `--lang auto` (default): Auto-detects script; routes Devanagari/Hinglish to Hindi index first
- `--lang hi`: Search Hindi/Hinglish corpus only
- `--lang en`: Search English corpus only
- `--lang both`: Blend results from all corpora equally

## Using Weaviate (Optional)

For production deployments or distributed setups, you can use Weaviate instead of FAISS:

**Step 1: Set up Weaviate**

```bash
# Docker (easiest)
docker run -d -p 8080:8080 semitechnologies/weaviate:latest

# Or use Weaviate Cloud
# Sign up at https://console.weaviate.cloud
```

**Step 2: Configure environment**

```env
VECTOR_BACKEND=weaviate
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your_key_here  # Only for cloud instances
```

**Step 3: Run with Weaviate**

```bash
python app.py --backend weaviate --rebuild --query "your query"
```

The system automatically falls back to FAISS if Weaviate is unreachable.

## Frontend Features

The React web UI includes:

- **Modern Design**: Glassmorphism effects, gradient animations, smooth transitions
- **Real-time Search**: Instant results as you type
- **Language Detection**: Automatic script detection and routing
- **Sample Queries**: Pre-loaded examples for Hindi, Hinglish, and English
- **Responsive Layout**: Works seamlessly on desktop and mobile
- **Professional Icons**: Lucide React icons throughout
- **Dark Theme**: Eye-friendly dark mode optimized for readability

## Data Sources

### Hindi Content

- Primary dataset: `Sourabh2/Hindi_Poems` (~1,100 entries)
- Includes classical and modern Hindi poetry
- Automatically transliterated to Hinglish for better romanized matching

### English Content

- Primary: `Santarabantoosoo/hf_song_lyrics_with_names`
- Secondary: `Annanay/aml_song_lyrics_balanced`
- Tertiary: `sheacon/song_lyrics`
- Combined corpus: ~20,000+ song lyrics
- Diverse genres and artists

You can add custom datasets by modifying `DATASET_ID` and `EN_DATASET_ID` in `.env` with Hugging Face dataset identifiers.

## Technical Details

### How It Works

1. **Embedding Generation**: Text chunks are encoded using multilingual sentence transformers
2. **Normalization**: Vectors are L2-normalized for cosine similarity via dot product
3. **Indexing**: FAISS builds an efficient similarity search index
4. **Query Processing**: User queries are embedded with the same model
5. **Retrieval**: Top-K most similar vectors are retrieved and ranked
6. **Post-processing**: Results include transliterations and metadata enrichment

### Performance Optimizations

- **Lazy Loading**: Embeddings load on first request, not at startup
- **Caching**: FAISS index persists to disk to avoid rebuilding
- **Batch Processing**: Documents are embedded in batches for efficiency
- **Fast Path**: Pre-built indexes skip dataset reloading entirely

### Similarity Computation

Cosine similarity is computed as:

```
similarity = (query_vector · document_vector) / (||query|| × ||document||)
```

Since vectors are pre-normalized:

```
similarity = query_vector · document_vector
```

## Troubleshooting

### Windows Symlink Warning

You may see warnings about symlinks from `huggingface_hub`. This is harmless - caching still works, just uses more disk space. To silence it, enable Windows Developer Mode.

### Path with `&` Character

If `npm run` commands fail due to the `Stats&AI` folder name, use the direct Vite binary:

```bash
node ./node_modules/vite/bin/vite.js dev
```

### Port Already in Use

If port 8000 or 5173 is occupied:

```bash
# Backend
uvicorn api:app --reload --port 8001

# Frontend (update VITE_API_URL accordingly)
node ./node_modules/vite/bin/vite.js dev --port 5174
```

### First Run is Slow

The initial run downloads:

- Sentence transformer model (~500MB)
- Datasets (~50-100MB combined)
- Builds FAISS index (~1-2 minutes)

Subsequent runs use cached artifacts and start instantly.

## Notes

- Cosine similarity achieved through L2-normalization before indexing
- Hindi chunks include automatic Hinglish transliteration for improved romanized query matching
- Multiple Hugging Face datasets can be specified via comma/newline separation in `.env`
- Language detection uses script analysis (Devanagari vs. Latin) and keyword matching
- Results include similarity scores, language tags, and transliterations where applicable

## License

MIT License - feel free to use this project for learning, development, or production.

## Contributing

Contributions welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

Built using FastAPI, React, FAISS, and Sentence Transformers
