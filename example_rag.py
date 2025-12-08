"""Test RAG functionality with HuggingFace"""

import sys
import os
import asyncio

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from api import engine, RAGRequest

# Load environment variables
load_dotenv()

async def test_rag():
    """Test RAG with HuggingFace"""

    # Check for API token
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("ERROR: HuggingFace token not found in environment")
        print("Please add one of these to your .env file:")
        print("HUGGINGFACEHUB_API_TOKEN=your_token_here")
        print("or")
        print("HUGGINGFACE_TOKEN=your_token_here")
        return

    print("=" * 80)
    print("Testing RAG with HuggingFace Llama-3.2-1B-Instruct")
    print("=" * 80)

    # Initialize engine
    print("\n[1] Initializing search engine...")
    await engine.ensure_ready()
    print("[OK] Engine ready")

    # Initialize RAG
    print("\n[2] Initializing RAG engine...")
    try:
        await engine.ensure_rag_ready()
        print("[OK] RAG engine ready")
    except Exception as e:
        print(f"[ERROR] RAG initialization failed: {e}")
        return

    # Test 1: Summary (English)
    print("\n" + "=" * 80)
    print("TEST 1: Generate Summary (English Query)")
    print("=" * 80)
    try:
        request = RAGRequest(query="love songs", mode="summary", top_k=3)
        result = await engine.generate_rag_response(request)
        print(f"\nQuery: {result.query}")
        print(f"Mode: {result.mode}")
        print(f"\nResponse:\n{result.response}")
        print(f"\nSources ({len(result.sources)}):")
        for i, src in enumerate(result.sources[:3], 1):
            print(f"  {i}. {src.get('title', 'N/A')} - {src.get('poet', 'N/A')} (score: {src.get('score', 0):.3f})")
    except Exception as e:
        print(f"[ERROR] Test 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Recommendation (Hindi)
    print("\n" + "=" * 80)
    print("TEST 2: Generate Recommendation (Hindi Query)")
    print("=" * 80)
    try:
        request = RAGRequest(query="प्यार की कविता", mode="recommendation", top_k=3)
        result = await engine.generate_rag_response(request)
        print(f"\nQuery: {result.query}")
        print(f"Mode: {result.mode}")
        print(f"\nResponse:\n{result.response}")
        print(f"\nSources ({len(result.sources)}):")
        for i, src in enumerate(result.sources[:3], 1):
            print(f"  {i}. {src.get('title', 'N/A')} - {src.get('poet', 'N/A')} (score: {src.get('score', 0):.3f})")
    except Exception as e:
        print(f"[ERROR] Test 2 failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Chat
    print("\n" + "=" * 80)
    print("TEST 3: Chat Mode")
    print("=" * 80)
    try:
        request = RAGRequest(
            query="romantic poetry",
            mode="chat",
            top_k=3,
            user_message="What makes these poems special?"
        )
        result = await engine.generate_rag_response(request)
        print(f"\nQuery: {result.query}")
        print(f"Mode: {result.mode}")
        print(f"User Message: {request.user_message}")
        print(f"\nResponse:\n{result.response}")
    except Exception as e:
        print(f"[ERROR] Test 3 failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("RAG Testing Complete!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_rag())
