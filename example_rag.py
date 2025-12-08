"""
Example usage of RAG features
"""

import asyncio
import os
from api import engine, RAGRequest

# Set your API key (or use environment variable)
# os.environ["OPENAI_API_KEY"] = "your-key-here"


async def test_rag():
    """Test RAG functionality"""
    
    # Ensure engine is ready
    await engine.ensure_ready()
    
    # Test 1: Summary mode
    print("=== RAG Summary Mode ===")
    request = RAGRequest(
        query="love songs about heartbreak",
        top_k=5,
        mode="summary"
    )
    
    try:
        response = await engine.generate_rag_response(request)
        print(f"Query: {response.query}")
        print(f"Response: {response.response}")
        print(f"Sources: {len(response.sources)} documents")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set OPENAI_API_KEY or HUGGINGFACE_TOKEN\n")
    
    # Test 2: Recommendation mode
    print("=== RAG Recommendation Mode ===")
    request = RAGRequest(
        query="पे्रम गीत",  # Hindi query
        top_k=3,
        mode="recommendation"
    )
    
    try:
        response = await engine.generate_rag_response(request)
        print(f"Query: {response.query}")
        print(f"Recommendation: {response.response}")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Test 3: Chat mode
    print("=== RAG Chat Mode ===")
    request = RAGRequest(
        query="What themes are common in Hindi poetry?",
        top_k=5,
        mode="chat"
    )
    
    try:
        response = await engine.generate_rag_response(request)
        print(f"Query: {response.query}")
        print(f"Chat Response: {response.response}")
        print()
    except Exception as e:
        print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(test_rag())
