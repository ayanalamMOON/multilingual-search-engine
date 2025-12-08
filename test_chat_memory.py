"""Test script for RAG chat with conversation memory"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag import create_rag_engine

def test_search_function(query: str, top_k: int = 5):
    """Mock search function for testing"""
    return [
        {
            "id": "1",
            "text": "प्रेम की बातें करते हैं हम दोनों",
            "title": "प्रेम गीत",
            "poet": "कबीर दास",
            "language": "hi",
            "score": 0.95,
        },
        {
            "id": "2",
            "text": "Love is like a beautiful dream",
            "title": "Dreams of Love",
            "poet": "William Shakespeare",
            "language": "en",
            "score": 0.87,
        }
    ]

def main():
    # Check for API token
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("❌ Error: HUGGINGFACEHUB_API_TOKEN or HUGGINGFACE_TOKEN environment variable required")
        return
    
    print("🚀 Testing RAG Chat with Conversation Memory\n")
    
    # Create RAG engine
    print("Creating RAG engine...")
    rag = create_rag_engine(search_func=test_search_function)
    print("✅ RAG engine created\n")
    
    # Test 1: First chat message (creates new session)
    print("=" * 60)
    print("Test 1: First message (new session)")
    print("=" * 60)
    result1 = rag.chat(
        query="romantic songs",
        user_message="What themes do these songs have?",
        session_id=None  # No session ID - will create new
    )
    session_id = result1["session_id"]
    print(f"Session ID: {session_id}")
    print(f"User: {result1['user_message']}")
    print(f"AI: {result1['response'][:200]}...\n")
    
    # Test 2: Follow-up message in same session
    print("=" * 60)
    print("Test 2: Follow-up message (same session)")
    print("=" * 60)
    result2 = rag.chat(
        query="romantic songs",
        user_message="Tell me more about the first one",
        session_id=session_id  # Use existing session
    )
    print(f"Session ID: {result2['session_id']}")
    print(f"User: {result2['user_message']}")
    print(f"AI: {result2['response'][:200]}...\n")
    
    # Test 3: Check chat history
    print("=" * 60)
    print("Test 3: Chat history")
    print("=" * 60)
    history = rag.get_chat_history(session_id)
    print(f"Total messages in history: {len(history)}")
    for i, msg in enumerate(history, 1):
        role = msg["role"]
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"{i}. [{role.upper()}] {content}\n")
    
    # Test 4: Another message to verify context
    print("=" * 60)
    print("Test 4: Third message (testing context)")
    print("=" * 60)
    result3 = rag.chat(
        query="romantic songs",
        user_message="Can you summarize our conversation?",
        session_id=session_id
    )
    print(f"User: {result3['user_message']}")
    print(f"AI: {result3['response']}\n")
    
    # Test 5: Clear history
    print("=" * 60)
    print("Test 5: Clear session history")
    print("=" * 60)
    cleared = rag.clear_session(session_id)
    print(f"Session cleared: {cleared}")
    history_after_clear = rag.get_chat_history(session_id)
    print(f"Messages after clear: {len(history_after_clear)}\n")
    
    # Test 6: New message after clear
    print("=" * 60)
    print("Test 6: Message after clearing history")
    print("=" * 60)
    result4 = rag.chat(
        query="romantic songs",
        user_message="What are these songs about?",
        session_id=session_id
    )
    print(f"User: {result4['user_message']}")
    print(f"AI: {result4['response'][:200]}...\n")
    
    history_after_new_msg = rag.get_chat_history(session_id)
    print(f"Messages after new message: {len(history_after_new_msg)}")
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    main()
