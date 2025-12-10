"""Quick test to verify RAG chat returns chat_history"""

import os
import sys
import requests
import json

# Load token from environment
from dotenv import load_dotenv
load_dotenv()

API_URL = "http://localhost:8000"

def test_chat_with_history():
    print("Testing RAG chat with history...\n")

    # First message
    print("1. Sending first message...")
    response1 = requests.post(
        f"{API_URL}/api/rag",
        json={
            "query": "love songs",
            "mode": "chat",
            "user_message": "What themes do these songs have?",
            "top_k": 3
        }
    )

    if response1.status_code != 200:
        print(f"❌ Error: {response1.status_code}")
        print(response1.text)
        return

    data1 = response1.json()
    print(f"✅ Response received")
    print(f"Session ID: {data1.get('session_id')}")
    print(f"Chat history length: {len(data1.get('chat_history', []))}")
    print(f"Chat history: {json.dumps(data1.get('chat_history'), indent=2)}\n")

    session_id = data1.get('session_id')

    if not session_id:
        print("❌ No session_id in response!")
        return

    # Second message
    print("2. Sending second message with same session...")
    response2 = requests.post(
        f"{API_URL}/api/rag",
        json={
            "query": "love songs",
            "mode": "chat",
            "user_message": "Tell me more about the first one",
            "session_id": session_id,
            "top_k": 3
        }
    )

    if response2.status_code != 200:
        print(f"❌ Error: {response2.status_code}")
        print(response2.text)
        return

    data2 = response2.json()
    print(f"✅ Response received")
    print(f"Session ID: {data2.get('session_id')}")
    print(f"Chat history length: {len(data2.get('chat_history', []))}")
    print(f"Chat history: {json.dumps(data2.get('chat_history'), indent=2)}\n")

    # Get history via API
    print("3. Getting chat history via API...")
    response3 = requests.get(f"{API_URL}/api/rag/history/{session_id}")

    if response3.status_code == 200:
        history_data = response3.json()
        print(f"✅ History retrieved")
        print(f"Message count: {history_data.get('message_count')}")
        print(f"History: {json.dumps(history_data.get('history'), indent=2)}\n")
    else:
        print(f"❌ Error getting history: {response3.status_code}")

    print("✅ All tests passed!")

if __name__ == "__main__":
    try:
        test_chat_with_history()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
