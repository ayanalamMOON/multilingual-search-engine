# Chat Memory and Conversation History

This document explains the chat memory feature in the RAG system, which enables multi-turn conversations with context retention.

## Overview

The RAG system now includes **conversation memory** using LangChain's `ConversationBufferMemory`. This allows the AI to remember previous messages in a conversation and provide contextually relevant responses.

## Features

- **Session-based Memory**: Each conversation is stored in a unique session
- **Persistent Context**: AI remembers all previous messages in the session
- **Multi-turn Conversations**: Ask follow-up questions that reference earlier messages
- **Session Management**: Create, view, clear, and delete chat sessions
- **Automatic Session Creation**: New sessions created automatically when not provided

## How It Works

### Backend (Python)

The `RAGEngine` class maintains a dictionary of conversation sessions:

```python
# Each session has its own ConversationBufferMemory
self.sessions: Dict[str, ConversationBufferMemory] = {}
```

When you chat:
1. A session ID is created (or reused if provided)
2. The user message and AI response are saved to memory
3. Previous messages are included in the context for future queries
4. The updated chat history is returned

### API Endpoints

#### 1. Chat with Memory (POST `/api/rag`)

```json
{
  "query": "romantic songs",
  "mode": "chat",
  "user_message": "What themes do these have?",
  "session_id": "optional-session-id",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "romantic songs",
  "response": "AI response here...",
  "sources": [...],
  "mode": "chat",
  "session_id": "abc-123-def-456",
  "chat_history": [
    {"role": "user", "content": "What themes do these have?"},
    {"role": "assistant", "content": "AI response here..."}
  ]
}
```

#### 2. Get Chat History (GET `/api/rag/history/{session_id}`)

Returns the full conversation history for a session.

**Response:**
```json
{
  "session_id": "abc-123-def-456",
  "history": [
    {"role": "user", "content": "Message 1"},
    {"role": "assistant", "content": "Response 1"},
    {"role": "user", "content": "Message 2"},
    {"role": "assistant", "content": "Response 2"}
  ],
  "message_count": 4
}
```

#### 3. Clear Chat History (POST `/api/rag/clear/{session_id}`)

Clears all messages from a session but keeps the session active.

**Response:**
```json
{
  "session_id": "abc-123-def-456",
  "status": "cleared",
  "message": "Chat history cleared successfully"
}
```

#### 4. Delete Session (DELETE `/api/rag/session/{session_id}`)

Completely removes a session and its history.

**Response:**
```json
{
  "session_id": "abc-123-def-456",
  "status": "deleted",
  "message": "Session deleted successfully"
}
```

## Frontend (React)

The web UI includes chat history display:

### Features:
- **Chat History Panel**: Shows all messages in the current session
- **Visual Distinction**: User messages (blue) vs AI responses (purple)
- **Clear Button**: One-click to clear conversation history
- **Auto-scroll**: Automatically scrolls to latest messages
- **Session Persistence**: Session ID maintained across messages

### State Management:

```javascript
const [sessionId, setSessionId] = useState(null)
const [chatHistory, setChatHistory] = useState([])
```

### Chat History Display:

The UI shows:
- User messages on the left (blue accent)
- AI responses on the right (purple accent)
- Message count in header
- Clear history button

## Usage Examples

### Python Example

```python
from rag import create_rag_engine

# Create RAG engine
rag = create_rag_engine(search_func=my_search_function)

# First message (creates new session)
result1 = rag.chat(
    query="romantic poetry",
    user_message="What are common themes?",
    session_id=None  # Will create new session
)
session_id = result1["session_id"]

# Follow-up message (uses existing session)
result2 = rag.chat(
    query="romantic poetry",
    user_message="Tell me more about the first one",
    session_id=session_id  # Reuse session
)

# AI now has context from first message!

# Get full history
history = rag.get_chat_history(session_id)
print(f"Messages: {len(history)}")

# Clear history but keep session
rag.clear_session(session_id)

# Delete session entirely
rag.delete_session(session_id)
```

### JavaScript Example

```javascript
// First message
const response1 = await fetch('/api/rag', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'romantic songs',
    mode: 'chat',
    user_message: 'What makes these special?',
    // session_id: null (will create new)
  })
})
const data1 = await response1.json()
const sessionId = data1.session_id

// Follow-up message
const response2 = await fetch('/api/rag', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'romantic songs',
    mode: 'chat',
    user_message: 'Can you elaborate?',
    session_id: sessionId  // Reuse session
  })
})

// Get history
const historyResponse = await fetch(`/api/rag/history/${sessionId}`)
const history = await historyResponse.json()

// Clear history
await fetch(`/api/rag/clear/${sessionId}`, { method: 'POST' })

// Delete session
await fetch(`/api/rag/session/${sessionId}`, { method: 'DELETE' })
```

## Technical Details

### Memory Implementation

Uses LangChain's `ConversationBufferMemory`:
- Stores complete conversation history
- Returns messages as LangChain message objects
- Converted to OpenAI-compatible format for HuggingFace API

### Chat History Format

Messages are converted from LangChain format to API format:

```python
# LangChain format
HumanMessage(content="Hello")
AIMessage(content="Hi there!")

# Converted to
[
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

### Session Storage

- **In-memory**: Sessions stored in Python dictionary
- **Per-engine**: Each RAGEngine instance has its own sessions
- **Not persistent**: Sessions reset when server restarts
- **Scalable**: Can be moved to Redis/database for production

## Best Practices

1. **Reuse Sessions**: Keep the session ID to maintain conversation context
2. **Clear When Done**: Clear history after conversation ends to save memory
3. **Monitor Size**: Long conversations consume more tokens
4. **Fresh Start**: Create new session for unrelated topics
5. **Error Handling**: Check for 404 when session doesn't exist

## Limitations

- Sessions are not persisted across server restarts
- Memory grows with conversation length
- Max context window limited by model (256 tokens for current model)
- One engine instance = one set of sessions

## Future Enhancements

Potential improvements:
- Persistent storage (Redis, PostgreSQL)
- Session expiration/TTL
- Memory summarization for long conversations
- Multi-user session management
- Conversation export/import
- Analytics on session usage

## Testing

Run the test script to verify functionality:

```bash
python test_chat_memory.py
```

This tests:
- Session creation
- Multi-turn conversations
- Chat history retrieval
- Session clearing
- Context retention
