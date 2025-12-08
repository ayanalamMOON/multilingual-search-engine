"""RAG (Retrieval-Augmented Generation) module using LangChain + HuggingFace
Enhances search results with AI-generated summaries and insights
Includes conversation memory for multi-turn interactions
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional
import requests

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory


class CustomVectorRetriever(BaseRetriever):
    """Custom retriever that wraps existing FAISS search functionality"""

    def __init__(self, search_func, top_k: int = 5):
        super().__init__()
        object.__setattr__(self, 'search_func', search_func)
        object.__setattr__(self, 'top_k', top_k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """Get relevant documents using our search function"""
        results = self.search_func(query, self.top_k)

        # Convert search results to LangChain Documents
        documents = []
        for result in results:
            content = f"Title: {result['title']}\n"
            content += f"Poet: {result['poet']}\n"
            if result.get("lines"):
                content += f"Content: {' '.join(result['lines'][:15])}\n"
            elif result.get("text"):
                content += f"Content: {result['text'][:500]}\n"

            metadata = {
                "title": result["title"],
                "poet": result["poet"],
                "score": result["score"],
                "language": result.get("language", "unknown"),
            }

            documents.append(Document(page_content=content, metadata=metadata))

        return documents


class HuggingFaceLLM:
    """Direct HuggingFace Inference API wrapper compatible with LangChain"""

    def __init__(self, api_token: str, model: str = "meta-llama/Llama-3.2-1B-Instruct"):
        self.api_token = api_token
        self.model = model
        # Use new HuggingFace router endpoint (OpenAI-compatible)
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }

    def __call__(self, prompt: str) -> str:
        """Call the model - LangChain compatible interface"""
        return self.predict(prompt)

    def predict(self, text: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate text from prompt using OpenAI-compatible chat completion API
        
        Args:
            text: The current user message or prompt
            chat_history: Optional list of previous messages [{"role": "user"|"assistant", "content": str}]
        """
        messages = []
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)
        
        # Add current message
        messages.append({"role": "user", "content": text})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": False
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)

            # Handle model loading
            if response.status_code == 503:
                result = response.json()
                if "estimated_time" in result:
                    return f"Model is loading (estimated {result['estimated_time']:.0f}s). Please try again..."
                return "Model is loading, please try again in a moment..."

            response.raise_for_status()
            result = response.json()

            # Parse OpenAI-style response
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                content = message.get("content", "")
                return content.strip() if content else "No response generated"
            elif "error" in result:
                return f"Model error: {result['error']}"
            else:
                return str(result)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503:
                return "Model is loading, please try again in a moment..."
            elif e.response.status_code == 410:
                return "Error: This API endpoint is deprecated. Please update to the new router endpoint."
            return f"API error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"


class RAGEngine:
    """RAG engine combining vector search with LangChain and HuggingFace LLM
    
    Includes conversation memory management for multi-turn interactions.
    """

    def __init__(
        self,
        search_func,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        api_key: Optional[str] = None,
    ):
        """
        Initialize RAG engine with LangChain + HuggingFace

        Args:
            search_func: Function that performs vector search (query, top_k) -> List[Dict]
            model_name: HuggingFace model to use
            api_key: HuggingFace API token (uses HUGGINGFACEHUB_API_TOKEN env var if not provided)
        """
        self.search_func = search_func
        api_key = api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not api_key:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN or HUGGINGFACE_TOKEN environment variable required")

        self.llm = HuggingFaceLLM(api_token=api_key, model=model_name)
        
        # Session-based memory storage
        self.sessions: Dict[str, ConversationBufferMemory] = {}

    def generate_summary(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Generate a summary of search results using LangChain"""
        # Create retriever
        retriever = CustomVectorRetriever(self.search_func, top_k=top_k)

        # Get documents
        docs = retriever._get_relevant_documents(query)

        # Format context
        context = "\n\n".join([
            f"Title: {doc.metadata['title']}\nPoet: {doc.metadata['poet']}\n{doc.page_content}"
            for doc in docs
        ])

        # Create prompt
        prompt_text = f"""Based on the following search results for the query "{query}", provide a concise summary of the themes, styles, and notable works found.

Search Results:
{context}

Summary:"""

        # Generate response directly
        response = self.llm.predict(prompt_text)

        return {
            "query": query,
            "summary": response,
            "sources": [doc.metadata for doc in docs],
        }

    def generate_recommendation(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Generate recommendations based on search results"""
        retriever = CustomVectorRetriever(self.search_func, top_k=top_k)
        docs = retriever._get_relevant_documents(query)

        context = "\n\n".join([
            f"Title: {doc.metadata['title']}\nPoet: {doc.metadata['poet']}\n{doc.page_content}"
            for doc in docs
        ])

        prompt_text = f"""You are a poetry and music recommendation expert.

Based on these search results for "{query}", recommend similar works the user might enjoy and explain why.

Search Results:
{context}

Recommendations:"""

        response = self.llm.predict(prompt_text)

        return {
            "query": query,
            "recommendations": response,
            "sources": [doc.metadata for doc in docs],
        }

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one
        
        Args:
            session_id: Optional session ID. If None, creates new session.
            
        Returns:
            Session ID (existing or newly created)
        """
        if session_id and session_id in self.sessions:
            return session_id
        
        new_session_id = session_id or str(uuid.uuid4())
        self.sessions[new_session_id] = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        return new_session_id
    
    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get chat history for a session
        
        Args:
            session_id: The session ID
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        if session_id not in self.sessions:
            return []
        
        memory = self.sessions[session_id]
        messages = memory.chat_memory.messages
        
        # Convert LangChain messages to dict format for HuggingFace API
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        
        return history
    
    def clear_session(self, session_id: str) -> bool:
        """Clear chat history for a session
        
        Args:
            session_id: The session ID to clear
            
        Returns:
            True if session was cleared, False if session not found
        """
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session entirely
        
        Args:
            session_id: The session ID to delete
            
        Returns:
            True if session was deleted, False if session not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def chat(self, query: str, user_message: str, session_id: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
        """Interactive chat about search results with conversation memory
        
        Args:
            query: Search query to retrieve relevant documents
            user_message: User's chat message
            session_id: Optional session ID for conversation continuity
            top_k: Number of documents to retrieve
            
        Returns:
            Dict with query, user_message, response, sources, and session_id
        """
        # Get or create session
        session_id = self.get_or_create_session(session_id)
        memory = self.sessions[session_id]
        
        # Retrieve relevant documents
        retriever = CustomVectorRetriever(self.search_func, top_k=top_k)
        docs = retriever._get_relevant_documents(query)

        context = "\n\n".join([
            f"Title: {doc.metadata['title']}\nPoet: {doc.metadata['poet']}\n{doc.page_content}"
            for doc in docs
        ])

        # Build prompt with context
        prompt_text = f"""You are a knowledgeable assistant helping users explore poetry and lyrics.

Relevant search results for "{query}":
{context}

User question: {user_message}

Response:"""

        # Get chat history in format for HuggingFace API
        chat_history = self.get_chat_history(session_id)
        
        # Generate response with chat history
        response = self.llm.predict(prompt_text, chat_history=chat_history)
        
        # Save to memory
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(response)

        return {
            "query": query,
            "user_message": user_message,
            "response": response,
            "sources": [doc.metadata for doc in docs],
            "session_id": session_id,
        }


def create_rag_engine(
    search_func,
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    api_key: Optional[str] = None,
) -> RAGEngine:
    """
    Factory function to create a RAG engine with LangChain + HuggingFace

    Args:
        search_func: Function that performs vector search
        model_name: HuggingFace model name
        api_key: Optional HuggingFace API token override

    Returns:
        Configured RAGEngine instance

    Example:
        >>> rag = create_rag_engine(engine.search)
        >>> result = rag.generate_summary("love poems", top_k=3)
    """
    return RAGEngine(
        search_func=search_func,
        model_name=model_name,
        api_key=api_key,
    )
