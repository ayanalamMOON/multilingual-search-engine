"""RAG (Retrieval-Augmented Generation) module using LangChain + HuggingFace
Enhances search results with AI-generated summaries and insights
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import requests

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


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

    def predict(self, text: str) -> str:
        """Generate text from prompt using OpenAI-compatible chat completion API"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": text}
            ],
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
    """RAG engine combining vector search with LangChain and HuggingFace LLM"""

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

    def chat(self, query: str, user_message: str, top_k: int = 3) -> Dict[str, Any]:
        """Interactive chat about search results"""
        retriever = CustomVectorRetriever(self.search_func, top_k=top_k)
        docs = retriever._get_relevant_documents(query)

        context = "\n\n".join([
            f"Title: {doc.metadata['title']}\nPoet: {doc.metadata['poet']}\n{doc.page_content}"
            for doc in docs
        ])

        prompt_text = f"""You are a knowledgeable assistant helping users explore poetry and lyrics.

Relevant search results for "{query}":
{context}

User question: {user_message}

Response:"""

        response = self.llm.predict(prompt_text)

        return {
            "query": query,
            "user_message": user_message,
            "response": response,
            "sources": [doc.metadata for doc in docs],
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
