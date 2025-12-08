"""
RAG (Retrieval-Augmented Generation) module using LangChain
Enhances search results with AI-generated summaries and insights
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class CustomVectorRetriever(BaseRetriever):
    """Custom retriever that wraps our existing FAISS/Weaviate search"""
    
    search_function: Any
    top_k: int = 5
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Retrieve documents using our existing search function"""
        results = self.search_function(query, self.top_k)
        
        documents = []
        for result in results:
            # Convert search results to LangChain Document format
            content = result.get("text", "")
            metadata = {
                "title": result.get("title", "Unknown"),
                "poet": result.get("poet", "Unknown"),
                "language": result.get("language", "unknown"),
                "score": result.get("score", 0.0),
                "period": result.get("period", ""),
            }
            
            # Add hinglish transliteration if available
            if result.get("hinglish"):
                metadata["hinglish"] = result["hinglish"]
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents


class RAGEngine:
    """RAG engine for generating AI-enhanced responses"""
    
    def __init__(
        self,
        search_function: Any,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        use_huggingface: bool = False,
        hf_model: str = "google/flan-t5-large",
    ):
        """
        Initialize RAG engine
        
        Args:
            search_function: Function that performs vector search
            api_key: OpenAI or HuggingFace API key
            model_name: OpenAI model name (if using OpenAI)
            use_huggingface: Whether to use HuggingFace instead of OpenAI
            hf_model: HuggingFace model to use
        """
        self.search_function = search_function
        
        # Initialize LLM
        if use_huggingface:
            if not api_key:
                api_key = os.getenv("HUGGINGFACE_TOKEN")
            self.llm = HuggingFaceHub(
                repo_id=hf_model,
                huggingfacehub_api_token=api_key,
                model_kwargs={"temperature": 0.7, "max_length": 512}
            )
        else:
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=0.7,
                openai_api_key=api_key,
            )
        
        # Create custom retriever
        self.retriever = CustomVectorRetriever(
            search_function=search_function,
            top_k=5
        )
        
        # Define prompt templates
        self.summary_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a knowledgeable assistant for a multilingual song and poem discovery platform.
Based on the retrieved songs and poems below, provide a helpful and creative response to the user's query.

Retrieved Content:
{context}

User Query: {question}

Instructions:
- Summarize the themes and emotions in the retrieved content
- Highlight any common patterns or connections between the pieces
- If the content is in Hindi or Hinglish, mention the cultural context
- Be creative and insightful, but stay grounded in the retrieved content
- Keep your response concise (3-4 sentences)

Response:"""
        )
        
        self.recommendation_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a music and poetry recommendation expert for a multilingual platform.

Retrieved Songs/Poems:
{context}

User Query: {question}

Based on the retrieved content, provide:
1. A brief analysis of the emotional themes
2. Why these pieces match the user's query
3. A creative suggestion for what mood or occasion these would be perfect for

Keep your response engaging and under 4 sentences.

Response:"""
        )
        
        # Create RAG chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.summary_template}
        )
    
    def generate_summary(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate an AI summary of search results
        
        Args:
            query: User's search query
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with summary and source documents
        """
        self.retriever.top_k = top_k
        
        result = self.qa_chain.invoke({"query": query})
        
        return {
            "summary": result["result"],
            "source_documents": [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "poet": doc.metadata.get("poet", "Unknown"),
                    "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "language": doc.metadata.get("language", "unknown"),
                    "score": doc.metadata.get("score", 0.0),
                }
                for doc in result["source_documents"]
            ]
        }
    
    def generate_recommendation(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate AI-powered recommendations
        
        Args:
            query: User's search query
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with recommendation and source documents
        """
        self.retriever.top_k = top_k
        
        # Use recommendation template
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.recommendation_template}
        )
        
        result = chain.invoke({"query": query})
        
        return {
            "recommendation": result["result"],
            "source_documents": [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "poet": doc.metadata.get("poet", "Unknown"),
                    "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "language": doc.metadata.get("language", "unknown"),
                    "score": doc.metadata.get("score", 0.0),
                }
                for doc in result["source_documents"]
            ]
        }
    
    def chat(self, query: str, custom_prompt: Optional[str] = None, top_k: int = 5) -> str:
        """
        General chat interface with custom prompts
        
        Args:
            query: User's query
            custom_prompt: Optional custom prompt template
            top_k: Number of results to retrieve
            
        Returns:
            AI-generated response
        """
        self.retriever.top_k = top_k
        
        if custom_prompt:
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=custom_prompt
            )
            chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            result = chain.invoke({"query": query})
        else:
            result = self.qa_chain.invoke({"query": query})
        
        return result["result"]


def create_rag_engine(
    search_function: Any,
    use_openai: bool = True,
    api_key: Optional[str] = None,
) -> Optional[RAGEngine]:
    """
    Factory function to create RAG engine
    
    Args:
        search_function: Function that performs vector search
        use_openai: Whether to use OpenAI (True) or HuggingFace (False)
        api_key: API key for the chosen provider
        
    Returns:
        RAGEngine instance or None if API key not available
    """
    try:
        if use_openai:
            return RAGEngine(
                search_function=search_function,
                api_key=api_key,
                model_name="gpt-3.5-turbo",
                use_huggingface=False,
            )
        else:
            return RAGEngine(
                search_function=search_function,
                api_key=api_key,
                use_huggingface=True,
                hf_model="google/flan-t5-large",
            )
    except ValueError as e:
        print(f"⚠️  RAG engine not initialized: {e}")
        print("💡 Set OPENAI_API_KEY or HUGGINGFACE_TOKEN to enable RAG features")
        return None
