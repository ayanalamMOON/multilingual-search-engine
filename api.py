from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Try to import RAG - optional feature
try:
    from rag import create_rag_engine, RAGEngine
    RAG_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] RAG features not available: {str(e)[:100]}")
    print("[INFO] RAG requires compatible langchain versions. Search still works!")
    RAG_AVAILABLE = False
    RAGEngine = None
    create_rag_engine = None

from app import (
    Settings,
    _is_devanagari,
    _looks_hinglish,
    build_embeddings,
    build_weaviate,
    connect_weaviate,
    faiss_index_exists,
    group_unique_results,
    load_english_songs,
    load_faiss_meta,
    load_or_build_faiss,
    load_poems,
    prepare_query,
    similarity_search,
    transliterate_to_hinglish,
)


@dataclass
class RecommenderState:
    backend: str
    hi_store: Any
    en_store: Any
    combined_store: Any
    embeddings: Any
    settings: Settings
    hi_docs: int
    en_docs: int
    client: Any = None
    rag_engine: Optional[RAGEngine] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query text in Hindi/Hinglish/English")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    lang: Literal["auto", "hi", "en", "both"] = Field(
        "auto", description="Language focus: auto routes Devanagari/Hinglish to Hindi first"
    )
    include_english: Optional[bool] = Field(
        None,
        description="Override whether to include the English lyrics corpus (defaults to .env INCLUDE_ENGLISH)",
    )


class SearchResult(BaseModel):
    id: str
    language: str
    title: Optional[str]
    poet: Optional[str]
    period: Optional[str]
    text: str
    hinglish: Optional[str]
    score: Optional[float]


class SearchResponse(BaseModel):
    backend: str
    results: List[SearchResult]
    counts: Dict[str, int]


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query text")
    top_k: int = Field(5, ge=1, le=10, description="Number of results to retrieve")
    mode: Literal["summary", "recommendation", "chat"] = Field(
        "summary", description="RAG mode: summary, recommendation, or chat"
    )
    user_message: Optional[str] = Field(None, description="User message for chat mode")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity in chat mode")


class RAGResponse(BaseModel):
    query: str
    response: str
    sources: List[Dict[str, Any]]
    mode: str
    rag_available: bool = True
    session_id: Optional[str] = Field(None, description="Session ID for chat mode")
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="Chat history for the session")


class RecommenderEngine:
    def __init__(self) -> None:
        self.state: Optional[RecommenderState] = None
        self._lock = asyncio.Lock()

    async def ensure_ready(self, rebuild: bool = False) -> RecommenderState:
        if self.state and not rebuild:
            return self.state

        async with self._lock:
            if self.state and not rebuild:
                return self.state

            settings = Settings.load()
            include_english = settings.include_english
            limit = settings.dataset_limit

            combined_path = settings.faiss_dir
            hi_path = settings.faiss_dir / "hi"
            en_path = settings.faiss_dir / "en"

            fast_path_possible = (
                not rebuild
                and faiss_index_exists(combined_path)
                and faiss_index_exists(hi_path)
                and (not include_english or faiss_index_exists(en_path))
            )

            hi_docs: List[Any] = []
            en_docs: List[Any] = []

            if fast_path_possible:
                hi_count = load_faiss_meta(hi_path) or 0
                en_count = load_faiss_meta(en_path) or 0
            else:
                for ds_id in settings.hindi_datasets:
                    hi_docs.extend(load_poems(ds_id, limit=limit))
                if include_english:
                    for ds_id in settings.english_datasets:
                        en_docs.extend(load_english_songs(ds_id, limit=limit))
                hi_count = len(hi_docs)
                en_count = len(en_docs)

            embeddings = build_embeddings(settings.embed_model, settings.hf_token)

            backend = settings.vector_backend
            client = None
            hi_store = None
            en_store = None
            combined_store = None

            if backend == "weaviate":
                try:
                    client = connect_weaviate(settings)
                    if not fast_path_possible:
                        combined_store = build_weaviate(hi_docs + en_docs, embeddings, client)
                    else:
                        combined_store = None
                except Exception as exc:  # pragma: no cover - optional path
                    print(f"Weaviate unavailable ({exc}); falling back to FAISS")
                    backend = "faiss"

            if backend == "faiss":
                if fast_path_possible:
                    hi_store = load_or_build_faiss([], embeddings, hi_path, rebuild=False) if faiss_index_exists(hi_path) else None
                    en_store = None
                    if include_english and faiss_index_exists(en_path):
                        en_store = load_or_build_faiss([], embeddings, en_path, rebuild=False)
                    combined_store = load_or_build_faiss([], embeddings, combined_path, rebuild=False)
                else:
                    if hi_docs:
                        hi_store = load_or_build_faiss(hi_docs, embeddings, hi_path, rebuild=rebuild)
                    if en_docs:
                        en_store = load_or_build_faiss(en_docs, embeddings, en_path, rebuild=rebuild)
                    combined_store = load_or_build_faiss(hi_docs + en_docs, embeddings, combined_path, rebuild=rebuild)

            self.state = RecommenderState(
                backend=backend,
                hi_store=hi_store,
                en_store=en_store,
                combined_store=combined_store,
                embeddings=embeddings,
                settings=settings,
                hi_docs=hi_count,
                en_docs=en_count,
                client=client,
                rag_engine=None,  # Will be initialized on first RAG request
            )
            return self.state

    def _format_result(self, doc, score: Optional[float], idx: int) -> SearchResult:
        meta = doc.metadata or {}
        display_text = meta.get("display_text", doc.page_content)
        language = (meta.get("language") or "").lower() or "hi"
        hinglish = transliterate_to_hinglish(display_text) if _is_devanagari(display_text) else None
        poet = meta.get("poet")
        period = meta.get("period")
        title = meta.get("title")
        result_id = hashlib.md5(f"{language}-{poet}-{title}-{idx}-{display_text[:32]}".encode("utf-8")).hexdigest()

        safe_score = None
        try:
            safe_score = float(score) if score is not None else None
        except Exception:
            safe_score = None

        return SearchResult(
            id=result_id,
            language=language,
            title=title,
            poet=poet,
            period=period,
            text=display_text.strip(),
            hinglish=hinglish.strip() if hinglish else None,
            score=safe_score,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        state = await self.ensure_ready()
        include_english = state.settings.include_english if request.include_english is None else request.include_english

        search_query = prepare_query(request.query)
        fetch_k = max(request.top_k * 3, request.top_k + 5)
        lang_pref = request.lang.lower()
        hinglish_hint = _looks_hinglish(request.query)

        def merge(primary, secondary):
            seen = set()
            merged = []
            for pair in primary + secondary:
                key = pair[0].page_content
                if key in seen:
                    continue
                seen.add(key)
                merged.append(pair)
                if len(merged) >= fetch_k:
                    break
            return merged

        result_pairs = []
        if state.backend == "faiss":
            hi_store = state.hi_store
            en_store = state.en_store if include_english else None
            combined_store = state.combined_store if include_english or not hi_store else state.hi_store

            if lang_pref == "hi" and hi_store:
                result_pairs = similarity_search(hi_store, search_query, k=fetch_k)
            elif lang_pref == "en" and en_store:
                result_pairs = similarity_search(en_store, search_query, k=fetch_k)
            elif lang_pref == "both" and hi_store and en_store:
                hi_results = similarity_search(hi_store, search_query, k=fetch_k)
                en_results = similarity_search(en_store, search_query, k=fetch_k)
                result_pairs = merge(hi_results, en_results)
            elif lang_pref == "auto" and hi_store and (_is_devanagari(request.query) or hinglish_hint):
                hi_results = similarity_search(hi_store, search_query, k=fetch_k)
                combined_results = similarity_search(combined_store, search_query, k=fetch_k)
                result_pairs = merge(hi_results, combined_results)
            else:
                result_pairs = similarity_search(combined_store, search_query, k=fetch_k)
        else:
            # Weaviate fallback: search combined and filter if needed
            combined_results = similarity_search(state.combined_store, search_query, k=fetch_k * 2)
            if not include_english:
                combined_results = [
                    (doc, score)
                    for doc, score in combined_results
                    if (doc.metadata or {}).get("language", "").lower().startswith("hi")
                ]
            result_pairs = combined_results

        ranked = group_unique_results(result_pairs, request.top_k)
        formatted = [self._format_result(doc, score, idx) for idx, (doc, score) in enumerate(ranked, start=1)]

        return SearchResponse(
            backend=state.backend,
            results=formatted,
            counts={"hi": state.hi_docs, "en": state.en_docs},
        )

    def _search_wrapper(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Wrapper function for RAG engine to use our search - runs synchronously"""
        # Use the state directly for synchronous search
        state = self.state
        if not state:
            return []

        try:
            # Prepare the search query
            search_query = prepare_query(query)

            # Perform similarity search
            result_pairs = similarity_search(state.combined_store, search_query, k=top_k)
            ranked = group_unique_results(result_pairs, top_k)

            # Format results
            results = []
            for idx, (doc, score) in enumerate(ranked, start=1):
                meta = doc.metadata or {}
                display_text = meta.get("display_text", doc.page_content)
                language = (meta.get("language") or "").lower() or "hi"
                hinglish = transliterate_to_hinglish(display_text) if _is_devanagari(display_text) else None

                results.append({
                    "id": hashlib.md5(f"{language}-{idx}-{display_text[:32]}".encode("utf-8")).hexdigest(),
                    "text": display_text,
                    "title": meta.get("title"),
                    "poet": meta.get("poet"),
                    "language": language,
                    "score": float(score) if score else None,
                    "period": meta.get("period"),
                    "hinglish": hinglish,
                })

            return results
        except Exception as e:
            print(f"Search wrapper error: {e}")
            return []

    async def ensure_rag_ready(self) -> bool:
        """Initialize RAG engine if not already done"""
        if not RAG_AVAILABLE:
            return False

        state = await self.ensure_ready()

        if state.rag_engine is not None:
            return True

        try:
            state.rag_engine = create_rag_engine(
                search_func=self._search_wrapper,
            )
            return state.rag_engine is not None
        except Exception as e:
            print(f"Failed to initialize RAG engine: {e}")
            return False

    async def generate_rag_response(self, request: RAGRequest) -> RAGResponse:
        """Generate RAG response"""
        rag_ready = await self.ensure_rag_ready()
        state = await self.ensure_ready()

        if not rag_ready or state.rag_engine is None:
            raise HTTPException(
                status_code=503,
                detail="RAG engine not available. Please set HUGGINGFACEHUB_API_TOKEN or HUGGINGFACE_TOKEN environment variable."
            )

        try:
            if request.mode == "summary":
                result = state.rag_engine.generate_summary(request.query, request.top_k)
                return RAGResponse(
                    query=request.query,
                    response=result["summary"],
                    sources=result["sources"],
                    mode="summary",
                )
            elif request.mode == "recommendation":
                result = state.rag_engine.generate_recommendation(request.query, request.top_k)
                return RAGResponse(
                    query=request.query,
                    response=result["recommendations"],
                    sources=result["sources"],
                    mode="recommendation",
                )
            else:  # chat mode
                result = state.rag_engine.chat(
                    query=request.query,
                    user_message=request.user_message or "Tell me about these results",
                    session_id=request.session_id,
                    top_k=request.top_k
                )
                # Get updated chat history
                session_id = result.get("session_id")
                chat_history = state.rag_engine.get_chat_history(session_id) if session_id else []
                
                return RAGResponse(
                    query=request.query,
                    response=result["response"],
                    sources=result["sources"],
                    mode="chat",
                    session_id=session_id,
                    chat_history=chat_history,
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG generation failed: {str(e)}")


engine = RecommenderEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.ensure_ready()
    try:
        yield
    finally:
        state = engine.state
        client = getattr(state, "client", None) if state else None
        if client:
            try:
                client.close()
            except Exception:
                pass


app = FastAPI(
    title="Hindi/Hinglish/English Song & Poem Recommender",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    state = await engine.ensure_ready()
    rag_available = await engine.ensure_rag_ready()
    return {
        "status": "ok",
        "backend": state.backend,
        "counts": {"hi": state.hi_docs, "en": state.en_docs},
        "rag_available": rag_available,
    }


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    return await engine.search(request)


@app.post("/api/rag", response_model=RAGResponse)
async def rag_generate(request: RAGRequest):
    """
    Generate AI-enhanced responses using RAG

    Modes:
    - summary: Generate a summary of retrieved content
    - recommendation: Get personalized recommendations
    - chat: Interactive chat about the content (with conversation memory)

    Chat mode supports session_id for multi-turn conversations.
    If no session_id is provided, a new session will be created.

    Requires: HUGGINGFACEHUB_API_TOKEN or HUGGINGFACE_TOKEN environment variable
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    return await engine.generate_rag_response(request)


@app.get("/api/rag/history/{session_id}")
async def get_chat_history(session_id: str):
    """
    Get chat history for a specific session
    
    Returns the full conversation history including user messages and AI responses.
    """
    rag_ready = await engine.ensure_rag_ready()
    state = await engine.ensure_ready()
    
    if not rag_ready or state.rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not available")
    
    history = state.rag_engine.get_chat_history(session_id)
    
    return {
        "session_id": session_id,
        "history": history,
        "message_count": len(history)
    }


@app.post("/api/rag/clear/{session_id}")
async def clear_chat_history(session_id: str):
    """
    Clear chat history for a specific session
    
    Removes all messages but keeps the session active.
    """
    rag_ready = await engine.ensure_rag_ready()
    state = await engine.ensure_ready()
    
    if not rag_ready or state.rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not available")
    
    cleared = state.rag_engine.clear_session(session_id)
    
    if not cleared:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return {
        "session_id": session_id,
        "status": "cleared",
        "message": "Chat history cleared successfully"
    }


@app.delete("/api/rag/session/{session_id}")
async def delete_chat_session(session_id: str):
    """
    Delete a chat session entirely
    
    Removes the session and all associated conversation history.
    """
    rag_ready = await engine.ensure_rag_ready()
    state = await engine.ensure_ready()
    
    if not rag_ready or state.rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not available")
    
    deleted = state.rag_engine.delete_session(session_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return {
        "session_id": session_id,
        "status": "deleted",
        "message": "Session deleted successfully"
    }


@app.get("/")
async def root():  # pragma: no cover - trivial
    return {"message": "Hindi/Hinglish/English song recommender API", "docs": "/docs"}
