from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import json

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError, DatasetGenerationError
from datasets.exceptions import DatasetNotFoundError
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Weaviate
import weaviate
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


@dataclass
class Settings:
    hindi_datasets: List[str]
    english_datasets: List[str]
    embed_model: str
    faiss_dir: Path
    weaviate_url: Optional[str]
    weaviate_api_key: Optional[str]
    weaviate_persist_path: Path
    hf_token: Optional[str]
    vector_backend: str
    include_english: bool
    dataset_limit: Optional[int]

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()

        def _as_bool(value: Optional[str], default: bool = True) -> bool:
            if value is None:
                return default
            return value.strip().lower() not in {"0", "false", "no", "off"}

        def _as_int(value: Optional[str]) -> Optional[int]:
            if value is None or str(value).strip() == "":
                return None
            try:
                return int(str(value).strip())
            except ValueError:
                return None

        def _split_ids(value: Optional[str], default: List[str]) -> List[str]:
            if value is None:
                return default
            parts = re.split(r"[,\n]+", value)
            cleaned = [p.strip() for p in parts if p and p.strip()]
            return cleaned or default

        return cls(
            hindi_datasets=_split_ids(os.getenv("DATASET_ID"), ["Sourabh2/Hindi_Poems"]),
            english_datasets=_split_ids(
                os.getenv("EN_DATASET_ID"), ["Santarabantoosoo/hf_song_lyrics_with_names"]
            ),
            embed_model=os.getenv(
                "EMBED_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            faiss_dir=Path(os.getenv("FAISS_INDEX_PATH", "artifacts/faiss_index")),
            weaviate_url=os.getenv("WEAVIATE_URL"),
            weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
            weaviate_persist_path=Path(os.getenv("WEAVIATE_PERSIST_PATH", ".weaviate")),
            hf_token=os.getenv("HUGGINGFACE_TOKEN"),
            vector_backend=os.getenv("VECTOR_BACKEND", "faiss").lower(),
            include_english=_as_bool(os.getenv("INCLUDE_ENGLISH"), True),
            dataset_limit=_as_int(os.getenv("DATASET_LIMIT")),
        )


def _safe_get(row, key, default=""):
    if isinstance(row, dict):
        return row.get(key, default)
    return row[key] if key in row else default


def _is_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text))


def _is_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text)) and not _is_devanagari(text)


HINGLISH_HINTS = {
    "prem",
    "geet",
    "pyar",
    "pyaar",
    "ishq",
    "mohabbat",
    "dil",
    "yaar",
    "dosti",
    "tum",
    "hum",
    "tera",
    "teri",
    "mera",
    "meri",
    "kya",
    "kaise",
    "safar",
    "zindagi",
    "yaad",
    "yaadein",
    "sapna",
    "sapne",
    "raat",
    "sajan",
    "sajna",
    "saajan",
    "barsaat",
    "baarish",
    "khwaab",
    "khwab",
}


def _looks_hinglish(text: str) -> bool:
    if _is_devanagari(text):
        return False
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    return any(tok in HINGLISH_HINTS for tok in tokens)


def _score_value(score):
    if score is None:
        return float("-inf")
    try:
        return float(score)
    except Exception:
        return float("-inf")


def group_unique_results(pairs, top_k: int):
    """Group by song/poet so we don't surface identical titles multiple times.

    For English, we group by normalized title.
    For Hindi, we group by poet + first 32 chars of the display_text (acts like a snippet id).
    Keeps best-scoring item per group and then ranks by score descending.
    """

    grouped = {}
    order = []
    for doc, score in pairs:
        meta = doc.metadata or {}
        lang = (meta.get("language") or "").lower()
        display_text = meta.get("display_text", doc.page_content)
        if lang.startswith("en"):
            title = (meta.get("title") or "").strip().lower()
            key = f"en::{title}" if title else f"en::{display_text[:40].lower()}"
        else:
            snippet = display_text[:32].strip().lower()
            poet = (meta.get("poet") or "").strip().lower()
            key = f"hi::{poet}::{snippet}"

        best = grouped.get(key)
        if best is None:
            grouped[key] = (doc, score)
            order.append(key)
        else:
            _, best_score = best
            if _score_value(score) > _score_value(best_score):
                grouped[key] = (doc, score)

    # Preserve insertion order but sort by score descending within that
    ordered_pairs = sorted(
        (grouped[k] for k in order),
        key=lambda pair: _score_value(pair[1]),
        reverse=True,
    )
    return ordered_pairs[:top_k]


def transliterate_to_devanagari(text: str) -> str:
    try:
        return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
    except Exception:
        return text


def transliterate_to_hinglish(text: str) -> str:
    try:
        return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    except Exception:
        return ""


def augment_with_hinglish(text: str) -> str:
    hinglish = transliterate_to_hinglish(text)
    return f"{text}\n{hinglish}" if hinglish else text


def load_poems(dataset_id: str, limit: Optional[int] = None) -> List[Document]:
    try:
        ds = load_dataset(dataset_id, split="train")
    except DatasetNotFoundError:
        print(f"[warn] Dataset not found or inaccessible: {dataset_id} — skipping")
        return []
    except DatasetGenerationError as exc:
        print(f"[warn] Dataset generation failed for {dataset_id}: {exc} — skipping")
        return []
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    for row in ds:
        text = _safe_get(row, "Poem Text") or _safe_get(row, "poem") or ""
        if not text:
            continue
        poet = _safe_get(row, "Poet's Name", "अज्ञात कवि")
        period = _safe_get(row, "Period", "")
        language = "hi"
        meta = {
            "poet": poet.strip(),
            "period": str(period).strip(),
            "language": language,
            "display_text": None,
        }
        for chunk in splitter.split_text(text):
            meta_chunk = dict(meta)
            meta_chunk["display_text"] = chunk
            doc_text = augment_with_hinglish(chunk)
            docs.append(Document(page_content=doc_text, metadata=meta_chunk))
    return docs


def load_english_songs(dataset_id: str, limit: Optional[int] = None) -> List[Document]:
    try:
        ds = load_dataset(dataset_id, split="train")
    except DatasetNotFoundError:
        print(f"[warn] Dataset not found or inaccessible: {dataset_id} — skipping")
        return []
    except DatasetGenerationError as exc:
        print(f"[warn] Dataset generation failed for {dataset_id}: {exc} — skipping")
        return []
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    for row in ds:
        text = _safe_get(row, "Lyric") or _safe_get(row, "lyrics") or _safe_get(row, "text") or ""
        if not text:
            continue
        title = _safe_get(row, "SName", _safe_get(row, "title", ""))
        meta = {
            "title": title.strip() if isinstance(title, str) else title,
            "language": "en",
            "display_text": None,
        }
        for chunk in splitter.split_text(text):
            meta_chunk = dict(meta)
            meta_chunk["display_text"] = chunk
            docs.append(Document(page_content=chunk, metadata=meta_chunk))
    return docs


def build_embeddings(model_name: str, hf_token: Optional[str] = None) -> HuggingFaceEmbeddings:
    # Normalize embeddings so FAISS can use inner-product as cosine similarity.
    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=os.path.join(str(Path.home()), ".cache", "hf"),
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"use_auth_token": hf_token} if hf_token else {},
    )


def load_or_build_faiss(
    docs: List[Document], embeddings: HuggingFaceEmbeddings, path: Path, rebuild: bool = False
):
    path.mkdir(parents=True, exist_ok=True)
    index_file = path / "index.faiss"
    if rebuild and path.exists():
        shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    if index_file.exists():
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(str(path))
    meta_path = path / "meta.json"
    try:
        meta_path.write_text(json.dumps({"count": len(docs)}))
    except Exception:
        pass
    return store


def load_faiss_meta(path: Path) -> Optional[int]:
    meta_path = path / "meta.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text())
        return int(data.get("count", 0))
    except Exception:
        return None


def faiss_index_exists(path: Path) -> bool:
    return (path / "index.faiss").exists()


def connect_weaviate(settings: Settings):
    if settings.weaviate_url:
        auth = weaviate.AuthApiKey(api_key=settings.weaviate_api_key) if settings.weaviate_api_key else None
        return weaviate.Client(url=settings.weaviate_url, auth_client_secret=auth)
    try:
        from weaviate.embedded import EmbeddedOptions

        settings.weaviate_persist_path.mkdir(parents=True, exist_ok=True)
        return weaviate.Client(embedded_options=EmbeddedOptions(
            persistence_data_path=str(settings.weaviate_persist_path)
        ))
    except Exception as exc:  # pragma: no cover - optional path
        raise RuntimeError("Embedded Weaviate is not available on this platform.") from exc


def build_weaviate(docs: List[Document], embeddings: HuggingFaceEmbeddings, client: weaviate.Client):
    index_name = "HindiPoems"
    try:
        # Clean existing schema if present to rebuild
        if client.schema.exists(index_name):
            client.schema.delete_class(index_name)
    except Exception:
        pass

    return Weaviate.from_documents(
        docs,
        embedding=embeddings,
        client=client,
        by_text=False,
        index_name=index_name,
    )


def similarity_search(store, query: str, k: int = 5):
    try:
        return store.similarity_search_with_score(query, k=k)
    except Exception:
        docs = store.similarity_search(query, k=k)
        return [(doc, None) for doc in docs]


def prepare_query(raw_query: str) -> str:
    if not raw_query:
        return raw_query
    augmented_parts = [raw_query]
    if _is_latin(raw_query):
        dev = transliterate_to_devanagari(raw_query)
        if dev and dev != raw_query:
            augmented_parts.append(dev)
    return " | ".join(augmented_parts)


def main():
    parser = argparse.ArgumentParser(description="Hindi song/poem recommender using LangChain")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the vector index")
    parser.add_argument("--query", type=str, help="Hindi/Hinglish/English query to search for similar songs/poems")
    parser.add_argument("--backend", choices=["faiss", "weaviate"], help="Vector backend to use")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows to load (per dataset)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of recommendations to return")
    parser.add_argument("--no-english", action="store_true", help="Skip English lyrics corpus")
    parser.add_argument("--include-english", action="store_true", help="Force include English lyrics corpus")
    parser.add_argument("--lang", choices=["auto", "hi", "en", "both"], default="auto", help="Force language focus (auto routes devanagari to Hindi)")
    args = parser.parse_args()

    settings = Settings.load()
    backend = (args.backend or settings.vector_backend).lower()
    print(f"Using backend: {backend}")

    include_english = args.include_english or (settings.include_english and not args.no_english)
    dataset_limit = args.limit if args.limit is not None else settings.dataset_limit

    combined_path = settings.faiss_dir
    hi_path = settings.faiss_dir / "hi"
    en_path = settings.faiss_dir / "en"

    fast_path_possible = (
        not args.rebuild
        and faiss_index_exists(combined_path)
        and faiss_index_exists(hi_path)
        and (not include_english or faiss_index_exists(en_path))
    )

    hi_docs: List[Document] = []
    en_docs: List[Document] = []

    if fast_path_possible:
        print("Using existing indexes; skipping dataset load. Use --rebuild to refresh.")
    else:
        print("Loading documents…")
        for ds_id in settings.hindi_datasets:
            hi_docs.extend(load_poems(ds_id, limit=dataset_limit))
        if include_english:
            for ds_id in settings.english_datasets:
                en_docs.extend(load_english_songs(ds_id, limit=dataset_limit))
        docs = hi_docs + en_docs
        print(f"Loaded {len(docs)} chunks")

    print("Loading embeddings… this may download the model on first run")
    embeddings = build_embeddings(settings.embed_model, settings.hf_token)

    store = None
    hi_store = None
    en_store = None
    client = None

    if backend == "weaviate":
        try:
            client = connect_weaviate(settings)
            if fast_path_possible:
                store = None
            else:
                store = build_weaviate(hi_docs + en_docs, embeddings, client)
            print("Weaviate index ready")
        except Exception as exc:
            print(f"Weaviate unavailable ({exc}), falling back to FAISS")
            backend = "faiss"

    if backend == "faiss":
        if fast_path_possible:
            hi_store = load_or_build_faiss([], embeddings, hi_path, rebuild=False) if faiss_index_exists(hi_path) else None
            if include_english and faiss_index_exists(en_path):
                en_store = load_or_build_faiss([], embeddings, en_path, rebuild=False)
            store = load_or_build_faiss([], embeddings, combined_path, rebuild=False)
        else:
            if hi_docs:
                hi_store = load_or_build_faiss(hi_docs, embeddings, hi_path, rebuild=args.rebuild)
            if en_docs:
                en_store = load_or_build_faiss(en_docs, embeddings, en_path, rebuild=args.rebuild)
            store = load_or_build_faiss(hi_docs + en_docs, embeddings, combined_path, rebuild=args.rebuild)
        print(f"FAISS index ready at {settings.faiss_dir}")

    if not args.query:
        print("No query provided. Use --query to search for recommendations.")
        return

    print("Searching…")
    search_query = prepare_query(args.query)
    fetch_k = max(args.top_k * 3, args.top_k + 5)
    lang_pref = (getattr(args, "lang", None) or "auto").lower()
    hinglish_hint = _looks_hinglish(args.query)

    def merge_results(primary, secondary):
        seen = set()
        merged = []
        for pair in primary + secondary:
            key = pair[0].page_content
            if key in seen:
                continue
            seen.add(key)
            merged.append(pair)
            if len(merged) >= args.top_k:
                break
        return merged

    result_pairs = []
    if backend == "faiss":
        if lang_pref == "hi" and hi_store:
            result_pairs = similarity_search(hi_store, search_query, k=fetch_k)
        elif lang_pref == "en" and en_store:
            result_pairs = similarity_search(en_store, search_query, k=fetch_k)
        else:
            combined_results = similarity_search(store, search_query, k=fetch_k)
            if lang_pref == "auto" and hi_store and (_is_devanagari(args.query) or hinglish_hint):
                hi_results = similarity_search(hi_store, search_query, k=fetch_k)
                result_pairs = merge_results(hi_results, combined_results)
            else:
                result_pairs = combined_results
    else:
        result_pairs = similarity_search(store, search_query, k=fetch_k)

    result_pairs = group_unique_results(result_pairs, args.top_k)

    for idx, (doc, score) in enumerate(result_pairs, start=1):
        meta = doc.metadata or {}
        poet = meta.get("poet", "अज्ञात")
        period = meta.get("period", "")
        language = meta.get("language", "")
        display_text = meta.get("display_text", doc.page_content)
        hinglish_line = transliterate_to_hinglish(display_text) if _is_devanagari(display_text) else None
        print("-" * 80)
        print(f"सिफ़ारिश #{idx}")
        if language.lower().startswith("en"):
            title = meta.get("title", "")
            prefix = f"Song: {title}" if title else "English lyric"
            print(prefix)
            print(display_text.strip())
        else:
            print(f"कवि: {poet} | काल: {period}")
            print(display_text.strip())
            if hinglish_line:
                print(f"(Hinglish) {hinglish_line.strip()}")

    if client:
        client.close()


if __name__ == "__main__":
    main()
