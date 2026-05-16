from __future__ import annotations

import glob as globmod
import os
import re
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sentence_transformers import SentenceTransformer

DEFAULT_DATA_DIR = "data"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_TOP_K = 4

DOCUMENT_FOLDERS = {
    "emails": "email",
    "notes": "note",
    "sms": "sms",
    "calendar": "calendar",
}

TAG_ALIASES = {
    "/email": "emails",
    "/emails": "emails",
    "/note": "notes",
    "/notes": "notes",
    "/sms": "sms",
    "/calendar": "calendar",
    "/calendars": "calendar",
}

MIN_RELEVANCE_SCORE = 0.12


def _parse_int_setting(name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    return parsed


def resolve_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolves runtime configuration with defaults and typed settings."""
    config = config or {}

    resolved = {
        "api_key": config.get("api_key", None),
        "base_url": config.get("base_url", None),
        "model": config.get("model", DEFAULT_LLM_MODEL),
        "embedding_model": config.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "top_k": _parse_int_setting(
            "TOP_K",
            config.get("top_k", DEFAULT_TOP_K),
        ),
        "chunk_size": _parse_int_setting(
            "CHUNK_SIZE",
            config.get("chunk_size", DEFAULT_CHUNK_SIZE),
        ),
        "chunk_overlap": _parse_int_setting(
            "CHUNK_OVERLAP",
            config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
        ),
    }

    if resolved["top_k"] <= 0:
        raise ValueError("TOP_K must be > 0")
    if resolved["chunk_size"] <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")
    if resolved["chunk_overlap"] < 0:
        raise ValueError("CHUNK_OVERLAP must be >= 0")
    if resolved["chunk_overlap"] >= resolved["chunk_size"]:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    return resolved


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as file:
            return file.read().strip()


def _normalize_path(path: str) -> str:
    return path.replace(os.sep, "/")


def load_documents(data_dir: str = DEFAULT_DATA_DIR) -> list[Document]:
    """Loads documents from the personal data folders.

    The collection contains one LangChain Document per `.txt` file in the
    emails, notes, SMS, and calendar folders. Each document stores the file text
    as `page_content` and includes metadata for the source file path and
    document type.
    """
    docs: list[Document] = []

    for folder_name, document_type in DOCUMENT_FOLDERS.items():
        folder_path = os.path.join(data_dir, folder_name)
        pattern = os.path.join(folder_path, "**", "*.txt")

        for file_path in sorted(globmod.glob(pattern, recursive=True)):
            normalized_path = _normalize_path(file_path)
            text = _read_text_file(file_path)
            metadata = {
                "source": normalized_path,
                "path": normalized_path,
                "file_path": normalized_path,
                "filename": os.path.basename(file_path),
                "type": folder_name,
                "doc_type": document_type,
                "category": folder_name,
            }
            docs.append(Document(page_content=text, metadata=metadata))

    return docs


def split_documents(
        docs: list[Document],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Splits documents into overlapping chunks.

    The resulting chunked Document objects use the configured chunk size and
    overlap while preserving the original document metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    chunk_counts_by_source: dict[str, int] = {}
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        source = str(metadata.get("source", "unknown"))
        chunk_index = chunk_counts_by_source.get(source, 0)
        chunk_counts_by_source[source] = chunk_index + 1
        metadata["chunk_index"] = chunk_index
        metadata["chunk_id"] = f"{source}#chunk-{chunk_index}"
        chunk.metadata = metadata

    return chunks


def _encode_texts(
        embedding_model: SentenceTransformer,
        texts: list[str],
) -> np.ndarray:
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype="float32")


def build_index(
        chunks: list[Document],
        embedding_model: SentenceTransformer,
) -> faiss.IndexFlatIP:
    """Creates a FAISS inner-product index for embedded document chunks.

    The index contains normalized float32 embeddings generated from each
    chunk's text with the provided embedding model.
    """
    dimension = embedding_model.get_sentence_embedding_dimension()
    if dimension is None:
        test_embedding = _encode_texts(embedding_model, ["dimension probe"])
        dimension = int(test_embedding.shape[1])

    index = faiss.IndexFlatIP(int(dimension))
    if not chunks:
        return index

    texts = [chunk.page_content for chunk in chunks]
    embeddings = _encode_texts(embedding_model, texts)
    index.add(embeddings)
    return index


def retrieve(
        query: str,
        index: faiss.IndexFlatIP,
        model: SentenceTransformer,
        chunks: list[Document],
        k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Gets the most relevant chunks for a query.

    Results are ordered by similarity and include the chunk text, similarity
    score, and metadata for each matching chunk.
    """
    if not query.strip() or not chunks or index.ntotal == 0:
        return []

    search_k = min(max(1, int(k)), index.ntotal)
    query_embedding = _encode_texts(model, [query])
    scores, indices = index.search(query_embedding, search_k)

    results: list[dict] = []
    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0 or index_position >= len(chunks):
            continue

        chunk = chunks[int(index_position)]
        results.append(
            {
                "text": chunk.page_content,
                "score": float(score),
                "metadata": dict(chunk.metadata),
            }
        )

    return results


SYSTEM_PROMPT = """
You are a personal digital assistant that answers questions using Retrieval-Augmented Generation.
Use only the retrieved context and the conversation history provided in the messages.
Do not invent information that is not supported by the retrieved context.
If the context does not contain enough information to answer, say that you do not have enough information in the available documents.
Answer in the same language as the user's latest question.
Be concise, but include the specific details that answer the question.
""".strip()


def _extract_tag_filter(question: str) -> tuple[str | None, str]:
    selected_category: str | None = None
    cleaned_question = question

    for tag, category in TAG_ALIASES.items():
        pattern = re.compile(rf"(?<!\w){re.escape(tag)}(?!\w)", re.IGNORECASE)
        if pattern.search(cleaned_question):
            selected_category = category
            cleaned_question = pattern.sub(" ", cleaned_question)

    cleaned_question = re.sub(r"\s+", " ", cleaned_question).strip()
    return selected_category, cleaned_question or question


def _metadata_matches_category(metadata: dict[str, Any], category: str) -> bool:
    values = {
        str(metadata.get("type", "")).lower(),
        str(metadata.get("category", "")).lower(),
        str(metadata.get("doc_type", "")).lower(),
    }

    if category == "emails":
        return bool(values.intersection({"email", "emails"}))
    if category == "notes":
        return bool(values.intersection({"note", "notes"}))
    if category == "sms":
        return "sms" in values
    if category == "calendar":
        return "calendar" in values

    return False


def _tokenize(text: str) -> set[str]:
    stop_words = {
        "the", "and", "for", "that", "this", "with", "from", "about", "what",
        "when", "where", "which", "should", "would", "could", "have", "has",
        "que", "para", "con", "una", "uno", "del", "las", "los", "por",
        "como", "donde", "cuál", "cual", "qué", "cuando", "sobre", "debo",
    }
    tokens = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ0-9]+", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in stop_words}


def _lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize(text)
    return len(query_tokens.intersection(text_tokens)) / len(query_tokens)


def _unique_references(results: list[dict]) -> list[str]:
    references: list[str] = []
    seen: set[str] = set()

    for result in results:
        metadata = result.get("metadata", {})
        source = str(
            metadata.get("source")
            or metadata.get("path")
            or metadata.get("file_path")
            or "unknown source"
        )
        if source not in seen:
            seen.add(source)
            references.append(source)

    return references


def _format_context(results: list[dict]) -> str:
    context_blocks: list[str] = []

    for number, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        source = metadata.get("source") or metadata.get("path") or "unknown source"
        doc_type = metadata.get("type") or metadata.get("doc_type") or "unknown"
        score = result.get("score", 0.0)
        text = result.get("text", "")
        context_blocks.append(
            f"[{number}] Source: {source}\n"
            f"Type: {doc_type}\n"
            f"Score: {score:.4f}\n"
            f"Text:\n{text}"
        )

    return "\n\n".join(context_blocks)


def _format_references(references: list[str]) -> str:
    if not references:
        return ""
    return "References:\n" + "\n".join(f"- {reference}" for reference in references)


class Assistant:
    """Stateful RAG assistant.

    The assistant owns the pipeline components, resolved configuration, and
    conversation history. Questions are answered with retrieved document context
    and the configured chat model.
    """

    def __init__(
            self,
            index: faiss.IndexFlatIP,
            model: SentenceTransformer,
            chunks: list[Document],
            client: OpenAI,
            config: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.model = model
        self.chunks = chunks
        self.client = client
        self.config = resolve_config(config)
        self.llm_model = self.config["model"]
        self.top_k = self.config["top_k"]
        self.history: list[dict[str, str]] = []

    def _expanded_query(self, question: str) -> str:
        recent_messages = self.history[-6:]
        recent_text = " ".join(
            message["content"] for message in recent_messages
            if message.get("role") in {"user", "assistant"}
        )
        if not recent_text:
            return question
        return f"{recent_text}\nCurrent question: {question}"

    def _rerank(self, query: str, results: list[dict], k: int) -> list[dict]:
        scored_results: list[dict] = []

        for result in results:
            embedding_score = float(result.get("score", 0.0))
            lexical_score = _lexical_overlap_score(query, str(result.get("text", "")))
            rerank_score = embedding_score + (0.15 * lexical_score)
            copy = dict(result)
            copy["rerank_score"] = rerank_score
            scored_results.append(copy)

        scored_results.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored_results[:k]

    def ask(self, question: str, k: int | None = None) -> str:
        """Generates an answer from the retrieved context and conversation history.

        The current question is combined with relevant document chunks, previous
        conversation messages, and the system prompt. The assistant response is
        appended to history alongside the user message.
        """
        selected_category, cleaned_question = _extract_tag_filter(question)
        requested_k = k or self.top_k
        overfetch_k = max(requested_k * 5, requested_k + 8) if selected_category else max(requested_k * 3, requested_k)
        expanded_query = self._expanded_query(cleaned_question)

        results = retrieve(
            expanded_query,
            self.index,
            self.model,
            self.chunks,
            k=overfetch_k,
        )

        if selected_category:
            results = [
                result for result in results
                if _metadata_matches_category(result.get("metadata", {}), selected_category)
            ]

        results = self._rerank(cleaned_question, results, requested_k)
        results = [result for result in results if float(result.get("score", 0.0)) >= MIN_RELEVANCE_SCORE]

        if not results:
            answer = "No encontré documentos relevantes para responder eso con la información disponible."
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})
            return answer

        context = _format_context(results)
        references = _unique_references(results)
        tag_note = f"\nSearch filter requested by user: {selected_category}" if selected_category else ""

        user_prompt = f"""
User question:
{cleaned_question}
{tag_note}

Retrieved context:
{context}

Instructions:
Answer the user question using only the retrieved context above and the previous conversation history.
If the answer cannot be supported by the retrieved context, say that you do not have enough information in the available documents.
Do not mention scores unless the user asks for them.
""".strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history,
            {"role": "user", "content": user_prompt},
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.2,
            )
            answer = completion.choices[0].message.content or ""
            answer = answer.strip()
        except Exception as exc:
            answer = f"Ocurrió un error al llamar al modelo de lenguaje: {exc}"

        reference_block = _format_references(references)
        if reference_block and "reference" not in answer.lower() and "referencia" not in answer.lower():
            answer = f"{answer}\n\n{reference_block}"

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def clear_history(self) -> None:
        """Empties the conversation history."""
        self.history.clear()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> Assistant:
        """Initializes the components required by the assistant and instantiates it

        The pipeline includes resolved configuration, loaded documents, chunked
        documents, an embedding model, a FAISS index, and an OpenAI-compatible
        client.
        """
        resolved_config = resolve_config(config)

        print("Cargando documentos...")
        docs = load_documents()
        print(f"  Se cargaron {len(docs)} documentos")

        print("Partiendo en pedacitos masticables...")
        chunks = split_documents(
            docs,
            chunk_size=resolved_config["chunk_size"],
            chunk_overlap=resolved_config["chunk_overlap"],
        )
        print(f"  Se crearon {len(chunks)} predacitos")

        embedding_model = SentenceTransformer(resolved_config["embedding_model"])

        print("Construyendo el índice FAISS...")
        index = build_index(chunks, embedding_model)
        print(f"  Se indizaron {index.ntotal} vectores (de dimensión {index.d})")

        client_kwargs = {}
        if resolved_config["api_key"]:
            client_kwargs["api_key"] = resolved_config["api_key"]
        if resolved_config["base_url"]:
            client_kwargs["base_url"] = resolved_config["base_url"]
        client = OpenAI(**client_kwargs)

        print("¡Listo!\n")
        return cls(index, embedding_model, chunks, client, resolved_config)
