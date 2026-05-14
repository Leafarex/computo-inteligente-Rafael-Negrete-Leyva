from typing import Optional
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


class Document:
    def __init__(self, text: str, metadata: dict[str, str]):
        self.text = text
        self.metadata = metadata


class SearchResult:
    def __init__(self, score: float, document: Document):
        self.score = score
        self.document = document


class FilteredVectorStore:
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents: list[Document]):
        if len(documents) == 0:
            return

        texts = [document.text for document in documents]

        new_embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self.documents.extend(documents)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None
    ) -> list[SearchResult]:
        if self.embeddings is None or len(self.documents) == 0:
            return []

        if metadata_filter is None:
            candidate_indices = list(range(len(self.documents)))
        else:
            candidate_indices = []

            for index, document in enumerate(self.documents):
                matches_filter = True

                for key, value in metadata_filter.items():
                    if key not in document.metadata:
                        matches_filter = False
                        break

                    if str(document.metadata[key]) != str(value):
                        matches_filter = False
                        break

                if matches_filter:
                    candidate_indices.append(index)

        if len(candidate_indices) == 0:
            return []

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        candidate_embeddings = self.embeddings[candidate_indices]
        scores = candidate_embeddings @ query_embedding

        sorted_positions = np.argsort(scores)[::-1][:top_k]

        results = []

        for position in sorted_positions:
            original_index = candidate_indices[position]

            result = SearchResult(
                score=float(scores[position]),
                document=self.documents[original_index]
            )

            results.append(result)

        return results


class DocumentMetadata(BaseModel):
    category: str = Field(min_length=1)
    source: str = Field(min_length=1)
    author: str = Field(min_length=1)


class CreateDocumentRequest(BaseModel):
    text: str = Field(min_length=1)
    metadata: DocumentMetadata


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    metadata_filter: Optional[dict[str, str]] = None


class StoredDocument(BaseModel):
    id: str
    text: str
    metadata: DocumentMetadata


class CreateDocumentResponse(BaseModel):
    id: str
    message: str
    total_chunks: int


class SearchResponseItem(BaseModel):
    similarity_percentage: float
    score: float
    text: str
    metadata: dict[str, str]


app = FastAPI(
    title="Filtered Vector Store API",
    description="API REST para crear documentos y realizar búsquedas semánticas con filtros de metadatos.",
    version="1.0.0"
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vector_store = FilteredVectorStore(embedding_model)
original_documents: dict[str, StoredDocument] = {}


def split_document(text: str) -> list[str]:
    if len(text) <= 500:
        return [text]

    chunks = []

    for start in range(0, len(text), 400):
        chunk = text[start:start + 400]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


@app.post("/documents", response_model=CreateDocumentResponse)
def create_document(request: CreateDocumentRequest):
    document_id = str(uuid4())

    stored_document = StoredDocument(
        id=document_id,
        text=request.text,
        metadata=request.metadata
    )

    original_documents[document_id] = stored_document

    chunks = split_document(request.text)
    fragment_documents = []

    for index, chunk in enumerate(chunks):
        fragment_metadata = {
            "original_document_id": document_id,
            "chunk_index": str(index),
            "total_chunks": str(len(chunks)),
            "category": request.metadata.category,
            "source": request.metadata.source,
            "author": request.metadata.author
        }

        fragment_document = Document(
            text=chunk,
            metadata=fragment_metadata
        )

        fragment_documents.append(fragment_document)

    vector_store.add_documents(fragment_documents)

    return CreateDocumentResponse(
        id=document_id,
        message="Documento creado correctamente.",
        total_chunks=len(chunks)
    )


@app.get("/documents/{document_id}", response_model=StoredDocument)
def get_document(document_id: str):
    if document_id not in original_documents:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")

    return original_documents[document_id]


@app.post("/documents/search", response_model=list[SearchResponseItem])
def search_documents(request: SearchRequest):
    results = vector_store.search(
        query=request.query,
        top_k=request.top_k,
        metadata_filter=request.metadata_filter
    )

    response = []

    for result in results:
        similarity_percentage = max(0.0, min(100.0, result.score * 100))

        item = SearchResponseItem(
            similarity_percentage=round(similarity_percentage, 2),
            score=round(result.score, 4),
            text=result.document.text,
            metadata=result.document.metadata
        )

        response.append(item)

    return response