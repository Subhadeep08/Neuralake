import uuid

from fastapi import APIRouter, Depends, HTTPException

from neuralake.api.middleware.auth import AuthContext, get_auth_context
from neuralake.api.schemas.common import PaginationParams
from neuralake.api.schemas.requests import IngestDocumentRequest
from neuralake.api.schemas.responses import DocumentListResponse, DocumentResponse
from neuralake.api.dependencies import get_collection_repo, get_document_repo
from neuralake.storage.repositories.base import BaseRepository

router = APIRouter(prefix="/collections/{collection_id}/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=201)
async def ingest_document(
    collection_id: uuid.UUID,
    body: IngestDocumentRequest,
    auth: AuthContext = Depends(get_auth_context),
    collection_repo: BaseRepository = Depends(get_collection_repo),
    doc_repo: BaseRepository = Depends(get_document_repo),
):
    collection = await collection_repo.get(collection_id)
    if not collection or collection.tenant_id != auth.tenant_id:
        raise HTTPException(status_code=404, detail="Collection not found")

    doc = await doc_repo.create(
        tenant_id=auth.tenant_id,
        collection_id=collection_id,
        title=body.title,
        source_type=body.source_type,
        source_uri=body.source_uri,
        raw_size_bytes=len(body.content.encode()),
        status="pending",
        metadata=body.metadata,
    )

    from neuralake.core.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    try:
        chunk_count = await pipeline.process_text(
            text=body.content,
            document_id=doc.id,
            collection=collection,
            tenant_id=auth.tenant_id,
            db=doc_repo.session,
        )
        await doc_repo.update(doc.id, status="completed", chunk_count=chunk_count)
    except Exception as e:
        await doc_repo.update(doc.id, status="failed", error_message=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    refreshed = await doc_repo.get(doc.id)
    return DocumentResponse.model_validate(refreshed)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    collection_id: uuid.UUID,
    pagination: PaginationParams = Depends(),
    auth: AuthContext = Depends(get_auth_context),
    doc_repo: BaseRepository = Depends(get_document_repo),
):
    items, total = await doc_repo.list(
        tenant_id=auth.tenant_id,
        collection_id=collection_id,
        offset=pagination.offset,
        limit=pagination.limit,
    )
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(i) for i in items],
        total=total,
        offset=pagination.offset,
        limit=pagination.limit,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    doc_repo: BaseRepository = Depends(get_document_repo),
):
    doc = await doc_repo.get(document_id)
    if not doc or doc.tenant_id != auth.tenant_id or doc.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse.model_validate(doc)


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    doc_repo: BaseRepository = Depends(get_document_repo),
):
    doc = await doc_repo.get(document_id)
    if not doc or doc.tenant_id != auth.tenant_id or doc.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="Document not found")
    await doc_repo.delete(document_id)
