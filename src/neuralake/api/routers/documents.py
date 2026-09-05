import asyncio
import logging
import mimetypes
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from neuralake.api.dependencies import get_collection_repo, get_document_repo
from neuralake.api.middleware.auth import AuthContext, get_auth_context
from neuralake.api.schemas.common import PaginationParams
from neuralake.api.schemas.requests import IngestDocumentRequest
from neuralake.api.schemas.responses import DocumentListResponse, DocumentResponse
from neuralake.storage.repositories.base import BaseRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections/{collection_id}/documents", tags=["documents"])

MIME_TO_SOURCE_TYPE = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/html": "html",
    "text/markdown": "markdown",
    "text/plain": "text",
}

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


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

    from neuralake.core.ingestion.pipeline import check_duplicate, compute_content_hash

    content_hash = compute_content_hash(body.content)
    duplicate = await check_duplicate(content_hash, collection_id, auth.tenant_id, doc_repo.session)
    if duplicate:
        raise HTTPException(status_code=409, detail="Duplicate content already ingested")

    doc = await doc_repo.create(
        tenant_id=auth.tenant_id,
        collection_id=collection_id,
        title=body.title,
        source_type=body.source_type,
        source_uri=body.source_uri,
        content_hash=content_hash,
        raw_size_bytes=len(body.content.encode()),
        status="processing",
        metadata=body.metadata,
    )

    async def _ingest():
        from neuralake.core.ingestion.pipeline import IngestionPipeline
        from neuralake.storage.database import get_session_factory

        factory = get_session_factory()
        async with factory() as db:
            try:
                pipeline = IngestionPipeline()
                coll = await collection_repo.get(collection_id)
                chunk_count = await pipeline.process(
                    content=body.content,
                    source_type=body.source_type,
                    document_id=doc.id,
                    collection=coll,
                    tenant_id=auth.tenant_id,
                    db=db,
                )
                from sqlalchemy import update
                from neuralake.storage.models.document import Document

                await db.execute(
                    update(Document)
                    .where(Document.id == doc.id)
                    .values(status="completed", chunk_count=chunk_count)
                )
                await db.commit()
            except Exception as e:
                logger.error("Ingestion failed for doc %s: %s", doc.id, e)
                from sqlalchemy import update
                from neuralake.storage.models.document import Document

                await db.execute(
                    update(Document)
                    .where(Document.id == doc.id)
                    .values(status="failed", error_message=str(e)[:2000])
                )
                await db.commit()

    asyncio.create_task(_ingest())

    refreshed = await doc_repo.get(doc.id)
    return DocumentResponse.model_validate(refreshed)


@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    collection_id: uuid.UUID,
    file: UploadFile,
    auth: AuthContext = Depends(get_auth_context),
    collection_repo: BaseRepository = Depends(get_collection_repo),
    doc_repo: BaseRepository = Depends(get_document_repo),
):
    collection = await collection_repo.get(collection_id)
    if not collection or collection.tenant_id != auth.tenant_id:
        raise HTTPException(status_code=404, detail="Collection not found")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit")

    mime = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "text/plain"
    source_type = MIME_TO_SOURCE_TYPE.get(mime, "text")

    from neuralake.core.ingestion.pipeline import check_duplicate, compute_content_hash

    content_hash = compute_content_hash(raw)
    duplicate = await check_duplicate(content_hash, collection_id, auth.tenant_id, doc_repo.session)
    if duplicate:
        raise HTTPException(status_code=409, detail="Duplicate content already ingested")

    title = file.filename or "Untitled"

    doc = await doc_repo.create(
        tenant_id=auth.tenant_id,
        collection_id=collection_id,
        title=title,
        source_type=source_type,
        content_hash=content_hash,
        raw_size_bytes=len(raw),
        status="processing",
        metadata={},
    )

    async def _ingest():
        from neuralake.core.ingestion.pipeline import IngestionPipeline
        from neuralake.storage.database import get_session_factory

        factory = get_session_factory()
        async with factory() as db:
            try:
                pipeline = IngestionPipeline()
                coll = await collection_repo.get(collection_id)
                content = raw if source_type in ("pdf", "docx") else raw.decode("utf-8", errors="replace")
                chunk_count = await pipeline.process(
                    content=content,
                    source_type=source_type,
                    document_id=doc.id,
                    collection=coll,
                    tenant_id=auth.tenant_id,
                    db=db,
                )
                from sqlalchemy import update
                from neuralake.storage.models.document import Document

                await db.execute(
                    update(Document)
                    .where(Document.id == doc.id)
                    .values(status="completed", chunk_count=chunk_count)
                )
                await db.commit()
            except Exception as e:
                logger.error("Upload ingestion failed for doc %s: %s", doc.id, e)
                from sqlalchemy import update
                from neuralake.storage.models.document import Document

                await db.execute(
                    update(Document)
                    .where(Document.id == doc.id)
                    .values(status="failed", error_message=str(e)[:2000])
                )
                await db.commit()

    asyncio.create_task(_ingest())

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
