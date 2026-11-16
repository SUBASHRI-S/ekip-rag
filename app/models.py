from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

EMBEDDING_DIMENSIONS = 384  # the size of the bge-small embedding model


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_name: Mapped[str] = mapped_column(unique=True)
    title: Mapped[str]
    entity_type: Mapped[str]  # commercial_bank, nbfc, small_finance_bank
    topic: Mapped[str]
    updated_as_on: Mapped[str | None]
    source_url: Mapped[str | None]
    page_count: Mapped[int]
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int]
    page_start: Mapped[int]
    page_end: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int]
    embedding = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    document: Mapped["Document"] = relationship(back_populates="chunks")