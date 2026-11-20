import json
import os
from collections import Counter
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db import SessionLocal
from app.llm import get_client
from app.models import Conversation, Message
from app.rag import answer_question, format_pages

app = FastAPI(title="EKIP RAG")


class DocumentOut(BaseModel):
    name: str
    pages: int


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["hybrid", "rerank"] = "hybrid"
    conversation_id: int | None = None


class SourceOut(BaseModel):
    number: int
    title: str
    file_name: str
    entity_type: str
    pages: str
    distance: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    answerable: bool
    sources: list[SourceOut]
    conversation_id: int
    message_id: int
    trace: dict


class FeedbackRequest(BaseModel):
    message_id: int
    rating: Literal[1, -1]


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime


class MessageOut(BaseModel):
    message_id: int
    role: str
    content: str
    answerable: bool | None = None
    sources: list | None = None
    trace: dict | None = None
    rating: int | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentOut])
def list_documents():
    counts = Counter()
    with open("data/processed/pages.jsonl", encoding="utf-8") as f:
        for line in f:
            counts[json.loads(line)["document"]] += 1
    return [DocumentOut(name=name, pages=pages) for name, pages in sorted(counts.items())]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = answer_question(request.question, mode=request.mode)
    sources = [
        SourceOut(
            number=number,
            title=p.title,
            file_name=p.file_name,
            entity_type=p.entity_type,
            pages=format_pages(p),
            distance=round(p.distance, 3),
            text=p.text,
        )
        for number, p in enumerate(result.passages, start=1)
    ]

    with SessionLocal() as session:
        conversation = session.get(Conversation, request.conversation_id) if request.conversation_id else None
        if conversation is None:
            conversation = Conversation()
            session.add(conversation)
            session.flush()
        session.add(Message(conversation_id=conversation.id, role="user", content=request.question))
        reply = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=result.answer,
            answerable=result.answerable,
            mode=request.mode,
            sources=[s.model_dump() for s in sources],
            trace=result.trace,
        )
        session.add(reply)
        session.flush()
        conversation_id, message_id = conversation.id, reply.id
        session.commit()

    return ChatResponse(
        answer=result.answer,
        answerable=result.answerable,
        sources=sources,
        conversation_id=conversation_id,
        message_id=message_id,
        trace=result.trace,
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    with SessionLocal() as session:
        message = session.get(Message, request.message_id)
        if message is None or message.role != "assistant":
            raise HTTPException(status_code=404, detail="Message not found")
        message.rating = request.rating
        session.commit()
    return {"status": "saved"}


@app.get("/conversations", response_model=list[ConversationOut])
def list_conversations():
    with SessionLocal() as session:
        conversations = session.scalars(
            select(Conversation).order_by(Conversation.id.desc()).limit(30)
        ).all()
        return [
            ConversationOut(
                id=c.id,
                title=next((m.content for m in c.messages if m.role == "user"), "(empty chat)")[:60],
                created_at=c.created_at,
            )
            for c in conversations
        ]


@app.get("/conversations/{conversation_id}", response_model=list[MessageOut])
def get_conversation(conversation_id: int):
    with SessionLocal() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return [
            MessageOut(
                message_id=m.id,
                role=m.role,
                content=m.content,
                answerable=m.answerable,
                sources=m.sources,
                trace=m.trace,
                rating=m.rating,
            )
            for m in conversation.messages
        ]


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: int):
    with SessionLocal() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        session.delete(conversation)
        session.commit()
    return {"status": "deleted"}


@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...)):
    data = audio.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty recording")
    try:
        result = get_client().audio.transcriptions.create(
            model=os.environ.get("STT_MODEL", "whisper-large-v3-turbo"),
            file=(audio.filename or "voice.wav", data, audio.content_type or "audio/wav"),
            language="en",
        )
    except Exception as error:  # the speech service can be busy, limited or missing the model
        raise HTTPException(status_code=503, detail=f"Speech service unavailable: {error}")
    return {"text": result.text.strip()}