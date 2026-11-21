import os
import re
from datetime import datetime
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError

from app.auth import check_collection_access, current_user_id
from app.db import SessionLocal
from app.ingest import ingest_pdf_bytes
from app.llm import get_client
from app.models import Collection, Conversation, Message, User
from app.rag import answer_question, format_pages
from app.retrieval import DEFAULT_COLLECTION
from app.security import hash_password, make_token, verify_password

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
USERNAME_PATTERN = re.compile(r"[a-z0-9_]+")

app = FastAPI(title="EKIP RAG")


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=30)
    password: str = Field(min_length=8, max_length=128)


class TokenOut(BaseModel):
    token: str
    username: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["hybrid", "rerank"] = "hybrid"
    conversation_id: int | None = None
    collection_id: int | None = None


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


class CollectionDocOut(BaseModel):
    file_name: str
    title: str
    pages: int


class CollectionOut(BaseModel):
    id: int
    name: str
    read_only: bool
    documents: list[CollectionDocOut]


def owned_conversation(session, conversation_id: int, user_id: int) -> Conversation:
    conversation = session.get(Conversation, conversation_id)
    if conversation is None or conversation.owner_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register", response_model=TokenOut)
def register(credentials: Credentials):
    username = credentials.username.strip().lower()
    if not USERNAME_PATTERN.fullmatch(username):
        raise HTTPException(status_code=400, detail="Use only letters, digits and underscores in the username")
    with SessionLocal() as session:
        user = User(username=username, password_hash=hash_password(credentials.password))
        session.add(user)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            raise HTTPException(status_code=400, detail="That username is taken")
        return TokenOut(token=make_token(user.id), username=username)


@app.post("/auth/login", response_model=TokenOut)
def login(credentials: Credentials):
    username = credentials.username.strip().lower()
    with SessionLocal() as session:
        user = session.scalar(select(User).where(User.username == username))
        if user is None or not verify_password(credentials.password, user.password_hash):
            # the same message for both cases, so nobody can find out which usernames exist
            raise HTTPException(status_code=401, detail="Incorrect username or password")
        return TokenOut(token=make_token(user.id), username=username)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, user_id: int = Depends(current_user_id)):
    with SessionLocal() as session:
        if request.collection_id is not None:
            check_collection_access(session, request.collection_id, user_id)
        if request.conversation_id:
            owned_conversation(session, request.conversation_id, user_id)

    result = answer_question(request.question, mode=request.mode, collection_id=request.collection_id)
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
            conversation = Conversation(owner_id=user_id)
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
def feedback(request: FeedbackRequest, user_id: int = Depends(current_user_id)):
    with SessionLocal() as session:
        message = session.get(Message, request.message_id)
        if message is None or message.role != "assistant" or message.conversation.owner_id != user_id:
            raise HTTPException(status_code=404, detail="Message not found")
        message.rating = request.rating
        session.commit()
    return {"status": "saved"}


@app.get("/conversations", response_model=list[ConversationOut])
def list_conversations(user_id: int = Depends(current_user_id)):
    with SessionLocal() as session:
        conversations = session.scalars(
            select(Conversation)
            .where(Conversation.owner_id == user_id)
            .order_by(Conversation.id.desc())
            .limit(30)
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
def get_conversation(conversation_id: int, user_id: int = Depends(current_user_id)):
    with SessionLocal() as session:
        conversation = owned_conversation(session, conversation_id, user_id)
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
def delete_conversation(conversation_id: int, user_id: int = Depends(current_user_id)):
    with SessionLocal() as session:
        session.delete(owned_conversation(session, conversation_id, user_id))
        session.commit()
    return {"status": "deleted"}


@app.get("/collections", response_model=list[CollectionOut])
def list_collections(user_id: int = Depends(current_user_id)):
    with SessionLocal() as session:
        collections = session.scalars(
            select(Collection)
            .where(or_(Collection.owner_id.is_(None), Collection.owner_id == user_id))
            .order_by(Collection.id)
        ).all()
        return [
            CollectionOut(
                id=c.id,
                name=c.name,
                read_only=c.owner_id is None,
                documents=[
                    CollectionDocOut(file_name=d.file_name, title=d.title, pages=d.page_count)
                    for d in sorted(c.documents, key=lambda d: d.file_name)
                ],
            )
            for c in collections
        ]


@app.post("/upload")
def upload(
    collection: str = Form(...),
    files: list[UploadFile] = File(...),
    user_id: int = Depends(current_user_id),
):
    name = " ".join(collection.split())
    if not name or len(name) > 80:
        raise HTTPException(status_code=400, detail="Give the collection a name of 1 to 80 characters")
    if name == DEFAULT_COLLECTION:
        raise HTTPException(status_code=400, detail="That collection is read-only. Choose another name")

    results = []
    for upload_file in files:
        label = upload_file.filename or "(no name)"
        if not label.lower().endswith(".pdf"):
            results.append({"file": label, "status": "skipped: not a PDF"})
            continue
        data = upload_file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            results.append({"file": label, "status": "skipped: larger than 25 MB"})
            continue
        try:
            outcome = ingest_pdf_bytes(data, label, name, owner_id=user_id)
        except ValueError as error:
            results.append({"file": label, "status": f"failed: {error}"})
        except Exception:
            results.append({"file": label, "status": "failed: this PDF could not be read"})
        else:
            if outcome is None:
                results.append({"file": label, "status": "already in this collection"})
            else:
                results.append({"file": label, "status": f"added ({outcome[1]} chunks)"})
    return {"collection": name, "results": results}


@app.delete("/collections/{collection_id}")
def delete_collection(collection_id: int, user_id: int = Depends(current_user_id)):
    with SessionLocal() as session:
        collection = session.get(Collection, collection_id)
        if collection is None or (collection.owner_id is not None and collection.owner_id != user_id):
            raise HTTPException(status_code=404, detail="Collection not found")
        if collection.owner_id is None:
            raise HTTPException(status_code=400, detail="The shared RBI collection cannot be deleted")
        session.delete(collection)
        session.commit()
    return {"status": "deleted"}


@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...), user_id: int = Depends(current_user_id)):
    data = audio.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty recording")
    try:
        result = get_client().audio.transcriptions.create(
            model=os.environ.get("STT_MODEL", "whisper-large-v3-turbo"),
            file=(audio.filename or "voice.wav", data, audio.content_type or "audio/wav"),
            language="en",
            prompt="RBI, NBFC, KYC, UAPA, SFB, small finance bank, commercial bank, "
            "loan to value ratio, lending service provider, credit facilities",
        )
    except Exception as error:  # the speech service can be busy, limited or missing the model
        raise HTTPException(status_code=503, detail=f"Speech service unavailable: {error}")
    return {"text": result.text.strip()}