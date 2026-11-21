import os
import uuid

os.environ["JWT_SECRET"] = "a-secret-used-only-in-tests"  # must be set before main is imported

import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

import main
from app.db import SessionLocal, engine
from app.models import User
from app.rag import RagAnswer
from app.retrieval import search

try:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1 FROM users LIMIT 1"))
    DATABASE_READY = True
except Exception:
    DATABASE_READY = False

pytestmark = pytest.mark.skipif(not DATABASE_READY, reason="needs the local PostgreSQL database")

SECRET_SENTENCE = "The zebra clause says the maximum penalty is 42 rupees per day."


def make_pdf(content: str) -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), content)
    data = doc.tobytes()
    doc.close()
    return data


def fake_answer(question, top_k=5, mode="hybrid", collection_id=None):
    return RagAnswer(
        question=question, answer="A test answer.", answerable=True, passages=[], trace={"test": True}
    )


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(main, "answer_question", fake_answer)  # no real LLM, no tokens
        client = TestClient(main.app)
        names = [f"iso_{uuid.uuid4().hex[:10]}" for _ in range(2)]
        headers = []
        for name in names:
            response = client.post("/auth/register", json={"username": name, "password": "test-password-123"})
            assert response.status_code == 200
            headers.append({"Authorization": f"Bearer {response.json()['token']}"})
        alice, bob = headers

        upload = client.post(
            "/upload",
            data={"collection": "Alice docs"},
            files=[("files", ("secret.pdf", make_pdf(SECRET_SENTENCE), "application/pdf"))],
            headers=alice,
        )
        assert upload.status_code == 200
        assert upload.json()["results"][0]["status"].startswith("added")
        collection_id = next(
            c["id"] for c in client.get("/collections", headers=alice).json() if c["name"] == "Alice docs"
        )
        chat = client.post(
            "/chat",
            json={"question": "What does the zebra clause say?", "collection_id": collection_id},
            headers=alice,
        )
        assert chat.status_code == 200

        yield {
            "client": client,
            "alice": alice,
            "bob": bob,
            "collection_id": collection_id,
            "conversation_id": chat.json()["conversation_id"],
            "message_id": chat.json()["message_id"],
        }

        with SessionLocal() as session:  # remove the test users; the database cascades to their data
            for user in session.scalars(select(User).where(User.username.in_(names))):
                session.delete(user)
            session.commit()


def test_requests_without_a_valid_token_are_rejected(world):
    client = world["client"]
    assert client.get("/collections").status_code == 401
    assert client.post("/chat", json={"question": "hello there"}).status_code == 401
    assert client.get("/collections", headers={"Authorization": "Bearer not-a-token"}).status_code == 401


def test_owner_sees_the_collection_and_other_users_do_not(world):
    client = world["client"]
    alice_names = [c["name"] for c in client.get("/collections", headers=world["alice"]).json()]
    bob_names = [c["name"] for c in client.get("/collections", headers=world["bob"]).json()]
    assert "Alice docs" in alice_names
    assert "Alice docs" not in bob_names


def test_other_user_cannot_chat_in_the_collection(world):
    response = world["client"].post(
        "/chat",
        json={"question": "What does the zebra clause say?", "collection_id": world["collection_id"]},
        headers=world["bob"],
    )
    assert response.status_code == 404


def test_other_user_cannot_read_rate_or_delete_the_conversation(world):
    client, conversation_id = world["client"], world["conversation_id"]
    assert client.get(f"/conversations/{conversation_id}", headers=world["alice"]).status_code == 200
    assert client.get(f"/conversations/{conversation_id}", headers=world["bob"]).status_code == 404
    listed = [c["id"] for c in client.get("/conversations", headers=world["bob"]).json()]
    assert conversation_id not in listed
    rating = client.post(
        "/feedback", json={"message_id": world["message_id"], "rating": 1}, headers=world["bob"]
    )
    assert rating.status_code == 404
    assert client.delete(f"/conversations/{conversation_id}", headers=world["bob"]).status_code == 404


def test_other_user_cannot_delete_the_collection(world):
    client = world["client"]
    assert client.delete(f"/collections/{world['collection_id']}", headers=world["bob"]).status_code == 404
    still_there = [c["name"] for c in client.get("/collections", headers=world["alice"]).json()]
    assert "Alice docs" in still_there


def test_the_shared_collection_cannot_be_uploaded_to(world):
    response = world["client"].post(
        "/upload",
        data={"collection": "RBI Master Directions"},
        files=[("files", ("x.pdf", make_pdf("hello"), "application/pdf"))],
        headers=world["alice"],
    )
    assert response.status_code == 400


def test_search_stays_inside_its_collection(world):
    inside = search("zebra clause maximum penalty", top_k=5, collection_id=world["collection_id"])
    assert inside and all(p.file_name == "secret.pdf" for p in inside)
    default = search("zebra clause maximum penalty", top_k=10)  # the shared RBI collection
    assert all(p.file_name != "secret.pdf" for p in default)