import os

os.environ["JWT_SECRET"] = "a-test-only-secret-that-is-longer-than-32-bytes"

import importlib

from app import security

importlib.reload(security)


def test_password_hash_is_not_the_password_and_verifies():
    hashed = security.hash_password("correct horse battery")
    assert hashed != "correct horse battery"
    assert security.verify_password("correct horse battery", hashed)
    assert not security.verify_password("wrong password", hashed)


def test_token_round_trip():
    token = security.make_token(42)
    assert security.read_token(token) == 42


def test_tampered_or_garbage_token_is_rejected():
    token = security.make_token(42)
    assert security.read_token(token + "x") is None
    assert security.read_token("not-a-token") is None