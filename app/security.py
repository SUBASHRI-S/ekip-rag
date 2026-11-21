import os
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

_hasher = PasswordHasher()
SECRET = os.environ.get("JWT_SECRET", "")
ALGORITHM = "HS256"
TOKEN_HOURS = 12


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def make_token(user_id: int) -> str:
    if not SECRET:
        raise RuntimeError("JWT_SECRET is not set in .env")
    expires = datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
    return jwt.encode({"sub": str(user_id), "exp": expires}, SECRET, algorithm=ALGORITHM)


def read_token(token: str) -> int | None:
    """The user id inside a valid, unexpired token, or None."""
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None