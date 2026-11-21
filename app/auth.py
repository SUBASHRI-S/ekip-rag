from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.db import SessionLocal
from app.models import Collection, User
from app.security import read_token

bearer = HTTPBearer(auto_error=False)


def current_user_id(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> int:
    """The id of the logged-in user, or a 401 error."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Please log in")
    user_id = read_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Your login has expired. Please log in again")
    with SessionLocal() as session:
        if session.get(User, user_id) is None:
            raise HTTPException(status_code=401, detail="Please log in again")
    return user_id


def check_collection_access(session, collection_id: int, user_id: int) -> Collection:
    """A collection is usable if it is shared (no owner) or belongs to this user."""
    collection = session.get(Collection, collection_id)
    if collection is None or collection.owner_id not in (None, user_id):
        raise HTTPException(status_code=404, detail="Collection not found")
    return collection