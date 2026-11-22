import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

# The defaults match .env.example, so importing this module never fails and the
# unit tests need no database. A real connection still needs the right password.
DATABASE_URL = URL.create(
    "postgresql+psycopg",
    username=os.environ.get("POSTGRES_USER", "ekip"),
    password=os.environ.get("POSTGRES_PASSWORD", ""),
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=int(os.environ.get("POSTGRES_PORT", "5433")),
    database=os.environ.get("POSTGRES_DB", "ekip"),
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass