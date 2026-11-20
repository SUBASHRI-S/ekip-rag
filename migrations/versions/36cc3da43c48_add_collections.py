"""add collections

Revision ID: 36cc3da43c48
Revises: f9ba66f7d6c8
Create Date: 2026-11-20 08:47:02.901692

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '36cc3da43c48'
down_revision: Union[str, Sequence[str], None] = 'f9ba66f7d6c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "collections",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.add_column("documents", sa.Column("collection_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "documents_collection_id_fkey", "documents", "collections",
        ["collection_id"], ["id"], ondelete="CASCADE",
    )
    op.drop_constraint("documents_file_name_key", "documents", type_="unique")
    op.create_unique_constraint("uq_documents_collection_file", "documents", ["collection_id", "file_name"])
    # put the existing documents into a first collection
    op.execute("INSERT INTO collections (name) VALUES ('RBI Master Directions')")
    op.execute(
        "UPDATE documents SET collection_id = "
        "(SELECT id FROM collections WHERE name = 'RBI Master Directions')"
    )


def downgrade() -> None:
    op.drop_constraint("uq_documents_collection_file", "documents", type_="unique")
    op.create_unique_constraint("documents_file_name_key", "documents", ["file_name"])
    op.drop_constraint("documents_collection_id_fkey", "documents", type_="foreignkey")
    op.drop_column("documents", "collection_id")
    op.drop_table("collections")