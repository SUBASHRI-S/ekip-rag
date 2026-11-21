"""add users and owners

Revision ID: f232cdb61b45
Revises: 36cc3da43c48
Create Date: 2026-11-21 11:26:57.165873

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f232cdb61b45'
down_revision: Union[str, Sequence[str], None] = '36cc3da43c48'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("username", sa.String(), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.add_column("collections", sa.Column("owner_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "collections_owner_id_fkey", "collections", "users", ["owner_id"], ["id"], ondelete="CASCADE"
    )
    op.drop_constraint("collections_name_key", "collections", type_="unique")
    op.create_unique_constraint("uq_collections_owner_name", "collections", ["owner_id", "name"])
    op.add_column("conversations", sa.Column("owner_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "conversations_owner_id_fkey", "conversations", "users", ["owner_id"], ["id"], ondelete="CASCADE"
    )


def downgrade() -> None:
    op.drop_constraint("conversations_owner_id_fkey", "conversations", type_="foreignkey")
    op.drop_column("conversations", "owner_id")
    op.drop_constraint("uq_collections_owner_name", "collections", type_="unique")
    op.create_unique_constraint("collections_name_key", "collections", ["name"])
    op.drop_constraint("collections_owner_id_fkey", "collections", type_="foreignkey")
    op.drop_column("collections", "owner_id")
    op.drop_table("users")