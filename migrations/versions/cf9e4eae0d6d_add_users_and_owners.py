"""add users and owners

Revision ID: cf9e4eae0d6d
Revises: f232cdb61b45
Create Date: 2026-11-21 11:34:16.809482

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cf9e4eae0d6d'
down_revision: Union[str, Sequence[str], None] = 'f232cdb61b45'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
