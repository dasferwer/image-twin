from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE object_deletions (
            image_id uuid PRIMARY KEY REFERENCES images(id),
            object_key text NOT NULL UNIQUE,
            attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            next_attempt_at timestamptz NOT NULL DEFAULT now(),
            lease_token uuid,
            lease_until timestamptz,
            last_error text,
            completed_at timestamptz,
            created_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX object_deletions_due ON object_deletions(next_attempt_at,image_id)
    """)
    # Старые deleted тоже могли остаться в S3 после отказа. DeleteObject идемпотентен.
    op.execute("""
        INSERT INTO object_deletions(image_id,object_key)
        SELECT id,object_key FROM images WHERE status='deleted'
    """)


def downgrade():
    raise RuntimeError(
        "Намерения удаления нельзя потерять: восстановите согласованную резервную копию"
    )
