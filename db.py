"""SQLite + sqlite-vec storage for text chunks and their embeddings."""

import sqlite3

import sqlite_vec
from sqlite_vec import serialize_float32

from embeddings import embed_passages, embed_query

DB_PATH = "rag.db"
EMBEDDING_DIM = 768


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL
        )
    """)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
            id INTEGER PRIMARY KEY,
            embedding FLOAT[{EMBEDDING_DIM}]
        )
    """)
    conn.commit()


def add_chunk(conn: sqlite3.Connection, text: str) -> int:
    """Embed a text chunk and store it. Returns the chunk id."""
    embedding = embed_passages([text])[0]

    cur = conn.execute("INSERT INTO chunks (text) VALUES (?)", (text,))
    chunk_id = cur.lastrowid
    conn.execute(
        "INSERT INTO chunk_vectors (id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_float32(embedding)),
    )
    conn.commit()
    return chunk_id


def search(conn: sqlite3.Connection, query: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Return the top_k most similar chunks as (text, distance) tuples, closest first."""
    query_embedding = serialize_float32(embed_query(query))

    rows = conn.execute(
        """
        SELECT chunks.text, chunk_vectors.distance
        FROM chunk_vectors
        JOIN chunks ON chunks.id = chunk_vectors.id
        WHERE chunk_vectors.embedding MATCH ?
          AND k = ?
        ORDER BY chunk_vectors.distance
        """,
        (query_embedding, top_k),
    ).fetchall()
    return rows
