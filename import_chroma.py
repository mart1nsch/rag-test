"""One-off import: pull chunk text out of an existing chroma.sqlite3 and
store it in our own store via db.add_chunk. Text only, no metadata.
"""

import sqlite3

import db

CHROMA_DB_PATH = "chroma.sqlite3"


def iter_chunks(chroma_path: str):
    conn = sqlite3.connect(chroma_path)
    rows = conn.execute(
        "SELECT string_value FROM embedding_metadata WHERE key = 'chroma:document' ORDER BY id"
    )
    for (text,) in rows:
        if text:
            yield text
    conn.close()


def main() -> None:
    conn = db.connect()
    db.init_db(conn)

    count = 0
    for text in iter_chunks(CHROMA_DB_PATH):
        db.add_chunk(conn, text)
        count += 1
        if count % 100 == 0:
            print(f"Imported {count} chunks...")

    print(f"Done. Imported {count} chunks.")


if __name__ == "__main__":
    main()
