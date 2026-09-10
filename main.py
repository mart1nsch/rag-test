"""Entry point: add text chunks to the store, or ask questions against them.

Usage:
    python main.py add "some chunk of text..."
    python main.py ask "some question?"
"""

import argparse

import db
import llm

TOP_K = 5


def add_chunk(text: str) -> None:
    conn = db.connect()
    db.init_db(conn)
    chunk_id = db.add_chunk(conn, text)
    print(f"Stored chunk {chunk_id}")


def ask(question: str) -> None:
    conn = db.connect()
    db.init_db(conn)
    results = db.search(conn, question, top_k=TOP_K)
    context_chunks = [text for text, _ in results]
    answer = llm.ask(question, context_chunks)
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple RAG over SQLite + sqlite-vec.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Store a text chunk.")
    add_parser.add_argument("text", help="The chunk text to embed and store.")

    ask_parser = subparsers.add_parser("ask", help="Ask a question against stored chunks.")
    ask_parser.add_argument("question", help="The question to ask.")

    args = parser.parse_args()

    if args.command == "add":
        add_chunk(args.text)
    elif args.command == "ask":
        ask(args.question)


if __name__ == "__main__":
    main()
