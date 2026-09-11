"""Subprocess worker used by seed_db.py to isolate docling/torch native crashes.

Reads one file path per line from stdin, processes it, and writes one JSON
result per line to stdout. Runs as a long-lived process so models are loaded
once and reused across files; if it dies (e.g. a Windows access violation
inside docling), the parent detects the closed stdout and restarts it.
"""
import json
import sys


def main() -> None:
    from app.services.document_processor import DocumentProcessor

    processor = DocumentProcessor()
    for line in sys.stdin:
        file_path = line.strip()
        if not file_path:
            continue
        try:
            chunks = processor.process_document(file_path)
            print(json.dumps({"ok": True, "chunks": chunks}), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    main()
