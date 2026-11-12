"""Serves the Prompt Gateway site plus a spaCy token-count endpoint.

Run from the repo root:
    pip install -r requirements.txt
    python3 website/server.py            # or: python3 website/server.py 8080
then open http://localhost:8000/#optimizer

POST /api/spacy/count  {"texts": ["...", ...]}  ->  {"counts": [n, ...], "tokenizer": "..."}
Counts are spaCy's linguistic tokens (words and punctuation, whitespace excluded). They are a
separate data point from the model tokenizer counts the site uses for cost.
"""
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import spacy

SITE_DIR = Path(__file__).resolve().parent / "site"
MAX_CHARS = 200_000

# A blank English pipeline is just spaCy's rule-based tokenizer: no model download needed,
# and it splits text the same way the en_core_web_* pipelines do.
NLP = spacy.blank("en")
TOKENIZER_NAME = f"spaCy {spacy.__version__} (en)"


def count_tokens(text):
    return sum(1 for tok in NLP.make_doc(text) if not tok.is_space)


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/spacy/count":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            texts = json.loads(self.rfile.read(length))["texts"]
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                raise ValueError
            if sum(len(t) for t in texts) > MAX_CHARS:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            self.send_error(400, 'Expected {"texts": [string, ...]}')
            return
        body = json.dumps({"counts": [count_tokens(t) for t in texts], "tokenizer": TOKENIZER_NAME}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(SITE_DIR)))
    print(f"Prompt Gateway running at http://localhost:{port}/  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
