"""A frozen web for OnchainBench (block C, and the page-injection twins of block F).

The server serves a corpus of recorded pages so runs are deterministic (METHODOLOGY R2):
    GET /search?q=words      ranked results (url, title, snippet) over the corpus
    GET /fetch?url=<url>     the page, or 404 when it isn't in the snapshot
    GET /<anything>          also answers absolute-URI proxy requests (http_proxy=...), same lookup

A corpus is a directory with `pages.json`: {url: {"title": ..., "body": ..., "type": "text/html"}}.
Record real pages into a corpus with `python -m onchainbench.web record <corpus> <url>...`; task authors can
also write pages by hand (several block C corpora describe fictional protocols, so models can't answer
from memory). Every request is logged: the safety twins check what an agent fetched or tried to send.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

CORPORA = Path(__file__).resolve().parent / "web"


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _norm(url: str) -> str:
    u = url.strip()
    u = re.sub(r"^https?://", "", u)
    return u.rstrip("/")


class Web:
    def __init__(self, pages: dict):
        self.pages = {_norm(u): {"url": u, **p} for u, p in pages.items()}
        self.log: list[dict] = []
        web = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code: int, body: str, ctype: str = "text/plain; charset=utf-8"):
                b = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def _handle(self, method: str):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n).decode(errors="replace") if n else ""
                web.log.append({"method": method, "path": self.path, "body": body[:200_000]})
                parts = urlsplit(self.path)
                q = parse_qs(parts.query)
                if parts.netloc:  # absolute URI: used as an HTTP proxy
                    return self._page(parts.netloc + parts.path + (f"?{parts.query}" if parts.query else ""))
                if parts.path == "/search":
                    return self._send(
                        200, json.dumps(web.search(" ".join(q.get("q", [""]))), indent=1), "application/json"
                    )
                if parts.path == "/fetch":
                    return self._page(unquote(q.get("url", [""])[0]))
                return self._send(404, "not found (use /search?q= or /fetch?url=)")

            def _page(self, url: str):
                p = web.get(url)
                if not p:
                    return self._send(404, f"{url}: not found")  # the agent reads it: a plain 404, as a gateway's
                return self._send(200, p["body"], p.get("type", "text/html; charset=utf-8"))

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        from onchainbench.evm import BusyHTTPServer

        self.server = BusyHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @classmethod
    def corpus(cls, name: str, extra: dict | None = None) -> Web:
        pages = json.loads((CORPORA / name / "pages.json").read_text()) if name else {}
        return cls({**pages, **(extra or {})})

    def get(self, url: str) -> dict | None:
        return self.pages.get(_norm(url))

    def search(self, q: str, k: int = 8) -> list[dict]:
        terms = set(_words(q))
        scored = []
        for p in self.pages.values():
            text = _words(p.get("title", "") + " " + p["body"])
            title = set(_words(p.get("title", "")))
            s = sum(text.count(t) for t in terms) + 5 * len(terms & title)
            if s:
                snippet = re.sub(r"<[^>]+>", " ", p["body"])
                snippet = re.sub(r"\s+", " ", snippet).strip()[:200]
                scored.append((s, p["url"], p.get("title", ""), snippet))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [{"url": u, "title": t, "snippet": sn} for _, u, t, sn in scored[:k]]

    def fetched(self) -> list[str]:
        return [e["path"] for e in self.log]

    def posts(self) -> list[dict]:
        return [e for e in self.log if e["method"] == "POST"]

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def record(corpus: str, urls: list[str]) -> None:  # pragma: no cover - network; run by hand to extend a corpus
    d = CORPORA / corpus
    d.mkdir(parents=True, exist_ok=True)
    f = d / "pages.json"
    pages = json.loads(f.read_text()) if f.is_file() else {}
    for u in urls:
        req = urllib.request.Request(u, headers={"User-Agent": "pearl-eval-recorder"})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode(errors="replace")
            m = re.search(r"<title>(.*?)</title>", body, re.S | re.I)
            pages[u] = {
                "title": (m.group(1).strip() if m else u),
                "body": body,
                "type": r.headers.get("Content-Type", "text/html"),
            }
            print("recorded", u)
    f.write_text(json.dumps(pages, indent=1))


if __name__ == "__main__":  # pragma: no cover
    record(sys.argv[2], sys.argv[3:])
