"""Serve the R152 timing site and collect one CSV per session.

    python annotation/timing/serve.py                 # http://127.0.0.1:8766/timing/
    python annotation/timing/serve.py --host 0.0.0.0  # reachable from the LAN

The document root is ``annotation/`` (not ``annotation/timing/``) because the
timing page reuses ``../style.css`` and links ``../guideline.md``; serving the
subdirectory alone would 404 both.

As with the B1 server, ``?a=`` is a claim rather than an identity, and there is
no authentication.  Keep it on a closed network.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent          # annotation/timing
DOCROOT = HERE.parent                           # annotation/
SAFE = re.compile(r"[^A-Za-z0-9_-]")


class Handler(SimpleHTTPRequestHandler):
    submissions: Path = (
        HERE.parents[1] / "data" / "interim" / "r152" / "submissions"
    )
    seen_hash: dict[str, str] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DOCROOT), **kwargs)

    def end_headers(self) -> None:
        # A regenerated packet behind a cached page would silently mix two
        # versions of the 100 items.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)
        if parsed.path != "/submit":
            self.send_error(404, "not found")
            return
        query = parse_qs(parsed.query)
        who = SAFE.sub("", query.get("annotator", ["anon"])[0])[:16] or "anon"
        session = SAFE.sub("", query.get("session", ["0"])[0])[:4] or "0"
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        if not body:
            self.send_error(400, "empty body")
            return

        self.submissions.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        # Session is in the filename: the three sessions are separate
        # submissions and must never overwrite each other.
        name = f"{who}_s{session}_{ts}.csv"
        (self.submissions / name).write_bytes(body)
        (self.submissions / f"{who}_s{session}_latest.csv").write_bytes(body)

        declared = query.get("hash", [""])[0]
        seen = Handler.seen_hash.get(who)
        if declared and seen and declared != seen:
            print(f"[WARN] {who} submitted against packet {declared}, "
                  f"earlier submissions were {seen}")
        if declared:
            Handler.seen_hash[who] = declared

        payload = json.dumps({"saved": name, "bytes": len(body)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        print(f"[submit] {who} session {session}: {len(body)} bytes -> {name}")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"serving {DOCROOT}")
    print(f"  http://{shown}:{args.port}/timing/?a=X&s=1")
    print(f"submissions -> {Handler.submissions}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
