"""Serve the B1 annotation site and collect submissions.

    python annotation/serve.py                 # http://127.0.0.1:8765
    python annotation/serve.py --host 0.0.0.0  # reachable from the LAN

The site itself works from ``file://`` (the item file is a plain script, not a
fetch), so the server is only needed if you want submissions written straight
to disk instead of mailed back as CSV.

Nothing here is authenticated.  Binding to 0.0.0.0 exposes the page to anyone
who can reach the port, and ``?annotator=`` is a claim, not an identity.  That
is acceptable for two known annotators on a closed network; do not put this on
the open internet.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
SAFE = re.compile(r"[^A-Za-z0-9_-]")


class Handler(SimpleHTTPRequestHandler):
    submissions: Path = HERE.parent / "data" / "interim" / "b1_round1" / "submissions"
    seen_hash: dict[str, str] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def end_headers(self) -> None:
        # Annotators reload mid-session; a cached old packet would silently
        # mismatch the submissions.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)
        if parsed.path != "/submit":
            self.send_error(404, "not found")
            return
        who = parse_qs(parsed.query).get("annotator", ["anon"])[0]
        who = SAFE.sub("", who)[:16] or "anon"
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        if not body:
            self.send_error(400, "empty body")
            return

        self.submissions.mkdir(parents=True, exist_ok=True)
        # Microseconds, not seconds: two rapid submits from the same annotator
        # used to collide on the same filename and the earlier copy was lost.
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        name = f"{who}_{ts}.csv"
        (self.submissions / name).write_bytes(body)
        (self.submissions / f"{who}_latest.csv").write_bytes(body)

        # A submission carries the packet hash it was annotated against.  If it
        # changes mid-run, items.js was regenerated and the two files are not
        # comparable -- say so loudly rather than merging them silently.
        declared = parse_qs(parsed.query).get("hash", [""])[0]
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
        print(f"[submit] {who}: {len(body)} bytes -> {name}")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"serving {HERE}")
    print(f"  http://{shown}:{args.port}/?a=A")
    print(f"submissions -> {Handler.submissions}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
