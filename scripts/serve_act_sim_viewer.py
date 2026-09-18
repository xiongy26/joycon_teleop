#!/usr/bin/env python3
"""Serve the ACT-sim web viewer (repo root HTTP server) with optional data export.

Usage:
    python scripts/serve_act_sim_viewer.py                # auto-latest run
    python scripts/serve_act_sim_viewer.py --run-dir runs/act_sim_xxx --port 8010

Then open http://127.0.0.1:<port>/web/viewer.html
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def latest_run_dir() -> Path | None:
    runs = sorted((PROJECT_ROOT / "runs").glob("act_sim_*"))
    return runs[-1] if runs else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--run-dir", default=None, help="Run dir to export into web/data (default: latest)")
    parser.add_argument("--no-export", action="store_true", help="Skip data export")
    args = parser.parse_args(argv)

    if not args.no_export:
        run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
        if run_dir and (run_dir / "data").exists():
            from scripts.export_act_sim_web import export_for_viewer

            print(f"exporting viewer data from {run_dir} ...")
            export_for_viewer(run_dir)
        else:
            print(f"no run data found ({run_dir}); viewer will start without replay data")

    import http.server
    import socketserver

    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {
            **http.server.SimpleHTTPRequestHandler.extensions_map,
            ".wasm": "application/wasm",
            ".mjs": "text/javascript",
        }

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    with Server(("127.0.0.1", args.port), Handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/web/viewer.html"
        print(f"serving {PROJECT_ROOT}")
        print(f"viewer: {url}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
