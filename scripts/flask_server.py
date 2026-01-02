#!/usr/bin/env python3
"""
Minimal Flask server to serve the dashboards + generated run artifacts.

Serves:
- /scripts/...  -> repo_root/scripts/...
- /runs/...     -> repo_root/runs/...
- /sweeps/...   -> repo_root/sweeps/...

This is intended to be used together with `scripts/wandb_live_dashboard.py`,
which keeps writing/refreshing files under runs/ and sweeps/.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional


try:
    from flask import Flask, Response, abort, redirect, send_from_directory
except ModuleNotFoundError:
    print("Flask is not installed. Install it first:\n  pip install flask", file=sys.stderr)
    raise


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
SWEEPS_DIR = os.path.join(REPO_ROOT, "sweeps")

app = Flask(__name__)


@app.after_request
def _no_cache(resp: Response) -> Response:
    # Dashboards poll frequently; disable caching to avoid stale reads.
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/")
def index():
    # Prefer the newest generated run folder (so opening "/" "just works").
    try:
        if os.path.isdir(RUNS_DIR):
            dirs = [
                d
                for d in os.listdir(RUNS_DIR)
                if os.path.isdir(os.path.join(RUNS_DIR, d)) and re.match(r"^\d{8}_\d{6}_", d)
            ]
            if dirs:
                dirs.sort()
                latest = dirs[-1]
                if os.path.exists(os.path.join(RUNS_DIR, latest, "dashboard.html")):
                    return redirect(f"/runs/{latest}/dashboard.html")
                return redirect(f"/scripts/dashboard.html?run=runs/{latest}")
    except Exception:
        pass
    return redirect("/scripts/dashboard.html")


def _safe_send(dir_path: str, rel_path: str):
    abs_dir = os.path.abspath(dir_path)
    if not os.path.isdir(abs_dir):
        abort(404)
    # send_from_directory defends against path traversal; keep it.
    return send_from_directory(abs_dir, rel_path)


@app.get("/scripts/<path:rel_path>")
def serve_scripts(rel_path: str):
    return _safe_send(SCRIPTS_DIR, rel_path)


@app.get("/runs/<path:rel_path>")
def serve_runs(rel_path: str):
    return _safe_send(RUNS_DIR, rel_path)


@app.get("/sweeps/<path:rel_path>")
def serve_sweeps(rel_path: str):
    return _safe_send(SWEEPS_DIR, rel_path)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    print(f"[flask] Serving repo root: {REPO_ROOT}")
    print(f"[flask] URL: http://{args.host}:{args.port}/")
    print()
    print("[flask] Dashboards:")
    print(f"  - http://{args.host}:{args.port}/scripts/dashboard.html")
    print(f"  - http://{args.host}:{args.port}/scripts/training_dynamics.html")
    print()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


