#!/usr/bin/env python3
"""
Live bridge: W&B -> local files consumed by scripts/dashboard.html and scripts/training_dynamics.html.

What it does (polling loop):
- Fetch latest logged `val/pass@1_table` and `val/pass@k_table` from a W&B run.
- Download the underlying W&B table json files.
- Convert them into:
  - runs/<ts>_<run_id>/progress.json
  - runs/<ts>_<run_id>/results.jsonl
- Also export a lightweight curve file for training_dynamics:
  - sweeps/<ts>_<run_id>/sweep_results.csv

Usage:
  export WANDB_API_KEY=...
  python scripts/wandb_live_dashboard.py --run <entity>/<project>/<run_id>

Then serve repo root and open:
  scripts/dashboard.html?run=runs/<folder>
  scripts/training_dynamics.html?sweep=sweeps/<folder>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import wandb


DISPLAY_KS = [8, 16, 32, 64, 128, 256]
STEPS_DIRNAME = "steps"


@dataclass(frozen=True)
class TableRef:
    step: int
    key: str
    file_path: str
    sha256: Optional[str] = None


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_run_path(s: str) -> str:
    s = s.strip()
    if not s:
        raise ValueError("--run is required")
    # Accept full URL as well.
    if s.startswith("https://wandb.ai/"):
        # https://wandb.ai/<entity>/<project>/runs/<run_id>
        parts = s.split("/")
        try:
            i = parts.index("wandb.ai")  # will raise
            _ = i
        except Exception:
            # Simpler: just parse by known layout
            pass
        m = re.match(r"^https://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", s)
        if not m:
            raise ValueError(f"Unrecognized W&B URL: {s}")
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return s


def _safe_dir_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def _find_existing_dir_with_run_id(base_dir: str, run_id: str) -> Optional[str]:
    """Pick the most recent existing output folder that ends with _<run_id>."""
    if not os.path.isdir(base_dir):
        return None
    suffix = f"_{run_id}"
    cands = []
    for name in os.listdir(base_dir):
        if not name.endswith(suffix):
            continue
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            cands.append(name)
    return sorted(cands)[-1] if cands else None


def _copy_file(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp.", dir=os.path.dirname(dst))
    try:
        with open(src, "rb") as fsrc, os.fdopen(fd, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)
        os.replace(tmp_path, dst)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp.", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _atomic_write_json(path: str, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _atomic_write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp.", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _resolve_table_file_path(table_obj: Any) -> Optional[tuple[str, Optional[str]]]:
    """
    W&B history stores tables as a "table-file" reference dict.
    We try to resolve the underlying run file path.
    """
    if table_obj is None:
        return None
    if isinstance(table_obj, str):
        return (table_obj, None)
    if isinstance(table_obj, dict):
        for k in ("path", "artifact_path", "_latest_artifact_path"):
            if k in table_obj and isinstance(table_obj[k], str) and table_obj[k]:
                return (table_obj[k], table_obj.get("sha256"))
        return None
    # Best-effort for unexpected objects
    path = getattr(table_obj, "path", None)
    if isinstance(path, str) and path:
        return (path, getattr(table_obj, "sha256", None))
    return None


def _download_run_file(run: Any, file_path: str, cache_dir: str) -> str:
    f = run.file(file_path)
    f.download(root=cache_dir, replace=True)
    return os.path.join(cache_dir, file_path)


def _load_table_json(local_path: str) -> tuple[list[str], list[list[Any]]]:
    with open(local_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    cols = obj.get("columns")
    data = obj.get("data")
    if not isinstance(cols, list) or not isinstance(data, list):
        raise ValueError(f"Invalid wandb table json: {local_path}")
    return [str(c) for c in cols], data


def _extract_pred(text: str) -> str:
    """Heuristic: extract a compact prediction for grouping/self-consistency."""
    if not text:
        return ""
    # Drop <think>...</think> blocks (common in R1-style outputs)
    text2 = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Prefer boxed content
    m = re.search(r"\\boxed\{([^}]*)\}", text2)
    if m:
        return m.group(1).strip()
    # Otherwise last non-empty line
    lines = [ln.strip() for ln in text2.splitlines() if ln.strip()]
    return lines[-1] if lines else text2


def _mode_key(keys: list[str]) -> Optional[str]:
    """Mode with tie-break by first appearance."""
    if not keys:
        return None
    counts: Counter[str] = Counter()
    first_idx: dict[str, int] = {}
    best_key: Optional[str] = None
    best = (0, 10**18)  # (count, first)
    for i, k in enumerate(keys):
        counts[k] += 1
        if k not in first_idx:
            first_idx[k] = i
        cur = (counts[k], first_idx[k])
        if cur[0] > best[0] or (cur[0] == best[0] and cur[1] < best[1]):
            best = cur
            best_key = k
    return best_key


def _build_records_from_tables(
    pass1_cols: list[str],
    pass1_rows: list[list[Any]],
    passk_cols: list[str],
    passk_rows: list[list[Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    pass1 = [dict(zip(pass1_cols, r)) for r in pass1_rows]
    passk = [dict(zip(passk_cols, r)) for r in passk_rows]

    pass1_by_idx: dict[int, dict[str, Any]] = {}
    for r in pass1:
        try:
            idx = int(r.get("idx"))
        except Exception:
            continue
        pass1_by_idx[idx] = r

    # Determine n from output_* columns
    n = 0
    for c in passk_cols:
        if c.startswith("output_"):
            try:
                n = max(n, int(c.split("_", 1)[1]))
            except Exception:
                pass
    if n <= 0:
        raise ValueError("Could not infer n from pass@k table columns")

    records: list[dict[str, Any]] = []
    first_hits: list[Optional[int]] = []
    pass1_corrects: list[float] = []
    sc_corrects: dict[int, float] = {k: 0.0 for k in DISPLAY_KS}
    passk_corrects: dict[int, float] = {k: 0.0 for k in DISPLAY_KS}

    for row in passk:
        idx = int(row.get("idx", len(records)))
        prompt = row.get("prompt", "")
        gt = row.get("ground_truth", "")

        p1_row = pass1_by_idx.get(idx, {})
        p1_text = p1_row.get("output", "")
        p1_ok = float(p1_row.get("pass@1", 0.0) or 0.0) >= 1.0
        p1_pred = _extract_pred(str(p1_text))
        pass1_corrects.append(1.0 if p1_ok else 0.0)

        gens = []
        first_hit = None
        for j in range(1, n + 1):
            text = row.get(f"output_{j}", "")
            acc = row.get(f"acc_{j}", 0.0)
            ok = float(acc or 0.0) >= 1.0
            if first_hit is None and ok:
                first_hit = j - 1  # 0-based
            gens.append(
                {
                    "index": j - 1,
                    "text": str(text),
                    "pred": _extract_pred(str(text)),
                    "correct": bool(ok),
                }
            )

        first_hits.append(first_hit)

        # pass@k correctness for DISPLAY_KS
        for k in DISPLAY_KS:
            kk = min(k, n)
            ok_k = first_hit is not None and first_hit < kk
            passk_corrects[k] += 1.0 if ok_k else 0.0

        # self-consistency vote@k (mode among first k) correctness (requires extracted preds)
        gold_key = _extract_pred(str(gt)) if gt is not None else ""
        for k in DISPLAY_KS:
            kk = min(k, n)
            mode = _mode_key([g["pred"] for g in gens[:kk]])
            ok_sc = bool(mode) and bool(gold_key) and mode == gold_key
            sc_corrects[k] += 1.0 if ok_sc else 0.0

        rec = {
            "sample_index": idx,
            "sample_id": str(idx),
            "sample": {"problem": str(prompt)},
            "gold": {"answer": str(gt)},
            "pass1": {"correct": bool(p1_ok), "pred": p1_pred, "text": str(p1_text)},
            "sc": {"generations": gens},
        }
        records.append(rec)

    n_samples = max(len(records), 1)
    aggregate = {
        "n": len(records),
        "pass@1": float(sum(pass1_corrects) / n_samples),
        "pass@k": {f"pass@{k}": float(passk_corrects[k] / n_samples) for k in DISPLAY_KS},
        "sc@k": {f"sc@{k}": float(sc_corrects[k] / n_samples) for k in DISPLAY_KS},
    }
    return records, aggregate, n


def _write_sweep_results_csv(run: Any, out_csv: str) -> None:
    keys = ["_step", "val/pass@1_temp0"] + [f"val/pass@{k}" for k in DISPLAY_KS if k != 1]
    rows: list[dict[str, Any]] = []
    hf_repo = getattr(run, "name", None) or getattr(run, "id", None) or ""
    last_step = -1
    for row in run.scan_history(keys=keys, page_size=1000):
        step = row.get("_step")
        if step is None:
            continue
        try:
            step_i = int(step)
        except Exception:
            continue
        last_step = max(last_step, step_i)
        out = {"step": step_i, "hf_repo": str(hf_repo)}
        for k in keys:
            if k == "_step":
                continue
            out[k] = row.get(k, None)
        rows.append(out)

    # stable sort and write CSV
    rows.sort(key=lambda r: int(r.get("step", 0)))
    fieldnames = ["step", "hf_repo"] + [k for k in keys if k != "_step"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp.", dir=os.path.dirname(out_csv))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        os.replace(tmp_path, out_csv)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _list_table_refs_by_step(run: Any) -> dict[int, tuple[TableRef, TableRef]]:
    """Collect (pass@1_table, pass@k_table) references for all steps that have both."""
    out: dict[int, tuple[TableRef, TableRef]] = {}
    for row in run.scan_history(keys=["_step", "val/pass@1_table", "val/pass@k_table"], page_size=1000):
        step = row.get("_step")
        if step is None:
            continue
        try:
            step_i = int(step)
        except Exception:
            continue

        p1 = row.get("val/pass@1_table")
        pk = row.get("val/pass@k_table")
        p1_res = _resolve_table_file_path(p1)
        pk_res = _resolve_table_file_path(pk)
        if p1_res is None or pk_res is None:
            continue
        out[step_i] = (
            TableRef(step=step_i, key="val/pass@1_table", file_path=p1_res[0], sha256=p1_res[1]),
            TableRef(step=step_i, key="val/pass@k_table", file_path=pk_res[0], sha256=pk_res[1]),
        )
    return out


def _find_latest_table_refs(run: Any) -> Optional[tuple[TableRef, TableRef]]:
    latest_step: Optional[int] = None
    latest_p1: Optional[TableRef] = None
    latest_pk: Optional[TableRef] = None
    for row in run.scan_history(keys=["_step", "val/pass@1_table", "val/pass@k_table"], page_size=1000):
        step = row.get("_step")
        if step is None:
            continue
        try:
            step_i = int(step)
        except Exception:
            continue

        p1 = row.get("val/pass@1_table")
        pk = row.get("val/pass@k_table")

        p1_res = _resolve_table_file_path(p1)
        pk_res = _resolve_table_file_path(pk)
        if p1_res is None or pk_res is None:
            continue

        if latest_step is None or step_i >= latest_step:
            latest_step = step_i
            latest_p1 = TableRef(step=step_i, key="val/pass@1_table", file_path=p1_res[0], sha256=p1_res[1])
            latest_pk = TableRef(step=step_i, key="val/pass@k_table", file_path=pk_res[0], sha256=pk_res[1])

    if latest_p1 is None or latest_pk is None:
        return None
    return latest_p1, latest_pk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="W&B run path: <entity>/<project>/<run_id> or full URL")
    ap.add_argument("--poll-seconds", type=float, default=20.0)
    ap.add_argument("--out-runs", default="runs", help="Output folder for dashboard.html")
    ap.add_argument("--out-sweeps", default="sweeps", help="Output folder for training_dynamics.html")
    ap.add_argument("--once", action="store_true", help="Fetch once and exit")
    args = ap.parse_args()

    run_path = _parse_run_path(args.run)
    api = wandb.Api()
    run = api.run(run_path)

    run_id = str(getattr(run, "id", "run"))
    existing = _find_existing_dir_with_run_id(args.out_runs, run_id)
    if existing:
        run_dir_name = existing
    else:
        tag = _now_tag()
        run_dir_name = _safe_dir_name(f"{tag}_{run_id}")
    sweep_dir_name = run_dir_name

    out_run_dir = os.path.abspath(os.path.join(args.out_runs, run_dir_name))
    out_sweep_dir = os.path.abspath(os.path.join(args.out_sweeps, sweep_dir_name))
    cache_dir = os.path.join(out_run_dir, ".wandb_cache")
    steps_dir = os.path.join(out_run_dir, STEPS_DIRNAME)
    os.makedirs(out_run_dir, exist_ok=True)
    os.makedirs(out_sweep_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(steps_dir, exist_ok=True)

    # Keep a run-local dashboard copy up-to-date for convenience.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_dashboard = os.path.join(repo_root, "scripts", "dashboard.html")
    if os.path.exists(src_dashboard):
        _copy_file(src_dashboard, os.path.join(out_run_dir, "dashboard.html"))

    print(f"[wandb-live] run: {run_path}")
    print(f"[wandb-live] dashboard out: {out_run_dir}")
    print(f"[wandb-live] dynamics out:  {out_sweep_dir}")
    print()
    print("[wandb-live] Open in browser (serve repo root via http):")
    print(f"  scripts/dashboard.html?run=runs/{run_dir_name}")
    print(f"  scripts/training_dynamics.html?sweep=sweeps/{sweep_dir_name}")
    print(f"  runs/{run_dir_name}/dashboard.html  (run-local copy)")
    print()

    def step_done(step: int) -> bool:
        return os.path.exists(os.path.join(steps_dir, str(step), "results.jsonl")) and os.path.exists(
            os.path.join(steps_dir, str(step), "progress.json")
        )

    def write_index(latest_step: Optional[int]) -> None:
        steps = []
        try:
            for name in os.listdir(steps_dir):
                try:
                    steps.append(int(name))
                except Exception:
                    pass
        except Exception:
            pass
        steps = sorted(set(steps))
        if latest_step is None and steps:
            latest_step = steps[-1]
        _atomic_write_json(
            os.path.join(out_run_dir, "index.json"),
            {
                "run": run_path,
                "run_id": run_id,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "steps": steps,
                "latest_step": latest_step,
            },
        )

    def write_step_snapshot(step: int, p1_ref: TableRef, pk_ref: TableRef) -> tuple[dict[str, Any], int]:
        step_dir = os.path.join(steps_dir, str(step))
        os.makedirs(step_dir, exist_ok=True)

        p1_local = _download_run_file(run, p1_ref.file_path, cache_dir)
        pk_local = _download_run_file(run, pk_ref.file_path, cache_dir)
        p1_cols, p1_rows = _load_table_json(p1_local)
        pk_cols, pk_rows = _load_table_json(pk_local)
        records, aggregate, n = _build_records_from_tables(p1_cols, p1_rows, pk_cols, pk_rows)

        _atomic_write_jsonl(os.path.join(step_dir, "results.jsonl"), records)
        prog = {
            "total": len(records),
            "completed": len(records),
            "errors": 0,
            "signature": step,
            "current": {"phase": "wandb", "sample_index": 0, "sc_done": n, "sc_total": n, "step": step},
            "aggregate": aggregate,
        }
        _atomic_write_json(os.path.join(step_dir, "progress.json"), prog)
        return prog, n

    # Backfill all available checkpoints on startup (so you can switch between any steps immediately).
    latest_step_on_disk: Optional[int] = None
    refs_by_step = _list_table_refs_by_step(run)
    for step in sorted(refs_by_step.keys()):
        if step_done(step):
            latest_step_on_disk = step if latest_step_on_disk is None else max(latest_step_on_disk, step)
            continue
        print(f"[wandb-live] Backfill step={step} ...")
        prog, _n = write_step_snapshot(step, refs_by_step[step][0], refs_by_step[step][1])
        latest_step_on_disk = step if latest_step_on_disk is None else max(latest_step_on_disk, step)
        # Also keep root files pointing to the latest step for "latest" view.
        _copy_file(os.path.join(steps_dir, str(step), "results.jsonl"), os.path.join(out_run_dir, "results.jsonl"))
        _atomic_write_json(os.path.join(out_run_dir, "progress.json"), prog)
        write_index(latest_step_on_disk)

    write_index(latest_step_on_disk)

    last_step = latest_step_on_disk
    while True:
        try:
            # Always refresh training dynamics curves (cheap: scalars only).
            _write_sweep_results_csv(run, os.path.join(out_sweep_dir, "sweep_results.csv"))

            refs = _find_latest_table_refs(run)
            if refs is None:
                print("[wandb-live] No tables found yet (waiting)...")
            else:
                p1_ref, pk_ref = refs
                if last_step != pk_ref.step or not step_done(pk_ref.step):
                    print(f"[wandb-live] New tables at step={pk_ref.step}. Downloading...")
                    prog, _n = write_step_snapshot(pk_ref.step, p1_ref, pk_ref)
                    # Update root files to point to latest
                    _copy_file(os.path.join(steps_dir, str(pk_ref.step), "results.jsonl"), os.path.join(out_run_dir, "results.jsonl"))
                    _atomic_write_json(os.path.join(out_run_dir, "progress.json"), prog)
                    last_step = pk_ref.step
                    write_index(last_step)
                else:
                    print(f"[wandb-live] step={pk_ref.step} (no change)")

        except KeyboardInterrupt:
            print("\n[wandb-live] Stopped.")
            return 0
        except Exception as e:
            print(f"[wandb-live] Error: {e}", file=sys.stderr)

        if args.once:
            return 0
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())


