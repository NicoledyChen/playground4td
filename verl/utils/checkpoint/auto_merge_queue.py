# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

from filelock import FileLock


QUEUE_VERSION = 1
QUEUE_CONFIG_FILENAME = "config.json"
QUEUE_LOCK_FILENAME = "queue.lock"


def _now_ts() -> float:
    return time.time()


def _safe_task_name(tag: str) -> str:
    # Avoid path traversal and weird filenames.
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", tag.strip())
    if not safe:
        raise ValueError(f"Invalid tag: {tag!r}")
    return safe


def _atomic_write_json(path: str, data: dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def ensure_queue_dirs(queue_dir: str) -> dict[str, str]:
    """
    Ensure the queue directory structure exists. Returns a dict of state->dir.
    """

    os.makedirs(queue_dir, exist_ok=True)
    state_dirs = {}
    for state in ("pending", "running", "done", "failed"):
        d = os.path.join(queue_dir, state)
        os.makedirs(d, exist_ok=True)
        state_dirs[state] = d
    return state_dirs


def queue_lock_path(queue_dir: str) -> str:
    return os.path.join(queue_dir, QUEUE_LOCK_FILENAME)


def queue_config_path(queue_dir: str) -> str:
    return os.path.join(queue_dir, QUEUE_CONFIG_FILENAME)


def write_queue_config(queue_dir: str, config: Any) -> None:
    """
    Persist worker configuration into `<queue_dir>/config.json`.
    Worker can read it to avoid repeating CLI args.
    """

    ensure_queue_dirs(queue_dir)
    if is_dataclass(config):
        payload = asdict(config)
    elif isinstance(config, dict):
        payload = config
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")

    # Put metadata last so user-provided keys can't override them.
    payload = {**payload, "version": QUEUE_VERSION, "updated_at": _now_ts()}
    _atomic_write_json(queue_config_path(queue_dir), payload)


def read_queue_config(queue_dir: str) -> Optional[dict[str, Any]]:
    path = queue_config_path(queue_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_task_file(queue_dir: str, task_name: str) -> Optional[str]:
    state_dirs = ensure_queue_dirs(queue_dir)
    for state, d in state_dirs.items():
        candidate = os.path.join(d, f"{task_name}.json")
        if os.path.exists(candidate):
            return candidate
    return None


def enqueue_tag(queue_dir: str, tag: str, *, extra: Optional[dict[str, Any]] = None) -> str:
    """
    Enqueue a tag as a merge task. If the task already exists in any state dir, returns the existing path.
    """

    ensure_queue_dirs(queue_dir)
    task_name = _safe_task_name(tag)
    lock = FileLock(queue_lock_path(queue_dir), timeout=60)
    with lock:
        existing = _find_task_file(queue_dir, task_name)
        if existing is not None:
            return existing

        task = {
            "version": QUEUE_VERSION,
            "tag": tag,
            "task_name": task_name,
            "status": "pending",
            "created_at": _now_ts(),
            "updated_at": _now_ts(),
            "attempt": 0,
            "progress": {},
        }
        if extra:
            task["extra"] = extra

        pending_path = os.path.join(queue_dir, "pending", f"{task_name}.json")
        _atomic_write_json(pending_path, task)
        return pending_path


def recover_running_to_pending(queue_dir: str) -> int:
    """
    If a worker crashed, tasks may be left in `running/`. Move them back to `pending/`.
    Returns number of tasks recovered.
    """

    ensure_queue_dirs(queue_dir)
    lock = FileLock(queue_lock_path(queue_dir), timeout=60)
    recovered = 0
    with lock:
        running_dir = os.path.join(queue_dir, "running")
        for fname in os.listdir(running_dir):
            if not fname.endswith(".json"):
                continue
            src = os.path.join(running_dir, fname)
            dst = os.path.join(queue_dir, "pending", fname)
            os.replace(src, dst)
            recovered += 1
    return recovered


def _extract_step_for_sorting(task_name: str) -> tuple[int, str]:
    # Prefer sorting by global_step_N if available.
    m = re.match(r"global_step_(\d+)$", task_name)
    if m:
        return int(m.group(1)), task_name
    return 2**31 - 1, task_name


def claim_next(queue_dir: str) -> Optional[str]:
    """
    Atomically move the next pending task to running and return its path.
    Returns None if empty.
    """

    ensure_queue_dirs(queue_dir)
    lock = FileLock(queue_lock_path(queue_dir), timeout=60)
    with lock:
        pending_dir = os.path.join(queue_dir, "pending")
        candidates = [f for f in os.listdir(pending_dir) if f.endswith(".json")]
        if not candidates:
            return None

        # Sort by step asc (older first)
        candidates.sort(key=lambda fn: _extract_step_for_sorting(os.path.splitext(fn)[0]))
        fname = candidates[0]
        src = os.path.join(pending_dir, fname)
        dst = os.path.join(queue_dir, "running", fname)
        os.replace(src, dst)
        return dst


def load_task(task_path: str) -> dict[str, Any]:
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_task(task_path: str, update: dict[str, Any]) -> None:
    task = load_task(task_path)
    task.update(update)
    task["updated_at"] = _now_ts()
    _atomic_write_json(task_path, task)


def finish_task(queue_dir: str, task_path: str, *, status: str, error: Optional[str] = None) -> str:
    """
    Move a running task to done/failed and return the new path.
    """

    assert status in ("done", "failed"), f"Unsupported finish status: {status}"
    task = load_task(task_path)
    task["status"] = status
    task["updated_at"] = _now_ts()
    if status == "failed" and error:
        task["error"] = error

    _atomic_write_json(task_path, task)

    fname = os.path.basename(task_path)
    dst = os.path.join(queue_dir, status, fname)
    os.replace(task_path, dst)
    return dst


