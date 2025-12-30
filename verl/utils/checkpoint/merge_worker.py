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

import argparse
import gc
import json
import os
import re
import time
import traceback
from typing import Any, Optional

from .auto_merge_queue import (
    claim_next,
    ensure_queue_dirs,
    finish_task,
    load_task,
    read_queue_config,
    recover_running_to_pending,
    update_task,
    write_queue_config,
)
from .checkpoint_manager import CHECKPOINT_TRACKER
from .model_merger import merge_fsdp_checkpoint_to_hf, upload_folder_to_huggingface


RAW_FILE_PATTERNS = (
    re.compile(r"model_world_size_\d+_rank_\d+\.pt$"),
    re.compile(r"optim_world_size_\d+_rank_\d+\.pt$"),
    re.compile(r"extra_state_world_size_\d+_rank_\d+\.pt$"),
)


def _extract_global_step(tag: str) -> Optional[int]:
    m = re.match(r"global_step_(\d+)$", tag)
    if not m:
        return None
    return int(m.group(1))


def _list_step_tags(save_checkpoint_path: str) -> list[str]:
    if not os.path.exists(save_checkpoint_path):
        return []
    tags = []
    for name in os.listdir(save_checkpoint_path):
        if re.match(r"global_step_\d+$", name) and os.path.isdir(os.path.join(save_checkpoint_path, name)):
            tags.append(name)
    tags.sort(key=lambda t: _extract_global_step(t) or 2**31 - 1)
    return tags


def _load_checkpoint_tracker(save_checkpoint_path: str) -> Optional[dict[str, Any]]:
    tracker_path = os.path.join(save_checkpoint_path, CHECKPOINT_TRACKER)
    if not os.path.exists(tracker_path):
        return None
    try:
        with open(tracker_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _protected_tags(
    save_checkpoint_path: str,
    *,
    keep_last_n_raw: int,
    keep_best_raw: bool,
) -> set[str]:
    protected: set[str] = set()

    # Always protect the newest N global_step_* folders (best-effort).
    if keep_last_n_raw > 0:
        tags = _list_step_tags(save_checkpoint_path)
        protected.update(tags[-keep_last_n_raw :])

    # Protect tracker-specified last/best if present.
    tracker = _load_checkpoint_tracker(save_checkpoint_path)
    if tracker is not None:
        last = tracker.get("last_global_step")
        if isinstance(last, int):
            protected.add(f"global_step_{last}")
        if keep_best_raw:
            best = tracker.get("best_global_step")
            if isinstance(best, int):
                protected.add(f"global_step_{best}")

    return protected


def _is_raw_file(filename: str) -> bool:
    return any(p.match(filename) for p in RAW_FILE_PATTERNS)


def _cleanup_raw_files(step_dir: str, *, components: Optional[list[str]] = None) -> dict[str, Any]:
    """
    Delete raw shard/optimizer files for the given components under a step folder.
    Returns stats dict.
    """

    deleted: list[str] = []
    missing_components: list[str] = []

    if components is None:
        # Default: clean all immediate sub-directories under the step folder (e.g. actor/critic).
        components = [
            d for d in os.listdir(step_dir) if os.path.isdir(os.path.join(step_dir, d)) and not d.startswith(".")
        ]

    cleaned_components: list[str] = []
    for comp in components:
        comp_dir = os.path.join(step_dir, comp)
        if not os.path.isdir(comp_dir):
            missing_components.append(comp)
            continue
        cleaned_components.append(comp)
        for fname in os.listdir(comp_dir):
            if not _is_raw_file(fname):
                continue
            fpath = os.path.join(comp_dir, fname)
            try:
                os.remove(fpath)
                deleted.append(fpath)
            except FileNotFoundError:
                continue

    # also remove dataloader state if present (it's only used for resume)
    dataloader_path = os.path.join(step_dir, "dataloader.pt")
    if os.path.exists(dataloader_path):
        try:
            os.remove(dataloader_path)
            deleted.append(dataloader_path)
        except FileNotFoundError:
            pass

    return {"deleted": deleted, "cleaned_components": cleaned_components, "missing_components": missing_components}


def _format_template(template: str, *, tag: str, component: str, hf_repo_id: Optional[str] = None) -> str:
    try:
        step = _extract_global_step(tag)
        fmt_kwargs: dict[str, Any] = {"tag": tag, "component": component}
        if hf_repo_id is not None:
            fmt_kwargs["hf_repo_id"] = hf_repo_id
        if step is not None:
            fmt_kwargs["step"] = step
            fmt_kwargs["step_padded"] = f"{step:08d}"
        return template.format(**fmt_kwargs)
    except KeyError as e:
        raise ValueError(f"Invalid template {template!r}, missing key: {e}") from e


def _merge_and_upload_one_component(
    *,
    save_checkpoint_path: str,
    tag: str,
    component: str,
    hf_repo_id: Optional[str],
    repo_per_step: bool,
    hf_repo_id_per_step_template: str,
    hf_latest_repo_id: Optional[str],
    multi_component: bool,
    hf_path_in_repo_template: str,
    update_latest: bool,
    hf_latest_path_in_repo_template: str,
    hf_private: bool,
) -> dict[str, Any]:
    step_dir = os.path.join(save_checkpoint_path, tag)
    local_dir = os.path.join(step_dir, component)
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Checkpoint component dir not found: {local_dir}")

    # Merge
    hf_path = merge_fsdp_checkpoint_to_hf(local_dir)

    upload_info: dict[str, Any] = {"hf_path": hf_path}
    if hf_repo_id or repo_per_step:
        # Determine target repo_id
        if repo_per_step:
            if not hf_repo_id and "{hf_repo_id}" in hf_repo_id_per_step_template:
                raise ValueError("`hf_repo_id` is required when `hf_repo_id_per_step_template` uses `{hf_repo_id}`.")
            repo_id = _format_template(
                hf_repo_id_per_step_template, tag=tag, component=component, hf_repo_id=hf_repo_id
            )
            # Default: upload to repo root for single-component; otherwise use component subfolder.
            if hf_path_in_repo_template in ("", "checkpoints/{tag}/{component}"):
                path_in_repo = component if multi_component else None
            else:
                path_in_repo = _format_template(
                    hf_path_in_repo_template, tag=tag, component=component, hf_repo_id=hf_repo_id
                )
        else:
            assert hf_repo_id is not None
            repo_id = hf_repo_id
            path_in_repo = _format_template(hf_path_in_repo_template, tag=tag, component=component, hf_repo_id=hf_repo_id)

        # If template returns empty string, treat as repo root.
        if path_in_repo == "":
            path_in_repo = None

        upload_folder_to_huggingface(
            local_path=hf_path,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            private=hf_private,
            commit_message=f"auto-merge {tag} ({component})",
        )
        upload_info["uploaded"] = True
        upload_info["repo_id"] = repo_id
        upload_info["path_in_repo"] = path_in_repo

        if update_latest:
            if repo_per_step:
                if hf_latest_repo_id is None:
                    if hf_repo_id is None:
                        raise ValueError("`hf_latest_repo_id` is required when `repo_per_step=true` and `hf_repo_id` is null.")
                    hf_latest_repo_id = f"{hf_repo_id}-latest"
                latest_repo_id = hf_latest_repo_id
                latest_path_in_repo = component if multi_component else None
            else:
                latest_repo_id = hf_repo_id
                latest_path_in_repo = _format_template(
                    hf_latest_path_in_repo_template, tag=tag, component=component, hf_repo_id=hf_repo_id
                )
                if latest_path_in_repo == "":
                    latest_path_in_repo = None

            upload_folder_to_huggingface(
                local_path=hf_path,
                repo_id=latest_repo_id,
                path_in_repo=latest_path_in_repo,
                private=hf_private,
                commit_message=f"auto-merge latest ({component})",
            )
            upload_info["latest_repo_id"] = latest_repo_id
            upload_info["latest_path_in_repo"] = latest_path_in_repo
    else:
        upload_info["uploaded"] = False

    return upload_info


def _process_task(
    *,
    queue_dir: str,
    task_path: str,
    save_checkpoint_path: str,
    merge_components: list[str],
    hf_repo_id: Optional[str],
    repo_per_step: bool,
    hf_repo_id_per_step_template: str,
    hf_latest_repo_id: Optional[str],
    hf_path_in_repo_template: Optional[str],
    update_latest: bool,
    hf_latest_path_in_repo_template: Optional[str],
    keep_last_n_raw: int,
    keep_best_raw: bool,
    delete_raw: bool,
    hf_private: bool,
) -> None:
    task = load_task(task_path)
    tag = task["tag"]
    update_task(
        task_path,
        {
            "status": "running",
            "attempt": int(task.get("attempt", 0)) + 1,
            "started_at": time.time(),
            "worker_pid": os.getpid(),
        },
    )

    progress = task.get("progress", {})
    progress.setdefault("components", {})
    update_task(task_path, {"progress": progress})

    multi_component = len(merge_components) > 1

    # Merge + upload per component
    for comp in merge_components:
        comp_prog = progress["components"].setdefault(comp, {})
        comp_prog["merge_started_at"] = time.time()
        update_task(task_path, {"progress": progress})

        info = _merge_and_upload_one_component(
            save_checkpoint_path=save_checkpoint_path,
            tag=tag,
            component=comp,
            hf_repo_id=hf_repo_id,
            repo_per_step=repo_per_step,
            hf_repo_id_per_step_template=hf_repo_id_per_step_template,
            hf_latest_repo_id=hf_latest_repo_id,
            multi_component=multi_component,
            hf_path_in_repo_template=hf_path_in_repo_template or "",
            update_latest=update_latest,
            hf_latest_path_in_repo_template=hf_latest_path_in_repo_template or "",
            hf_private=hf_private,
        )

        comp_prog["merge_finished_at"] = time.time()
        comp_prog["upload"] = info
        update_task(task_path, {"progress": progress})

        # try to reduce resident memory between components
        gc.collect()

    # Cleanup raw files (for non-protected checkpoints)
    step_dir = os.path.join(save_checkpoint_path, tag)
    cleanup_info: dict[str, Any] = {"skipped": True}
    if delete_raw:
        protected = _protected_tags(save_checkpoint_path, keep_last_n_raw=keep_last_n_raw, keep_best_raw=keep_best_raw)
        if tag in protected:
            cleanup_info = {"skipped": True, "reason": "protected_for_resume", "protected_tags": sorted(protected)}
        else:
            # For cleanup, delete raw files for all components under the step directory (actor/critic/...).
            cleanup_info = _cleanup_raw_files(step_dir, components=None)
            cleanup_info["skipped"] = False
    progress["cleanup"] = cleanup_info
    update_task(task_path, {"progress": progress, "finished_at": time.time()})

    finish_task(queue_dir, task_path, status="done")


def _try_deferred_cleanup(
    *,
    queue_dir: str,
    save_checkpoint_path: str,
    keep_last_n_raw: int,
    keep_best_raw: bool,
) -> int:
    """
    Tasks may skip cleanup when they were the latest checkpoint at processing time.
    This function revisits DONE tasks and cleans up raw files once they are no longer protected.
    Returns number of tasks cleaned up.
    """

    done_dir = os.path.join(queue_dir, "done")
    if not os.path.isdir(done_dir):
        return 0

    protected = _protected_tags(save_checkpoint_path, keep_last_n_raw=keep_last_n_raw, keep_best_raw=keep_best_raw)
    cleaned = 0
    for fname in os.listdir(done_dir):
        if not fname.endswith(".json"):
            continue
        task_path = os.path.join(done_dir, fname)
        try:
            task = load_task(task_path)
            tag = task.get("tag")
            if not isinstance(tag, str) or tag in protected:
                continue

            progress = task.get("progress", {})
            cleanup = progress.get("cleanup", {})
            if not (isinstance(cleanup, dict) and cleanup.get("skipped") is True and cleanup.get("reason") == "protected_for_resume"):
                continue

            step_dir = os.path.join(save_checkpoint_path, tag)
            cleanup_info = _cleanup_raw_files(step_dir, components=None)
            cleanup_info["skipped"] = False
            cleanup_info["deferred_cleanup"] = True
            progress["cleanup"] = cleanup_info
            update_task(task_path, {"progress": progress, "finished_at": time.time()})
            cleaned += 1
        except Exception:
            # best-effort only
            continue

    return cleaned


def _apply_config_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if v is None:
            continue
        out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Background worker to merge checkpoints and push to Hugging Face.")
    parser.add_argument("--save_checkpoint_path", type=str, required=True, help="Checkpoint root, e.g. checkpoints/..")
    parser.add_argument(
        "--queue_dir",
        type=str,
        default=None,
        help="Queue dir (default: `<save_checkpoint_path>/.merge_queue`).",
    )
    parser.add_argument("--hf_repo_id", type=str, default=None, help="HF repo id like `org/model`.")
    parser.add_argument("--hf_private", action="store_true", help="Create HF repo as private (if creating).")
    parser.add_argument(
        "--repo_per_step",
        type=int,
        default=None,
        help="1/0. If 1, upload each checkpoint into its own repo named by step (prefix-step-N).",
    )
    parser.add_argument(
        "--hf_repo_id_per_step_template",
        type=str,
        default=None,
        help="Repo id template when repo_per_step=1 (default: `{hf_repo_id}-step-{step}`).",
    )
    parser.add_argument(
        "--hf_latest_repo_id",
        type=str,
        default=None,
        help="Repo id used for latest upload when repo_per_step=1 (default: `<hf_repo_id>-latest`).",
    )
    parser.add_argument(
        "--merge_components",
        type=str,
        default=None,
        help="Comma-separated component folders to merge (default from queue config or `actor`).",
    )
    parser.add_argument("--keep_last_n_raw", type=int, default=None, help="Keep newest N raw checkpoints (default: 1).")
    parser.add_argument("--keep_best_raw", type=int, default=None, help="1/0. Keep best raw checkpoint too.")
    parser.add_argument("--delete_raw", type=int, default=None, help="1/0. Delete raw files after merge (default: 1).")
    parser.add_argument(
        "--hf_path_in_repo_template",
        type=str,
        default=None,
        help="Upload path template (default: `checkpoints/{tag}/{component}`).",
    )
    parser.add_argument("--update_latest", type=int, default=None, help="1/0. Also upload to latest/ path.")
    parser.add_argument(
        "--hf_latest_path_in_repo_template",
        type=str,
        default=None,
        help="Latest upload path template (default: `latest/{component}`).",
    )
    parser.add_argument("--poll_interval_sec", type=float, default=None, help="Poll interval when queue empty.")
    parser.add_argument("--once", action="store_true", help="Process current queue then exit.")

    args = parser.parse_args()

    save_checkpoint_path = os.path.abspath(args.save_checkpoint_path)
    queue_dir = args.queue_dir or os.path.join(save_checkpoint_path, ".merge_queue")
    queue_dir = os.path.abspath(queue_dir)
    ensure_queue_dirs(queue_dir)

    # Load config persisted by training (if any), then override by CLI.
    persisted = read_queue_config(queue_dir) or {}
    persisted.pop("version", None)

    overrides: dict[str, Any] = {
        "hf_repo_id": args.hf_repo_id,
        "hf_private": bool(args.hf_private) if args.hf_private else None,
        "hf_repo_id_per_step_template": args.hf_repo_id_per_step_template,
        "hf_latest_repo_id": args.hf_latest_repo_id,
        "hf_path_in_repo_template": args.hf_path_in_repo_template,
        "hf_latest_path_in_repo_template": args.hf_latest_path_in_repo_template,
        "poll_interval_sec": args.poll_interval_sec,
    }

    # bool/int overrides
    if args.repo_per_step is not None:
        overrides["repo_per_step"] = bool(int(args.repo_per_step))
    if args.keep_best_raw is not None:
        overrides["keep_best_raw"] = bool(int(args.keep_best_raw))
    if args.delete_raw is not None:
        overrides["delete_raw"] = bool(int(args.delete_raw))
    if args.update_latest is not None:
        overrides["update_latest"] = bool(int(args.update_latest))
    if args.keep_last_n_raw is not None:
        overrides["keep_last_n_raw"] = int(args.keep_last_n_raw)

    cfg = _apply_config_overrides(persisted, overrides)

    merge_components = cfg.get("merge_components") or ["actor"]
    if args.merge_components is not None:
        merge_components = [c.strip() for c in args.merge_components.split(",") if c.strip()]

    # Normalize config and write back (so subsequent workers see it)
    normalized_cfg = {
        "save_checkpoint_path": save_checkpoint_path,
        "queue_dir": queue_dir,
        "hf_repo_id": cfg.get("hf_repo_id"),
        "hf_private": bool(cfg.get("hf_private", False)),
        "repo_per_step": bool(cfg.get("repo_per_step", False)),
        "hf_repo_id_per_step_template": cfg.get("hf_repo_id_per_step_template", "{hf_repo_id}-step-{step}"),
        "hf_latest_repo_id": cfg.get("hf_latest_repo_id"),
        "hf_path_in_repo_template": cfg.get("hf_path_in_repo_template", "checkpoints/{tag}/{component}"),
        "hf_latest_path_in_repo_template": cfg.get("hf_latest_path_in_repo_template", "latest/{component}"),
        "update_latest": bool(cfg.get("update_latest", True)),
        "merge_components": merge_components,
        "keep_last_n_raw": int(cfg.get("keep_last_n_raw", 1)),
        "keep_best_raw": bool(cfg.get("keep_best_raw", True)),
        "delete_raw": bool(cfg.get("delete_raw", True)),
        "poll_interval_sec": float(cfg.get("poll_interval_sec", 30.0)),
    }
    write_queue_config(queue_dir, normalized_cfg)

    # best-effort recover for crashed workers
    recover_running_to_pending(queue_dir)

    while True:
        task_path = claim_next(queue_dir)
        if task_path is None:
            if args.once:
                # Best-effort deferred cleanup before exit.
                _try_deferred_cleanup(
                    queue_dir=queue_dir,
                    save_checkpoint_path=save_checkpoint_path,
                    keep_last_n_raw=normalized_cfg["keep_last_n_raw"],
                    keep_best_raw=normalized_cfg["keep_best_raw"],
                )
                break

            # No tasks: still try deferred cleanup (in case the previously-latest ckpt became deletable).
            _try_deferred_cleanup(
                queue_dir=queue_dir,
                save_checkpoint_path=save_checkpoint_path,
                keep_last_n_raw=normalized_cfg["keep_last_n_raw"],
                keep_best_raw=normalized_cfg["keep_best_raw"],
            )
            time.sleep(normalized_cfg["poll_interval_sec"])
            continue

        try:
            _process_task(
                queue_dir=queue_dir,
                task_path=task_path,
                save_checkpoint_path=save_checkpoint_path,
                merge_components=merge_components,
                hf_repo_id=normalized_cfg["hf_repo_id"],
                repo_per_step=normalized_cfg["repo_per_step"],
                hf_repo_id_per_step_template=normalized_cfg["hf_repo_id_per_step_template"],
                hf_latest_repo_id=normalized_cfg["hf_latest_repo_id"],
                hf_path_in_repo_template=normalized_cfg["hf_path_in_repo_template"],
                update_latest=normalized_cfg["update_latest"],
                hf_latest_path_in_repo_template=normalized_cfg["hf_latest_path_in_repo_template"],
                keep_last_n_raw=normalized_cfg["keep_last_n_raw"],
                keep_best_raw=normalized_cfg["keep_best_raw"],
                delete_raw=normalized_cfg["delete_raw"],
                hf_private=normalized_cfg["hf_private"],
            )
        except Exception as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            try:
                update_task(task_path, {"status": "failed", "error": err, "finished_at": time.time()})
                finish_task(queue_dir, task_path, status="failed", error=err)
            except Exception:
                # last resort: do not crash the whole worker
                pass
        finally:
            gc.collect()


if __name__ == "__main__":
    main()


