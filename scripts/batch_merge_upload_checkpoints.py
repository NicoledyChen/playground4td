#!/usr/bin/env python3
"""
Batch merge + upload EasyR1/veRL checkpoints into HuggingFace format.

It scans `<save_checkpoint_path>/global_step_*` folders, merges each checkpoint component
(default: auto-detect subfolders that contain a `huggingface/` directory), uploads the merged
HF folder to Hugging Face Hub, and then deletes everything except the merged HF folder to save disk.

It will KEEP the newest checkpoint folder unmerged (default: keep_last_n=1), so you can still resume.

Example:
  export HF_TOKEN=...   # or HUGGINGFACE_HUB_TOKEN=...
  python scripts/batch_merge_upload_checkpoints.py \
    --save_checkpoint_path checkpoints/easy_r1/exp_name \
    --hf_repo_id org/model \
    --hf_path_in_repo_template "checkpoints/{tag}/{component}"
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Optional

from verl.utils.checkpoint.model_merger import merge_fsdp_checkpoint_to_hf, upload_folder_to_huggingface


STEP_RE = re.compile(r"^global_step_(\d+)$")
RANK0_SHARD_RE = re.compile(r"^model_world_size_\d+_rank_0\.pt$")


@dataclass(frozen=True)
class StepTag:
    step: int
    tag: str


def _list_step_tags(save_checkpoint_path: str) -> list[StepTag]:
    if not os.path.isdir(save_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint root not found: {save_checkpoint_path}")
    out: list[StepTag] = []
    for name in os.listdir(save_checkpoint_path):
        m = STEP_RE.match(name)
        if not m:
            continue
        full = os.path.join(save_checkpoint_path, name)
        if not os.path.isdir(full):
            continue
        out.append(StepTag(step=int(m.group(1)), tag=name))
    out.sort(key=lambda x: x.step)
    return out


def _detect_components(step_dir: str) -> list[str]:
    comps: list[str] = []
    for name in os.listdir(step_dir):
        if name.startswith("."):
            continue
        comp_dir = os.path.join(step_dir, name)
        if not os.path.isdir(comp_dir):
            continue
        # Heuristic: only treat dirs with a saved HF config/tokenizer as mergeable components.
        if os.path.isdir(os.path.join(comp_dir, "huggingface")):
            comps.append(name)
    comps.sort()
    return comps


def _hf_has_weights(hf_dir: str) -> bool:
    """Best-effort check whether a HF folder already contains merged model weights."""
    if not os.path.isdir(hf_dir):
        return False
    for fname in os.listdir(hf_dir):
        # Common outputs of `save_pretrained`
        if fname.endswith(".safetensors"):
            return True
        if fname.startswith("pytorch_model") and fname.endswith(".bin"):
            return True
    return False


def _has_rank0_shard(local_dir: str) -> bool:
    if not os.path.isdir(local_dir):
        return False
    return any(RANK0_SHARD_RE.match(fn) for fn in os.listdir(local_dir))


def _format_path_in_repo(template: str, *, tag: str, step: int, component: str) -> str:
    # Available fields: {tag}, {step}, {step_padded}, {component}
    return template.format(tag=tag, step=step, step_padded=f"{step:08d}", component=component)


def _delete_except(path: str, keep_names: set[str], *, dry_run: bool) -> None:
    for name in os.listdir(path):
        if name in keep_names:
            continue
        target = os.path.join(path, name)
        if dry_run:
            print(f"[dry-run] delete: {target}")
            continue
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
        else:
            try:
                os.remove(target)
            except FileNotFoundError:
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_checkpoint_path", required=True, type=str, help="e.g. checkpoints/<proj>/<exp>")
    ap.add_argument("--hf_repo_id", required=True, type=str, help="HF repo id like `org/model`")
    ap.add_argument(
        "--hf_path_in_repo_template",
        type=str,
        default="checkpoints/{tag}/{component}",
        help="Upload path template inside repo. Fields: {tag},{step},{step_padded},{component}.",
    )
    ap.add_argument("--hf_private", action="store_true", help="Create repo as private (if creating).")
    ap.add_argument(
        "--components",
        nargs="*",
        default=None,
        help="Optional component subfolders to merge (e.g. actor critic). Default: auto-detect.",
    )
    ap.add_argument(
        "--keep_last_n",
        type=int,
        default=1,
        help="Keep newest N checkpoints untouched (not merged, not deleted). Default: 1.",
    )
    ap.add_argument("--no_delete", action="store_true", help="Do not delete raw files after merge.")
    ap.add_argument("--dry_run", action="store_true", help="Print actions without merging/uploading/deleting.")
    ap.add_argument("--continue_on_error", action="store_true", help="Continue processing next checkpoint on errors.")
    args = ap.parse_args()

    save_checkpoint_path = os.path.abspath(args.save_checkpoint_path)
    steps = _list_step_tags(save_checkpoint_path)
    if not steps:
        print(f"No global_step_* folders found under {save_checkpoint_path}", file=sys.stderr)
        return 1

    keep_last_n = max(0, int(args.keep_last_n))
    protected = {s.tag for s in steps[-keep_last_n:]} if keep_last_n > 0 else set()
    to_process = [s for s in steps if s.tag not in protected]

    print(f"[batch-merge] save_checkpoint_path: {save_checkpoint_path}")
    print(f"[batch-merge] found steps: {[s.step for s in steps]}")
    if protected:
        print(f"[batch-merge] protected (skip merge/delete): {sorted(protected)}")
    print(f"[batch-merge] will process: {[s.step for s in to_process]}")
    print()

    for s in to_process:
        tag = s.tag
        step_dir = os.path.join(save_checkpoint_path, tag)
        try:
            comps = list(args.components) if args.components else _detect_components(step_dir)
            if not comps:
                print(f"[batch-merge] {tag}: no mergeable components found (skip)")
                continue

            print(f"[batch-merge] {tag}: components={comps}")
            processed_components: list[str] = []

            for comp in comps:
                local_dir = os.path.join(step_dir, comp)
                hf_dir = os.path.join(local_dir, "huggingface")
                if not os.path.isdir(hf_dir):
                    print(f"[batch-merge]  - {comp}: missing {hf_dir} (skip)")
                    continue

                raw_ok = _has_rank0_shard(local_dir)
                already_merged = _hf_has_weights(hf_dir)

                if args.dry_run:
                    print(
                        f"[dry-run]  - {comp}: raw_rank0={raw_ok}, already_merged={already_merged} -> merge+upload+cleanup"
                    )
                    processed_components.append(comp)
                    continue

                if not already_merged:
                    if not raw_ok:
                        raise RuntimeError(
                            f"{tag}/{comp}: no rank-0 shard found and HF weights not present; cannot merge."
                        )
                    print(f"[batch-merge]  - {comp}: merging shards -> huggingface/")
                    hf_path = merge_fsdp_checkpoint_to_hf(local_dir)
                else:
                    hf_path = hf_dir
                    print(f"[batch-merge]  - {comp}: already merged (found weights), skip merge")

                # Upload
                path_in_repo = _format_path_in_repo(args.hf_path_in_repo_template, tag=tag, step=s.step, component=comp)
                if path_in_repo == "":
                    path_in_repo = None
                print(f"[batch-merge]  - {comp}: uploading to {args.hf_repo_id}::{path_in_repo or '/'}")
                upload_folder_to_huggingface(
                    local_path=hf_path,
                    repo_id=args.hf_repo_id,
                    path_in_repo=path_in_repo,
                    private=bool(args.hf_private),
                    commit_message=f"batch-merge {tag} ({comp})",
                )

                # Cleanup component dir: keep only huggingface/
                if not args.no_delete:
                    print(f"[batch-merge]  - {comp}: cleanup local files (keep huggingface/ only)")
                    _delete_except(local_dir, keep_names={"huggingface"}, dry_run=False)

                processed_components.append(comp)

            # Cleanup step dir: delete everything except processed component dirs (which now only contain huggingface/)
            if (not args.no_delete) and processed_components:
                keep = set(processed_components)
                print(f"[batch-merge] {tag}: cleanup step folder (keep {sorted(keep)} only)")
                _delete_except(step_dir, keep_names=keep, dry_run=False)

            print(f"[batch-merge] {tag}: done")
            print()

        except Exception as e:
            print(f"[batch-merge] {tag}: ERROR: {e}", file=sys.stderr)
            if not args.continue_on_error:
                return 2
            print("[batch-merge] continue_on_error=true, moving on.\n", file=sys.stderr)
            continue

    print("[batch-merge] all done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


