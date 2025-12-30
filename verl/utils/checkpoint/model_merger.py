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

"""
Utilities to merge EasyR1/veRL-style FSDP shard checkpoints into HuggingFace format,
and (optionally) upload the merged folder to Hugging Face Hub.

This is extracted from `scripts/model_merger.py` so it can be reused by background workers.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
)


def merge_by_placement(tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
    if placement.is_replicate():
        return tensors[0]
    if placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    if placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    raise ValueError(f"Unsupported placement: {placement}")


def merge_fsdp_shards_to_state_dict(local_dir: str, *, torch_dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """
    Merge `model_world_size_*_rank_*.pt` shards under `local_dir` into a single full state_dict.

    Notes:
    - It loads shards on CPU (`map_location="cpu"`) and casts to `torch_dtype`.
    - It supports pure FSDP and DDP+FSDP (replicated at DDP dimension).
    """

    assert not local_dir.endswith("huggingface"), "The local_dir should not end with `huggingface`."

    rank = 0
    world_size: Optional[str] = None
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break
    assert world_size, f"No model shard found in {local_dir} with format `model_world_size_*_rank_0.pt`."

    rank0_weight_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    state_dict_rank0 = torch.load(rank0_weight_path, map_location="cpu", weights_only=False)

    pivot_key = sorted(state_dict_rank0.keys())[0]
    pivot_weight = state_dict_rank0[pivot_key]
    if isinstance(pivot_weight, DTensor):
        device_mesh = pivot_weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}."

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp only
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    model_state_dict_lst: list[dict[str, object]] = [state_dict_rank0]
    model_state_dict_lst.extend([""] * (total_shards - 1))  # type: ignore[list-item]

    def process_one_shard(rank_: int) -> None:
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank_}.pt")
        model_state_dict_lst[rank_] = torch.load(model_path, map_location="cpu", weights_only=False)

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
        for rank_ in range(1, total_shards):
            executor.submit(process_one_shard, rank_)

    # Collect and merge tensors
    state_dict: dict[str, list[torch.Tensor] | torch.Tensor] = {}
    param_placements: dict[str, tuple[Placement, ...]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for sd in model_state_dict_lst:
            tensor = sd.pop(key)  # type: ignore[union-attr]
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.to(dtype=torch_dtype))  # type: ignore[union-attr]
                placements = tuple(tensor.placements)
                if mesh_dim_names[0] == "ddp":
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                assert isinstance(tensor, torch.Tensor)
                state_dict[key].append(tensor.to(dtype=torch_dtype))  # type: ignore[union-attr]

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            continue

        if key in param_placements:
            placements = param_placements[key]
            if len(mesh_shape) == 1:
                assert len(placements) == 1
                shards = state_dict[key]
                assert isinstance(shards, list)
                state_dict[key] = merge_by_placement(shards, placements[0])
            else:
                raise NotImplementedError("FSDP + TP is not supported yet.")
        else:
            shards = state_dict[key]
            assert isinstance(shards, list)
            state_dict[key] = torch.cat(shards, dim=0)

    # Cast type for return
    merged_state_dict: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        assert isinstance(v, torch.Tensor), f"Unexpected non-tensor value for key={k}: {type(v)}"
        merged_state_dict[k] = v
    return merged_state_dict


def save_state_dict_to_hf_folder(local_dir: str, state_dict: dict[str, torch.Tensor], *, torch_dtype: torch.dtype) -> str:
    """
    Save merged `state_dict` to `<local_dir>/huggingface` using the config/tokenizer already saved there.
    Returns the hf folder path.
    """

    hf_path = os.path.join(local_dir, "huggingface")
    config: PretrainedConfig = AutoConfig.from_pretrained(hf_path)
    architectures: list[str] = getattr(config, "architectures", ["Unknown"])

    if "ForTokenClassification" in architectures[0]:
        AutoClass = AutoModelForTokenClassification
    elif "ForConditionalGeneration" in architectures[0]:
        AutoClass = AutoModelForImageTextToText
    elif "ForCausalLM" in architectures[0]:
        AutoClass = AutoModelForCausalLM
    else:
        raise NotImplementedError(f"Unknown architecture {architectures}.")

    with torch.device("meta"):
        model: PreTrainedModel = AutoClass.from_config(config, torch_dtype=torch_dtype)
    assert isinstance(model, PreTrainedModel)
    model.to_empty(device="cpu")

    model.save_pretrained(hf_path, state_dict=state_dict)
    return hf_path


def merge_fsdp_checkpoint_to_hf(local_dir: str, *, torch_dtype: torch.dtype = torch.bfloat16) -> str:
    """
    Merge shards under `local_dir` and save to `<local_dir>/huggingface`.
    Returns the HF folder path.
    """

    state_dict = merge_fsdp_shards_to_state_dict(local_dir, torch_dtype=torch_dtype)
    try:
        return save_state_dict_to_hf_folder(local_dir, state_dict, torch_dtype=torch_dtype)
    finally:
        # help GC
        del state_dict


def upload_folder_to_huggingface(
    *,
    local_path: str,
    repo_id: str,
    repo_type: str = "model",
    path_in_repo: Optional[str] = None,
    commit_message: Optional[str] = None,
    private: bool = False,
) -> None:
    """
    Upload a local folder to Hugging Face Hub.

    - If repo does not exist, it will be created (private configurable).
    - If `path_in_repo` is provided, uploads into that subfolder.
    """

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type=repo_type)

    kwargs = {}
    if path_in_repo is not None:
        kwargs["path_in_repo"] = path_in_repo
    if commit_message is not None:
        kwargs["commit_message"] = commit_message

    api.upload_folder(repo_id=repo_id, folder_path=local_path, repo_type=repo_type, **kwargs)


