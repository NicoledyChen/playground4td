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
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    @dataclass
    class AutoMergeConfig:
        """
        Auto-merge FSDP shard checkpoints into HuggingFace format in a background worker process.

        The training process ONLY enqueues a tag (e.g. `global_step_123`) into a persistent on-disk queue.
        A separate worker consumes the queue and performs:
        - merge shards -> `huggingface/` folder
        - optional push to Hugging Face Hub
        - optional cleanup of raw shard/optimizer files (while keeping the latest checkpoint raw for resume)
        """

        enabled: bool = False
        """Enable enqueueing merge tasks when saving checkpoints."""

        queue_dir: Optional[str] = None
        """Queue directory. If not set, defaults to `<save_checkpoint_path>/.merge_queue`."""

        hf_repo_id: Optional[str] = None
        """Hugging Face repo id, e.g. `org/model`. If not set, worker can still merge locally but won't upload."""

        repo_per_step: bool = False
        """
        If True, upload each checkpoint into its own Hugging Face repo.

        Example repo names:
        - `<hf_repo_id>-step-1`
        - `<hf_repo_id>-step-2`
        """

        hf_repo_id_per_step_template: str = "{hf_repo_id}-step-{step}"
        """
        Repo id template used when `repo_per_step=true`.

        Available fields:
        - `{hf_repo_id}`: the `hf_repo_id` value (often used as prefix/base)
        - `{tag}`: checkpoint tag (e.g. `global_step_123`)
        - `{step}`: int step index (only available when tag matches `global_step_\\d+`)
        - `{step_padded}`: 8-digit padded step string (e.g. `00000123`)
        """

        hf_latest_repo_id: Optional[str] = None
        """
        Optional repo id used for `update_latest=true` when `repo_per_step=true`.
        If not set, defaults to `<hf_repo_id>-latest`.
        """

        hf_path_in_repo_template: Optional[str] = "checkpoints/{tag}/{component}"
        """
        Remote path template when uploading in a single repo mode (`repo_per_step=false`).

        Available fields: `{tag}`, `{component}`, `{step}`, `{step_padded}`.

        Set to `null` to upload to the repo root.
        """

        update_latest: bool = True
        """Also upload a copy to a `latest/` path (useful for always having the newest model)."""

        hf_latest_path_in_repo_template: Optional[str] = "latest/{component}"
        """
        Remote path template for latest upload (single repo mode).
        Available fields: `{component}`, `{tag}`, `{step}`, `{step_padded}`.
        """

        merge_components: Tuple[str, ...] = ("actor",)
        """Which sub-folders under each `global_step_*` to merge. Defaults to only `actor`."""

        keep_last_n_raw: int = 1
        """Keep the newest N checkpoints in raw (unmerged shard) format for resume. Default: 1."""

        keep_best_raw: bool = True
        """Keep the best checkpoint (from `checkpoint_tracker.json`) raw as well."""

        delete_raw: bool = True
        """After successful merge (+ optional upload), delete raw shard/optimizer files for non-protected checkpoints."""

        delete_after_upload: bool = False
        """Only clean local checkpoints after a successful upload."""

        delete_hf_after_upload: bool = False
        """Delete merged Hugging Face folders after a successful upload."""

        poll_interval_sec: float = 30.0
        """Worker poll interval (seconds) when the queue is empty. Stored for convenience."""

    auto_merge: AutoMergeConfig = field(default_factory=AutoMergeConfig)

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")

        # Fill auto-merge queue dir after `save_checkpoint_path` is resolved.
        if self.auto_merge.queue_dir is None:
            self.auto_merge.queue_dir = os.path.join(self.save_checkpoint_path, ".merge_queue")
        elif not os.path.isabs(self.auto_merge.queue_dir):
            # treat as relative to save_checkpoint_path
            self.auto_merge.queue_dir = os.path.abspath(os.path.join(self.save_checkpoint_path, self.auto_merge.queue_dir))


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
