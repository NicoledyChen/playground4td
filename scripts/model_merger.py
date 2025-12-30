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

import argparse

from verl.utils.checkpoint.model_merger import merge_fsdp_checkpoint_to_hf, upload_folder_to_huggingface


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str, help="The path for your saved model")
    parser.add_argument("--hf_upload_path", default=False, type=str, help="The path of the huggingface repo to upload")
    parser.add_argument(
        "--hf_path_in_repo",
        default=None,
        type=str,
        help="Optional subfolder path in the HF repo (e.g. `checkpoints/global_step_100/actor`).",
    )
    parser.add_argument("--hf_private", action="store_true", help="Create the HF repo as private (if creating).")
    parser.add_argument(
        "--hf_commit_message", default=None, type=str, help="Optional commit message for HF upload."
    )
    args = parser.parse_args()

    local_dir: str = args.local_dir
    print(f"Merging checkpoint shards under {local_dir} ...")
    hf_path = merge_fsdp_checkpoint_to_hf(local_dir)
    print(f"Saved HuggingFace format model to {hf_path}.")

    if args.hf_upload_path:
        print(f"Uploading {hf_path} to Hugging Face repo {args.hf_upload_path} ...")
        upload_folder_to_huggingface(
            local_path=hf_path,
            repo_id=args.hf_upload_path,
            path_in_repo=args.hf_path_in_repo,
            commit_message=args.hf_commit_message,
            private=args.hf_private,
        )
        print("Upload completed.")
