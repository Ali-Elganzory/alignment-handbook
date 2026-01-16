from __future__ import annotations

import argparse
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import yaml

# HuggingFace
UPLOAD_TO_HF = False
HF_USERNAME = "ali-elganzory"

# Distribution
EFFECTIVE_BATCH_SIZE = 128
NUM_GPUS = 4


class ModelSize(Enum):
    S1_7B = "1.7b"


class GPUType(Enum):
    A100 = "A100"
    H100 = "H100"
    H200 = "H200"

    @property
    def active(self) -> list[GPUType]:
        return [self.H100]

    def max_batch_size(
        self,
        model_size: ModelSize,
    ) -> int:
        return {
            self.A100: {
                ModelSize.S1_7B: 4,
            },
            self.H100: {
                ModelSize.S1_7B: 16,
            },
            self.H200: {
                ModelSize.S1_7B: 16,
            },
        }[self][model_size]

    def accumulation_steps(
        self,
        model_size: ModelSize,
    ) -> int:
        one_step_batch_size = self.max_batch_size(model_size) * NUM_GPUS
        if EFFECTIVE_BATCH_SIZE % one_step_batch_size != 0:
            raise ValueError(
                f"Effective batch size {EFFECTIVE_BATCH_SIZE} is not divisible"
                f" by one step batch size {one_step_batch_size}"
            )
        return EFFECTIVE_BATCH_SIZE // one_step_batch_size


# Tulu 3 chat template
CHAT_TEMPLATE = """
{%- for message in messages -%}
	{%- if message["role"] == "system" -%}
		{{- "<|system|>\n" + message["content"] + "\n" -}}
	{%- elif message["role"] == "user" -%}
		{{- "<|user|>\n" + message["content"] + "\n" -}}
	{%- elif message["role"] == "assistant" -%}
		{%- if not loop.last -%}
			{{- "<|assistant|>\n" + message["content"] + eos_token + "\n" -}}
		{%- else -%}
			{{- "<|assistant|>\n" + message["content"] + eos_token -}}
		{%- endif -%}
	{%- endif -%}
	{%- if loop.last and add_generation_prompt -%}
		{{- "<|assistant|>\n" -}}
	{%- endif -%}
{%- endfor -%}
"""

ADDITIONAL_SPECIAL_TOKENS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


def make_base_config() -> dict:
    """Returns a mutable base configuration."""

    return {
        # Model arguments
        "model_name_or_path": "",
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "trust_remote_code": True,
        # Data arguments
        "chat_template": CHAT_TEMPLATE,
        "additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS,
        "dataset_mixture": {"datasets": []},
        "dataset_num_proc": 32,
        # SFT trainer config
        "eval_strategy": "no",
        "remove_unused_columns": True,
        "dataset_kwargs": {
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        # Saving
        "output_dir": "",
        "overwrite_output_dir": False,
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 2,
        "push_to_hub": UPLOAD_TO_HF,
        # Reporting
        "log_level": "info",
        "report_to": ["wandb"],
        "logging_steps": 10,
        "logging_strategy": "steps",
        # Dynamics
        "seed": 42,
        "bf16": True,
        "gradient_checkpointing": True,
        "learning_rate": 5.0e-6,
        "lr_scheduler_type": "linear",
        "max_length": 4096,
        "warmup_ratio": 0.03,
        # TODO: MODIFY TO EXPERIMENT TARGET
        "num_train_epochs": 2,
        # "max_steps": 10,
    }


################################################################################
# Data mixtures
################################################################################

DATA_MIXTURES: List[Dict[str, Any]] = [
    {
        "name": "tulu-3-sft-mixture-decontaminated",
        "datasets": [
            {
                "id": "ali-elganzory/tulu-3-sft-mixture-decontaminated",
                "config": "default",
                "split": "train",
                "columns": ["messages"],
            },
        ],
    },
]


################################################################################
# Models
################################################################################

MODELS: List[Dict[str, Any]] = [
    {
        "name": "HuggingFaceTB/SmolLM2-1.7B",
        "size": ModelSize.S1_7B,
    },
    {
        "name": "Qwen/Qwen2.5-1.5B",
        "size": ModelSize.S1_7B,
    },
    {
        "name": "Qwen/Qwen3-1.7B-Base",
        "size": ModelSize.S1_7B,
    },
]

################################################################################
# Generate recipes
################################################################################


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def create_recipe(
    model: Dict[str, Any],
    mixture: Dict[str, Any],
    gpu_type: GPUType,
) -> dict:
    config = deepcopy(make_base_config())
    config["dataset_mixture"]["datasets"] = mixture["datasets"]
    config["per_device_train_batch_size"] = gpu_type.max_batch_size(model["size"])
    config["gradient_accumulation_steps"] = gpu_type.accumulation_steps(model["size"])
    config["model_name_or_path"] = model["name"]

    if UPLOAD_TO_HF:
        config["hub_model_id"] = (
            f"{HF_USERNAME}/{model['name'].split('/')[-1]}-SFT-Tulu3-decontaminated"
        )
        config["hub_strategy"] = "every_save"

    if "max_length" in model:
        config["max_length"] = model["max_length"]

    model_slug = sanitize_model_name(model["name"])
    config["output_dir"] = (
        f"results/mv_exp/sft/{model_slug}_{mixture['name']}_{gpu_type.value}"
    )

    return config


def write_recipe(
    recipe: dict,
    model_name: str,
    mixture_name: str,
    gpu_type: GPUType,
    target_dir: Path,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sanitize_model_name(model_name)}_{mixture_name}_{gpu_type.value}.yaml"
    output_path = target_dir / filename
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(recipe, handle, sort_keys=False)
    return output_path


def write_script(
    recipe_path: Path,
    model_name: str,
    mixture_name: str,
    gpu_type: GPUType,
    target_dir: Path,
) -> Path:
    """
    Generate a bash script for running this SFT recipe on a cloud (lambda) single node;
    this does NOT use SLURM and assumes only one machine and that you use accelerate with a config for single-node.
    """
    model_slug = sanitize_model_name(model_name)
    mixture_slug = sanitize_model_name(mixture_name)
    gpu_slug = gpu_type.value
    script_slug = f"{model_slug}_{mixture_slug}_{gpu_slug}"
    script_name = f"{script_slug}.sh"
    script_path = target_dir / script_name
    grad_acc_steps = gpu_type.accumulation_steps(
        next(model for model in MODELS if model["name"] == model_name)["size"]
    )

    # The script assumes .venv exists and is correct.
    script_content = f"""#!/bin/bash
set -e

# Activate environment
if [ -f .env ]; then export $(grep -v '^#' .env | xargs); fi
source .venv/bin/activate

echo "Starting single-node SFT training on model: {model_name}, dataset: {mixture_name}, gpu type: {gpu_type.value}"
echo "Recipe config: {recipe_path}"

accelerate launch \\
    --config_file recipes/accelerate_configs/zero3.yaml \\
    --num_processes {NUM_GPUS} \\
    --gradient_accumulation_steps {grad_acc_steps} \\
    scripts/sft.py --config {recipe_path}

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training finished successfully!"
    exit 0
else
    echo "Training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
"""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with script_path.open("w", encoding="utf-8") as fh:
        fh.write(script_content)
    script_path.chmod(0o770)
    return script_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate HuggingFace TRL recipes and Slurm scripts for each model/data mixture combination."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to place generated YAML and SLURM files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for model in MODELS:
        for mixture in DATA_MIXTURES:
            for gpu_type in GPUType:
                recipe = create_recipe(model, mixture, gpu_type)
                recipe_path = write_recipe(
                    recipe,
                    model["name"],
                    mixture["name"],
                    gpu_type,
                    args.output_dir,
                )
                print(
                    f"✓ Recipe: {recipe_path.name}\n"
                    f"   Model: {model['name']}\n"
                    f"   Mixture: {mixture['name']}\n"
                    f"   GPU: {gpu_type.value}"
                )
                script_path = write_script(
                    recipe_path=recipe_path,
                    model_name=model["name"],
                    mixture_name=mixture["name"],
                    gpu_type=gpu_type,
                    target_dir=args.output_dir,
                )
                print(f"  Script written to: {script_path.name}")


if __name__ == "__main__":
    main()
