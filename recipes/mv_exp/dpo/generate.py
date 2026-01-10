from __future__ import annotations

import argparse
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Distributed settings
EFFECTIVE_BATCH_SIZE = 128
NUM_GPUS = 4


class ModelSize(Enum):
    S1_7B = "1.7b"


class GPUType(Enum):
    A100 = "A100"
    H100 = "H100"

    def max_batch_size(
        self,
        model_size: ModelSize,
    ) -> int:
        return {
            self.A100: {
                ModelSize.S1_7B: -1,  # TODO: add max batch size
            },
            self.H100: {
                ModelSize.S1_7B: 32,
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


def make_base_config() -> dict:
    """Returns a mutable base configuration."""

    return {
        # Model arguments
        "model_name_or_path": "",
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "trust_remote_code": True,
        # Data arguments
        "chat_template": CHAT_TEMPLATE,
        "dataset_mixture": {"datasets": []},
        "dataset_num_proc": 32,
        # SFT trainer config
        "eval_strategy": "no",
        "remove_unused_columns": True,
        # Saving
        "output_dir": "",
        "overwrite_output_dir": False,
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 2,
        # Reporting
        "log_level": "info",
        "report_to": ["wandb"],
        "logging_steps": 10,
        "logging_strategy": "steps",
        # Dynamics
        "seed": 42,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": False,
        },
        "learning_rate": 5.0e-7,
        "lr_scheduler_type": "linear",
        "max_length": 2048,
        "beta": 5,
        "warmup_ratio": 0.1,
        "per_device_train_batch_size": 0,
        "gradient_accumulation_steps": 0,
        "num_train_epochs": 1,
        # "max_steps": 10, # NOTE: For debugging
    }


################################################################################
# Data mixtures
################################################################################

DATA_MIXTURES: List[Dict[str, Any]] = [
    {
        "name": "allenai-llama-3.1-tulu-3-8b-preference-mixture",
        "datasets": [
            {
                "id": "allenai/llama-3.1-tulu-3-8b-preference-mixture",
                "config": "default",
                "split": "train",
                "columns": ["chosen", "rejected"],
            },
        ],
    },
]


################################################################################
# Models
################################################################################

MODELS: List[Dict[str, Any]] = [
    {
        "name": "ontocord-1.7b-MixtureVitae-300BT-v1",
        "path": "/home/hk-project-p0024002/fr_ae293/work/hkfswork/fr_ae293-ra/alignment-handbook/results/mv_exp/sft/ontocord-1.7b-MixtureVitae-300BT-v1_allenai-tulu-3-sft-mixture",
        "size": ModelSize.S1_7B,
    },
    {
        "name": "ontocord-1.7b-MixtureVitae-300BT-v1-16k",
        "path": "/home/hk-project-p0024002/fr_ae293/work/hkfswork/fr_ae293-ra/alignment-handbook/results/mv_exp/sft/ontocord-1.7b-MixtureVitae-300BT-v1-16k_allenai-tulu-3-sft-mixture",
        "size": ModelSize.S1_7B,
    },
    {
        "name": "open-sci-open-sci-ref-v0.01-1.7b-nemotron-hq-300B-16384",
        "path": "/home/hk-project-p0024002/fr_ae293/work/hkfswork/fr_ae293-ra/alignment-handbook/results/mv_exp/sft/open-sci-open-sci-ref-v0.01-1.7b-nemotron-hq-300B-16384_allenai-tulu-3-sft-mixture",
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
    config["model_name_or_path"] = model["path"]
    if "max_length" in model:
        config["max_length"] = model["max_length"]

    model_slug = sanitize_model_name(model["name"])
    config["output_dir"] = (
        f"results/mv_exp/dpo/{model_slug}_{mixture['name']}_{gpu_type.value}"
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


def write_slurm_script(
    recipe_path: Path,
    model_name: str,
    mixture_name: str,
    gpu_type: GPUType,
    target_dir: Path,
) -> Path:
    """Generate a SLURM script for running this SFT recipe."""
    model_slug = sanitize_model_name(model_name)
    mixture_slug = sanitize_model_name(mixture_name)
    gpu_slug = gpu_type.value
    script_slug = f"{model_slug}_{mixture_slug}_{gpu_slug}"
    script_name = f"{script_slug}.sh"
    script_path = target_dir / script_name

    partition = f"accelerated{'-h100' if gpu_type == GPUType.H100 else ''}"

    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={script_slug}_dpo
#SBATCH --output=slurm_logs/mv_exp/dpo/{script_slug}/%j.%x.%N.out
#SBATCH --error=slurm_logs/mv_exp/dpo/{script_slug}/%j.%x.%N.err
#SBATCH --time=00-48:00:00
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{NUM_GPUS}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alielganzory@hotmail.com

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

source handbook/bin/activate

accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes={NUM_GPUS} \
    scripts/dpo.py --config {recipe_path}
"""

    script_path.parent.mkdir(parents=True, exist_ok=True)
    with script_path.open("w", encoding="utf-8") as fh:
        fh.write(slurm_content)
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
                slurm_path = write_slurm_script(
                    recipe_path=recipe_path,
                    model_name=model["name"],
                    mixture_name=mixture["name"],
                    gpu_type=gpu_type,
                    target_dir=args.output_dir,
                )
                print(f"  SLURM script written to: {slurm_path.name}")


if __name__ == "__main__":
    main()
