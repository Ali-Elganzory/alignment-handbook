from __future__ import annotations

import argparse
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import yaml

# SLURM
SLURM_ACCOUNT = "transfernetx"
SLURM_PARTITION = "booster"
SLURM_MAX_TIME = "00-24:00:00"

# HuggingFace
UPLOAD_TO_HF = False
HF_USERNAME = "ali-elganzory"

# Distribution
EFFECTIVE_BATCH_SIZE = 128
NUM_GPUS = 4
NUM_NODES = 2


class ModelSize(Enum):
    S1_7B = "1.7b"


class GPUType(Enum):
    A100 = "A100"
    # H100 = "H100"

    def max_batch_size(
        self,
        model_size: ModelSize,
    ) -> int:
        return {
            self.A100: {
                ModelSize.S1_7B: 4,
            },
            # self.H100: {
            #     ModelSize.S1_7B: 16,
            # },
        }[self][model_size]

    def accumulation_steps(
        self,
        model_size: ModelSize,
    ) -> int:
        one_step_batch_size = self.max_batch_size(model_size) * NUM_GPUS * NUM_NODES
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
        "torch_dtype": "bfloat16",
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
            f"{HF_USERNAME}/{model['name'].split('/')[-1]}-SFT-Tulu3"
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
    grad_acc_steps = gpu_type.accumulation_steps(
        next(model for model in MODELS if model["name"] == model_name)["size"]
    )

    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={script_slug}_sft
#SBATCH --output=slurm_logs/mv_exp/sft/{script_slug}/%j.%x.%N.out
#SBATCH --error=slurm_logs/mv_exp/sft/{script_slug}/%j.%x.%N.err
#SBATCH --time={SLURM_MAX_TIME}
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --nodes={NUM_NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{NUM_GPUS}
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alielganzory@hotmail.com
#SBATCH --open-mode=append
#SBATCH --signal=B:SIGUSR1@90

# -----------------------------------------------------------------------------
# DISTRIBUTED CONFIGURATION
# -----------------------------------------------------------------------------
# 1. Get the list of nodes
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)

# 2. Identify the Master Node (The first one in the list)
MASTER_NODE=$(echo $NODELIST | head -n 1)

# 3. Resolve that specific hostname to an IPv4 address using Python
# This runs once on the launch node and hardcodes the IP for everyone.
MASTER_ADDR=$(python -c "import socket; print(socket.gethostbyname('$MASTER_NODE'))")

MASTER_PORT=29500

# 4. Strict Network Bindings (Force IPv4 & InfiniBand)
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_SOCKET_FAMILY=AF_INET

# Force IPv4 just to be safe
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET

# 4. Calculate World Size
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE={NUM_GPUS}
WORLD_SIZE=$(($NUM_NODES * $GPUS_PER_NODE))

echo "Master Node: $MASTER_ADDR"
echo "Network Interface: $NCCL_SOCKET_IFNAME"

# -----------------------------------------------------------------------------
# SELF-HEALING & RETRY LOGIC
# -----------------------------------------------------------------------------
RETRY_FILE="retry_count_${{SLURM_JOB_NAME}}.txt"
MAX_RETRIES=10
if [ ! -f "$RETRY_FILE" ]; then echo "0" > "$RETRY_FILE"; fi
CURRENT_RETRIES=$(cat "$RETRY_FILE")

handler() {{
    echo "Function 'handler' triggered."
    if [ "$CURRENT_RETRIES" -ge "$MAX_RETRIES" ]; then
        echo "Max retries reached. Stopping."
        rm "$RETRY_FILE"
        exit 1
    fi
    echo $((CURRENT_RETRIES + 1)) > "$RETRY_FILE"
    sbatch $0
    exit 0
}}
trap 'handler' SIGUSR1

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------
module load CUDA
if [ -f .env ]; then export $(grep -v '^#' .env | xargs); fi
source .venv/bin/activate

# DEFINE THE LAUNCHER
# We wrap 'accelerate launch' inside 'srun'.
# srun ensures this runs exactly once per node.
# accelerate then manages the local GPUs on that node.

export LAUNCHER="accelerate launch \\
    --config_file recipes/accelerate_configs/zero3.yaml \\
    --gradient_accumulation_steps {grad_acc_steps} \\
    --num_machines $NUM_NODES \\
    --num_processes $WORLD_SIZE \\
    --main_process_ip $MASTER_ADDR \\
    --main_process_port $MASTER_PORT \\
    --machine_rank \$SLURM_PROCID \\
    --same_network \\
    --rdzv_backend c10d \\
    --rdzv_conf \"rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT\" \\
    --max_restarts 0 \\
    --role \$(hostname -s): "

export CMD="scripts/sft.py --config {recipe_path}"

# RUN IT
# --ntasks=$NUM_NODES means "Run 1 copy of this command per node"
# --kill-on-bad-exit=1 means "If one node crashes, kill them all"
srun --ntasks=$NUM_NODES --kill-on-bad-exit=1 bash -c "$LAUNCHER $CMD" &

CHILD_PID=$!
wait $CHILD_PID
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Job finished successfully!"
    rm "$RETRY_FILE"
    exit 0
else
    echo "Job Failed with Exit Code $EXIT_CODE"
    handler
fi
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
