from __future__ import annotations

import argparse
import re
import subprocess
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from jinja2 import Environment, FileSystemLoader
import yaml

# Debug mode
DEBUG_MODE = False

# SLURM
SLURM_ACCOUNT = "reformo"
SLURM_MAX_TIME = "00-01:00:00" if DEBUG_MODE else "00-12:00:00"
SLURM_MAIL_USER = "alielganzory@hotmail.com"
SLURM_CPUS_PER_GPU = 18

# HuggingFace
UPLOAD_TO_HF = False
HF_USERNAME = "ali-elganzory"

# Distribution
EFFECTIVE_BATCH_SIZE = 128
NUM_GPUS = 4
NUM_NODES = 2

# Repo layout (generate_jupiter.py lives in recipes/mv_exp/sft/)
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_TEMPLATE_DIR = _THIS_FILE.parent / "templates"


class ModelSize(Enum):
    S1_7B = "1.7b"
    S0_4B = "0.4b"


class GPUType(Enum):
    GH200 = "GH200"

    def max_batch_size(
        self,
        model_size: ModelSize,
    ) -> int:
        return {
            self.GH200: {
                ModelSize.S1_7B: 16,
                ModelSize.S0_4B: 16,
            },
        }[self][model_size]

    @property
    def partition(
        self,
    ) -> str:
        return {
            self.GH200: "booster",
        }[self]

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
        **({"max_steps": 10} if DEBUG_MODE else {"num_train_epochs": 2}),
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
    # #### Main ####


    # {
    #     "new_id": "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096",
    #     "size": ModelSize.S1_7B,
    # },

    {
        "new_id": "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-long_sft_16k-SFT-Tulu3-decontaminated",
        "old_id": "open-sci/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-long_sft_16k",
        "size": ModelSize.S1_7B,
    },
    
    # {
    #     "new_id": "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16384-rope_theta-1M-long_sft_16k",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-4096-longsft_16k",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/1.7b-Comma0.1-300BT-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/1.7b-Comma0.1-300BT-WithChatTemplate",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/SmolLM2-1.7B-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/SmolLM2-1.7B-WithChatTemplate",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/SmolLM2-1.7B-16k-SFT-Tulu3-decontaminated",
    #     "old_id": "ontocord/SmolLM2-1.7B-16k",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/Qwen2.5-1.5B-SFT-Tulu3-decontaminated",
    #     "old_id": "Qwen/Qwen2.5-1.5B",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/Qwen3-1.7B-Base-SFT-Tulu3-decontaminated",
    #     "old_id": "Qwen/Qwen3-1.7B-Base",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k",
    #     "size": ModelSize.S1_7B,
    # },

    # #### Ablation ####

    # {
    #     "new_id": "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k-SFT-Tulu3-decontaminated",
    #     "old_id": "ontocord/1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/1.7b-MixtureVitae-100BT-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/1.7b-MixtureVitae-100BT",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/1.7b-MixtureVitae-100BT-longsft_16k-SFT-Tulu3-decontaminated",
    #     "old_id": "ontocord/1.7b-MixtureVitae-100BT-longsft_16k",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-longsft_16k-SFT-Tulu3-decontaminated",
    #     "old_id": "ontocord/1.7b-MixtureVitae-web_curated-100BT-longsft_16k",
    #     "size": ModelSize.S1_7B,
    # },

    # #### Not Decontaminated ####

    # {
    #     "new_id": "ali-elganzory/1.7b-MixtureVitae-300BT-v1-SFT-Tulu3",
    #     "old_id": "ali-elganzory/1.7b-MixtureVitae-300BT-v1-WithChatTemplate",
    #     "size": ModelSize.S1_7B,
    # },

    # {
    #     "new_id": "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-SFT-Tulu3",
    #     "old_id": "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-WithChatTemplate",
    #     "size": ModelSize.S1_7B,
    # },

    # #### 0.4B ####

    # {
    #     "new_id": "ali-elganzory/Baguettotron-SFT-Tulu3-decontaminated",
    #     "old_id": "PleIAs/Baguettotron",
    #     "size": ModelSize.S0_4B,
    # },

    # {
    #     "new_id": "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-SFT-Tulu3-decontaminated",
    #     "old_id": "ontocord/0.4b-mixturevitae-v1-decontaminated-300B-4096",
    #     "size": ModelSize.S0_4B,
    # },

    # {
    #     "new_id": "ali-elganzory/Baguettotron-longsft_16k-SFT-Tulu3-decontaminated",
    #     "old_id": "ontocord/Baguettotron-longsft_16k",
    #     "size": ModelSize.S0_4B,
    # },

    # {
    #     "new_id": "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k-SFT-Tulu3-decontaminated",
    #     "old_id": "ontocord/0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k",
    #     "size": ModelSize.S0_4B,
    # }

    #### Merged ####
    # {
    #     "new_id": "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-merged-SFT-Tulu3-decontaminated",
    #     "old_id": "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-merged",
    #     "size": ModelSize.S1_7B,
    # },
]

################################################################################
# Generate recipes
################################################################################


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def generate_new_id(old_id: str, suffix: str) -> str:
    """Generate new_id from old_id by replacing owner with ali-elganzory and appending suffix."""
    if "/" in old_id:
        owner, model_part = old_id.split("/", 1)
        return f"ali-elganzory/{model_part}{suffix}"
    else:
        return f"ali-elganzory/{old_id}{suffix}"


def parse_model_size(size_str: str) -> ModelSize:
    """Convert model size string to ModelSize enum."""
    size_map = {
        "1.7b": ModelSize.S1_7B,
        "0.4b": ModelSize.S0_4B,
    }
    if size_str not in size_map:
        raise ValueError(
            f"Invalid model size: {size_str}. Valid options: {list(size_map.keys())}"
        )
    return size_map[size_str]


def create_recipe(
    model: Dict[str, Any],
    mixture: Dict[str, Any],
    gpu_type: GPUType,
) -> dict:
    config = deepcopy(make_base_config())
    config["dataset_mixture"]["datasets"] = mixture["datasets"]
    config["per_device_train_batch_size"] = gpu_type.max_batch_size(model["size"])
    config["gradient_accumulation_steps"] = gpu_type.accumulation_steps(model["size"])
    config["model_name_or_path"] = model["old_id"]

    config["hub_model_id"] = model["new_id"]
    config["hub_strategy"] = "every_save"

    if "max_length" in model:
        config["max_length"] = model["max_length"]

    model_slug = sanitize_model_name(model["old_id"])
    config["output_dir"] = (
        f"/e/project1/reformo/ali/alignment-handbook/results{'_debug' if DEBUG_MODE else ''}/mv_exp/sft/{model_slug}_{mixture['name']}_{gpu_type.value}"
    )

    return config


def write_recipe(recipe: dict, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = (run_dir / "config.yaml").resolve()
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(recipe, handle, sort_keys=False)
    return output_path


_jinja_env: Environment | None = None


def _get_jinja_env() -> Environment:
    global _jinja_env
    if _jinja_env is None:
        _jinja_env = Environment(
            loader=FileSystemLoader(_TEMPLATE_DIR),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
    return _jinja_env


def write_slurm_script(
    *,
    recipe_path: Path,
    model_name: str,
    mixture_name: str,
    gpu_type: GPUType,
    run_dir: Path,
    model_size: ModelSize,
) -> Path:
    """Render the SLURM batch script under run_dir/slurm/."""
    model_slug = sanitize_model_name(model_name)
    mixture_slug = sanitize_model_name(mixture_name)
    gpu_slug = gpu_type.value
    script_slug = f"{model_slug}_{mixture_slug}_{gpu_slug}"
    slurm_dir = run_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    script_path = slurm_dir / f"{script_slug}.sh"

    run_dir_resolved = run_dir.resolve()
    run_posix = run_dir_resolved.as_posix()
    grad_acc_steps = gpu_type.accumulation_steps(model_size)
    partition = gpu_type.partition
    recipe_abs = recipe_path.resolve().as_posix()
    repo_root_abs = _REPO_ROOT.as_posix()

    template = _get_jinja_env().get_template("jupiter_sft.slurm.j2")
    content = template.render(
        script_slug=script_slug,
        slurm_out_pattern=f"{run_posix}/slurm/%j.%x.%N.out",
        slurm_err_pattern=f"{run_posix}/slurm/%j.%x.%N.err",
        slurm_max_time=SLURM_MAX_TIME,
        partition=partition,
        slurm_account=SLURM_ACCOUNT,
        num_nodes=NUM_NODES,
        num_gpus=NUM_GPUS,
        cpus_per_gpu=SLURM_CPUS_PER_GPU,
        mail_user=SLURM_MAIL_USER,
        repo_root_abs=repo_root_abs,
        run_dir_posix=run_posix,
        grad_acc_steps=grad_acc_steps,
        recipe_path_abs=recipe_abs,
        debug_mode=DEBUG_MODE,
    )

    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o770)
    return script_path.resolve()


def prefetch_caches(models: List[Dict[str, Any]], *, verbose: bool, skip: bool) -> None:
    if skip:
        return
    if verbose:
        print("Prefetching dataset caches...")
    else:
        print("Prefetching Hugging Face caches (datasets + selected models)...")
    for mixture in DATA_MIXTURES:
        for dataset in mixture["datasets"]:
            if verbose:
                print(f"  dataset: {dataset['id']}")
            load_dataset(dataset["id"])
    for model in models:
        if verbose:
            print(f"  model: {model['old_id']}")
        snapshot_download(model["old_id"], repo_type="model")
        AutoTokenizer.from_pretrained(model["old_id"], trust_remote_code=True)
        AutoModelForCausalLM.from_pretrained(model["old_id"], trust_remote_code=True)
    print("Prefetch complete.")


def prompt_model_selection(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(models) <= 1:
        return models
    print(f"\n{len(models)} models are defined. Choose scope:")
    print("  [a] All models  —  generate/submit jobs for every model")
    print("  [p] Pick one    —  choose a single model by index\n")
    for i, m in enumerate(models, start=1):
        print(f"  {i}. {m['old_id']}")
    while True:
        choice = input("\nEnter a or p: ").strip().lower()
        if choice == "a":
            return models
        if choice == "p":
            while True:
                raw = input(f"Model index [1-{len(models)}]: ").strip()
                try:
                    idx = int(raw)
                    if 1 <= idx <= len(models):
                        return [models[idx - 1]]
                except ValueError:
                    pass
                print("Invalid index; try again.")
        else:
            print("Please enter 'a' or 'p'.")


def print_run_confirmation(
    *,
    run_dir: Path,
    config_path: Path,
    slurm_path: Path,
    recipe: dict,
) -> None:
    print("\n--- Run artifacts ---")
    print(f"Run directory:   {run_dir}")
    print(f"Config:          {config_path}")
    print(f"Slurm script:    {slurm_path}")
    print(f"Slurm logs:      {run_dir / 'slurm' / '%j.%x.%N.out'} (.err alongside)\n")
    print("--- config.yaml ---")
    print(yaml.safe_dump(recipe, sort_keys=False), end="")
    print("--- end config ---\n")


def prompt_submit_run() -> bool:
    answer = input("Submit this job with sbatch? [y/N]: ").strip().lower()
    return answer in ("y", "yes")


def submit_slurm(script_path: Path) -> str:
    result = subprocess.run(
        ["sbatch", str(script_path)],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    match = re.search(r"Submitted batch job (\d+)", combined)
    if not match:
        raise RuntimeError(f"Unexpected sbatch output: {combined!r}")
    return match.group(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SFT config.yaml and Slurm scripts under each run output_dir, "
            "then optionally submit with sbatch."
        )
    )
    parser.add_argument(
        "--old-id",
        type=str,
        default=None,
        help="Model id (Hub path) overriding MODELS; generates a single model entry.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="1.7b",
        help="Model size when using --old-id (default: 1.7b). Options: 1.7b, 0.4b.",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Write config and Slurm script only; do not call sbatch.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Do not prompt before sbatch; submit (or skip if --no-submit).",
    )
    parser.add_argument(
        "--skip-prefetch",
        action="store_true",
        help="Skip Hugging Face dataset/model cache prefetch.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose prefetch logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.old_id:
        model_size = parse_model_size(args.model_size)
        new_id = generate_new_id(args.old_id, "-SFT-Tulu3-decontaminated")
        models_to_process = [
            {
                "old_id": args.old_id,
                "new_id": new_id,
                "size": model_size,
            }
        ]
    else:
        models_to_process = MODELS

    models_to_process = prompt_model_selection(models_to_process)
    prefetch_caches(
        models_to_process,
        verbose=args.verbose,
        skip=args.skip_prefetch,
    )

    for model in models_to_process:
        for mixture in DATA_MIXTURES:
            for gpu_type in GPUType:
                recipe = create_recipe(model, mixture, gpu_type)
                run_dir = Path(recipe["output_dir"]).resolve()
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "slurm").mkdir(parents=True, exist_ok=True)

                config_path = write_recipe(recipe, run_dir)
                slurm_path = write_slurm_script(
                    recipe_path=config_path,
                    model_name=model["old_id"],
                    mixture_name=mixture["name"],
                    gpu_type=gpu_type,
                    run_dir=run_dir,
                    model_size=model["size"],
                )

                print_run_confirmation(
                    run_dir=run_dir,
                    config_path=config_path,
                    slurm_path=slurm_path,
                    recipe=recipe,
                )

                if args.no_submit:
                    print("(--no-submit) Skipped sbatch.\n")
                    continue

                if not args.yes and not prompt_submit_run():
                    print("Skipped sbatch for this run.\n")
                    continue

                job_id = submit_slurm(slurm_path)
                print(f"Submitted batch job {job_id}\n")


if __name__ == "__main__":
    main()
