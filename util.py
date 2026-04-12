"""mv_exp run discovery, training-finished checks, checkpoint steps, and Slurm submission.

CLI helpers parse 1-based run indices as shown in ``push.py`` / ``submit.py`` listings.
Expressions may use commas, spaces, single integers, or inclusive ranges ``A-B`` (``A <= B``).
"""

from __future__ import annotations

import re
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parent

# Hugging Face `Trainer` writes this after a successful training loop (see `Trainer.train`).
FINISHED_TRAINING_MARKERS = ("train_results.json",)


class IndexParseError(ValueError):
    """Invalid run index expression or out-of-range 1-based index."""


def _expand_index_token(token: str) -> list[int]:
    token = token.strip()
    if not token:
        raise IndexParseError("empty index token")
    if token.count("-") == 1:
        left, _, right = token.partition("-")
        if not left or not right:
            raise IndexParseError(f"invalid range: {token!r}")
        try:
            lo = int(left, 10)
            hi = int(right, 10)
        except ValueError as exc:
            raise IndexParseError(f"invalid range: {token!r}") from exc
        if lo > hi:
            raise IndexParseError(f"range must be low-high (got {token!r})")
        return list(range(lo, hi + 1))
    try:
        return [int(token, 10)]
    except ValueError as exc:
        raise IndexParseError(f"invalid index token: {token!r}") from exc


def parse_run_index_expression(raw: str) -> list[int]:
    """Parse a user string like ``1-4,5 8`` into a flat list of integers (before deduplication)."""
    if not raw.strip():
        raise IndexParseError("empty selection")
    tokens = [t for t in re.split(r"[\s,]+", raw.strip()) if t]
    if not tokens:
        raise IndexParseError("empty selection")
    out: list[int] = []
    for t in tokens:
        out.extend(_expand_index_token(t))
    return out


def dedupe_preserve_order(indices: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def parse_indices_from_cli_parts(parts: list[str]) -> list[int]:
    """Parse ``argparse`` ``--index`` fragments (each may contain commas)."""
    if not parts:
        raise IndexParseError("empty selection")
    tokens: list[str] = []
    for p in parts:
        tokens.extend(t for t in re.split(r"[\s,]+", p.strip()) if t)
    if not tokens:
        raise IndexParseError("empty selection")
    out: list[int] = []
    for t in tokens:
        out.extend(_expand_index_token(t))
    return out


def validate_1_based(indices: list[int], n: int) -> None:
    bad = [i for i in indices if not (1 <= i <= n)]
    if bad:
        raise IndexParseError(f"index out of range (valid 1-{n}): {bad}")


def hub_model_repo_exists(repo_id: str) -> bool:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()
    try:
        api.repo_info(repo_id, repo_type="model")
        return True
    except RepositoryNotFoundError:
        return False


def load_run_config(run_dir: Path) -> dict[str, Any] | None:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        return None
    try:
        with cfg_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def training_finished(run_dir: Path) -> tuple[bool, str]:
    for name in FINISHED_TRAINING_MARKERS:
        if (run_dir / name).is_file():
            return True, f"found {name}"
    return False, f"missing {FINISHED_TRAINING_MARKERS[0]} (training not finished or failed)"


def latest_checkpoint_step(run_dir: Path) -> int | None:
    """Max step N from immediate subdirs named ``checkpoint-N`` (numeric suffix only)."""
    prefix = "checkpoint-"
    best: int | None = None
    for p in run_dir.iterdir():
        if not p.is_dir() or not p.name.startswith(prefix):
            continue
        suffix = p.name[len(prefix) :]
        if suffix.isdigit():
            step = int(suffix)
            best = step if best is None else max(best, step)
    return best


def hub_model_id_from_config(cfg: dict[str, Any] | None) -> str | None:
    if cfg is None:
        return None
    raw = cfg.get("hub_model_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def find_slurm_script(
    kind: Literal["sft", "dpo"],
    run_dir: Path,
    repo_root: Path,
) -> Path | None:
    """SFT: ``run_dir/slurm/*.sh``. DPO: ``recipes/mv_exp/dpo/<run_basename>.sh`` then SFT-style fallback."""
    run_dir = run_dir.resolve()
    repo_root = repo_root.resolve()

    def first_sh_in_slurm(base: Path) -> Path | None:
        slurm_dir = base / "slurm"
        if not slurm_dir.is_dir():
            return None
        candidates = sorted(slurm_dir.glob("*.sh"))
        if not candidates:
            return None
        if len(candidates) > 1:
            warnings.warn(
                f"Multiple Slurm scripts in {slurm_dir}; using {candidates[0].name!r}.",
                UserWarning,
                stacklevel=2,
            )
        return candidates[0].resolve()

    if kind == "sft":
        return first_sh_in_slurm(run_dir)

    dpo_script = repo_root / "recipes" / "mv_exp" / "dpo" / f"{run_dir.name}.sh"
    if dpo_script.is_file():
        return dpo_script.resolve()
    return first_sh_in_slurm(run_dir)


@dataclass(frozen=True)
class MVRun:
    kind: Literal["sft", "dpo"]
    path: Path
    hub_model_id: str | None
    finished: bool
    finish_reason: str
    latest_checkpoint_step: int | None
    slurm_script: Path | None

    @property
    def label(self) -> str:
        return f"{self.kind.upper()}  {self.path.name}"


def _discover_in_base(
    kind: Literal["sft", "dpo"],
    base: Path,
    repo_root: Path,
) -> list[MVRun]:
    if not base.is_dir():
        return []
    out: list[MVRun] = []
    for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        cfg = load_run_config(child)
        hub_id = hub_model_id_from_config(cfg)
        finished, reason = training_finished(child)
        step = latest_checkpoint_step(child)
        slurm = find_slurm_script(kind, child, repo_root)
        out.append(
            MVRun(
                kind=kind,
                path=child.resolve(),
                hub_model_id=hub_id,
                finished=finished,
                finish_reason=reason,
                latest_checkpoint_step=step,
                slurm_script=slurm,
            )
        )
    return out


def discover_mv_runs(repo_root: Path | None = None) -> list[MVRun]:
    root = repo_root if repo_root is not None else REPO_ROOT
    sft_base = root / "results" / "mv_exp" / "sft"
    dpo_base = root / "results" / "mv_exp" / "dpo"
    return _discover_in_base("sft", sft_base, root) + _discover_in_base(
        "dpo", dpo_base, root
    )


def submit_slurm_script(script: Path, *, cwd: Path) -> str:
    """Run ``sbatch`` and return the job id string."""
    result = subprocess.run(
        ["sbatch", str(script)],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    match = re.search(r"Submitted batch job (\d+)", combined)
    if not match:
        raise RuntimeError(f"Unexpected sbatch output: {combined!r}")
    return match.group(1)
