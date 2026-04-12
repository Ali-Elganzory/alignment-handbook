#!/usr/bin/env python3
"""Interactively submit mv_exp SFT/DPO training jobs via sbatch.

Pass ``--exclude-finished`` to drop finished runs (✅) from the listing and from ``--all`` /
``--index`` (indices count only the remaining rows).

Non-interactive: ``submit.py --all -y`` or ``submit.py --index 1-4,5 8 -y``
(same index syntax as interactive: ranges, commas, spaces).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Literal

from util import (
    MVRun,
    REPO_ROOT,
    IndexParseError,
    dedupe_preserve_order,
    discover_mv_runs,
    parse_indices_from_cli_parts,
    parse_run_index_expression,
    submit_slurm_script,
    validate_1_based,
)


def submit_status_icon(r: MVRun) -> str:
    if r.finished:
        return "✅"
    if r.latest_checkpoint_step is not None:
        return "🔄"
    return "❌"


def _checkpoint_summary(r: MVRun) -> str:
    if r.finished:
        return r.finish_reason
    if r.latest_checkpoint_step is not None:
        return f"checkpoint max step: {r.latest_checkpoint_step} — {r.finish_reason}"
    return f"no checkpoints yet — {r.finish_reason}"


def _slurm_line(r: MVRun) -> str:
    if r.slurm_script is not None:
        return f"slurm: {r.slurm_script}"
    return "slurm: (no script found)"


def _eligible_for_batch(r: MVRun) -> bool:
    return r.slurm_script is not None and not r.finished


def prompt_scope(
    runs: list[MVRun],
) -> tuple[Literal["all"], None] | tuple[Literal["pick"], list[int]] | None:
    """Return ('all', None), ('pick', indices), or ``None`` if the user cancelled (empty input)."""
    n = len(runs)
    print(f"\n{n} run(s) listed. Choose scope:")
    print("  [a] All eligible  —  sbatch every run that has a Slurm script and is not finished")
    print(
        "  [p] Pick runs     —  indices: integers, ranges (1-4), commas or spaces (e.g. 1-4,5 8)"
    )
    print("(empty Enter = cancel)\n")
    while True:
        choice = input("Enter a or p (empty = cancel): ").strip().lower()
        if not choice:
            return None
        if choice == "a":
            return "all", None
        if choice == "p":
            while True:
                raw = input(f"Run index(es) [1-{n}] (empty = cancel): ").strip()
                if not raw:
                    return None
                try:
                    parsed = dedupe_preserve_order(parse_run_index_expression(raw))
                    validate_1_based(parsed, n)
                except IndexParseError as err:
                    print(f"Invalid input: {err}")
                    continue
                return "pick", parsed
        print("Please enter 'a', 'p', or empty to cancel.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--exclude-finished",
        action="store_true",
        help=(
            "Do not list or select runs that already finished (train_results.json / ✅). "
            "Useful to focus on jobs still needing submission."
        ),
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Submit all eligible runs (non-interactive scope); use with -y to skip prompts.",
    )
    p.add_argument(
        "--index",
        nargs="+",
        type=str,
        metavar="SPEC",
        help=(
            "Submit run(s) by 1-based index(es): each SPEC can be an integer, range (e.g. 1-4), "
            "or comma-separated groups. Example: --index 1-4,5 8"
        ),
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation when any selected run is already finished.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.all and args.index is not None:
        print("Use only one of --all and --index.", file=sys.stderr)
        return 1

    runs = discover_mv_runs()

    if not runs:
        print("No runs found under results/mv_exp/sft or results/mv_exp/dpo.", file=sys.stderr)
        print("(Missing directories are treated as having no runs.)", file=sys.stderr)
        return 1

    if args.exclude_finished:
        n_before = len(runs)
        runs = [r for r in runs if not r.finished]
        excluded = n_before - len(runs)
        if excluded:
            print(f"Excluding {excluded} finished run(s) (--exclude-finished).\n")
        if not runs:
            print(
                "No runs left after --exclude-finished (all discovered runs are finished).",
                file=sys.stderr,
            )
            return 1

    print("Finished runs: looking for train_results.json at run root.")
    print(
        "Legend: ✅ finished  ·  🔄 in progress (has checkpoint-*)  ·  ❌ no checkpoints yet\n"
    )

    for i, r in enumerate(runs, start=1):
        icon = submit_status_icon(r)
        print(f"  {i:3}.  {icon}  {r.label}")
        print(f"       {_checkpoint_summary(r)}")
        print(f"       {_slurm_line(r)}")

    eligible = [r for r in runs if _eligible_for_batch(r)]
    print(
        f"\n{len(eligible)} run(s) eligible for batch submit (Slurm script + not finished)."
    )

    scope: str
    pick_indices: list[int] | None
    if args.all:
        scope, pick_indices = "all", None
    elif args.index is not None:
        try:
            pick_indices = dedupe_preserve_order(parse_indices_from_cli_parts(args.index))
            validate_1_based(pick_indices, len(runs))
        except IndexParseError as err:
            print(f"Invalid --index: {err}", file=sys.stderr)
            return 1
        scope, pick_indices = "pick", pick_indices
    else:
        prompted = prompt_scope(runs)
        if prompted is None:
            print("Cancelled.")
            return 0
        scope, pick_indices = prompted
        if scope == "pick":
            assert pick_indices is not None

    to_submit: list[MVRun] = []
    if scope == "all":
        to_submit = list(eligible)
        if not to_submit:
            print("No eligible runs to submit.", file=sys.stderr)
            return 1
        skipped_finished = sum(1 for r in runs if r.finished and r.slurm_script)
        if skipped_finished:
            print(f"Skipping {skipped_finished} finished run(s) with a Slurm script.")
    else:
        assert pick_indices is not None
        selected_list = [runs[i - 1] for i in pick_indices]
        missing = [i for i, r in zip(pick_indices, selected_list) if r.slurm_script is None]
        if missing:
            print(
                f"No Slurm script for run index(es) {missing}; cannot submit.",
                file=sys.stderr,
            )
            return 1
        finished_pairs = [(i, r) for i, r in zip(pick_indices, selected_list) if r.finished]
        if finished_pairs and not args.yes:
            labels = ", ".join(f"{i} ({r.label})" for i, r in finished_pairs)
            ans = input(
                f"Some selected runs already finished (train_results.json): {labels}\n"
                f"Resubmit anyway? [y/N]: "
            ).strip().lower()
            if ans not in ("y", "yes"):
                print("Aborted.")
                return 0
        to_submit = selected_list

    cwd = REPO_ROOT
    for r in to_submit:
        assert r.slurm_script is not None
        print(f"Submitting {r.slurm_script} ...")
        try:
            job_id = submit_slurm_script(r.slurm_script, cwd=cwd)
        except subprocess.CalledProcessError as e:
            print(f"sbatch failed: {e}", file=sys.stderr)
            if e.stderr:
                print(e.stderr, file=sys.stderr)
            return 1
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1
        print(f"  Submitted batch job {job_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
