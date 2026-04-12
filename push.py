#!/usr/bin/env python3
"""Interactively push finished mv_exp SFT/DPO runs from results/ to the Hugging Face Hub.

By default, queries the Hub once per ``hub_model_id`` to show whether the repo already exists.
Pass ``--no-check-hub`` to skip those calls (offline / faster).
Pass ``--exclude-on-hub`` to omit runs whose repo already exists (☁️) from the listing and from
``--all`` / ``--index`` (indices count only the remaining rows).

Non-interactive: ``push.py --all -y`` or ``push.py --index 1-4,5 8 -y``
(same index syntax as ``submit.py``: ranges, commas, spaces).
"""

from __future__ import annotations

import argparse
import sys
from typing import Literal

from util import (
    FINISHED_TRAINING_MARKERS,
    MVRun,
    IndexParseError,
    dedupe_preserve_order,
    discover_mv_runs,
    hub_model_repo_exists,
    parse_indices_from_cli_parts,
    parse_run_index_expression,
    validate_1_based,
)

# Do not upload intermediate checkpoints or cluster logs (large and usually redundant).
UPLOAD_IGNORE_PATTERNS = ["checkpoint-*", "slurm/**"]


def run_pushable(r: MVRun) -> bool:
    return r.finished and r.hub_model_id is not None


def run_status_icon(r: MVRun) -> str:
    """Visual run state: ready to push, blocked on config, or not finished."""
    if run_pushable(r):
        return "✅"
    if r.finished:
        return "⚠️"
    return "❌"


def _hub_column_icon(
    *,
    check_hub: bool,
    run: MVRun,
    on_hub: bool | None,
) -> str:
    if not check_hub:
        return "—"
    if run.hub_model_id is None:
        return "n/a"
    if on_hub is True:
        return "☁️"
    return "·"


def push_run(run: MVRun) -> None:
    from huggingface_hub import HfApi

    assert run.hub_model_id is not None
    api = HfApi()
    api.create_repo(repo_id=run.hub_model_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(run.path),
        repo_id=run.hub_model_id,
        repo_type="model",
        ignore_patterns=UPLOAD_IGNORE_PATTERNS,
    )


def prompt_push_scope(
    runs: list[MVRun],
    n_pushable: int,
) -> tuple[Literal["all"], None] | tuple[Literal["pick"], list[int]] | None:
    """Return selection, or ``None`` if the user cancelled (empty input)."""
    n = len(runs)
    print(f"\n{n} run(s) listed; {n_pushable} can be pushed (finished + hub_model_id). Choose scope:")
    print("  [a] All pushable  —  upload every finished run that has hub_model_id")
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


def _confirm_push(
    *,
    to_push: list[tuple[int, MVRun]],
    check_hub: bool,
    on_hub: list[bool | None],
    assume_yes: bool,
) -> bool:
    if assume_yes:
        return True
    if check_hub:
        on_hub_indices = [i for i, _ in to_push if on_hub[i - 1] is True]
        if on_hub_indices:
            print(
                "\nHub check: some selected repos already exist (☁️). "
                "upload_folder will add or update files; local checkpoints may be newer than the Hub."
            )
            ans = input("Continue with upload(s)? [y/N]: ").strip().lower()
            return ans in ("y", "yes")
    if len(to_push) > 1:
        ans = input(f"\nPush {len(to_push)} model(s) to the Hub? [y/N]: ").strip().lower()
        return ans in ("y", "yes")
    if len(to_push) == 1:
        i, _r = to_push[0]
        if check_hub and on_hub[i - 1] is True:
            ans = input("\nThis repo exists on the Hub. Continue with upload? [y/N]: ").strip().lower()
            return ans in ("y", "yes")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--no-check-hub",
        action="store_true",
        help="Skip Hub API lookups (no ☁️ column; use when offline or to save time).",
    )
    p.add_argument(
        "--exclude-on-hub",
        action="store_true",
        help=(
            "Do not list or select runs whose hub_model_id already exists on the Hub (☁️). "
            "Requires Hub lookup; cannot be combined with --no-check-hub."
        ),
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Push all pushable runs (non-interactive); use with -y to skip confirmations.",
    )
    p.add_argument(
        "--index",
        nargs="+",
        type=str,
        metavar="SPEC",
        help=(
            "Push run(s) by 1-based index(es): each SPEC can be an integer, range (e.g. 1-4), "
            "or comma-separated groups. Example: --index 1-4,5 8"
        ),
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts before uploading.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.all and args.index is not None:
        print("Use only one of --all and --index.", file=sys.stderr)
        return 1

    check_hub = not args.no_check_hub
    if args.exclude_on_hub and not check_hub:
        print(
            "--exclude-on-hub needs Hub lookups; omit --no-check-hub.",
            file=sys.stderr,
        )
        return 1

    runs = discover_mv_runs()
    if not runs:
        print("No runs found under results/mv_exp/sft or results/mv_exp/dpo.", file=sys.stderr)
        print("(Missing directories are treated as having no runs.)", file=sys.stderr)
        return 1

    n = len(runs)
    on_hub: list[bool | None] = [None] * n
    if check_hub:
        for idx, r in enumerate(runs):
            if r.hub_model_id:
                on_hub[idx] = hub_model_repo_exists(r.hub_model_id)

    if args.exclude_on_hub:
        excluded = sum(1 for v in on_hub if v is True)
        new_runs: list[MVRun] = []
        new_on_hub: list[bool | None] = []
        for idx, r in enumerate(runs):
            if on_hub[idx] is True:
                continue
            new_runs.append(r)
            new_on_hub.append(on_hub[idx])
        runs = new_runs
        on_hub = new_on_hub
        if excluded:
            print(f"Excluding {excluded} run(s) already on the Hub (--exclude-on-hub).\n")
        if not runs:
            print(
                "No runs left after --exclude-on-hub (all discovered runs are already on the Hub).",
                file=sys.stderr,
            )
            return 1

    print("Finished-run check: looking for", FINISHED_TRAINING_MARKERS[0])
    print(
        "(Stricter alternative: require model weights at the run root, e.g. model.safetensors "
        "or pytorch_model.bin, in addition to the training marker.)"
    )
    print(
        "Legend: ✅ ready to push  ·  ⚠️ finished but missing hub_model_id  ·  ❌ not finished"
    )
    if check_hub:
        print(
            "Hub column: ☁️ repo exists  ·  · not on Hub  ·  n/a no hub_model_id in config"
        )
        print(
            "Note: ☁️ only means the repo name exists on the Hub; your local run may be newer. "
            "You can push again to update.\n"
        )
    else:
        print(
            "Hub column: — (skipped; omit --no-check-hub to query the Hub for ☁️ / ·)\n"
        )

    for i, r in enumerate(runs, start=1):
        status = "finished" if r.finished else "NOT finished"
        hub = r.hub_model_id or "(no hub_model_id in config.yaml)"
        st_icon = run_status_icon(r)
        hub_icon = _hub_column_icon(check_hub=check_hub, run=r, on_hub=on_hub[i - 1])
        print(f"  {i:3}.  {hub_icon}  {st_icon}  {r.label}")
        print(f"       hub_model_id: {hub}")
        print(f"       status: {status} — {r.finish_reason}")

    pushable = [r for r in runs if run_pushable(r)]
    if not pushable:
        print("\nNo runs are both finished and have hub_model_id; nothing to push.", file=sys.stderr)
        return 1

    print(
        f"\n✅  {len(pushable)} run(s) can be pushed (finished + hub_model_id). "
        "Others are listed above for context."
    )

    pick_indices: list[int]
    if args.all:
        pick_indices = [i for i, r in enumerate(runs, start=1) if run_pushable(r)]
    elif args.index is not None:
        try:
            pick_indices = dedupe_preserve_order(parse_indices_from_cli_parts(args.index))
            validate_1_based(pick_indices, n)
        except IndexParseError as err:
            print(f"Invalid --index: {err}", file=sys.stderr)
            return 1
    else:
        prompted = prompt_push_scope(runs, len(pushable))
        if prompted is None:
            print("Cancelled.")
            return 0
        scope, picked = prompted
        if scope == "all":
            pick_indices = [i for i, r in enumerate(runs, start=1) if run_pushable(r)]
        else:
            assert picked is not None
            pick_indices = picked

    selected_pairs = [(i, runs[i - 1]) for i in pick_indices]
    not_pushable_idx = [i for i, r in selected_pairs if not run_pushable(r)]
    if not_pushable_idx:
        print(
            f"Cannot push: run index(es) {not_pushable_idx} are not finished "
            f"or lack hub_model_id.",
            file=sys.stderr,
        )
        return 1

    to_push = selected_pairs
    if not to_push:
        print("No runs selected.", file=sys.stderr)
        return 1

    if not _confirm_push(
        to_push=to_push,
        check_hub=check_hub,
        on_hub=on_hub,
        assume_yes=args.yes,
    ):
        print("Aborted.")
        return 0

    for i, selected in to_push:
        repo_id = selected.hub_model_id
        assert repo_id is not None
        print(f"Pushing [{i}] {selected.path} -> {repo_id} ...")
        try:
            push_run(selected)
        except Exception as err:
            print(f"Push failed: {err}", file=sys.stderr)
            return 1
        print("  Done.")

    print("All selected uploads finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
