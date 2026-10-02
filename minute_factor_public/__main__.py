"""Command-line entry points for synthetic training and checkpoint evaluation."""

import argparse
import json

from .data import synthetic_market
from .discovery import FactorModel, discover
from .evaluation import evaluate_model
from .splits import chronological_split, walk_forward_splits


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic minute-factor research example")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "walk-forward"):
        command = commands.add_parser(name)
        command.add_argument("--days", type=int, default=84)
        command.add_argument("--minutes", type=int, default=32)
        command.add_argument("--stocks", type=int, default=24)
        command.add_argument("--data-seed", type=int, default=7)
        if name in {"train", "walk-forward"}:
            command.add_argument("--trials", type=int, default=48)
            command.add_argument("--capacity", type=int, default=5)
            command.add_argument("--search", choices=("random", "ppo"), default="random")
            command.add_argument("--mode", choices=("pooled", "single"), default="pooled")
            command.add_argument("--search-seed", type=int, default=11)
            command.add_argument("--chunk-minutes", type=int, default=16)
        if name in {"train", "evaluate"}:
            command.add_argument("--model", default="artifacts/model.json")
        if name == "evaluate":
            command.add_argument("--split", choices=("train", "validation", "test"), default="test")
        if name == "walk-forward":
            command.add_argument("--year-days", type=int, default=12)
    return parser


def main() -> None:
    args = _parser().parse_args()
    panel = synthetic_market(args.days, args.minutes, args.stocks, args.data_seed)
    if args.command == "train":
        train, validation, _ = chronological_split(panel.days)
        model = discover(
            panel, train, trials=args.trials, capacity=args.capacity, mode=args.mode,
            search=args.search, seed=args.search_seed, chunk_minutes=args.chunk_minutes,
        )
        model.save(args.model)
        result = {
            "synthetic_data": True,
            "model_path": args.model,
            "selected_factors": len(model.expressions),
            "train": evaluate_model(panel, model, train),
            "validation": evaluate_model(panel, model, validation),
            "next_command": (
                f'python -m minute_factor_public evaluate --model "{args.model}" --split test '
                f"--days {args.days} --minutes {args.minutes} --stocks {args.stocks} "
                f"--data-seed {args.data_seed}"
            ),
        }
    elif args.command == "evaluate":
        model = FactorModel.load(args.model)
        splits = dict(zip(("train", "validation", "test"), chronological_split(panel.days)))
        result = {
            "synthetic_data": True,
            "split": args.split,
            "result": evaluate_model(panel, model, splits[args.split]),
        }
    else:
        folds = []
        for number, (train, validation, test) in enumerate(
            walk_forward_splits(panel.days, args.year_days), start=1
        ):
            model = discover(
                panel, train, trials=args.trials, capacity=args.capacity, mode=args.mode,
                search=args.search, seed=args.search_seed + number,
                chunk_minutes=args.chunk_minutes,
            )
            folds.append({
                "fold": number,
                "selected_factors": len(model.expressions),
                "validation": evaluate_model(panel, model, validation),
                "test": evaluate_model(panel, model, test),
            })
        result = {"synthetic_data": True, "year_days": args.year_days, "folds": folds}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
