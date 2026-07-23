import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from garsyn.config import DEFAULT_CONFIG
from garsyn.constants import LABELS
from garsyn.data import prepare_data
from garsyn.metrics import evaluate_prediction_frame, summarize
from garsyn.train import run_experiment


def parse_csv_list(value, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def load_config(args):
    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    overrides = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "gpu": args.gpu,
        "num_workers": args.workers,
        "lambda_perm": args.lambda_perm,
        "lambda_gate_sparse": args.lambda_gate_sparse,
        "lambda_gate_balance": args.lambda_gate_balance,
        "perm_every": args.perm_every,
        "use_gate_regularization": args.use_gate_regularization,
        "use_amp": args.amp,
        "gpu_cache": args.gpu_cache,
        "graph_cache_mode": args.graph_cache_mode,
        "allow_tf32": args.tf32,
        "use_interaction_type_bias": args.use_interaction_type_bias,
        "use_gate": args.use_gate,
        "use_depth_residual": args.use_depth_residual,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def cmd_prepare(args):
    folds = parse_csv_list(args.folds, int)
    manifest = prepare_data(
        output_dir=args.output_dir,
        dataset=args.dataset,
        source_raw_dir=args.source_raw_dir,
        source_repeat_dir=args.source_repeat_dir,
        split_mode=args.split_mode,
        folds=folds,
        labels=LABELS,
    )
    print(f"Prepared {len(manifest['fold_files'])} split files under {Path(args.output_dir).resolve()}")


def cmd_train(args):
    labels = LABELS if args.labels == "all" else parse_csv_list(args.labels)
    folds = parse_csv_list(args.folds, int)
    cfg = load_config(args)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    rows = run_experiment(args, cfg, labels, folds)
    if rows:
        pd.DataFrame(rows).to_csv(Path(args.results_dir) / "metrics_latest_run.csv", index=False)


def cmd_evaluate(args):
    labels = LABELS if args.labels == "all" else parse_csv_list(args.labels)
    rows = []
    for pred_path in Path(args.results_dir).glob("*/predict/repeat*_predict.csv"):
        method = pred_path.parts[-3]
        fold = int(pred_path.stem.replace("_predict", "").replace("repeat", ""))
        df = pd.read_csv(pred_path)
        metrics = evaluate_prediction_frame(df, labels)
        if metrics.empty:
            continue
        metrics.insert(0, "method", method)
        metrics.insert(1, "fold", fold)
        metrics.insert(2, "prediction_csv", str(pred_path))
        rows.append(metrics)
    if not rows:
        raise SystemExit(f"No prediction files found under {args.results_dir}")
    out = pd.concat(rows, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    mean_path = Path(args.output).with_name(Path(args.output).stem + "_mean_std.csv")
    summarize(out).to_csv(mean_path, index=False)
    print(f"Saved {args.output}")
    print(f"Saved {mean_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Clean GAR-Syn implementation.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--dataset", default="drugcomb", choices=["drugcomb", "oneil", "nci-almanac"])
    p.add_argument("--source-raw-dir", default=None)
    p.add_argument("--source-repeat-dir", default=None)
    p.add_argument("--split-mode", default="random", choices=["precomputed", "random", "drugpairout", "drugout", "cellout", "bothout"])
    p.add_argument("--output-dir", default="./data_random")
    p.add_argument("--folds", default="1,2,3,4,5")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train")
    p.add_argument("--config", default=None)
    p.add_argument("--labels", default="S_mean")
    p.add_argument("--folds", default="1,2,3,4,5")
    p.add_argument("--data-dir", default="./data_random")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--lambda-perm", type=float, default=None)
    p.add_argument("--lambda-gate-sparse", type=float, default=None)
    p.add_argument("--lambda-gate-balance", type=float, default=None)
    p.add_argument("--perm-every", type=int, default=None)
    p.add_argument("--gpu-cache", choices=["none", "indices", "features", "all"], default=None)
    p.add_argument("--graph-cache-mode", choices=["batch", "full_batch"], default=None)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--use-interaction-type-bias", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--use-gate", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--use-depth-residual", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--use-gate-regularization", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--labels", default="S_mean")
    p.add_argument("--output", default="./results/summary_all.csv")
    p.set_defaults(func=cmd_evaluate)
    return parser


def main():
    os.chdir(ROOT)
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
