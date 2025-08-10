# ============================
# file: run_ablation.py
# ============================
"""
Ablation runner that:
- Discovers tickers under data/<TICKER> (having train.csv & test.csv)
- For each ticker, runs 4 experiments (baseline, headline, techsent, all)
- Each experiment runs N seeds (default 3) and picks the best model by a metric
- Copies **only the best** checkpoints into checkpoints/<TICKER>/
- Creates plots **only for the best** of each experiment and a combined plot vs B&H
- Writes manifests and summaries suitable for a paper
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from shutil import copy2
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from train import train_one
from evaluate import evaluate_one

RUN_TYPES = ["baseline", "headline", "techsent", "all"]


def discover_tickers(data_root: str | Path = "data") -> List[str]:
    root = Path(data_root)
    tickers = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if d.name.lower() == "cache":
            continue
        if (d / "train.csv").exists() and (d / "test.csv").exists():
            tickers.append(d.name)
    return sorted(tickers)


def pick_best(run_records: List[Dict], metric: str = "CAGR_%") -> Dict:
    best = None
    best_val = -np.inf
    for r in run_records:
        val = r["eval"]["metrics"].get(metric, -np.inf)
        if val > best_val:
            best_val = val
            best = r
    return best


def plot_combined_best(
    *, ticker: str, best_by_type: Dict[str, Dict], results_root: str | Path = "results"
):
    results_root = Path(results_root)
    plots_dir = results_root / "plots" / ticker
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))
    bh_equity = None
    dates = None
    for run_type, rec in best_by_type.items():
        eq = rec["eval"]["equity"]
        dt = rec["eval"]["dates"]
        plt.plot(dt, eq, linewidth=2, label=f"{run_type}")
        if bh_equity is None:
            bh_equity = rec["eval"]["bh_equity"]
            dates = dt
    if bh_equity is not None:
        plt.plot(dates, bh_equity, label="Buy&Hold", alpha=0.85)
    plt.title(f"{ticker} vs B&H")
    plt.ylabel("Portfolio Value ($)")
    plt.legend(); plt.grid(True, alpha=0.3)
    out = plots_dir / f"combined_best_{ticker}.png"
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    return str(out)


def run_for_ticker(
    *,
    ticker: str,
    seeds: List[int] = [26, 927, 2025],
    selection_metric: str = "CAGR_%",
    data_root: str | Path = "data",
    results_root: str | Path = "results",
    checkpoints_root: str | Path = "checkpoints",
) -> Dict:
    print(f"===== {ticker}: starting ablation =====")
    tdir = Path(data_root) / ticker
    train_csv = tdir / "train.csv"
    test_csv = tdir / "test.csv"

    exp_records: Dict[str, List[Dict]] = {k: [] for k in RUN_TYPES}

    # 1) run all seeds WITHOUT plots to avoid clutter
    for run_type in RUN_TYPES:
        print(f"--- Experiment: {run_type} ---")
        for s in seeds:
            print(f"Seed {s}…")
            train_meta = train_one(
                ticker=ticker,
                run_type=run_type,
                train_csv=train_csv,
                seed=s,
            )
            eval_meta = evaluate_one(
                ticker=ticker,
                run_type=run_type,
                test_csv=test_csv,
                ckpt_path=train_meta["ckpt_path"],
                debug=False,
                save_plots=False,
                train_csv=train_csv,
            )
            rec = {"train": train_meta, "eval": eval_meta, "seed": s}
            per_run_json = Path(results_root) / f"{ticker}_{run_type}_seed{s}_record.json"
            with per_run_json.open("w") as f:
                json.dump(rec, f, indent=2)
            exp_records[run_type].append(rec)

    # 2) choose best per run_type
    best_by_type: Dict[str, Dict] = {
        rt: pick_best(exp_records[rt], selection_metric) for rt in RUN_TYPES
    }

    # 3) copy best checkpoints and **create plots only for best**
    best_ckpt_dir = Path(checkpoints_root) / ticker
    best_ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpts = {}
    for rt, rec in best_by_type.items():
        src = Path(rec["train"]["ckpt_path"]).resolve()
        dst = best_ckpt_dir / (f"utrans_{ticker}_{rt}_best_seed{rec['seed']}.pt")
        copy2(src, dst)
        best_ckpts[rt] = str(dst)

        # re-evaluate best with plots enabled
        _ = evaluate_one(
            ticker=ticker,
            run_type=rt,
            test_csv=test_csv,
            ckpt_path=dst,
            save_plots=True,
        )

    # 4) combined plot
    combined_png = plot_combined_best(
        ticker=ticker, best_by_type=best_by_type, results_root=results_root
    )

    # 5) aggregate table (best runs only)
    rows = []
    for rt, rec in best_by_type.items():
        m = rec["eval"]["metrics"]
        rows.append({"run_type": rt, **m, "seed": rec["seed"]})
    best_table = pd.DataFrame(rows).sort_values("run_type")
    table_csv = Path(results_root) / f"{ticker}_best_summary.csv"
    best_table.to_csv(table_csv, index=False)

    # 6) manifest
    manifest = {
        "ticker": ticker,
        "selection_metric": selection_metric,
        "seeds": seeds,
        "experiments": exp_records,
        "best": {rt: {
            "seed": rec["seed"],
            "ckpt": best_ckpts[rt],
            "metrics": rec["eval"]["metrics"],
            "equity_csv": rec["eval"]["equity_csv"],
            "metrics_json": rec["eval"]["metrics_path"],
        } for rt, rec in best_by_type.items()},
        "combined_plot": combined_png,
        "best_table_csv": str(table_csv),
    }

    man_path = Path(results_root) / f"{ticker}_ablation_manifest.json"
    with man_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest → {man_path}")
    return manifest


def main():

    seeds = [26, 927, 2025] 

    tickers = ["BABA", "BRK", "GOOG", "KO", "MRK", "MS", "NVDA", "QQQ", "T", "WFC"]


    print("Tickers:", tickers)
    all_manifests = {}
    for t in tickers:
        all_manifests[t] = run_for_ticker(
            ticker=t,
            seeds=seeds,
            selection_metric="final_return_%",
            data_root="data",
            results_root="results",
            checkpoints_root="checkpoints",
        )

    rows = []
    for t, man in all_manifests.items():
        for rt, rec in man["best"].items():
            r = {"ticker": t, "run_type": rt, "seed": rec["seed"], **rec["metrics"]}
            rows.append(r)
    summary = pd.DataFrame(rows)
    out_csv = Path("results") / "ALL_best_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Saved overall summary → {out_csv}")


if __name__ == "__main__":
    main()
