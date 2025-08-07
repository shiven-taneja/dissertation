import os
from pathlib import Path
from datetime import datetime

from fetch_data_groq import build_datasets_with_llm
from train_groq import train_one_variant
from evaluate_groq import combined_test_plot, train_then_test_plot

TICKERS = ["NVDA","QQQ","BABA","KO","MRK","WFC","GOOG","T","MS","BRK"]

DATA_ROOT = Path("data")

def create_datasets(train_ratio: float = 0.7):
    train_datasets = []
    test_datasets = []
    for ticker in TICKERS:
        train_csv, test_csv = build_datasets_with_llm(ticker, train_ratio=train_ratio)
        train_datasets.append(train_csv)
        test_datasets.append(test_csv)
    return train_datasets, test_datasets

def read_dataset(ticker: str):
    out_dir = DATA_ROOT / ticker.upper()
    train_csv = out_dir / f"{ticker}_train.csv"
    test_csv = out_dir / f"{ticker}_test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Dataset files {train_csv} or {test_csv} do not exist.")
    return train_csv, test_csv

def run_all(
    epsilon_end: float = 0.05,
    weight_loss_coef: float = 0.0,
    rand_weights: bool = False,
    train_ratio: float = 0.7,
    variants = ("baseline","headline","tech","both"),  # 4 runs
):
    dt_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_tag = f"eps{epsilon_end}_wl{weight_loss_coef}_rw{rand_weights}_{dt_tag}"

    for tic in TICKERS:
        tr, te = build_datasets_with_llm(tic, train_ratio=train_ratio)

        # Train/eval each variant
        for v in variants:
            _ = train_one_variant(
                ticker=tic,
                train_csv=tr,
                test_csv=te,
                feature_variant=v,
                out_tag=out_tag,
                epsilon_end=epsilon_end,
                weight_loss_coef=weight_loss_coef,
                rand_weights=rand_weights,
            )

        # Combined plots after all variants for this ticker
        combined_test_plot(tic, out_tag=out_tag, variants=list(variants))
        train_then_test_plot(tic, out_tag=out_tag, variants=list(variants))

    print("Done. Check results/ and data/<TICKER>/")

if __name__ == "__main__":
    tr, te = create_datasets(train_ratio=0.7)  # Create datasets if not already present
    run_all()