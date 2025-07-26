from train import main as train_main
from fetch_data import main as fetch_data_main
from evaluate import main as evaluate_main

from pathlib import Path

if __name__ == "__main__":
    ticker = "AAPL"
    start = "2010-01-01"
    end = "2025-01-01"
    train_csv = f"data/{ticker}_train.csv"
    test_csv = f"data/{ticker}_test.csv"
    split_date = "2018-01-01"
    commission_rate = 0.001
    investment_capacity = 500

    # fetch_data_main(ticker, start, end, split_date)
    # ckpt_path = train_main(ticker, start, end, train_csv, split_date, commission_rate, investment_capacity, epochs=50)
    ckpt_path = "checkpoints/utrans_AAPL_final.pt"  # Assuming the checkpoint is saved here
    evaluate_main(ticker, test_csv, ckpt_path, commission_rate, investment_capacity)

    # Run the feature generation script
