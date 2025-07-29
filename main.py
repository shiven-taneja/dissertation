from train import main as train_main
from fetch_data import main as fetch_data_main
from evaluate import main as evaluate_main

from pathlib import Path

if __name__ == "__main__":
    ticker = "AAPL"
    start = "2010-01-01"
    end = "2020-08-25"
    split_date = "2018-01-01"
    commission_rate = 0.001
    investment_capacity = 500

    train_csv, test_csv = fetch_data_main(ticker, start, end, split_date)
    ckpt_path = train_main(ticker, start, end, train_csv, split_date, commission_rate, investment_capacity, epochs=50)
    evaluate_main(ticker, test_csv, ckpt_path, commission_rate, investment_capacity)

    # Run the feature generation script
