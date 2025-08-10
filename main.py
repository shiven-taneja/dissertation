from fetch_data import main as fetch_data_main

from train import main as train_ppo_main
from evaluate import main as evaluate_ppo_main


if __name__ == "__main__":
    ticker = "AAPL"
    start = "2010-01-01"
    end = "2020-08-25"
    split_date = "2018-01-01"

    train_csv, test_csv = fetch_data_main(ticker, start, end, split_date)

    ckpt_path = train_ppo_main(
        ticker,
        train_csv,)
    
    evaluate_ppo_main(
        ticker,
        test_csv,
        ckpt_path)
    
 