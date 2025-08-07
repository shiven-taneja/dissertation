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
    window_size = 12
    feature_dim = 14     # do not change unless indicators file changes
    learning_rate = 0.001
    batch_size = 20
    replay_memory_size = 10000
    epochs = 50
    gamma = 0.9
    target_update_freq = 500
    epsilon_start = 1.0
    epsilon_end: 0.4
    # epsilon_decay = 0.99

    train_csv, test_csv = fetch_data_main(ticker, start, end, split_date)
    ckpt_path = train_main(ticker, train_csv, commission_rate, investment_capacity, epochs=50, )
    # ckpt_path = f"checkpoints/utrans_{ticker}_final.pt"
    evaluate_main(ticker, test_csv, ckpt_path, commission_rate, investment_capacity)

    # Run the feature generation script

 