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
    # feature_dim = 1  # only delta_oc feature
    learning_rate = 0.001
    batch_size = 20
    replay_memory_size = 10000
    epochs = 50
    gamma = 0.99
    target_update_freq = 200
    epsilon_start = 1.0
    epsilon_end = 0.05
    # epsilon_decay = 0.99
    weight_loss_coef = 0.5  # Coefficient for the weight loss term in the loss function
    rand_weights = True  # Use random weights during exploration


    # WITH RANDOM WEIGHTS FROM 0.1 TO 1.0
    # Best so far: epsilon_start = 1.0, epsilon_end = 0.05, weight_loss_coef = 1.0 -> 198.2% return (total buys = 13, total sells = 7)
    # Other good configurations:
    # epsilon_start = 1.0, epsilon_end = 0.01, weight_loss_coef = 1.0 -> 188.48% return (total buys = 18, total sells = 7)
    # epsilon_start = 1.0, epsilon_end = 0.2, weight_loss_coef = 1.0 -> 194.18% return (total buys = 28, total sells = 13)

    # WITH FIXED WEIGHT OF 0.1
    # Best so far: epsilon_start = 1.0, epsilon_end = 0.01, weight_loss_coef = 0.5 -> 199.48% return (total buys = 5, total sells = 0)
    # Other good configurations:
    # epsilon_start = 1.0, epsilon_end = 0.01, weight_loss_coef = 1.0 -> 193.93% return (total buys = 12, total sells = 7)
    # epsilon_start = 1.0, epsilon_end = 0.02, weight_loss_coef = 1.0, rand_weights = False -> 188.08% return (total buys = 7, total sells = 2)

    train_csv, test_csv = fetch_data_main(ticker, start, end, split_date)
    # for params in [
    #     (0.05, 1.0, True),  # epsilon_end, weight_loss_coef, 
    #     (0.01, 1.0, True),
    #     (0.02, 1.0, True),
    #     (0.01, 0.5, False),
    #     (0.01, 1.0, False),
    #     (0.02, 1.0, False),]:
    
    for params in [(0.05, 0.0, True), (0.01, 0.0, True), (0.05, 0.0, False), (0.01, 0.0, False)]:
        epsilon_end, weight_loss_coef, rand_weights = params
        print(f"\nRunning with params: epsilon_end={epsilon_end}, weight_loss_coef={weight_loss_coef}, rand_weights={rand_weights}")

        ckpt_path = train_main(ticker, train_csv, commission_rate, investment_capacity, epochs=50, epsilon_start=epsilon_start, epsilon_end=epsilon_end, feature_dim=feature_dim, weight_loss_coef=weight_loss_coef, rand_weights = rand_weights, output = False)
        # ckpt_path = f"checkpoints/utrans_{ticker}_final.pt"
        evaluate_main(ticker, test_csv, ckpt_path, commission_rate, investment_capacity, feature_dim=feature_dim, window_size=window_size)

    # Run the feature generation script

 