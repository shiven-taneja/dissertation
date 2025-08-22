# ===============================================================
# file: generate_dissertation_visuals.py
# ===============================================================
"""
Generates all metrics and graphics for the dissertation's 
'Results and Analysis' chapter.

This script performs the following steps:
1.  Iterates through all specified tickers and model types.
2.  Runs an evaluation for each model, collecting performance data.
    - Includes a critical fix for Buy & Hold equity curve alignment.
3.  Aggregates results across all tickers.
4.  Generates and saves all required plots to a structured 'diss_plots' directory:
    - Individual & Combined Equity Curves
    - Comparative Bar Charts (Sharpe, CAGR, MDD)
    - Performance Distribution Box Plots
    - Market Regime Performance Analysis
5.  Generates and saves a master CSV table with aggregate metrics.
"""
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import torch
from tqdm import tqdm

# Local project imports (ensure your project structure allows this)
from drl_utrans.utils.metrics import cagr, sharpe, max_drawdown
from drl_utrans.envs.single_stock import PaperSingleStockEnv
from drl_utrans.agent.drl_utrans import DrlUTransAgent

# --- Script Configuration ---

# Ignore harmless warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define tickers and model types
TICKERS = ["BABA", "GE", "GOOG", "KO", "MRK", "MS", "NVDA", "QQQ", "T", "WFC"]
RUN_TYPES = ["baseline", "techsent", "headline", "all"]
MODEL_NAME_MAP = {
    "baseline": "Baseline",
    "techsent": "Indicator-LLM",
    "headline": "Headline-LLM",
    "all": "Combined-Fusion",
    "bh": "Buy & Hold"
}

# Define root paths
BASE_DIR = Path(__file__).parent
CHECKPOINT_ROOT = BASE_DIR / "checkpoints"
DATA_ROOT = BASE_DIR / "data"
OUTPUT_ROOT = BASE_DIR / "diss_plots"

# Define model parameters
WINDOW_SIZE = 12
INVESTMENT_CAPACITY = 500  # As per your original code
COMMISSION_RATE = 0.001

# --- Feature Column Definitions ---
HEADLINE_COLS = ["news_sent_mean_1d", "news_sent_std_1d", "news_sent_maxabs_1d", "news_count_1d"]
TECH_SENT_COL = ["tech_sent_score"]
BASE_FEATURE_COLS = ["macd", "kdj_k", "open_close_diff", "rsi_14"]

# --- Market Regime Definitions ---
# These dates are defined based on QQQ's performance relative to its 200-day SMA.
# You can adjust these based on your specific test periods.
MARKET_REGIMES = {
    "Bull (Mar 2020 - Dec 2021)": ("2020-03-23", "2021-12-31"),
    "Bear (Jan 2022 - Oct 2022)": ("2022-01-01", "2022-10-13"),
    "Volatile/Recovery (Oct 2022 - Jun 2023)": ("2022-10-14", "2023-06-30"),
}


# --- Core Evaluation Logic ---

def get_feature_cols(run_type: str, ticker) -> List[str]:
    """Returns the correct feature columns for a given run type."""
    base = BASE_FEATURE_COLS.copy()
    if ticker in ["KO", "MRK", "MS", "QQQ", "WFC"]:
        base = ["open_close_diff"]
    if run_type == "baseline":
        return base
    if run_type == "techsent":
        return base + TECH_SENT_COL
    if run_type == "headline":
        return base + HEADLINE_COLS
    if run_type == "all":
        return base + HEADLINE_COLS + TECH_SENT_COL
    raise ValueError(f"Unknown run_type: {run_type}")

def run_evaluation_episode(env: PaperSingleStockEnv, agent: DrlUTransAgent) -> np.ndarray:
    """Runs a single evaluation episode and returns the equity curve."""
    state_nd = env.reset()
    state = torch.from_numpy(state_nd).float()
    equity_curve = [env.portfolio_value()]

    while True:
        act, w = agent.select_action(state, eval_mode=True)
        nxt, _, done, _ = env.step((act, w))
        state = torch.from_numpy(nxt).float()
        equity_curve.append(env.portfolio_value())
        if done:
            break
    return np.asarray(equity_curve)

def evaluate_model(ticker: str, run_type: str) -> Dict:
    """
    Evaluates a single model checkpoint and returns its performance data.
    Includes the fix for B&H alignment and dynamic model initialization.
    """
    test_csv_path = DATA_ROOT /ticker/"test.csv"
    if not test_csv_path.exists():
        print(f"Warning: Test data not found for {ticker} at {test_csv_path}. Skipping.")
        return None
        
    df_test = pd.read_csv(test_csv_path, parse_dates=['date'], index_col='date')

    # Find the best checkpoint for the given type
    ckpt_dir = CHECKPOINT_ROOT / ticker
    search_pattern = f"utrans_{ticker}_{run_type}_best_seed*.pt"
    model_files = list(ckpt_dir.glob(search_pattern))
    
    if not model_files:
        print(f"Warning: Checkpoint not found for {ticker}/{run_type}. Skipping.")
        return None
    ckpt_path = model_files[0]
    
    # --- DYNAMIC FEATURE AND MODEL INITIALIZATION (THE FIX) ---
    feature_cols = get_feature_cols(run_type, ticker)
    num_market_features = len(feature_cols)
    total_features = num_market_features + 3

    # Initialize the agent with the CORRECT state dimension for THIS model
    agent = DrlUTransAgent(state_dim=(WINDOW_SIZE, total_features))
    # -----------------------------------------------------------

    feats = df_test[feature_cols].to_numpy(dtype=np.float32)
    prices = df_test["close"].to_numpy(dtype=np.float32)

    env = PaperSingleStockEnv(
        feats, prices, window_size=WINDOW_SIZE, ic_shares=INVESTMENT_CAPACITY, commission=COMMISSION_RATE
    )
    
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("policy", checkpoint)
        agent.policy_net.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading checkpoint {ckpt_path}: {e}")
        return None

    agent.policy_net.eval()
    agent.epsilon = 0.0

    equity_curve = run_evaluation_episode(env, agent)

    # --- B&H ALIGNMENT FIX ---
    start_idx = WINDOW_SIZE - 1
    sim_dates = df_test.index[start_idx : start_idx + len(equity_curve)]
    sim_prices = prices[start_idx : start_idx + len(equity_curve)]

    initial_capital = equity_curve[0]
    bh_equity = initial_capital * (sim_prices / sim_prices[0])

    # Calculate metrics
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    bh_daily_returns = np.diff(bh_equity) / bh_equity[:-1]

    metrics = {
        "Final Return %": (equity_curve[-1] / equity_curve[0] - 1) * 100,
        "CAGR %": cagr(equity_curve, sim_dates),
        "Sharpe Ratio": sharpe(daily_returns),
        "Sortino Ratio": sharpe(daily_returns, downside_only=True),
        "Maximum Drawdown %": max_drawdown(equity_curve) * 100,
    }
    bh_metrics = {
        "Final Return %": (bh_equity[-1] / bh_equity[0] - 1) * 100,
        "CAGR %": cagr(bh_equity, sim_dates),
        "Sharpe Ratio": sharpe(bh_daily_returns),
        "Sortino Ratio": sharpe(bh_daily_returns, downside_only=True),
        "Maximum Drawdown %": max_drawdown(bh_equity) * 100,
    }

    return {
        "metrics": metrics,
        "bh_metrics": bh_metrics,
        "equity": equity_curve,
        "bh_equity": bh_equity,
        "dates": sim_dates
    }

# --- Plotting Functions ---

def setup_plot(title: str, ylabel: str = "Portfolio Value ($)", figsize: Tuple[int, int] = (12, 7)):
    """Sets up a standard matplotlib plot."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.xticks(rotation=45)

def save_and_close_plot(path: Path):
    """Saves a plot to a file and closes the figure."""
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_individual_equity_curves(ticker: str, results: Dict, output_dir: Path):
    """Plots and saves the equity curve for each model type individually."""
    for run_type, data in results.items():
        if data is None: continue
        
        setup_plot(f"Equity Curve: {MODEL_NAME_MAP[run_type]} vs. Buy & Hold ({ticker})")
        plt.plot(data["dates"], data["equity"], label=MODEL_NAME_MAP[run_type], linewidth=2.5, zorder=5)
        plt.plot(data["dates"], data["bh_equity"], label="Buy & Hold", color='gray', linestyle='--', alpha=0.8)
        save_and_close_plot(output_dir / f"equity_individual_{run_type}.png")

def plot_combined_equity_curve(ticker: str, results: Dict, output_dir: Path):
    """Plots and saves all model equity curves on a single chart."""
    setup_plot(f"Combined Equity Curves ({ticker})")
    for run_type, data in results.items():
        if data is None: continue
        plt.plot(data["dates"], data["equity"], label=MODEL_NAME_MAP[run_type], linewidth=2)
    
    # Add B&H once
    first_valid_run = next((r for r in results.values() if r is not None), None)
    if first_valid_run:
        plt.plot(first_valid_run["dates"], first_valid_run["bh_equity"], label="Buy & Hold", color='black', linestyle=':', linewidth=2.5)
        
    save_and_close_plot(output_dir / "equity_combined.png")

def plot_annotated_equity_curve(ticker: str, results: Dict, output_dir: Path):
    """Plots combined equity curve with shaded market regime periods."""
    setup_plot(f"Combined Equity Curves with Market Regimes ({ticker})")
    
    for run_type, data in results.items():
        if data is None: continue
        plt.plot(data["dates"], data["equity"], label=MODEL_NAME_MAP[run_type], linewidth=2, alpha=0.9)

    first_valid_run = next((r for r in results.values() if r is not None), None)
    if first_valid_run:
        plt.plot(first_valid_run["dates"], first_valid_run["bh_equity"], label="Buy & Hold", color='black', linestyle=':', linewidth=2.5)

        # Add shaded regions for regimes
        for regime, (start, end) in MARKET_REGIMES.items():
            plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), facecolor='gray', alpha=0.15, label=regime if regime not in plt.gca().get_legend_handles_labels()[1] else "")
    
    save_and_close_plot(output_dir / "equity_annotated_regimes.png")

def plot_comparative_bars(df: pd.DataFrame, output_dir: Path):
    """Plots bar charts comparing key metrics across all models."""
    metrics_to_plot = {
        "Sharpe Ratio": "Average Sharpe Ratio",
        "CAGR %": "Average Compound Annual Growth Rate (%)",
        "Maximum Drawdown %": "Average Maximum Drawdown (%)"
    }
    
    for metric, title in metrics_to_plot.items():
        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-whitegrid')
        
        avg_metrics = df.groupby('Model')[metric].mean().sort_values(ascending=False if metric != "Maximum Drawdown %" else True)
        
        colors = sns.color_palette("viridis", len(avg_metrics))
        bars = plt.bar(avg_metrics.index, avg_metrics.values, color=colors)
        
        plt.title(title, fontsize=16)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=0)
        
        # Add values on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom' if yval >= 0 else 'top', ha='center')

        plt.tight_layout()
        plt.savefig(output_dir / f"bar_{metric.replace(' %', '').replace(' ', '_').lower()}.png", dpi=200)
        plt.close()

def plot_performance_boxplots(df: pd.DataFrame, output_dir: Path):
    """Plots box plots of metric distributions across all tickers."""
    metrics_to_plot = ["Sharpe Ratio", "CAGR %"]
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 7))
        plt.style.use('seaborn-v0_8-whitegrid')
        
        sns.boxplot(x='Model', y=metric, data=df, palette="viridis", order=df.groupby('Model')[metric].median().sort_values(ascending=False).index)
        sns.stripplot(x='Model', y=metric, data=df, color=".25", size=5, order=df.groupby('Model')[metric].median().sort_values(ascending=False).index)
        
        plt.title(f"Distribution of {metric} Across All Tickers", fontsize=16)
        plt.ylabel(metric, fontsize=12)
        plt.xlabel("Model", fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / f"boxplot_{metric.replace(' %', '').replace(' ', '_').lower()}.png", dpi=200)
        plt.close()

def plot_regime_performance(df: pd.DataFrame, output_dir: Path):
    """Plots grouped bar chart of performance in different market regimes."""
    if 'Regime' not in df.columns:
        print("Warning: Regime data not available. Skipping regime plot.")
        return
        
    plt.figure(figsize=(14, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Filter out empty regimes and calculate mean returns
    regime_df = df[df['Regime'] != 'N/A'].copy()
    regime_df['Return %'] = regime_df.apply(lambda row: (row['Equity'].iloc[-1] / row['Equity'].iloc[0] - 1) * 100, axis=1)

    grouped_perf = regime_df.groupby(['Regime', 'Model'])['Return %'].mean().unstack()
    
    grouped_perf.plot(kind='bar', figsize=(14, 8), colormap='viridis', rot=0)
    
    plt.title("Average Model Performance by Market Regime", fontsize=16)
    plt.ylabel("Average Return During Period (%)", fontsize=12)
    plt.xlabel("Market Regime", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(output_dir / "bar_regime_performance.png", dpi=200)
    plt.close()


# --- Main Execution ---

def main():
    """Main function to run all evaluations and generate all visuals."""
    print("🚀 Starting dissertation visual generation process...")
    OUTPUT_ROOT.mkdir(exist_ok=True)
    
    all_results = {}
    
    # Step 1: Run evaluation for all tickers and models
    print("\nStep 1: Running evaluations...")
    for ticker in tqdm(TICKERS, desc="Processing Tickers"):
        ticker_results = {}
        for run_type in RUN_TYPES:
            result = evaluate_model(ticker, run_type)
            if result:
                ticker_results[run_type] = result
        all_results[ticker] = ticker_results

    # Step 2: Create a master DataFrame for aggregate analysis
    print("\nStep 2: Aggregating results into master DataFrame...")
    records = []
    regime_records = []
    for ticker, ticker_results in all_results.items():
        for run_type, data in ticker_results.items():
            if data is None: continue
            
            # Record for main metrics
            rec = data['metrics'].copy()
            rec['Ticker'] = ticker
            rec['Model'] = MODEL_NAME_MAP[run_type]
            records.append(rec)
            
            # Record for B&H
            if run_type == 'baseline': # Only add B&H once per ticker
                bh_rec = data['bh_metrics'].copy()
                bh_rec['Ticker'] = ticker
                bh_rec['Model'] = 'Buy & Hold'
                records.append(bh_rec)
            
            # Records for regime analysis
            full_equity_series = pd.Series(data['equity'], index=data['dates'])
            for regime_name, (start_str, end_str) in MARKET_REGIMES.items():
                start_date, end_date = pd.to_datetime(start_str), pd.to_datetime(end_str)
                regime_equity = full_equity_series.loc[start_date:end_date]
                if not regime_equity.empty:
                    regime_records.append({
                        'Ticker': ticker,
                        'Model': MODEL_NAME_MAP[run_type],
                        'Regime': regime_name,
                        'Equity': regime_equity
                    })
    
    master_df = pd.DataFrame(records)
    regime_df_full = pd.DataFrame(regime_records)
    
    # Step 3: Generate and save the master metrics table
    print("\nStep 3: Generating master metrics table...")
    agg_dir = OUTPUT_ROOT / "_aggregate"
    agg_dir.mkdir(exist_ok=True)
    
    master_table = master_df.groupby('Model').mean(numeric_only=True)
    master_table.to_csv(agg_dir / "master_comparative_metrics.csv")
    print(f"✅ Master metrics table saved to {agg_dir / 'master_comparative_metrics.csv'}")
    print("\n--- Aggregate Performance Metrics ---")
    print(master_table.round(2).to_string())
    print("-----------------------------------")

    # Step 4: Generate aggregate plots
    print("\nStep 4: Generating aggregate plots...")
    plot_comparative_bars(master_df, agg_dir)
    plot_performance_boxplots(master_df, agg_dir)
    plot_regime_performance(regime_df_full, agg_dir)
    print(f"✅ Aggregate plots saved to {agg_dir}")

    # Step 5: Generate ticker-specific plots
    print("\nStep 5: Generating plots for each ticker...")
    for ticker in tqdm(TICKERS, desc="Generating Ticker Plots"):
        if ticker in all_results and all_results[ticker]:
            ticker_output_dir = OUTPUT_ROOT / ticker
            ticker_output_dir.mkdir(exist_ok=True)
            
            plot_individual_equity_curves(ticker, all_results[ticker], ticker_output_dir)
            plot_combined_equity_curve(ticker, all_results[ticker], ticker_output_dir)
            plot_annotated_equity_curve(ticker, all_results[ticker], ticker_output_dir)
    
    print("\n🎉 All visuals and metrics have been successfully generated!")

if __name__ == "__main__":
    main()