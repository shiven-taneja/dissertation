"""
Generate daily news sentiment & volatility labels
and append them to {ticker}_{train|test}.csv
"""
import os, csv, openai, json, tqdm, pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"  # or "gpt-4o"

def prompt_from_headlines(date, headlines):
    joined = "\n".join(f"- {h}" for h in headlines)
    return f"""You are a professional macro analyst.
Date: {date}
Headlines:
{joined}

Classify overall MARKET SENTIMENT as Positive, Neutral, or Negative
and expected DAILY VOLATILITY as High, Medium, or Low.
Respond exactly on one line:
Sentiment=<Positive/Neutral/Negative>;Volatility=<High/Medium/Low>
"""

def classify_day(date, headlines):
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt_from_headlines(date, headlines)}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content
    # crude parse
    sent = text.split("Sentiment=")[1].split(";")[0].strip()
    vol  = text.split("Volatility=")[1].split("\n")[0].strip()
    return sent, vol

def encode(sent, vol):
    sent_map = {"Positive":1, "Neutral":0, "Negative":-1}
    vol_map  = {"Low":0, "Medium":1, "High":2}
    return sent_map[sent], vol_map[vol]       # simple numeric
    # OR return one-hot arrays if you prefer

def run(csv_path, headlines_df):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    sentiments, vols = [], []
    for d in tqdm.tqdm(df["date"]):
        hlines = headlines_df.loc[headlines_df["date"]==d, "headline"].tolist()
        sent, vol = classify_day(d.date(), hlines)
        s_val, v_val = encode(sent, vol)
        sentiments.append(s_val)
        vols.append(v_val)
    df["sentiment"] = sentiments
    df["volatility"] = vols
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    run("data/AAPL_train.csv", pd.read_csv("data/news_headlines.csv"))
    run("data/AAPL_test.csv",  pd.read_csv("data/news_headlines.csv"))
