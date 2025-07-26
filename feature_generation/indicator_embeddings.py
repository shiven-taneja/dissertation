"""
Convert daily indicator rows into ChatGPT commentary,
then embed with FinBERT CLS vector and append to CSV.
"""
import os, torch, tqdm, pandas as pd, pickle
from transformers import AutoTokenizer, AutoModel
import openai, numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = "gpt-3.5-turbo"   # or 4
TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
FINBERT   = AutoModel.from_pretrained("ProsusAI/finbert").eval()

def prompt_from_indicators(row):
    return (
        "You are a seasoned technical analyst.\n"
        f"Date: {row['date']}\n"
        f"SMA_20={row['SMA_20']:.2f}, SMA_50={row['SMA_50']:.2f}, "
        f"RSI={row['RSI']:.1f}, MACD={row['MACD']:.3f}, "
        f"UpperBB={row['BB_upper']:.2f}, LowerBB={row['BB_lower']:.2f}.\n"
        "Give a concise analysis (1-2 sentences)."
    )

def analyse(row):
    resp = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt_from_indicators(row)}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def embed(text):
    with torch.no_grad():
        tok = TOKENIZER(text, return_tensors="pt", truncation=True, max_length=128)
        out = FINBERT(**tok)       # last_hidden_state: (1, seq_len, 768)
        vec = out.last_hidden_state[0,0]  # CLS token
        return vec.numpy()

def run(csv_path):
    df = pd.read_csv(csv_path)
    embs = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        analysis = analyse(row)
        vec = embed(analysis)
        embs.append(vec)
    embs = np.vstack(embs)                      # (N, 768)
    for i in range(embs.shape[1]):
        df[f"emb_{i}"] = embs[:,i]
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    run("data/AAPL_train.csv")
    run("data/AAPL_test.csv")
