import pandas as pd

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("campaigns_clean.csv")

# -------------------------
# CLEANING
# -------------------------
numeric_cols = ["return_pct", "RS", "TR", "CMP", "VQS", "RMV", "DCR", "ORC"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["return_pct"])

# =========================
# FLAGS
# =========================
df["is_winner"] = df["return_pct"] > 0

print(f"TOTAL ROWS: {len(df)}")
print(f"TOTAL WINNERS: {df['is_winner'].sum()}")
print(f"WIN RATE: {df['is_winner'].mean():.3f}")

# =========================
# VALIDATE SYMBOL COLUMN
# =========================
if "symbol" not in df.columns:
    raise ValueError("Column 'symbol' not found in CSV")

# =========================
# GROUP
# =========================
ticker_stats = (
    df.groupby("symbol")
      .agg(
          total_trades=("return_pct", "count"),
          win_trades=("is_winner", "sum"),
          win_rate=("is_winner", "mean"),
          avg_return=("return_pct", "mean"),
          total_return=("return_pct", "sum")
      )
)

print(df["return_pct"].describe())

# =========================
# FILTER (VALIDATION UNIVERSE)
# =========================
min_trades = 5
min_win_rate = 0.30
min_avg_return = 0.005   # 0.5% avg return (decimal form)

ticker_stats = ticker_stats[
    (ticker_stats["total_trades"] >= min_trades) &
    (ticker_stats["win_rate"] >= min_win_rate) &
    (ticker_stats["avg_return"] >= min_avg_return)
]

# Sort for usability (not filtering)
ticker_stats = ticker_stats.sort_values(
    by=["total_return", "avg_return"],
    ascending=False
)

# Sort after filtering
ticker_stats = ticker_stats.sort_values(
    by=["total_return", "avg_return"],
    ascending=False
)

print(f"\nFinal universe size: {len(ticker_stats)}")
print("Avg avg_return:", ticker_stats["avg_return"].mean() if len(ticker_stats) else "No rows")
print("Median avg_return:", ticker_stats["avg_return"].median() if len(ticker_stats) else "No rows")

# =========================
# SAVE OUTPUTS
# =========================
ticker_stats.to_csv("winner_ticker_stats.csv")

ticker_stats.index.to_series().to_csv(
    "winner_tickers.txt",
    index=False,
    header=False
)

print("\nTOP WINNERS:")
print(ticker_stats.head(20))

# =========================
# OPTIONAL SCORE CHECK
# =========================
if all(col in df.columns for col in ["RS", "CMP", "TR"]):
    df["score_proxy"] = df["RS"] + df["CMP"] + df["TR"]

    top = df.nlargest(int(len(df) * 0.2), "score_proxy")
    bottom = df.nsmallest(int(len(df) * 0.2), "score_proxy")

    print("\nScore Proxy Check:")
    print("Top 20% expectancy:", top["return_pct"].mean())
    print("Bottom 20% expectancy:", bottom["return_pct"].mean())