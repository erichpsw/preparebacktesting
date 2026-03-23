import pandas as pd

# Load data
df = pd.read_csv("campaigns_clean.csv")

# Get unique symbols
tickers = sorted(df["symbol"].dropna().unique())

# Print count
print(f"Total unique tickers: {len(tickers)}")

# Save to file (one ticker per line — TradingView ready)
with open("tickers_list.txt", "w") as f:
    for t in tickers:
        f.write(f"{t}\n")

print("Saved to tickers_list.txt")