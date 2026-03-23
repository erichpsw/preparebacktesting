import pandas as pd

# ==============================
# CONFIG
# ==============================
INPUT_FILE = "campaigns_clean.csv"

# define winner threshold (adjust if needed)
WIN_THRESHOLD = 0.0   # >0% return = winner

# factor columns
FACTOR_COLS = ["RS", "TR", "CMP", "VQS"]

# thresholds for enrichment check (can tweak)
THRESHOLDS = {
    "RS": 73,
    "TR": 37,
    "CMP": 60,
    "VQS": 50
}

# ==============================
# LOAD
# ==============================
df = pd.read_csv(INPUT_FILE)

# ensure numeric
for col in FACTOR_COLS + ["return_pct"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["return_pct", "sid"])

# ==============================
# WINNER SPLIT
# ==============================
df["is_winner"] = df["return_pct"] > WIN_THRESHOLD

# ==============================
# ANALYSIS
# ==============================
results = []

for sid in sorted(df["sid"].unique()):
    sid_df = df[df["sid"] == sid]
    winners = sid_df[sid_df["is_winner"]]

    if len(sid_df) < 50 or len(winners) < 20:
        continue

    for factor in FACTOR_COLS:
        all_mean = sid_df[factor].mean()
        win_mean = winners[factor].mean()

        # threshold enrichment
        thresh = THRESHOLDS[factor]

        all_pct = (sid_df[factor] >= thresh).mean()
        win_pct = (winners[factor] >= thresh).mean()

        enrichment = (win_pct / all_pct) if all_pct > 0 else 0

        results.append({
            "SID": sid,
            "Factor": factor,
            "All_Mean": round(all_mean, 2),
            "Winner_Mean": round(win_mean, 2),
            "Mean_Delta": round(win_mean - all_mean, 2),
            "All_Pct_Above": round(all_pct, 3),
            "Winner_Pct_Above": round(win_pct, 3),
            "Enrichment": round(enrichment, 3)
        })

out = pd.DataFrame(results)

# ==============================
# RANK FACTORS PER SID
# ==============================
ranked = []

for sid in out["SID"].unique():
    sub = out[out["SID"] == sid].copy()

    sub = sub.sort_values(by="Enrichment", ascending=False)

    sub["Rank"] = range(1, len(sub) + 1)

    ranked.append(sub)

ranked_df = pd.concat(ranked)

# ==============================
# SAVE
# ==============================
ranked_df.to_csv("sid_factor_enrichment.csv", index=False)

print("\nSaved: sid_factor_enrichment.csv\n")
print(ranked_df.head(20))