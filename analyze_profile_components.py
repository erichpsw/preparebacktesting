import pandas as pd
import numpy as np

df = pd.read_csv("campaigns_clean.csv")

# numeric cleanup
for col in ["CMP", "TR", "DCR", "ORC", "PRF", "return_pct"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["is_winner"] = df["return_pct"] > 0

print("\nTOTAL ROWS:", len(df))
print("WIN RATE:", df["is_winner"].mean())

# --------------------------------------------------
# 1) Means: winners vs losers
# --------------------------------------------------
print("\n" + "="*70)
print("WINNERS VS LOSERS — MEAN COMPARISON")
print("="*70)

rows = []
for col in ["CMP", "TR", "DCR", "ORC", "PRF"]:
    w = df.loc[df["is_winner"], col]
    l = df.loc[~df["is_winner"], col]
    rows.append({
        "factor": col,
        "winner_mean": w.mean(),
        "loser_mean": l.mean(),
        "delta": w.mean() - l.mean()
    })

means_df = pd.DataFrame(rows).sort_values("delta", ascending=False)
print(means_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

# --------------------------------------------------
# 2) Expectancy by buckets
# --------------------------------------------------
print("\n" + "="*70)
print("EXPECTANCY BY EXISTING BUCKETS")
print("="*70)

for col in ["CMP_bucket", "TR_bucket", "DCR_bucket", "ORC_bucket", "PRF_bucket"]:
    if col in df.columns:
        print(f"\n--- {col} ---")
        print(df.groupby(col)["return_pct"].agg(["count", "mean", "median"]).sort_index())

# --------------------------------------------------
# 3) Threshold enrichment: winners vs losers
# --------------------------------------------------
thresholds = {
    "CMP": [50, 65, 80, 90],
    "TR": [55, 65, 75, 85],
    "DCR": [45, 60, 75, 90],
    "ORC": [0.25, 0.50, 0.75],
    "PRF": [70, 80, 85, 90]
}

print("\n" + "="*70)
print("THRESHOLD ENRICHMENT — WINNERS VS LOSERS")
print("="*70)

out = []
for factor, cuts in thresholds.items():
    for t in cuts:
        w_pct = (df.loc[df["is_winner"], factor] >= t).mean()
        l_pct = (df.loc[~df["is_winner"], factor] >= t).mean()
        enrichment = np.nan if l_pct == 0 else w_pct / l_pct
        out.append({
            "factor": factor,
            "threshold": t,
            "winner_pct_ge": w_pct,
            "loser_pct_ge": l_pct,
            "enrichment": enrichment
        })

enrich_df = pd.DataFrame(out).sort_values(["enrichment", "winner_pct_ge"], ascending=[False, False])
print(enrich_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

# --------------------------------------------------
# 4) Simple candidate rules
# --------------------------------------------------
print("\n" + "="*70)
print("SIMPLE PROFILE RULE TESTS")
print("="*70)

rules = {
    "CMP>=65": df["CMP"] >= 65,
    "CMP>=80": df["CMP"] >= 80,
    "TR>=75": df["TR"] >= 75,
    "DCR>=75": df["DCR"] >= 75,
    "DCR>=90": df["DCR"] >= 90,
    "ORC_0.25_0.75": (df["ORC"] >= 0.25) & (df["ORC"] <= 0.75),
    "PRF>=80": df["PRF"] >= 80,
    "CMP>=65 & DCR>=75": (df["CMP"] >= 65) & (df["DCR"] >= 75),
    "CMP>=65 & ORC_0.25_0.75": (df["CMP"] >= 65) & (df["ORC"] >= 0.25) & (df["ORC"] <= 0.75),
    "DCR>=75 & ORC_0.25_0.75": (df["DCR"] >= 75) & (df["ORC"] >= 0.25) & (df["ORC"] <= 0.75),
    "CMP>=65 & DCR>=75 & ORC_0.25_0.75": (df["CMP"] >= 65) & (df["DCR"] >= 75) & (df["ORC"] >= 0.25) & (df["ORC"] <= 0.75),
}

rule_rows = []
for name, mask in rules.items():
    sub = df.loc[mask.fillna(False)]
    if len(sub) == 0:
        continue
    rule_rows.append({
        "rule": name,
        "count": len(sub),
        "win_rate": sub["is_winner"].mean(),
        "expectancy": sub["return_pct"].mean(),
        "median_return": sub["return_pct"].median()
    })

rules_df = pd.DataFrame(rule_rows).sort_values(["expectancy", "win_rate"], ascending=[False, False])
print(rules_df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))