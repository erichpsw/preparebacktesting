import pandas as pd
import numpy as np

df = pd.read_csv("campaigns_clean.csv")

# -----------------------------
# Define runner from actual outcome
# -----------------------------
df["is_runner"] = (df["exit_type"].astype(str).str.upper() == "T3")

print("\nRUNNER COUNTS")
print(df["is_runner"].value_counts(dropna=False))

# -----------------------------
# Fields to inspect
# -----------------------------
numeric_fields = [
    "PRF", "RUN", "PRS",
    "SS", "TR", "RS", "CMP", "VQS",
    "ADR", "DCR", "RMV", "ORC", "GRP", "DDV"
]

categorical_fields = [
    "STATE", "SQZ", "MODE",
    "SQZQ", "LTFEXP", "RUNOK", "PROK"
]

# Convert numeric fields safely
for col in numeric_fields:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# 1) Mean comparison
# -----------------------------
print("\n" + "=" * 80)
print("NUMERIC FACTOR COMPARISON: RUNNERS VS NON-RUNNERS")
print("=" * 80)

rows = []
for col in numeric_fields:
    if col not in df.columns:
        continue
    r = df.loc[df["is_runner"], col]
    n = df.loc[~df["is_runner"], col]

    rows.append({
        "factor": col,
        "runner_count": int(r.notna().sum()),
        "non_runner_count": int(n.notna().sum()),
        "runner_mean": r.mean(),
        "non_runner_mean": n.mean(),
        "delta": r.mean() - n.mean()
    })

means_df = pd.DataFrame(rows).sort_values("delta", ascending=False)
print(means_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

# -----------------------------
# 2) Runner rate by bucket / category
# -----------------------------
print("\n" + "=" * 80)
print("RUNNER RATE BY CATEGORICAL / BOOLEAN FIELDS")
print("=" * 80)

for col in categorical_fields:
    if col not in df.columns:
        continue
    tmp = (
        df.groupby(col, dropna=False)["is_runner"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "runner_rate"})
        .sort_values("runner_rate", ascending=False)
    )
    print(f"\n--- {col} ---")
    print(tmp.to_string(float_format=lambda x: f"{x:0.3f}"))

# Also inspect existing bucket fields if present
bucket_fields = [c for c in df.columns if c.endswith("_bucket")]
interesting_buckets = [
    "PRF_bucket", "RUN_bucket", "PRS_bucket",
    "SS_bucket", "TR_bucket", "RS_bucket", "CMP_bucket",
    "VQS_bucket", "ADR_bucket", "DCR_bucket", "RMV_bucket",
    "ORC_bucket", "GRP_bucket", "DDV_bucket"
]

print("\n" + "=" * 80)
print("RUNNER RATE BY BUCKET")
print("=" * 80)

for col in interesting_buckets:
    if col not in df.columns:
        continue
    tmp = (
        df.groupby(col, dropna=False)["is_runner"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "runner_rate"})
        .sort_index()
    )
    print(f"\n--- {col} ---")
    print(tmp.to_string(float_format=lambda x: f"{x:0.3f}"))

# -----------------------------
# 3) Threshold enrichment
# -----------------------------
thresholds = {
    "PRF": [50, 60, 70, 80],
    "RUN": [20, 40, 60, 80],
    "PRS": [50, 60, 70, 80],
    "SS": [55, 65, 75, 85],
    "TR": [55, 65, 75, 85],
    "RS": [70, 80, 90],
    "CMP": [50, 65, 80, 90],
    "VQS": [40, 60, 75],
    "ADR": [2, 4, 6, 8],
    "DCR": [45, 60, 75, 90],
    "RMV": [40, 60, 75, 90],
    "ORC": [0.25, 0.50, 0.75],
    "DDV": [20, 50, 100, 200]
}

print("\n" + "=" * 80)
print("THRESHOLD ENRICHMENT: RUNNERS VS NON-RUNNERS")
print("=" * 80)

enrichment_rows = []
for col, cuts in thresholds.items():
    if col not in df.columns:
        continue

    rmask = df["is_runner"] == True
    nmask = df["is_runner"] == False

    for t in cuts:
        runner_pct = (df.loc[rmask, col] >= t).mean()
        non_runner_pct = (df.loc[nmask, col] >= t).mean()

        enrichment = np.nan
        if pd.notna(non_runner_pct) and non_runner_pct > 0:
            enrichment = runner_pct / non_runner_pct

        enrichment_rows.append({
            "factor": col,
            "threshold": t,
            "runner_pct_ge": runner_pct,
            "non_runner_pct_ge": non_runner_pct,
            "enrichment": enrichment
        })

enrich_df = pd.DataFrame(enrichment_rows).sort_values(
    ["enrichment", "runner_pct_ge"], ascending=[False, False]
)

print(enrich_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

# -----------------------------
# 4) Simple candidate rules
# -----------------------------
print("\n" + "=" * 80)
print("SIMPLE RULE TESTS")
print("=" * 80)

candidate_rules = {
    "PRF>=80": df["PRF"] >= 80 if "PRF" in df.columns else None,
    "PRF>=70": df["PRF"] >= 70 if "PRF" in df.columns else None,
    "TR>=75": df["TR"] >= 75 if "TR" in df.columns else None,
    "RS>=90": df["RS"] >= 90 if "RS" in df.columns else None,
    "CMP>=80": df["CMP"] >= 80 if "CMP" in df.columns else None,
    "DCR>=75": df["DCR"] >= 75 if "DCR" in df.columns else None,
    "ORC_0.25_0.75": ((df["ORC"] >= 0.25) & (df["ORC"] <= 0.75)) if "ORC" in df.columns else None,
    "DDV>=50": df["DDV"] >= 50 if "DDV" in df.columns else None,
    "STATE_READY_NOW": (df["STATE"] == "Ready Now") if "STATE" in df.columns else None,
    "MODE_BREAKOUT": (df["MODE"] == "Breakout") if "MODE" in df.columns else None,
}

# add a few combos
if all(c in df.columns for c in ["PRF", "CMP", "DCR"]):
    candidate_rules["PRF>=80 & CMP>=80"] = (df["PRF"] >= 80) & (df["CMP"] >= 80)
    candidate_rules["PRF>=80 & DCR>=75"] = (df["PRF"] >= 80) & (df["DCR"] >= 75)
    candidate_rules["CMP>=80 & DCR>=75"] = (df["CMP"] >= 80) & (df["DCR"] >= 75)

rule_rows = []
for name, mask in candidate_rules.items():
    if mask is None:
        continue
    sub = df.loc[mask.fillna(False)]
    if len(sub) == 0:
        continue
    rule_rows.append({
        "rule": name,
        "count": len(sub),
        "runner_rate": sub["is_runner"].mean(),
        "expectancy": sub["return_pct"].mean(),
        "median_return": sub["return_pct"].median()
    })

rules_df = pd.DataFrame(rule_rows).sort_values(
    ["runner_rate", "expectancy"], ascending=[False, False]
)
print(rules_df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))