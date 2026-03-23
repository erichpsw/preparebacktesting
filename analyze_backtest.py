import pandas as pd

# =============================
# LOAD DATA
# =============================
df = pd.read_csv("campaigns_clean.csv")

print("\nTOTAL ROWS:", len(df))

# =============================
# BASIC SANITY
# =============================
print("\n=== BASIC COUNTS ===")
print(df["sid"].value_counts().sort_index())

print("\nPRF distribution:")
print(df["PRF"].describe())

print("\nRUNOK distribution:")
print(df["RUNOK"].value_counts(dropna=False))

# =============================
# PROFILE ANALYSIS
# =============================
print("\n" + "="*60)
print("PRF BUCKET EXPECTANCY")
print("="*60)

if "PRF_bucket" in df.columns:
    print(df.groupby("PRF_bucket")["return_pct"].agg(["count","mean","median"]).sort_index())

print("\nPROFILE OK PERFORMANCE")
print(df.groupby("PROK")["return_pct"].agg(["count","mean","median"]))

# =============================
# RUNNER ANALYSIS
# =============================
print("\n" + "="*60)
print("RUNOK EXPECTANCY")
print("="*60)

print(df.groupby("RUNOK")["return_pct"].agg(["count","mean","median"]))

print("\nRUNOK EXIT TYPE DISTRIBUTION (RAW)")
print(df.groupby("RUNOK")["exit_type"].value_counts())

print("\nRUNOK EXIT TYPE DISTRIBUTION (NORMALIZED)")
print(df.groupby("RUNOK")["exit_type"].value_counts(normalize=True))

# =============================
# TRADE PLAN PERFORMANCE
# =============================
print("\n" + "="*60)
print("EXIT TYPE PERFORMANCE")
print("="*60)

print(df.groupby("exit_type")["return_pct"].agg(["count","mean","median"]))

# =============================
# SETUP PERFORMANCE
# =============================
print("\n" + "="*60)
print("SETUP ID PERFORMANCE")
print("="*60)

print(df.groupby("sid")["return_pct"].agg(["count","mean","median"]).sort_index())

# =============================
# OPTIONAL: WIN RATE
# =============================
if "is_win" in df.columns:
    print("\nWIN RATE BY SETUP")
    print(df.groupby("sid")["is_win"].mean())

# =============================
# QUICK CHECK: DO RUNNERS ACTUALLY RUN?
# =============================
print("\n" + "="*60)
print("RUNNER CHECK (T3 FREQUENCY)")
print("="*60)

runner_check = df[df["exit_type"] == "T3"].groupby("RUNOK")["exit_type"].count()
print(runner_check)