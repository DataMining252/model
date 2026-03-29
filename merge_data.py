import pandas as pd

# Load data
gold = pd.read_csv("raw/gold_cleaned.csv", parse_dates=["Date"])
macro = pd.read_csv("raw/macro_data.csv", parse_dates=["Date"])

# Sort
gold = gold.sort_values("Date")
macro = macro.sort_values("Date")

# Merge (ưu tiên gold)
df = gold.merge(macro, on="Date", how="left")

# Fill missing macro data
df = df.ffill()

# Drop nếu vẫn còn NaN đầu (do macro bắt đầu trễ)
df = df.dropna()

# Save
df.to_csv("raw/final_dataset.csv", index=False)

print("Saved to final_dataset.csv")
print(df.head())