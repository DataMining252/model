import pandas as pd

df = pd.read_csv("gold_with_season.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

train_df.to_csv("statistical/data/gold_train.csv", index=False)
test_df.to_csv("statistical/data/gold_test.csv", index=False)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)