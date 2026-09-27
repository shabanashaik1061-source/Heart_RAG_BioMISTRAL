import pandas as pd

# Load training and testing datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Training data loaded successfully!")
print("Training shape:", train_df.shape)

print("\nTesting data loaded successfully!")
print("Testing shape:", test_df.shape)

# Check required columns
required_columns = ["instruction", "input", "output", "source"]

for column in required_columns:
    if column in train_df.columns:
        print(f"✓ {column} found")
    else:
        print(f"✗ {column} NOT found")

# Remove missing values
train_df = train_df.dropna(
    subset=["instruction", "input", "output"]
)

test_df = test_df.dropna(
    subset=["instruction", "input", "output"]
)

print("\nAfter removing missing values:")
print("Training:", train_df.shape)
print("Testing:", test_df.shape)

# Create instruction format
def create_prompt(row):
    return f"""### Instruction:
{row['instruction']}

### Input:
{row['input']}

### Response:
{row['output']}"""

train_df["text"] = train_df.apply(create_prompt, axis=1)
test_df["text"] = test_df.apply(create_prompt, axis=1)

# Display one example
print("\n===== SAMPLE TRAINING EXAMPLE =====")
print(train_df["text"].iloc[0])

# Save prepared datasets
train_df.to_csv("train_prepared.csv", index=False)
test_df.to_csv("test_prepared.csv", index=False)

print("\nPrepared datasets saved successfully!")