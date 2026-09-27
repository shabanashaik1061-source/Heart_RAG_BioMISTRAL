import pandas as pd

heart_df = pd.read_csv("dataset_full.csv")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")





# Display basic information
print("\n===== HEART DATASET =====")
print("Shape:", heart_df.shape)
print("Columns:", heart_df.columns.tolist())
print(heart_df.head())

print("\n===== TRAIN DATASET =====")
print("Shape:", train_df.shape)
print("Columns:", train_df.columns.tolist())
print(train_df.head())

print("\n===== TEST DATASET =====")
print("Shape:", test_df.shape)
print("Columns:", test_df.columns.tolist())
print(test_df.head())

# Missing values
print("\n===== MISSING VALUES =====")
print("Heart:")
print(heart_df.isnull().sum())

print("\nTrain:")
print(train_df.isnull().sum())

print("\nTest:")
print(test_df.isnull().sum())

# Data types
print("\n===== DATA TYPES =====")
print(heart_df.dtypes)