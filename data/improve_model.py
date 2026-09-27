import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# 1. LOAD DATA
# =========================================================

print("Loading dataset...")

df = pd.read_csv("dataset_full.csv")

print("Dataset loaded!")
print("Total records:", len(df))


# =========================================================
# 2. PREPARE DATA
# =========================================================

X = df["input"].fillna("")
y = df["source"].fillna("")


# =========================================================
# 3. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# 4. IMPROVED TF-IDF + LOGISTIC REGRESSION
# =========================================================

print("\nTraining improved prediction model...")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        )
    )
])

model.fit(X_train, y_train)

print("Improved model ready!")


# =========================================================
# 5. PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)

probabilities = model.predict_proba(X_test)

classes = model.classes_


# =========================================================
# 6. TOP-1 ACCURACY
# =========================================================

top1_accuracy = accuracy_score(
    y_test,
    y_pred
)


# =========================================================
# 7. TOP-3 ACCURACY
# =========================================================

top3_correct = 0

for i, true_label in enumerate(y_test):
    
    top3_indices = probabilities[i].argsort()[-3:][::-1]
    
    top3_labels = classes[top3_indices]
    
    if true_label in top3_labels:
        top3_correct += 1


top3_accuracy = top3_correct / len(y_test)


# =========================================================
# 8. RESULTS
# =========================================================

print("\n======================================")
print("IMPROVED MODEL RESULTS")
print("======================================")

print(
    f"Top-1 Accuracy: {top1_accuracy * 100:.2f} %"
)

print(
    f"Top-3 Accuracy: {top3_accuracy * 100:.2f} %"
)


# =========================================================
# 9. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# =========================================================
# 10. COMPARISON
# =========================================================

print("\n======================================")
print("BASELINE vs IMPROVED")
print("======================================")

print("Baseline Top-1 : 62.42 %")
print(
    f"Improved Top-1 : {top1_accuracy * 100:.2f} %"
)

print("Baseline Top-3 : 84.71 %")
print(
    f"Improved Top-3 : {top3_accuracy * 100:.2f} %"
)

print("\nEvaluation completed successfully!")