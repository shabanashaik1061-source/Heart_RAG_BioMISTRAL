import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# =========================================================
# 1. LOAD DATA
# =========================================================

print("Loading dataset...")

df = pd.read_csv("dataset_full.csv")

print("Dataset loaded!")
print("Total records:", len(df))
print("Total categories:", df["source"].nunique())


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

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =========================================================
# 4. CREATE MODEL
# =========================================================

model = Pipeline([

    (
        "tfidf",

        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",

        LogisticRegression(
            max_iter=2000
        )
    )

])


# =========================================================
# 5. TRAIN MODEL
# =========================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# =========================================================
# 6. PREDICTIONS
# =========================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

probabilities = model.predict_proba(X_test)

classes = model.classes_


# =========================================================
# 7. TOP-1 ACCURACY
# =========================================================

top1_accuracy = accuracy_score(
    y_test,
    y_pred
)


# =========================================================
# 8. TOP-3 ACCURACY
# =========================================================

top3_correct = 0

for row, actual_label in zip(
    probabilities,
    y_test
):

    top3_indices = row.argsort()[-3:][::-1]

    top3_classes = classes[top3_indices]

    if actual_label in top3_classes:

        top3_correct += 1


top3_accuracy = (
    top3_correct / len(y_test)
)


# =========================================================
# 9. DISPLAY RESULTS
# =========================================================

print("\n======================================")
print("       MODEL EVALUATION")
print("======================================")

print(
    "Top-1 Accuracy:",
    round(top1_accuracy * 100, 2),
    "%"
)

print(
    "Top-3 Accuracy:",
    round(top3_accuracy * 100, 2),
    "%"
)


# =========================================================
# 10. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# =========================================================
# 11. CONFUSION MATRIX
# =========================================================

labels = sorted(
    y.unique()
)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=classes)

# Short category names for better readability
short_labels = [
    label.replace("synthetic_", "").replace("_", " ").title()
    for label in classes
]

plt.figure(figsize=(16, 13))

plt.imshow(cm, interpolation="nearest", cmap="Reds")

plt.title("Confusion Matrix", fontsize=18, pad=15)
plt.xlabel("Predicted Category", fontsize=13)
plt.ylabel("Actual Category", fontsize=13)

plt.xticks(
    range(len(classes)),
    short_labels,
    rotation=45,
    ha="right",
    fontsize=9
)

plt.yticks(
    range(len(classes)),
    short_labels,
    fontsize=9
)

# Display numbers inside cells
for i in range(len(classes)):
    for j in range(len(classes)):
        if cm[i, j] > 0:
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=8
            )

plt.colorbar(label="Number of Predictions")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved as confusion_matrix.png")


