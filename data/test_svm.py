import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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
# 4. LINEAR SVM MODEL
# =========================================================

print("\nTraining Linear SVM model...")

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
        LinearSVC(
            C=1.0,
            class_weight="balanced"
        )
    )
])

model.fit(X_train, y_train)

print("Linear SVM model ready!")


# =========================================================
# 5. PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)

decision_scores = model.decision_function(X_test)

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

    top3_indices = decision_scores[i].argsort()[-3:][::-1]

    top3_labels = classes[top3_indices]

    if true_label in top3_labels:
        top3_correct += 1


top3_accuracy = top3_correct / len(y_test)


# =========================================================
# 8. RESULTS
# =========================================================

print("\n======================================")
print("LINEAR SVM RESULTS")
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
# 10. MODEL COMPARISON
# =========================================================

print("\n======================================")
print("MODEL COMPARISON")
print("======================================")

print("Logistic Regression Top-1 : 63.69 %")
print(
    f"Linear SVM Top-1          : {top1_accuracy * 100:.2f} %"
)

print("Logistic Regression Top-3 : 84.71 %")
print(
    f"Linear SVM Top-3          : {top3_accuracy * 100:.2f} %"
)

print("\nEvaluation completed successfully!")