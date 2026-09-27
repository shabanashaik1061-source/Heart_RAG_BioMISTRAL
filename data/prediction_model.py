import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# =========================================================
# 1. LOAD DATA
# =========================================================

print("Loading heart health dataset...")

df = pd.read_csv("dataset_full.csv")

print("Dataset loaded!")
print("Total records:", len(df))
print("Categories:", df["source"].nunique())


# =========================================================
# 2. PREPARE DATA
# =========================================================

# Use patient/user question as input
X = df["input"].fillna("")

# Use source category as prediction label
y = df["source"].fillna("")


print("\nInput examples:")
print(X.head(3))

print("\nCategories:")
print(y.value_counts())


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
# 4. CREATE ML PIPELINE
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

print("\nTraining prediction model...")

model.fit(X_train, y_train)

print("Model training completed!")


# =========================================================
# 6. TEST MODEL
# =========================================================

print("\nTesting model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("MODEL PERFORMANCE")
print("======================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# =========================================================
# 7. TEST WITH USER QUESTION
# =========================================================

test_question = input(
    "\nEnter a health question to test prediction: "
)

prediction = model.predict([test_question])[0]

probabilities = model.predict_proba([test_question])[0]

confidence = probabilities.max() * 100

print("\n======================================")
print("PREDICTION RESULT")
print("======================================")

print("Question:")
print(test_question)

print("\nPredicted category:")
print(prediction)

print("\nConfidence:")
print(round(confidence, 2), "%")

print("\n======================================")