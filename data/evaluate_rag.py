import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss


# =========================================================
# 1. LOAD DATA
# =========================================================

print("Loading RAG dataset...")

df = pd.read_csv("train_prepared.csv")

documents = []

for _, row in df.iterrows():

    text = f"""
Instruction: {row['instruction']}
Input: {row['input']}
Output: {row['output']}
Source: {row['source']}
"""

    documents.append(text)

print("Total documents:", len(documents))


# =========================================================
# 2. LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# =========================================================
# 3. CREATE EMBEDDINGS
# =========================================================

print("\nCreating embeddings...")

embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)


# =========================================================
# 4. CREATE FAISS INDEX
# =========================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS index ready!")


# =========================================================
# 5. TEST QUESTIONS
# =========================================================

test_questions = [

    {
        "question": "What are the risk factors for heart disease?",
        "expected_category": "synthetic_cardiovascular"
    },

    {
        "question": "What are common symptoms of diabetes?",
        "expected_category": "synthetic_diabetes"
    },

    {
        "question": "What can cause a headache?",
        "expected_category": "synthetic_headache"
    },

    {
        "question": "What are common symptoms of a cold or flu?",
        "expected_category": "synthetic_cold_flu"
    },

    {
        "question": "What can help with back pain?",
        "expected_category": "synthetic_back_joint"
    }
]


# =========================================================
# 6. EVALUATE RETRIEVAL
# =========================================================

correct = 0

print("\n==========================================")
print("RAG RETRIEVAL EVALUATION")
print("==========================================")


for test in test_questions:

    question = test["question"]
    expected = test["expected_category"]

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        k=3
    )

    retrieved_categories = []

    for idx in indices[0]:

        category = df.iloc[idx]["source"]

        retrieved_categories.append(category)


    # Check whether expected category appears
    # in the top 3 retrieved documents

    if expected in retrieved_categories:

        correct += 1
        status = "CORRECT"

    else:

        status = "INCORRECT"


    print("\n------------------------------------------")

    print("Question:")
    print(question)

    print("\nExpected category:")
    print(expected)

    print("\nRetrieved categories:")

    for category in retrieved_categories:

        print("-", category)

    print("\nResult:", status)


# =========================================================
# 7. CALCULATE RAG TOP-3 CATEGORY HIT RATE
# =========================================================

accuracy = (
    correct / len(test_questions)
) * 100


print("\n==========================================")
print("FINAL RAG RESULT")
print("==========================================")

print(
    "Top-3 Retrieval Category Hit Rate:",
    round(accuracy, 2),
    "%"
)

print(
    "Correct:",
    correct,
    "/",
    len(test_questions)
)