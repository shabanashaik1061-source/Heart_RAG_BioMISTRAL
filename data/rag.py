import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss


# =========================================================
# 1. LOAD PREPARED TRAINING DATA
# =========================================================

print("Loading RAG dataset...")

df = pd.read_csv("train_prepared.csv")

print("RAG dataset loaded!")
print("Total documents:", len(df))


# =========================================================
# 2. CREATE DOCUMENTS
# =========================================================

documents = []

for _, row in df.iterrows():

    text = f"""
Instruction: {row['instruction']}
Input: {row['input']}
Output: {row['output']}
Source: {row['source']}
"""

    documents.append(text)


print("Documents created:", len(documents))


# =========================================================
# 3. LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# =========================================================
# 4. CREATE DOCUMENT EMBEDDINGS
# =========================================================

print("\nCreating embeddings...")

embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embeddings created:", embeddings.shape)


# =========================================================
# 5. CREATE FAISS INDEX
# =========================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("\nFAISS index created successfully!")
print("Number of vectors:", index.ntotal)


# =========================================================
# 6. RAG RETRIEVAL FUNCTION
# =========================================================

def retrieve_documents(
    query,
    predicted_categories=None,
    k=3
):
    """
    Retrieve the most relevant documents.

    Uses:
    1. Semantic similarity
    2. ML predicted categories
    """

    # Create query embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Search more documents first
    search_k = min(30, len(documents))

    scores, indices = index.search(
        query_embedding,
        search_k
    )

    candidates = []

    # =====================================================
    # CATEGORY-AWARE RETRIEVAL
    # =====================================================

    for score, idx in zip(scores[0], indices[0]):

        source = str(df.iloc[idx]["source"])

        category_match = False

        if predicted_categories:

            for category in predicted_categories:

                if source == category:
                    category_match = True
                    break

        # Category matching boost
        if category_match:
            final_score = float(score) + 0.15
        else:
            final_score = float(score)

        candidates.append(
            {
                "index": int(idx),
                "score": final_score,
                "source": source
            }
        )

    # =====================================================
    # SORT BY FINAL SCORE
    # =====================================================

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # =====================================================
    # SELECT TOP DOCUMENTS
    # =====================================================

    selected = candidates[:k]

    retrieved_documents = []

    for item in selected:

        idx = item["index"]

        retrieved_documents.append(
            {
                "instruction": str(df.iloc[idx]["instruction"]),
                "input": str(df.iloc[idx]["input"]),
                "output": str(df.iloc[idx]["output"]),
                "source": item["source"],
                "score": round(item["score"], 4)
            }
        )

    return retrieved_documents


# =========================================================
# 7. TEST RAG
# =========================================================

if __name__ == "__main__":

    query = "What are the risk factors for heart disease?"

    test_categories = [
        "synthetic_cardiovascular"
    ]

    results = retrieve_documents(
        query,
        predicted_categories=test_categories,
        k=3
    )

    print("\n======================================")
    print("RAG RETRIEVAL TEST")
    print("======================================")

    print("Query:", query)

    print("\nPredicted category:")
    print(test_categories[0])

    print("\nRetrieved documents:")

    for i, document in enumerate(results, start=1):

        print(f"\n----- Document {i} -----")
        print(document)