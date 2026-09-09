import pickle
from pathlib import Path

import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


RETRIEVAL_DIR = Path(
    "data/processed/applesupport/retrieval"
)

PAIRS_FILE = RETRIEVAL_DIR / "retrieval_pairs.csv"
VECTORIZER_FILE = RETRIEVAL_DIR / "tfidf_vectorizer.pkl"
MATRIX_FILE = RETRIEVAL_DIR / "tfidf_matrix.npz"


class HistoricalRetriever:

    def __init__(self):

        self.pairs = pd.read_csv(
            PAIRS_FILE
        )

        with open(
            VECTORIZER_FILE,
            "rb"
        ) as file:
            self.vectorizer = pickle.load(file)

        self.matrix = load_npz(
            MATRIX_FILE
        )

    def search(
        self,
        query,
        top_k=5,
        min_similarity=0.20
    ):

        query = str(query).strip()

        if not query:
            return []

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        ).ravel()

        top_indices = similarities.argsort()[
            ::-1
        ]

        results = []

        for index in top_indices:

            similarity = float(
                similarities[index]
            )

            if similarity < min_similarity:
                break

            row = self.pairs.iloc[index]

            results.append(
                {
                    "rank": len(results) + 1,
                    "tweet_id": str(
                        row.get("tweet_id", "")
                    ),
                    "customer_text": str(
                        row["customer_text"]
                    ),
                    "historical_response": str(
                        row["agent_response"]
                    ),
                    "similarity": round(
                        similarity,
                        4
                    ),
                }
            )

            if len(results) >= top_k:
                break

        return results

    def evidence_summary(
        self,
        query,
        top_k=5
    ):

        results = self.search(
            query,
            top_k=top_k
        )

        if not results:
            return {
                "evidence_level": "none",
                "top_similarity": 0.0,
                "average_similarity": 0.0,
                "results": [],
            }

        similarities = [
            item["similarity"]
            for item in results
        ]

        top_similarity = max(
            similarities
        )

        average_similarity = sum(
            similarities
        ) / len(similarities)

        if top_similarity >= 0.45:
            evidence_level = "strong"

        elif top_similarity >= 0.30:
            evidence_level = "moderate"

        else:
            evidence_level = "weak"

        return {
            "evidence_level": evidence_level,
            "top_similarity": round(
                top_similarity,
                4
            ),
            "average_similarity": round(
                average_similarity,
                4
            ),
            "results": results,
        }


if __name__ == "__main__":

    retriever = HistoricalRetriever()

    queries = [
        "My iPhone battery is draining very quickly",
        "WiFi keeps disconnecting after the update",
        "I cannot sign into my Apple ID",
        "My iPhone screen is cracked",
        "My issue is completely unusual and I need help",
    ]

    for query in queries:

        print("\n" + "=" * 70)
        print("CUSTOMER:")
        print(query)
        print("=" * 70)

        evidence = retriever.evidence_summary(
            query,
            top_k=3
        )

        print(
            f"Evidence level: "
            f"{evidence['evidence_level']}"
        )

        print(
            f"Top similarity: "
            f"{evidence['top_similarity']}"
        )

        print(
            f"Average similarity: "
            f"{evidence['average_similarity']}"
        )

        for result in evidence["results"]:

            print(
                f"\n[{result['rank']}] "
                f"Similarity: "
                f"{result['similarity']}"
            )

            print(
                f"Historical customer: "
                f"{result['customer_text']}"
            )

            print(
                f"Historical resolution: "
                f"{result['historical_response']}"
            )