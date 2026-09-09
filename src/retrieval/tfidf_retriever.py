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


class TfidfRetriever:

    def __init__(self):

        print("Loading retrieval index...")

        self.pairs = pd.read_csv(
            PAIRS_FILE
        )

        with open(
            VECTORIZER_FILE,
            "rb"
        ) as f:
            self.vectorizer = pickle.load(f)

        self.matrix = load_npz(
            MATRIX_FILE
        )

        print(
            f"Loaded {len(self.pairs):,} historical cases."
        )

    def search(
        self,
        query,
        top_k=5,
        min_similarity=0.0
    ):
        """
        Retrieve historically similar customer issues.

        Returns:
            List of dictionaries containing:
            - customer_text
            - agent_response
            - similarity
            - tweet_id
        """

        query = str(query).strip()

        if not query:
            return []

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        # Get candidate indices efficiently.
        candidate_count = min(
            top_k * 10,
            len(similarities)
        )

        candidate_indices = similarities.argsort()[
            -candidate_count:
        ][::-1]

        results = []

        for idx in candidate_indices:

            score = float(
                similarities[idx]
            )

            if score < min_similarity:
                continue

            row = self.pairs.iloc[idx]

            results.append(
                {
                    "tweet_id": row.get(
                        "tweet_id",
                        ""
                    ),
                    "customer_text": row[
                        "customer_text"
                    ],
                    "agent_response": row[
                        "agent_response"
                    ],
                    "similarity": round(
                        score,
                        4
                    ),
                }
            )

            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":

    retriever = TfidfRetriever()

    test_queries = [
        "My iPhone battery is draining very quickly",
        "WiFi keeps disconnecting after the update",
        "I cannot sign into my Apple ID",
        "My iPhone screen is cracked",
    ]

    for query in test_queries:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = retriever.search(
            query,
            top_k=3
        )

        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\n{i}. Similarity: "
                f"{result['similarity']}"
            )

            print(
                f"Customer: "
                f"{result['customer_text']}"
            )

            print(
                f"Historical response: "
                f"{result['agent_response']}"
            )