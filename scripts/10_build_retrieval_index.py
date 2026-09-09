import pandas as pd
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


INPUT_FILE = Path(
    "data/processed/applesupport/qa_pairs.csv"
)

OUTPUT_DIR = Path(
    "data/processed/applesupport/retrieval"
)

OUTPUT_PAIRS = OUTPUT_DIR / "retrieval_pairs.csv"
OUTPUT_VECTORIZER = OUTPUT_DIR / "tfidf_vectorizer.npz"
OUTPUT_MATRIX = OUTPUT_DIR / "tfidf_matrix.npz"


MAX_PAIRS = 30000


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())

    return text.strip()


def main():

    print("=" * 70)
    print("BUILD HISTORICAL RESOLUTION RETRIEVAL DATASET")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nOriginal QA pairs: {len(df):,}")

    # ------------------------------------------------------------
    # Clean fields
    # ------------------------------------------------------------

    df["customer_text"] = (
        df["customer_text"]
        .apply(clean_text)
    )

    df["agent_response"] = (
        df["agent_response"]
        .apply(clean_text)
    )

    # Remove unusable rows
    df = df[
        (df["customer_text"].str.len() >= 5)
        & (df["agent_response"].str.len() >= 5)
    ].copy()

    # Remove duplicate customer/response pairs
    df = df.drop_duplicates(
        subset=["customer_text", "agent_response"]
    )

    print(
        f"Usable unique pairs: {len(df):,}"
    )

    # ------------------------------------------------------------
    # Limit size for fast reproducibility
    # ------------------------------------------------------------

    if len(df) > MAX_PAIRS:

        # Deterministic sampling
        df = df.sample(
            n=MAX_PAIRS,
            random_state=42
        )

    df = df.reset_index(drop=True)

    print(
        f"Retrieval index size: {len(df):,}"
    )

    # ------------------------------------------------------------
    # Build TF-IDF representation
    # ------------------------------------------------------------

    print("\nBuilding TF-IDF retrieval representation...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=60000,
    )

    matrix = vectorizer.fit_transform(
        df["customer_text"]
    )

    print(
        f"TF-IDF matrix shape: {matrix.shape}"
    )

    # ------------------------------------------------------------
    # Save retrieval data
    # ------------------------------------------------------------

    df.to_csv(
        OUTPUT_PAIRS,
        index=False
    )

    # Save vectorizer vocabulary/configuration
    import pickle

    with open(
        OUTPUT_DIR / "tfidf_vectorizer.pkl",
        "wb"
    ) as f:
        pickle.dump(vectorizer, f)

    # Save sparse matrix
    from scipy.sparse import save_npz

    save_npz(
        OUTPUT_MATRIX,
        matrix
    )

    print("\nRetrieval artifacts saved:")
    print(f"- {OUTPUT_PAIRS}")
    print(f"- {OUTPUT_DIR / 'tfidf_vectorizer.pkl'}")
    print(f"- {OUTPUT_MATRIX}")

    print("\n" + "=" * 70)
    print("RETRIEVAL DATASET READY")
    print("=" * 70)


if __name__ == "__main__":
    main()