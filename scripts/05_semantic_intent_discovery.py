from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans


INPUT_FILE = Path(
    "data/processed/applesupport/customer_messages.csv"
)

OUTPUT_DIR = Path(
    "data/processed/applesupport"
)

OUTPUT_FILE = OUTPUT_DIR / "semantic_intent_clusters.csv"

# Use a manageable but representative sample for discovery.
SAMPLE_SIZE = 15000

# More clusters than our eventual taxonomy.
# We will merge them manually after inspection.
N_CLUSTERS = 20

MODEL_NAME = "all-MiniLM-L6-v2"


def main():

    print("=" * 70)
    print("HIVER SUPPORT AGENT - SEMANTIC INTENT DISCOVERY")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nTotal messages: {len(df):,}")

    # ---------------------------------------------------------
    # Sample messages
    # ---------------------------------------------------------

    sample_size = min(
        SAMPLE_SIZE,
        len(df)
    )

    sample = df.sample(
        sample_size,
        random_state=42
    ).copy()

    sample = sample.reset_index(
        drop=True
    )

    print(
        f"Discovery sample: {len(sample):,}"
    )

    # ---------------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------------

    print(
        f"\nLoading model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # ---------------------------------------------------------
    # Generate embeddings
    # ---------------------------------------------------------

    print(
        "\nGenerating semantic embeddings..."
    )

    embeddings = model.encode(
        sample["cleaned_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # ---------------------------------------------------------
    # Cluster
    # ---------------------------------------------------------

    print(
        f"\nCreating {N_CLUSTERS} semantic clusters..."
    )

    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        batch_size=512,
        n_init=10
    )

    sample["cluster"] = kmeans.fit_predict(
        embeddings
    )

    # ---------------------------------------------------------
    # Print clusters
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SEMANTIC CLUSTERS")
    print("=" * 70)

    for cluster_id in range(N_CLUSTERS):

        cluster = sample[
            sample["cluster"] == cluster_id
        ]

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"CLUSTER {cluster_id}"
        )

        print(
            f"Messages: {len(cluster):,}"
        )

        print(
            f"{'-' * 70}"
        )

        examples = cluster.sample(
            min(20, len(cluster)),
            random_state=42
        )

        for i, (_, row) in enumerate(
            examples.iterrows(),
            start=1
        ):

            print(
                f"{i:02d}. {row['cleaned_text']}"
            )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    sample.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("FILE CREATED")
    print("=" * 70)

    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()