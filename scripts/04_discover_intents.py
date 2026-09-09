from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans


INPUT_FILE = Path(
    "data/processed/applesupport/customer_messages.csv"
)

OUTPUT_DIR = Path(
    "data/processed/applesupport"
)

OUTPUT_FILE = OUTPUT_DIR / "intent_clusters.csv"


# Number of candidate clusters.
# We will NOT treat these as the final intents.
N_CLUSTERS = 12


def main():

    print("=" * 60)
    print("HIVER SUPPORT AGENT - INTENT DISCOVERY")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nLoaded messages: {len(df):,}")

    texts = (
        df["cleaned_text"]
        .fillna("")
        .astype(str)
    )

    # ---------------------------------------------------------
    # TF-IDF representation
    # ---------------------------------------------------------

    print("\nBuilding TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.95,
        max_features=50000,
    )

    X = vectorizer.fit_transform(texts)

    print(
        f"TF-IDF matrix: "
        f"{X.shape[0]:,} messages x "
        f"{X.shape[1]:,} features"
    )

    # ---------------------------------------------------------
    # Clustering
    # ---------------------------------------------------------

    print(
        f"\nCreating {N_CLUSTERS} candidate clusters..."
    )

    model = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        batch_size=2048,
        n_init=10,
    )

    df["cluster"] = model.fit_predict(X)

    # ---------------------------------------------------------
    # Get important terms for each cluster
    # ---------------------------------------------------------

    terms = vectorizer.get_feature_names_out()

    cluster_centers = model.cluster_centers_

    print("\n" + "=" * 60)
    print("CANDIDATE INTENT CLUSTERS")
    print("=" * 60)

    for cluster_id in range(N_CLUSTERS):

        cluster_mask = (
            df["cluster"] == cluster_id
        )

        cluster_size = cluster_mask.sum()

        center = cluster_centers[
            cluster_id
        ]

        top_indices = center.argsort()[
            ::-1
        ][:15]

        top_terms = [
            terms[i]
            for i in top_indices
        ]

        print(
            f"\nCLUSTER {cluster_id}"
        )

        print(
            f"Messages: {cluster_size:,}"
        )

        print(
            "Terms: "
            + ", ".join(top_terms)
        )

        print("\nExamples:")

        examples = df[
            cluster_mask
        ].sample(
            min(10, cluster_size),
            random_state=42
        )

        for _, row in examples.iterrows():

            print(
                f"  - {row['cleaned_text']}"
            )

    # ---------------------------------------------------------
    # Save clustered dataset
    # ---------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("FILE CREATED")
    print("=" * 60)

    print(OUTPUT_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()