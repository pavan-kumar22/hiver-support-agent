from pathlib import Path
import pandas as pd


INPUT_FILE = Path("data/raw/twcs.csv")
OUTPUT_DIR = Path("data/processed/applesupport")

BRAND = "AppleSupport"


def main():
    print("=" * 60)
    print("HIVER SUPPORT AGENT - APPLESUPPORT DATA PREPARATION")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nReading: {INPUT_FILE}")
    print("This may take a little while...")

    df = pd.read_csv(INPUT_FILE)

    print(f"\nDataset loaded: {len(df):,} rows")

    # ---------------------------------------------------------
    # Basic cleanup
    # ---------------------------------------------------------

    df["text"] = df["text"].fillna("").astype(str)

    df["tweet_id"] = pd.to_numeric(
        df["tweet_id"],
        errors="coerce"
    )

    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # AppleSupport responses
    # ---------------------------------------------------------

    apple_responses = df[
        (df["author_id"] == BRAND) &
        (df["in_response_to_tweet_id"].notna())
    ].copy()

    print(
        f"\nAppleSupport response tweets: "
        f"{len(apple_responses):,}"
    )

    # ---------------------------------------------------------
    # Create lookup table containing original tweets
    # ---------------------------------------------------------

    parent_tweets = df[
        [
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
        ]
    ].copy()

    parent_tweets = parent_tweets.rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "author_id": "customer_author_id",
            "inbound": "customer_inbound",
            "created_at": "customer_created_at",
            "text": "customer_text",
        }
    )

    # ---------------------------------------------------------
    # Join AppleSupport responses to the tweets they answered
    # ---------------------------------------------------------

    qa = apple_responses.merge(
        parent_tweets,
        left_on="in_response_to_tweet_id",
        right_on="customer_tweet_id",
        how="inner",
    )

    print(
        f"Matched response-parent pairs: "
        f"{len(qa):,}"
    )

    # ---------------------------------------------------------
    # Keep customer -> AppleSupport interactions
    # ---------------------------------------------------------

    qa = qa[
        qa["customer_inbound"] == True
    ].copy()

    qa = qa[
        (qa["customer_text"].str.strip() != "") &
        (qa["text"].str.strip() != "")
    ].copy()

    # ---------------------------------------------------------
    # Build clean final dataset
    # ---------------------------------------------------------

    qa = qa.rename(
        columns={
            "tweet_id": "agent_tweet_id",
            "created_at": "agent_created_at",
            "text": "agent_response",
        }
    )

    qa = qa[
        [
            "customer_tweet_id",
            "customer_author_id",
            "customer_created_at",
            "customer_text",
            "agent_tweet_id",
            "agent_created_at",
            "agent_response",
        ]
    ]

    # ---------------------------------------------------------
    # Convert timestamps
    # ---------------------------------------------------------

    qa["customer_created_at"] = pd.to_datetime(
        qa["customer_created_at"],
        errors="coerce",
        utc=True,
    )

    qa["agent_created_at"] = pd.to_datetime(
        qa["agent_created_at"],
        errors="coerce",
        utc=True,
    )

    # ---------------------------------------------------------
    # Remove invalid rows
    # ---------------------------------------------------------

    qa = qa.dropna(
        subset=[
            "customer_tweet_id",
            "agent_tweet_id",
            "customer_created_at",
            "agent_created_at",
        ]
    )

    # ---------------------------------------------------------
    # Sort chronologically
    # ---------------------------------------------------------

    qa = qa.sort_values(
        by="customer_created_at"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_file = OUTPUT_DIR / "qa_pairs.csv"

    qa.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"Total tweets:                 "
        f"{len(df):,}"
    )

    print(
        f"AppleSupport responses:       "
        f"{len(apple_responses):,}"
    )

    print(
        f"Matched pairs:                "
        f"{len(qa):,}"
    )

    print(
        f"Unique customer tweets:       "
        f"{qa['customer_tweet_id'].nunique():,}"
    )

    print(
        f"Unique agent responses:       "
        f"{qa['agent_tweet_id'].nunique():,}"
    )

    print(
        f"Unique customers:             "
        f"{qa['customer_author_id'].nunique():,}"
    )

    if len(qa) > 0:
        print(
            f"Date range:                   "
            f"{qa['customer_created_at'].min()} "
            f"to "
            f"{qa['customer_created_at'].max()}"
        )

    # ---------------------------------------------------------
    # Display sample conversations
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("SAMPLE CUSTOMER → APPLESUPPORT INTERACTIONS")
    print("=" * 60)

    for i, (_, row) in enumerate(
        qa.head(5).iterrows(),
        start=1
    ):
        print(f"\n--- Example {i} ---")

        print("\nCUSTOMER:")
        print(row["customer_text"])

        print("\nAPPLESUPPORT:")
        print(row["agent_response"])

    print("\n" + "=" * 60)
    print("FILE CREATED")
    print("=" * 60)

    print(output_file)

    print("=" * 60)


if __name__ == "__main__":
    main()