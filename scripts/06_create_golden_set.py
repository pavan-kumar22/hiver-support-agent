from pathlib import Path
import pandas as pd


INPUT_FILE = Path(
    "data/processed/applesupport/customer_messages.csv"
)

OUTPUT_DIR = Path("evaluation")
OUTPUT_FILE = OUTPUT_DIR / "golden_set.csv"

TOTAL_EXAMPLES = 200


def main():
    df = pd.read_csv(INPUT_FILE)

    print("=" * 60)
    print("GOLDEN EVALUATION SET CREATION")
    print("=" * 60)

    print(f"\nTotal available messages: {len(df):,}")

    # Use the actual column names from customer_messages.csv
    if "word_count" not in df.columns:
        df["word_count"] = (
            df["customer_text"]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
        )

    # Make sure cleaned text is available
    if "cleaned_text" not in df.columns:
        df["cleaned_text"] = (
            df["customer_text"]
            .fillna("")
            .astype(str)
        )
    # ---------------------------------------------------------
    # Remove extremely short messages from the main sample.
    # We will separately include difficult short messages.
    # ---------------------------------------------------------

    normal = df[
        df["word_count"] >= 8
    ].copy()

    short = df[
        df["word_count"] <= 3
    ].copy()

    medium = df[
        (df["word_count"] >= 4) &
        (df["word_count"] <= 7)
    ].copy()

    # ---------------------------------------------------------
    # Sample:
    #
    # 140 normal examples
    # 30 medium-length examples
    # 20 very short examples
    # 10 difficult/context-heavy examples
    #
    # Total = 200
    # ---------------------------------------------------------

    normal_sample = normal.sample(
        min(140, len(normal)),
        random_state=42
    )

    medium_sample = medium.sample(
        min(30, len(medium)),
        random_state=43
    )

    short_sample = short.sample(
        min(20, len(short)),
        random_state=44
    )

    # Difficult/context-heavy examples:
    # prioritize messages containing uncertainty,
    # questions, multiple symptoms, or conversational language.

    difficult_candidates = df[
        df["cleaned_text"].str.contains(
            r"\?|how|why|what|still|after|but|also|"
            r"tried|already|however|can't|cannot",
            case=False,
            regex=True,
            na=False
        )
    ]

    difficult_sample = difficult_candidates.sample(
        min(10, len(difficult_candidates)),
        random_state=45
    )

    golden = pd.concat(
        [
            normal_sample,
            medium_sample,
            short_sample,
            difficult_sample
        ],
        ignore_index=True
    )

    # ---------------------------------------------------------
    # Remove duplicates
    # ---------------------------------------------------------

    golden = golden.drop_duplicates(
        subset=["tweet_id"]
    )

    # If duplicate removal reduced the count,
    # fill from remaining data.
    if len(golden) < TOTAL_EXAMPLES:

        remaining = df[
            ~df["tweet_id"].isin(
                golden["tweet_id"]
            )
        ]

        extra = remaining.sample(
            TOTAL_EXAMPLES - len(golden),
            random_state=46
        )

        golden = pd.concat(
            [golden, extra],
            ignore_index=True
        )

    golden = golden.head(
        TOTAL_EXAMPLES
    )

    # ---------------------------------------------------------
    # Create annotation columns
    # ---------------------------------------------------------

    golden["gold_intent"] = ""
    golden["gold_action"] = ""
    golden["gold_reason"] = ""
    golden["annotator_notes"] = ""

    golden["model_intent"] = ""
    golden["model_reply"] = ""
    golden["model_action"] = ""
    golden["model_reason"] = ""

    # ---------------------------------------------------------
    # Evaluation metadata
    # ---------------------------------------------------------

    golden["example_id"] = [
        f"gold_{i:03d}"
        for i in range(1, len(golden) + 1)
    ]

    golden["split"] = "golden"

    # ---------------------------------------------------------
    # Select columns
    # ---------------------------------------------------------

    columns = [
        "example_id",
        "conversation_id",
        "customer_id",
        "tweet_id",
        "timestamp",
        "customer_text",
        "cleaned_text",
        "next_agent_response",
        "word_count",
        "gold_intent",
        "gold_action",
        "gold_reason",
        "annotator_notes",
        "model_intent",
        "model_reply",
        "model_action",
        "model_reason",
        "split"
    ]

    golden = golden[columns]

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    golden.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("GOLDEN SET CREATED")
    print("=" * 60)

    print(
        f"Examples: {len(golden)}"
    )

    print(
        f"Normal examples: "
        f"{len(normal_sample)}"
    )

    print(
        f"Medium examples: "
        f"{len(medium_sample)}"
    )

    print(
        f"Short examples: "
        f"{len(short_sample)}"
    )

    print(
        f"Difficult examples: "
        f"{len(difficult_sample)}"
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()