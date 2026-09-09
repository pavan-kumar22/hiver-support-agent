from pathlib import Path
import json
import re
from collections import Counter

import pandas as pd


INPUT_FILE = Path(
    "data/processed/applesupport/conversations.jsonl"
)

OUTPUT_DIR = Path(
    "data/processed/applesupport"
)

OUTPUT_FILE = OUTPUT_DIR / "customer_messages.csv"


def clean_text(text):
    """Basic text normalization for analysis."""

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove Twitter-style user mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def main():

    print("=" * 60)
    print("HIVER SUPPORT AGENT - INTENT DATA ANALYSIS")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load conversations
    # ---------------------------------------------------------

    conversations = []

    print(f"\nReading: {INPUT_FILE}")

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                conversations.append(
                    json.loads(line)
                )

    print(
        f"Conversations loaded: "
        f"{len(conversations):,}"
    )

    # ---------------------------------------------------------
    # Extract customer messages
    # ---------------------------------------------------------

    records = []

    for conversation in conversations:

        conversation_id = conversation[
            "conversation_id"
        ]

        customer_id = conversation[
            "customer_id"
        ]

        messages = conversation[
            "messages"
        ]

        for index, message in enumerate(
            messages
        ):

            if message["speaker"] != "customer":
                continue

            original_text = message[
                "text"
            ]

            cleaned = clean_text(
                original_text
            )

            if not cleaned:
                continue

            # Find next agent response
            next_agent_response = None

            for next_message in messages[
                index + 1:
            ]:

                if next_message[
                    "speaker"
                ] == "agent":

                    next_agent_response = (
                        next_message["text"]
                    )

                    break

            records.append(
                {
                    "conversation_id":
                        conversation_id,

                    "customer_id":
                        customer_id,

                    "tweet_id":
                        message["tweet_id"],

                    "timestamp":
                        message["timestamp"],

                    "customer_text":
                        original_text,

                    "cleaned_text":
                        cleaned,

                    "next_agent_response":
                        next_agent_response,

                    "message_position":
                        index
                }
            )

    customer_df = pd.DataFrame(
        records
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    customer_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CUSTOMER MESSAGE DATASET")
    print("=" * 60)

    print(
        f"Customer messages: "
        f"{len(customer_df):,}"
    )

    print(
        f"Unique conversations: "
        f"{customer_df['conversation_id'].nunique():,}"
    )

    print(
        f"Unique customers: "
        f"{customer_df['customer_id'].nunique():,}"
    )

    print(
        f"Messages with agent response: "
        f"{customer_df['next_agent_response'].notna().sum():,}"
    )

    # ---------------------------------------------------------
    # Message length analysis
    # ---------------------------------------------------------

    customer_df["word_count"] = (
        customer_df["cleaned_text"]
        .str.split()
        .str.len()
    )

    print(
        f"\nAverage words/message: "
        f"{customer_df['word_count'].mean():.2f}"
    )

    print(
        f"Median words/message: "
        f"{customer_df['word_count'].median():.0f}"
    )

    print(
        f"Maximum words/message: "
        f"{customer_df['word_count'].max():.0f}"
    )

    # ---------------------------------------------------------
    # Very short messages
    # ---------------------------------------------------------

    short_messages = customer_df[
        customer_df["word_count"] <= 3
    ]

    print(
        f"\nMessages <= 3 words: "
        f"{len(short_messages):,}"
    )

    # ---------------------------------------------------------
    # Common words
    # ---------------------------------------------------------

    words = []

    for text in customer_df[
        "cleaned_text"
    ]:

        tokens = re.findall(
            r"\b[a-zA-Z]{3,}\b",
            text.lower()
        )

        words.extend(tokens)

    stop_words = {
        "the",
        "and",
        "for",
        "you",
        "this",
        "that",
        "with",
        "have",
        "has",
        "are",
        "was",
        "were",
        "but",
        "not",
        "can",
        "its",
        "from",
        "your",
        "what",
        "how",
        "why",
        "when",
        "where",
        "does",
        "did",
        "just",
        "they",
        "will",
        "would",
        "could",
        "about",
        "apple",
        "support",
        "please",
        "help",
        "need",
    }

    filtered_words = [
        word
        for word in words
        if word not in stop_words
    ]

    word_counts = Counter(
        filtered_words
    )

    print("\n" + "=" * 60)
    print("MOST COMMON TERMS")
    print("=" * 60)

    for word, count in word_counts.most_common(50):

        print(
            f"{word:<25} {count:>8,}"
        )

    # ---------------------------------------------------------
    # Save word counts
    # ---------------------------------------------------------

    word_counts_df = pd.DataFrame(
        word_counts.most_common(),
        columns=[
            "word",
            "count"
        ]
    )

    word_counts_df.to_csv(
        OUTPUT_DIR / "word_counts.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Sample messages
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("RANDOM CUSTOMER MESSAGE SAMPLES")
    print("=" * 60)

    sample_size = min(
        50,
        len(customer_df)
    )

    sample = customer_df.sample(
        sample_size,
        random_state=42
    )

    for i, (_, row) in enumerate(
        sample.iterrows(),
        start=1
    ):

        print(
            f"\n--- Example {i} ---"
        )

        print(
            row["cleaned_text"]
        )

    print("\n" + "=" * 60)
    print("FILES CREATED")
    print("=" * 60)

    print(OUTPUT_FILE)

    print(
        OUTPUT_DIR / "word_counts.csv"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()