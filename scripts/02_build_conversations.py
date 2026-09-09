from pathlib import Path
import pandas as pd
import json


INPUT_FILE = Path(
    "data/processed/applesupport/qa_pairs.csv"
)

OUTPUT_DIR = Path(
    "data/processed/applesupport"
)

OUTPUT_FILE = OUTPUT_DIR / "conversations.jsonl"


def main():
    print("=" * 60)
    print("HIVER SUPPORT AGENT - CONVERSATION RECONSTRUCTION")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nReading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded pairs: {len(df):,}")

    # ---------------------------------------------------------
    # Convert IDs to strings
    # ---------------------------------------------------------

    df["customer_tweet_id"] = (
        df["customer_tweet_id"]
        .astype(str)
    )

    df["agent_tweet_id"] = (
        df["agent_tweet_id"]
        .astype(str)
    )

    # ---------------------------------------------------------
    # Parse dates
    # ---------------------------------------------------------

    df["customer_created_at"] = pd.to_datetime(
        df["customer_created_at"],
        errors="coerce",
        utc=True
    )

    df["agent_created_at"] = pd.to_datetime(
        df["agent_created_at"],
        errors="coerce",
        utc=True
    )

    # ---------------------------------------------------------
    # Sort chronologically
    # ---------------------------------------------------------

    df = df.sort_values(
        by=[
            "customer_author_id",
            "customer_created_at"
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Build conversations
    #
    # For this first version we group interactions by customer.
    # Later we will use tweet relationships to improve thread
    # reconstruction.
    # ---------------------------------------------------------

    conversations = []

    grouped = df.groupby(
        "customer_author_id",
        sort=False
    )

    for customer_id, group in grouped:

        group = group.sort_values(
            "customer_created_at"
        )

        messages = []

        for _, row in group.iterrows():

            customer_text = str(
                row["customer_text"]
            ).strip()

            agent_text = str(
                row["agent_response"]
            ).strip()

            if customer_text:
                messages.append(
                    {
                        "speaker": "customer",
                        "tweet_id": row[
                            "customer_tweet_id"
                        ],
                        "timestamp": str(
                            row["customer_created_at"]
                        ),
                        "text": customer_text
                    }
                )

            if agent_text:
                messages.append(
                    {
                        "speaker": "agent",
                        "tweet_id": row[
                            "agent_tweet_id"
                        ],
                        "timestamp": str(
                            row["agent_created_at"]
                        ),
                        "text": agent_text
                    }
                )

        if len(messages) >= 2:

            conversations.append(
                {
                    "conversation_id":
                        f"customer_{customer_id}",

                    "customer_id":
                        str(customer_id),

                    "message_count":
                        len(messages),

                    "messages":
                        messages
                }
            )

    # ---------------------------------------------------------
    # Sort messages inside every conversation
    # ---------------------------------------------------------

    for conversation in conversations:

        conversation["messages"].sort(
            key=lambda x: x["timestamp"]
        )

    # ---------------------------------------------------------
    # Save JSONL
    # ---------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for conversation in conversations:

            f.write(
                json.dumps(
                    conversation,
                    ensure_ascii=False
                ) + "\n"
            )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    message_counts = [
        c["message_count"]
        for c in conversations
    ]

    if message_counts:

        total_messages = sum(
            message_counts
        )

        avg_messages = (
            total_messages /
            len(message_counts)
        )

        max_messages = max(
            message_counts
        )

        multi_turn = sum(
            count >= 4
            for count in message_counts
        )

    else:

        total_messages = 0
        avg_messages = 0
        max_messages = 0
        multi_turn = 0

    print("\n" + "=" * 60)
    print("CONVERSATION RESULTS")
    print("=" * 60)

    print(
        f"Customer-agent pairs: "
        f"{len(df):,}"
    )

    print(
        f"Unique customers: "
        f"{df['customer_author_id'].nunique():,}"
    )

    print(
        f"Conversations created: "
        f"{len(conversations):,}"
    )

    print(
        f"Total messages: "
        f"{total_messages:,}"
    )

    print(
        f"Average messages/conversation: "
        f"{avg_messages:.2f}"
    )

    print(
        f"Maximum messages/conversation: "
        f"{max_messages}"
    )

    print(
        f"Multi-turn conversations (4+ msgs): "
        f"{multi_turn:,}"
    )

    # ---------------------------------------------------------
    # Print examples
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("SAMPLE CONVERSATIONS")
    print("=" * 60)

    for i, conversation in enumerate(
        conversations[:5],
        start=1
    ):

        print(
            f"\n--- Conversation {i} ---"
        )

        for message in conversation[
            "messages"
        ]:

            speaker = message[
                "speaker"
            ].upper()

            print(
                f"\n{speaker}:"
            )

            print(
                message["text"]
            )

    print("\n" + "=" * 60)
    print("FILE CREATED")
    print("=" * 60)

    print(OUTPUT_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()