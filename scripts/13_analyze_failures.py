import pandas as pd

from pathlib import Path


RESULTS_FILE = Path(
    "evaluation/agent_results.csv"
)


def main():

    print("=" * 70)
    print("AGENT FAILURE ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(
        RESULTS_FILE
    )

    # ------------------------------------------------------------
    # Intent failures
    # ------------------------------------------------------------

    intent_failures = df[
        df["gold_intent"] != df["model_intent"]
    ].copy()

    print(
        f"\nIntent failures: "
        f"{len(intent_failures)} / {len(df)}"
    )

    print(
        f"Intent accuracy: "
        f"{1 - len(intent_failures) / len(df):.4f}"
    )

    # ------------------------------------------------------------
    # Action failures
    # ------------------------------------------------------------

    action_failures = df[
        df["gold_action"] != df["model_action"]
    ].copy()

    print(
        f"\nAction failures: "
        f"{len(action_failures)} / {len(df)}"
    )

    print(
        f"Action accuracy: "
        f"{1 - len(action_failures) / len(df):.4f}"
    )

    # ------------------------------------------------------------
    # Confusion pairs
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP INTENT CONFUSIONS")
    print("=" * 70)

    confusion = (
        intent_failures
        .groupby(
            [
                "gold_intent",
                "model_intent"
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(confusion.head(15))

    # ------------------------------------------------------------
    # Confidence on correct vs incorrect predictions
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONFIDENCE ANALYSIS")
    print("=" * 70)

    correct = df[
        df["gold_intent"] == df["model_intent"]
    ]

    incorrect = df[
        df["gold_intent"] != df["model_intent"]
    ]

    print(
        f"\nCorrect prediction confidence: "
        f"{correct['model_confidence'].mean():.4f}"
    )

    print(
        f"Incorrect prediction confidence: "
        f"{incorrect['model_confidence'].mean():.4f}"
    )

    # ------------------------------------------------------------
    # Most dangerous false auto-handles
    # ------------------------------------------------------------

    false_auto = df[
        (df["gold_action"] == "escalate")
        &
        (df["model_action"] == "auto_handle")
    ].copy()

    print("\n" + "=" * 70)
    print("FALSE AUTO-HANDLES")
    print("=" * 70)

    print(
        f"\nFalse auto-handles: "
        f"{len(false_auto)}"
    )

    columns = [
        "example_id",
        "customer_text",
        "gold_intent",
        "model_intent",
        "model_confidence",
        "top_similarity",
        "evidence_level",
        "model_action",
        "model_reason",
    ]

    if len(false_auto) > 0:

        print(
            false_auto[
                columns
            ].to_string(
                index=False
            )
        )

    # ------------------------------------------------------------
    # Missed auto-handles
    # ------------------------------------------------------------

    missed_auto = df[
        (df["gold_action"] == "auto_handle")
        &
        (df["model_action"] == "escalate")
    ].copy()

    print("\n" + "=" * 70)
    print("MISSED AUTO-HANDLES")
    print("=" * 70)

    print(
        f"\nMissed auto-handles: "
        f"{len(missed_auto)}"
    )

    if len(missed_auto) > 0:

        print(
            missed_auto[
                columns
            ].to_string(
                index=False
            )
        )

    # ------------------------------------------------------------
    # Low confidence examples
    # ------------------------------------------------------------

    low_confidence = df[
        df["model_confidence"] < 0.60
    ].copy()

    print("\n" + "=" * 70)
    print("LOW-CONFIDENCE EXAMPLES")
    print("=" * 70)

    print(
        f"\nExamples below 0.60 confidence: "
        f"{len(low_confidence)}"
    )

    if len(low_confidence) > 0:

        print(
            low_confidence[
                [
                    "example_id",
                    "customer_text",
                    "gold_intent",
                    "model_intent",
                    "model_confidence",
                    "top_similarity",
                    "model_action",
                ]
            ].to_string(
                index=False
            )
        )

    # ------------------------------------------------------------
    # Weak evidence examples
    # ------------------------------------------------------------

    weak_evidence = df[
        df["evidence_level"] == "weak"
    ].copy()

    print("\n" + "=" * 70)
    print("WEAK-EVIDENCE EXAMPLES")
    print("=" * 70)

    print(
        f"\nWeak evidence examples: "
        f"{len(weak_evidence)}"
    )

    if len(weak_evidence) > 0:

        print(
            weak_evidence[
                [
                    "example_id",
                    "customer_text",
                    "gold_intent",
                    "model_intent",
                    "model_confidence",
                    "top_similarity",
                    "model_action",
                ]
            ].to_string(
                index=False
            )
        )

    # ------------------------------------------------------------
    # Save failure datasets
    # ------------------------------------------------------------

    intent_failures.to_csv(
        "evaluation/intent_failures.csv",
        index=False
    )

    action_failures.to_csv(
        "evaluation/action_failures.csv",
        index=False
    )

    false_auto.to_csv(
        "evaluation/false_auto_handles.csv",
        index=False
    )

    missed_auto.to_csv(
        "evaluation/missed_auto_handles.csv",
        index=False
    )

    print("\nFailure datasets saved:")
    print("- evaluation/intent_failures.csv")
    print("- evaluation/action_failures.csv")
    print("- evaluation/false_auto_handles.csv")
    print("- evaluation/missed_auto_handles.csv")


if __name__ == "__main__":
    main()