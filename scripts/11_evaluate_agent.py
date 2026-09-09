import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.agent.support_agent import AppleSupportAgent


GOLDEN_FILE = Path(
    "evaluation/golden_set_v1.csv"
)

OUTPUT_FILE = Path(
    "evaluation/agent_results.csv"
)


def main():

    print("=" * 70)
    print("APPLE SUPPORT AGENT — GOLDEN SET EVALUATION")
    print("=" * 70)

    golden = pd.read_csv(
        GOLDEN_FILE
    )

    print(
        f"\nGolden examples: {len(golden):,}"
    )

    print("\nInitializing agent...")

    start_time = time.time()

    agent = AppleSupportAgent()

    initialization_time = (
        time.time() - start_time
    )

    print(
        f"Agent initialization time: "
        f"{initialization_time:.2f}s"
    )

    results = []

    evaluation_start = time.time()

    for index, row in golden.iterrows():

        customer_text = str(
            row["customer_text"]
        )

        result = agent.analyze(
            customer_text
        )

        results.append(
            {
                "example_id": row["example_id"],
                "tweet_id": row["tweet_id"],
                "customer_text": customer_text,
                "next_agent_response": row[
                    "next_agent_response"
                ],

                "gold_intent": row[
                    "gold_intent"
                ],

                "gold_action": row[
                    "gold_action"
                ],

                "gold_reason": row[
                    "gold_reason"
                ],

                "model_intent": result[
                    "intent"
                ],

                "model_confidence": result[
                    "intent_confidence"
                ],

                "evidence_level": result[
                    "evidence_level"
                ],

                "top_similarity": result[
                    "top_similarity"
                ],

                "model_action": result[
                    "action"
                ],

                "model_reason": result[
                    "reason"
                ],

                "model_reply": result[
                    "reply"
                ],

                "evidence_count": len(
                    result["evidence"]
                ),

                "top_evidence_customer": (
                    result["evidence"][0][
                        "customer_text"
                    ]
                    if result["evidence"]
                    else ""
                ),

                "top_evidence_response": (
                    result["evidence"][0][
                        "historical_response"
                    ]
                    if result["evidence"]
                    else ""
                ),
            }
        )

        completed = index + 1

        if (
            completed % 10 == 0
            or completed == len(golden)
        ):

            elapsed = (
                time.time() -
                evaluation_start
            )

            print(
                f"Processed "
                f"{completed}/{len(golden)} "
                f"examples "
                f"({elapsed:.1f}s)"
            )

    results = pd.DataFrame(
        results
    )

    total_time = (
        time.time() -
        evaluation_start
    )

    # ------------------------------------------------------------
    # Intent metrics
    # ------------------------------------------------------------

    intent_accuracy = accuracy_score(
        results["gold_intent"],
        results["model_intent"]
    )

    intent_precision, intent_recall, intent_f1, _ = (
        precision_recall_fscore_support(
            results["gold_intent"],
            results["model_intent"],
            average="macro",
            zero_division=0
        )
    )

    # ------------------------------------------------------------
    # Action metrics
    # ------------------------------------------------------------

    action_accuracy = accuracy_score(
        results["gold_action"],
        results["model_action"]
    )

    action_precision, action_recall, action_f1, _ = (
        precision_recall_fscore_support(
            results["gold_action"],
            results["model_action"],
            average="binary",
            pos_label="auto_handle",
            zero_division=0
        )
    )

    # ------------------------------------------------------------
    # Evidence metrics
    # ------------------------------------------------------------

    strong_evidence_rate = (
        results["evidence_level"] == "strong"
    ).mean()

    moderate_or_strong_rate = (
        results["evidence_level"].isin(
            ["moderate", "strong"]
        )
    ).mean()

    average_similarity = (
        results["top_similarity"].mean()
    )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL AGENT RESULTS")
    print("=" * 70)

    print(
        f"\nEvaluation time: "
        f"{total_time:.2f}s"
    )

    print(
        f"\nIntent accuracy: "
        f"{intent_accuracy:.4f}"
    )

    print(
        f"Macro intent precision: "
        f"{intent_precision:.4f}"
    )

    print(
        f"Macro intent recall: "
        f"{intent_recall:.4f}"
    )

    print(
        f"Macro intent F1: "
        f"{intent_f1:.4f}"
    )

    print(
        f"\nAction accuracy: "
        f"{action_accuracy:.4f}"
    )

    print(
        f"Auto-handle precision: "
        f"{action_precision:.4f}"
    )

    print(
        f"Auto-handle recall: "
        f"{action_recall:.4f}"
    )

    print(
        f"Auto-handle F1: "
        f"{action_f1:.4f}"
    )

    print(
        f"\nStrong evidence rate: "
        f"{strong_evidence_rate:.4f}"
    )

    print(
        f"Moderate/strong evidence rate: "
        f"{moderate_or_strong_rate:.4f}"
    )

    print(
        f"Average top similarity: "
        f"{average_similarity:.4f}"
    )

    print("\nModel action distribution:")
    print(
        results["model_action"]
        .value_counts()
    )

    print("\nGold action distribution:")
    print(
        results["gold_action"]
        .value_counts()
    )

    print("\nModel intent distribution:")
    print(
        results["model_intent"]
        .value_counts()
    )

    print("\nIntent classification report:")
    print(
        classification_report(
            results["gold_intent"],
            results["model_intent"],
            zero_division=0
        )
    )

    print("\nIntent confusion matrix:")

    labels = sorted(
        results["gold_intent"]
        .unique()
    )

    matrix = confusion_matrix(
        results["gold_intent"],
        results["model_intent"],
        labels=labels
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    print(matrix_df)

    print(
        f"\nSaved results to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()