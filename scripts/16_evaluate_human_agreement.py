import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "evaluation" / "human_evidence_review.csv"
OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "human_agreement_results.txt"


def normalize(value):
    value = str(value).strip().lower()

    if value in {"yes", "y", "1", "true"}:
        return "Yes"

    if value in {"no", "n", "0", "false"}:
        return "No"

    return ""


def main():
    print("=" * 70)
    print("APPLE SUPPORT AGENT — HUMAN / AUTOMATED AGREEMENT")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}\n"
            "Make sure the human evidence review CSV exists."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nReview examples: {len(df)}")

    # Normalize human labels.
    df["human_evidence_relevant"] = (
        df["human_evidence_relevant"]
        .apply(normalize)
    )

    df["human_reply_supported_by_evidence"] = (
        df["human_reply_supported_by_evidence"]
        .apply(normalize)
    )

    # ---------------------------------------------------------
    # Check completeness.
    # ---------------------------------------------------------

    missing_relevance = (
        df["human_evidence_relevant"] == ""
    ).sum()

    missing_support = (
        df["human_reply_supported_by_evidence"] == ""
    ).sum()

    print("\nMissing human relevance labels:", missing_relevance)
    print("Missing human support labels:", missing_support)

    if missing_relevance > 0 or missing_support > 0:
        print(
            "\nHuman review is incomplete."
            "\nFill the human-review columns before calculating agreement."
        )
        return

    # ---------------------------------------------------------
    # Automated evidence judgment.
    # ---------------------------------------------------------

    automated = df["automated_evidence_judgment"].apply(normalize)

    human_relevance = df[
        "human_evidence_relevant"
    ]

    # Agreement between automated evidence judgment and
    # human evidence relevance.
    relevance_accuracy = (
        automated == human_relevance
    ).mean()

    relevance_kappa = cohen_kappa_score(
        human_relevance,
        automated
    )

    # ---------------------------------------------------------
    # Human reply-support judgment.
    #
    # This is a human quality measure rather than agreement
    # with the automated evidence classifier.
    # ---------------------------------------------------------

    reply_support_rate = (
        df["human_reply_supported_by_evidence"] == "Yes"
    ).mean()

    # ---------------------------------------------------------
    # Agreement by evidence level.
    # ---------------------------------------------------------

    agreement_by_level = (
        df.assign(
            agreement=(
                automated == human_relevance
            )
        )
        .groupby("evidence_level")["agreement"]
        .mean()
    )

    # ---------------------------------------------------------
    # Disagreements.
    # ---------------------------------------------------------

    disagreements = df[
        automated != human_relevance
    ].copy()

    # ---------------------------------------------------------
    # Print results.
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("AGREEMENT RESULTS")
    print("=" * 70)

    print(
        f"\nAutomated vs human evidence relevance agreement: "
        f"{relevance_accuracy:.4f}"
    )

    print(
        f"Cohen's kappa: "
        f"{relevance_kappa:.4f}"
    )

    print(
        f"\nHuman evidence relevance rate: "
        f"{(human_relevance == 'Yes').mean():.4f}"
    )

    print(
        f"Human reply-support rate: "
        f"{reply_support_rate:.4f}"
    )

    print("\nHuman evidence relevance:")
    print(human_relevance.value_counts())

    print("\nAgreement by evidence level:")
    print(agreement_by_level)

    print(
        f"\nAutomated/human disagreements: "
        f"{len(disagreements)}"
    )

    # ---------------------------------------------------------
    # Show disagreements.
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DISAGREEMENT EXAMPLES")
    print("=" * 70)

    for _, row in disagreements.head(10).iterrows():
        print(f"\n{row['example_id']}")
        print(f"Customer: {row['customer_text']}")
        print(
            f"Automated: "
            f"{automated.loc[row.name]}"
        )
        print(
            f"Human: "
            f"{row['human_evidence_relevant']}"
        )
        print(
            f"Evidence level: "
            f"{row['evidence_level']}"
        )
        print(
            f"Similarity: "
            f"{row['top_similarity']:.4f}"
        )
        print(
            f"Retrieved customer: "
            f"{row['top_evidence_customer']}"
        )
        print(
            f"Human note: "
            f"{row['human_notes']}"
        )

    # ---------------------------------------------------------
    # Save compact results.
    # ---------------------------------------------------------

    report = []

    report.append(
        "APPLE SUPPORT AGENT — HUMAN / AUTOMATED AGREEMENT"
    )
    report.append("=" * 60)
    report.append(
        f"Review examples: {len(df)}"
    )
    report.append(
        f"Automated vs human relevance agreement: "
        f"{relevance_accuracy:.4f}"
    )
    report.append(
        f"Cohen's kappa: {relevance_kappa:.4f}"
    )
    report.append(
        f"Human evidence relevance rate: "
        f"{(human_relevance == 'Yes').mean():.4f}"
    )
    report.append(
        f"Human reply-support rate: "
        f"{reply_support_rate:.4f}"
    )
    report.append(
        f"Disagreements: {len(disagreements)}"
    )

    report.append("\nAgreement by evidence level:")
    for level, value in agreement_by_level.items():
        report.append(
            f"{level}: {value:.4f}"
        )

    OUTPUT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print(
        f"\nSaved agreement report to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()