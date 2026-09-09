import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "evaluation" / "evidence_review.csv"
OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "human_evidence_review.csv"


def main():
    print("=" * 70)
    print("APPLE SUPPORT AGENT — HUMAN EVIDENCE REVIEW")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}\n"
            "Run scripts/14_create_evidence_review.py first."
        )

    evidence = pd.read_csv(INPUT_FILE)

    # Select only high-priority cases.
    review = evidence[
        evidence["review_priority"] == "high"
    ].copy()

    # Keep the review focused and easy to label.
    review = review[
        [
            "example_id",
            "customer_text",
            "gold_intent",
            "model_intent",
            "model_confidence",
            "evidence_level",
            "top_similarity",
            "top_evidence_customer",
            "top_evidence_response",
            "model_reply",
            "historical_grounding",
        ]
    ].copy()

    # Human-review columns.
    review["human_evidence_relevant"] = ""
    review["human_reply_supported_by_evidence"] = ""
    review["human_notes"] = ""

    # Automated judge decision converted to Yes/No.
    review["automated_evidence_judgment"] = review[
        "historical_grounding"
    ].map({
        1: "Yes",
        0: "No",
    })

    # Add a simple instruction row as metadata through a separate file.
    review.to_csv(OUTPUT_FILE, index=False)

    print(f"\nHigh-priority cases selected: {len(review)}")

    print("\nHuman review instructions:")
    print("- human_evidence_relevant: Yes if the retrieved historical")
    print("  customer issue is genuinely similar/relevant.")
    print("- human_reply_supported_by_evidence: Yes if the retrieved")
    print("  historical response reasonably supports the generated reply.")
    print("- human_notes: briefly explain difficult or borderline cases.")
    print("- Leave no human-review fields blank after reviewing.")

    print("\nAutomated evidence judgment:")
    print(review["automated_evidence_judgment"].value_counts())

    print("\n" + "=" * 70)
    print("FIRST 10 CASES")
    print("=" * 70)

    for _, row in review.head(10).iterrows():
        print(f"\n{row['example_id']}")
        print("-" * 50)
        print(f"Customer: {row['customer_text']}")
        print(f"Gold intent: {row['gold_intent']}")
        print(f"Model intent: {row['model_intent']}")
        print(f"Model confidence: {row['model_confidence']:.4f}")
        print(f"Evidence level: {row['evidence_level']}")
        print(f"Similarity: {row['top_similarity']:.4f}")
        print(f"Retrieved customer: {row['top_evidence_customer']}")
        print(f"Retrieved response: {row['top_evidence_response']}")
        print(f"Model reply: {row['model_reply']}")
        print(
            f"Automated judgment: "
            f"{row['automated_evidence_judgment']}"
        )

    print("\n" + "=" * 70)
    print(f"Saved human review file:")
    print(OUTPUT_FILE)
    print("=" * 70)


if __name__ == "__main__":
    main()