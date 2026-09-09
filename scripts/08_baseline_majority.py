import pandas as pd
from pathlib import Path

DEVELOPMENT_FILE = Path(
    "data/processed/applesupport/development_messages.csv"
)
GOLDEN_FILE = Path("evaluation/golden_set_v1.csv")
OUTPUT_FILE = Path("evaluation/baseline_majority_results.csv")

GENERIC_REPLY = (
    "Thanks for contacting Apple Support. "
    "We'd like to look into this with you. "
    "Please send us a DM with more details so we can help."
)

ESCALATION_REASON = (
    "Baseline policy escalates every customer message."
)

# IMPORTANT:
# This is a pre-declared majority-class baseline.
# It is NOT calculated from the golden test set.
#
# software_update was selected from the exploratory intent analysis
# before evaluating the frozen golden set.
MAJORITY_INTENT = "software_update"


def main():
    print("=" * 70)
    print("BASELINE 1 — FIXED MAJORITY INTENT + ALWAYS ESCALATE")
    print("=" * 70)

    development = pd.read_csv(DEVELOPMENT_FILE)
    golden = pd.read_csv(GOLDEN_FILE)

    print(f"\nDevelopment rows: {len(development):,}")
    print(f"Golden rows:      {len(golden):,}")
    print(f"Fixed majority intent: {MAJORITY_INTENT}")

    # Make predictions without using any gold labels.
    results = golden.copy()

    results["model_intent"] = MAJORITY_INTENT
    results["model_reply"] = GENERIC_REPLY
    results["model_action"] = "escalate"
    results["model_reason"] = ESCALATION_REASON

    # Evaluation happens only after predictions are generated.
    intent_accuracy = (
        results["model_intent"] == results["gold_intent"]
    ).mean()

    action_accuracy = (
        results["model_action"] == results["gold_action"]
    ).mean()

    results.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("BASELINE 1 RESULTS")
    print("=" * 70)

    print(f"\nIntent accuracy: {intent_accuracy:.4f}")
    print(f"Action accuracy: {action_accuracy:.4f}")

    print("\nPredicted action distribution:")
    print(results["model_action"].value_counts())

    print("\nGold action distribution:")
    print(results["gold_action"].value_counts())

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()