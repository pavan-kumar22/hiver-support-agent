import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


INPUT_FILE = PROJECT_ROOT / "evaluation" / "agent_results.csv"
OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "evidence_review.csv"


def assess_evidence(row):
    """
    Create a deterministic evidence-review rubric.

    This is NOT an LLM judge.
    It is used to identify examples that should be
    reviewed by a human for evidence quality.
    """

    evidence_level = str(row["evidence_level"])
    similarity = float(row["top_similarity"])
    evidence_customer = str(row["top_evidence_customer"])
    evidence_response = str(row["top_evidence_response"])
    customer_text = str(row["customer_text"])
    model_intent = str(row["model_intent"])

    customer_lower = customer_text.lower()
    evidence_customer_lower = evidence_customer.lower()
    evidence_response_lower = evidence_response.lower()

    # ---------------------------------------------------------
    # 1. Evidence exists
    # ---------------------------------------------------------
    evidence_exists = (
        evidence_level in {"moderate", "strong"}
        and bool(evidence_customer.strip())
        and bool(evidence_response.strip())
    )

    # ---------------------------------------------------------
    # 2. Lexical overlap between incoming issue and
    #    retrieved historical customer issue.
    # ---------------------------------------------------------
    stopwords = {
        "the", "a", "an", "and", "or", "is", "it", "to", "of",
        "in", "on", "for", "my", "i", "me", "this", "that",
        "with", "why", "what", "how", "can", "you", "please",
        "apple", "support", "https", "http"
    }

    customer_words = {
        word.strip(".,!?():;\"'").lower()
        for word in customer_lower.split()
        if len(word.strip(".,!?():;\"'")) >= 4
        and word.strip(".,!?():;\"'").lower() not in stopwords
    }

    evidence_words = {
        word.strip(".,!?():;\"'").lower()
        for word in evidence_customer_lower.split()
        if len(word.strip(".,!?():;\"'")) >= 4
        and word.strip(".,!?():;\"'").lower() not in stopwords
    }

    overlap = customer_words.intersection(evidence_words)

    if customer_words:
        overlap_ratio = len(overlap) / len(customer_words)
    else:
        overlap_ratio = 0.0

    # ---------------------------------------------------------
    # 3. Intent-specific keyword grounding.
    # ---------------------------------------------------------
    intent_keywords = {
        "software_update": {
            "update", "ios", "ios11", "ios12", "upgrade",
            "downgrade", "install", "version", "software"
        },
        "battery_charging": {
            "battery", "charge", "charging", "drain",
            "power", "battery-life"
        },
        "messaging": {
            "message", "messages", "imessage", "sms",
            "text", "texts"
        },
        "keyboard_input": {
            "keyboard", "typing", "type", "autocorrect",
            "words", "letters"
        },
        "apps_services": {
            "app", "apps", "itunes", "music", "safari",
            "photos", "mail", "facetime", "calendar",
            "notes", "store"
        },
        "account_icloud": {
            "icloud", "appleid", "apple", "account",
            "password", "signin", "login", "storage"
        },
        "connectivity_calls": {
            "wifi", "bluetooth", "network", "signal",
            "call", "calls", "cellular", "sim"
        },
        "hardware_device": {
            "iphone", "ipad", "screen", "speaker",
            "camera", "button", "device", "phone",
            "broken", "cracked"
        },
        "purchases_billing": {
            "purchase", "payment", "refund", "billing",
            "subscription", "charge", "charged", "card"
        },
        "repair_support": {
            "repair", "service", "replacement", "store",
            "warranty", "support"
        },
        "other": set(),
    }

    keywords = intent_keywords.get(model_intent, set())

    keyword_matches = [
        keyword
        for keyword in keywords
        if keyword in customer_lower
    ]

    intent_grounded = (
        model_intent == "other"
        or len(keyword_matches) > 0
    )

    # ---------------------------------------------------------
    # 4. Historical evidence relevance.
    #
    # We require both reasonable similarity and lexical
    # overlap with the retrieved customer issue.
    # ---------------------------------------------------------
    historical_grounding = (
        evidence_exists
        and similarity >= 0.30
        and overlap_ratio >= 0.05
    )

    # Strong evidence candidate
    strong_evidence_candidate = (
        evidence_level == "strong"
        and similarity >= 0.45
        and overlap_ratio >= 0.10
    )

    # ---------------------------------------------------------
    # 5. Human-review recommendation
    # ---------------------------------------------------------
    if not evidence_exists:
        review_priority = "high"
    elif strong_evidence_candidate:
        review_priority = "low"
    elif historical_grounding:
        review_priority = "medium"
    else:
        review_priority = "high"

    return pd.Series({
        "evidence_exists": int(evidence_exists),
        "overlap_ratio": round(overlap_ratio, 4),
        "intent_keyword_matches": ", ".join(keyword_matches),
        "intent_grounded": int(intent_grounded),
        "historical_grounding": int(historical_grounding),
        "strong_evidence_candidate": int(strong_evidence_candidate),
        "review_priority": review_priority,
    })


def main():
    print("=" * 70)
    print("APPLE SUPPORT AGENT — EVIDENCE REVIEW SET")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}\n"
            "Run scripts/11_evaluate_agent.py first."
        )

    results = pd.read_csv(INPUT_FILE)

    print(f"\nAgent results: {len(results):,}")

    review = results.apply(assess_evidence, axis=1)

    output = pd.concat(
        [
            results[
                [
                    "example_id",
                    "tweet_id",
                    "customer_text",
                    "gold_intent",
                    "model_intent",
                    "model_confidence",
                    "evidence_level",
                    "top_similarity",
                    "top_evidence_customer",
                    "top_evidence_response",
                    "model_action",
                    "model_reply",
                ]
            ],
            review,
        ],
        axis=1,
    )

    # Save complete review file.
    output.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("EVIDENCE REVIEW RESULTS")
    print("=" * 70)

    print(
        f"\nEvidence exists: "
        f"{output['evidence_exists'].mean():.4f}"
    )

    print(
        f"Intent grounded: "
        f"{output['intent_grounded'].mean():.4f}"
    )

    print(
        f"Historical grounding: "
        f"{output['historical_grounding'].mean():.4f}"
    )

    print(
        f"Strong evidence candidates: "
        f"{output['strong_evidence_candidate'].mean():.4f}"
    )

    print("\nReview priority:")
    print(output["review_priority"].value_counts())

    print("\nEvidence level:")
    print(output["evidence_level"].value_counts())

    print("\nAverage overlap ratio:")
    print(f"{output['overlap_ratio'].mean():.4f}")

    print("\n" + "=" * 70)
    print("HIGH-PRIORITY HUMAN REVIEW EXAMPLES")
    print("=" * 70)

    high_priority = output[
        output["review_priority"] == "high"
    ].head(15)

    for _, row in high_priority.iterrows():
        print(f"\n{row['example_id']}")
        print(f"Customer: {row['customer_text']}")
        print(f"Model intent: {row['model_intent']}")
        print(f"Evidence level: {row['evidence_level']}")
        print(f"Similarity: {row['top_similarity']:.4f}")
        print(f"Evidence customer: {row['top_evidence_customer']}")
        print(f"Evidence response: {row['top_evidence_response']}")

    print("\n" + "=" * 70)
    print("STRONG EVIDENCE CANDIDATES")
    print("=" * 70)

    strong = output[
        output["strong_evidence_candidate"] == 1
    ].head(10)

    for _, row in strong.iterrows():
        print(f"\n{row['example_id']}")
        print(f"Customer: {row['customer_text']}")
        print(f"Evidence customer: {row['top_evidence_customer']}")
        print(f"Evidence response: {row['top_evidence_response']}")
        print(f"Similarity: {row['top_similarity']:.4f}")

    print(f"\nSaved review set to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()