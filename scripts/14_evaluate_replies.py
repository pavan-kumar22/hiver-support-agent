import re
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "evaluation/agent_results.csv"
)

OUTPUT_FILE = Path(
    "evaluation/reply_evaluation.csv"
)


# ------------------------------------------------------------
# Intent-specific keywords
# ------------------------------------------------------------

INTENT_KEYWORDS = {

    "software_update": [
        "software",
        "update",
        "ios",
        "version",
        "device",
    ],

    "battery_charging": [
        "battery",
        "charging",
        "drain",
        "iphone",
        "ios",
    ],

    "messaging": [
        "messaging",
        "message",
        "messages",
        "sending",
        "receiving",
        "device",
    ],

    "keyboard_input": [
        "keyboard",
        "typing",
        "type",
        "ios",
        "device",
    ],

    "apps_services": [
        "app",
        "service",
        "device",
        "error",
    ],

    "account_icloud": [
        "apple id",
        "sign",
        "login",
        "error",
        "account",
    ],

    "connectivity_calls": [
        "connectivity",
        "network",
        "wifi",
        "device",
    ],

    "hardware_device": [
        "device",
        "problem",
        "issue",
    ],

    "purchases_billing": [
        "purchase",
        "billing",
        "subscription",
        "details",
    ],

    "repair_support": [
        "support",
        "service",
        "repair",
        "details",
    ],

    "other": [
        "details",
        "issue",
        "experiencing",
    ],
}


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def normalize(text):

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ------------------------------------------------------------
# Reply quality checks
# ------------------------------------------------------------

def check_non_empty(reply):

    return bool(
        normalize(reply)
    )


def check_professional(reply):

    text = normalize(reply)

    bad_patterns = [
        "fuck",
        "shit",
        "damn",
        "wtf",
        "stupid",
        "idiot",
    ]

    return not any(
        word in text
        for word in bad_patterns
    )


def check_has_helpful_request(reply):

    text = normalize(reply)

    request_patterns = [
        "please let us know",
        "please provide",
        "let us know",
        "provide more details",
        "which device",
        "what you're experiencing",
        "error message",
        "ios version",
        "country",
    ]

    return any(
        pattern in text
        for pattern in request_patterns
    )


def check_intent_grounding(reply, intent):

    text = normalize(reply)

    keywords = INTENT_KEYWORDS.get(
        intent,
        []
    )

    if not keywords:
        return False

    matches = sum(
        1
        for keyword in keywords
        if keyword in text
    )

    return matches >= 1


def check_escalation_appropriate(
    reply,
    action
):

    if action != "escalate":
        return True

    text = normalize(reply)

    escalation_patterns = [
        "take a closer look",
        "assist you",
        "support specialist",
        "provide more details",
        "look into",
    ]

    return any(
        pattern in text
        for pattern in escalation_patterns
    )


def check_historical_grounding(
    reply,
    evidence
):

    reply_text = normalize(reply)
    evidence_text = normalize(evidence)

    if not reply_text or not evidence_text:
        return False

    # Check whether the generated reply shares
    # meaningful support vocabulary with the
    # retrieved historical response.
    reply_words = set(
        reply_text.split()
    )

    evidence_words = set(
        evidence_text.split()
    )

    # Remove common conversational words.
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "with",
        "we",
        "you",
        "your",
        "us",
        "is",
        "are",
        "this",
        "that",
        "it",
        "on",
        "in",
        "be",
        "can",
        "will",
        "please",
    }

    reply_words -= stop_words
    evidence_words -= stop_words

    if not reply_words:
        return False

    overlap = (
        len(
            reply_words & evidence_words
        )
        /
        len(reply_words)
    )

    return overlap >= 0.10


def check_unsupported_claims(reply):

    text = normalize(reply)

    # The local responder should avoid promising
    # refunds, repairs, replacements, guarantees,
    # or definite resolutions.
    risky_patterns = [
        "we will refund",
        "we will replace",
        "guaranteed",
        "this will fix",
        "will definitely fix",
        "your refund",
        "replacement approved",
    ]

    return not any(
        pattern in text
        for pattern in risky_patterns
    )


# ------------------------------------------------------------
# Evaluate one reply
# ------------------------------------------------------------

def evaluate_row(row):

    reply = row["model_reply"]

    intent = row["model_intent"]

    action = row["model_action"]

    evidence = row[
        "top_evidence_response"
    ]

    checks = {

        "non_empty":
            check_non_empty(reply),

        "professional":
            check_professional(reply),

        "helpful_request":
            check_has_helpful_request(reply),

        "intent_grounded":
            check_intent_grounding(
                reply,
                intent
            ),

        "escalation_appropriate":
            check_escalation_appropriate(
                reply,
                action
            ),

        "historical_grounding":
            check_historical_grounding(
                reply,
                evidence
            ),

        "no_unsupported_claims":
            check_unsupported_claims(
                reply
            ),
    }

    score = sum(
        checks.values()
    )

    max_score = len(
        checks
    )

    quality_score = (
        score / max_score
    )

    return checks, quality_score


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "APPLE SUPPORT AGENT — "
        "OFFLINE REPLY & EVIDENCE EVALUATION"
    )
    print("=" * 70)

    results = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nExamples: {len(results):,}"
    )

    evaluation_rows = []

    for _, row in results.iterrows():

        checks, quality_score = (
            evaluate_row(row)
        )

        output = {
            "example_id":
                row["example_id"],

            "customer_text":
                row["customer_text"],

            "gold_intent":
                row["gold_intent"],

            "model_intent":
                row["model_intent"],

            "model_action":
                row["model_action"],

            "intent_confidence":
                row["model_confidence"],

            "top_similarity":
                row["top_similarity"],

            "evidence_level":
                row["evidence_level"],

            "model_reply":
                row["model_reply"],

            "historical_response":
                row["top_evidence_response"],

            "quality_score":
                quality_score,
        }

        output.update(
            checks
        )

        evaluation_rows.append(
            output
        )

    evaluation = pd.DataFrame(
        evaluation_rows
    )

    evaluation.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    check_columns = [
        "non_empty",
        "professional",
        "helpful_request",
        "intent_grounded",
        "escalation_appropriate",
        "historical_grounding",
        "no_unsupported_claims",
    ]

    print("\n" + "=" * 70)
    print("REPLY QUALITY RESULTS")
    print("=" * 70)

    for column in check_columns:

        rate = (
            evaluation[column]
            .mean()
        )

        print(
            f"{column}: "
            f"{rate:.4f}"
        )

    print(
        f"\nAverage offline quality score: "
        f"{evaluation['quality_score'].mean():.4f}"
    )

    print(
        f"Median offline quality score: "
        f"{evaluation['quality_score'].median():.4f}"
    )

    # --------------------------------------------------------
    # Action-specific results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("QUALITY BY ACTION")
    print("=" * 70)

    action_summary = (
        evaluation
        .groupby("model_action")
        ["quality_score"]
        .agg(
            ["count", "mean"]
        )
    )

    print(
        action_summary
    )

    # --------------------------------------------------------
    # Evidence-specific results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("QUALITY BY EVIDENCE LEVEL")
    print("=" * 70)

    evidence_summary = (
        evaluation
        .groupby("evidence_level")
        ["quality_score"]
        .agg(
            ["count", "mean"]
        )
    )

    print(
        evidence_summary
    )

    # --------------------------------------------------------
    # Strongest examples
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STRONGEST REPLIES")
    print("=" * 70)

    strongest = (
        evaluation
        .sort_values(
            "quality_score",
            ascending=False
        )
        .head(5)
    )

    for _, row in strongest.iterrows():

        print(
            f"\n{row['example_id']} "
            f"score={row['quality_score']:.2f}"
        )

        print(
            f"Customer: "
            f"{row['customer_text']}"
        )

        print(
            f"Reply: "
            f"{row['model_reply']}"
        )

    # --------------------------------------------------------
    # Weakest examples
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("WEAKEST REPLIES")
    print("=" * 70)

    weakest = (
        evaluation
        .sort_values(
            "quality_score",
            ascending=True
        )
        .head(10)
    )

    for _, row in weakest.iterrows():

        print(
            f"\n{row['example_id']} "
            f"score={row['quality_score']:.2f}"
        )

        print(
            f"Customer: "
            f"{row['customer_text']}"
        )

        print(
            f"Reply: "
            f"{row['model_reply']}"
        )

        print(
            "Checks failed:"
        )

        failed = [
            column
            for column in check_columns
            if not row[column]
        ]

        print(
            ", ".join(failed)
        )

    print(
        f"\nSaved results to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()