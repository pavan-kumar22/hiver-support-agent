import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


DEVELOPMENT_FILE = Path(
    "data/processed/applesupport/development_messages.csv"
)

GOLDEN_FILE = Path(
    "evaluation/golden_set_v1.csv"
)

OUTPUT_FILE = Path(
    "evaluation/baseline_tfidf_results.csv"
)


def build_weak_labels(df):
    """
    Create transparent weak labels from customer-message text.

    These labels are used ONLY to train the baseline.
    The frozen golden set is never used for training.
    """

    rules = {
        "software_update": [
            "update",
            "ios",
            "ios 11",
            "ios 10",
            "upgrade",
            "downgrade",
            "installing ios",
            "software update",
            "updated my iphone",
            "after update",
        ],
        "battery_charging": [
            "battery",
            "charging",
            "charge",
            "charger",
            "battery life",
            "battery drain",
            "power",
            "overheating",
        ],
        "messaging": [
            "imessage",
            "message",
            "messages",
            "sms",
            "text message",
            "texting",
        ],
        "keyboard_input": [
            "keyboard",
            "autocorrect",
            "typing",
            "type",
            "key",
            "predictive text",
        ],
        "account_icloud": [
            "icloud",
            "apple id",
            "appleid",
            "password",
            "account",
            "verification",
            "verify my account",
            "locked account",
        ],
        "connectivity_calls": [
            "wifi",
            "wi-fi",
            "bluetooth",
            "network",
            "signal",
            "cellular",
            "carrier",
            "call",
            "calls",
            "sim",
            "activation",
        ],
        "hardware_device": [
            "screen",
            "display",
            "camera",
            "speaker",
            "button",
            "home button",
            "touch id",
            "iphone broken",
            "cracked",
            "physical damage",
            "headphones",
            "port",
        ],
        "purchases_billing": [
            "purchase",
            "purchased",
            "payment",
            "charged",
            "charge",
            "refund",
            "subscription",
            "billing",
            "invoice",
            "app store purchase",
        ],
        "repair_support": [
            "repair",
            "replacement",
            "service center",
            "apple store",
            "support",
            "genius",
            "warranty",
        ],
        "apps_services": [
            "app store",
            "itunes",
            "apple music",
            "safari",
            "photos",
            "mail app",
            "facetime",
            "calendar",
            "notes",
        ],
    }

    def assign_label(text):
        text = str(text).lower()

        scores = {}

        for intent, keywords in rules.items():
            score = 0

            for keyword in keywords:
                if keyword in text:
                    score += 1

            scores[intent] = score

        best_intent = max(scores, key=scores.get)

        if scores[best_intent] == 0:
            return "other"

        return best_intent

    df = df.copy()
    df["weak_intent"] = df["customer_text"].apply(assign_label)

    return df


def main():

    print("=" * 70)
    print("BASELINE 2 — TF-IDF + LOGISTIC REGRESSION")
    print("=" * 70)

    development = pd.read_csv(DEVELOPMENT_FILE)
    golden = pd.read_csv(GOLDEN_FILE)

    print(f"\nDevelopment rows: {len(development):,}")
    print(f"Golden rows:      {len(golden):,}")

    # ------------------------------------------------------------
    # 1. Create weak training labels from development data only
    # ------------------------------------------------------------

    development = build_weak_labels(development)

    print("\nWeak-label distribution:")
    print(development["weak_intent"].value_counts())

    # ------------------------------------------------------------
    # 2. Remove extremely noisy / unsupported examples
    # ------------------------------------------------------------

    training = development[
        development["weak_intent"] != "other"
    ].copy()

    print(
        f"\nTraining rows after removing 'other': "
        f"{len(training):,}"
    )

    # ------------------------------------------------------------
    # 3. Build TF-IDF + Logistic Regression pipeline
    # ------------------------------------------------------------

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=50000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining TF-IDF + Logistic Regression...")

    model.fit(
        training["customer_text"].astype(str),
        training["weak_intent"],
    )

    print("Training complete.")

    # ------------------------------------------------------------
    # 4. Predict ONLY the frozen golden set
    # ------------------------------------------------------------

    predictions = model.predict(
        golden["customer_text"].astype(str)
    )

    probabilities = model.predict_proba(
        golden["customer_text"].astype(str)
    )

    confidence = probabilities.max(axis=1)

    results = golden.copy()

    results["model_intent"] = predictions
    results["model_confidence"] = confidence

    # ------------------------------------------------------------
    # 5. Simple action policy
    # ------------------------------------------------------------
    #
    # High confidence = auto-handle
    # Low confidence = escalate
    #
    # We deliberately do NOT tune this threshold on the golden set.
    # ------------------------------------------------------------

    AUTO_HANDLE_THRESHOLD = 0.70

    results["model_action"] = np.where(
        results["model_confidence"] >= AUTO_HANDLE_THRESHOLD,
        "auto_handle",
        "escalate",
    )

    results["model_reason"] = np.where(
        results["model_action"] == "auto_handle",
        "TF-IDF classifier confidence is above the fixed threshold.",
        "TF-IDF classifier confidence is below the fixed threshold.",
    )

    results["model_reply"] = (
        "Thanks for contacting Apple Support. "
        "We identified your issue as: "
        + results["model_intent"].astype(str)
        + ". "
        "Please send us more details so we can help."
    )

    # ------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------

    intent_accuracy = accuracy_score(
        results["gold_intent"],
        results["model_intent"],
    )

    action_accuracy = accuracy_score(
        results["gold_action"],
        results["model_action"],
    )

    print("\n" + "=" * 70)
    print("BASELINE 2 RESULTS")
    print("=" * 70)

    print(f"\nIntent accuracy: {intent_accuracy:.4f}")
    print(f"Action accuracy: {action_accuracy:.4f}")

    print("\nPredicted action distribution:")
    print(results["model_action"].value_counts())

    print("\nGold action distribution:")
    print(results["gold_action"].value_counts())

    print("\nIntent classification report:")
    print(
        classification_report(
            results["gold_intent"],
            results["model_intent"],
            zero_division=0,
        )
    )

    print("\nConfidence statistics:")
    print(
        results["model_confidence"].describe()
    )

    # ------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()