import pandas as pd

from pathlib import Path


DEVELOPMENT_FILE = Path(
    "data/processed/applesupport/development_messages.csv"
)

OUTPUT_FILE = Path(
    "data/processed/applesupport/"
    "balanced_training_messages.csv"
)

SAMPLES_PER_INTENT = 3000


def build_weak_labels(df):

    rules = {

        "software_update": [
            "update",
            "ios",
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

        best_intent = max(
            scores,
            key=scores.get
        )

        if scores[best_intent] == 0:
            return "other"

        return best_intent

    df = df.copy()

    df["weak_intent"] = (
        df["customer_text"]
        .apply(assign_label)
    )

    return df


def main():

    print("=" * 70)
    print("BUILD BALANCED WEAK-LABEL TRAINING SET")
    print("=" * 70)

    df = pd.read_csv(
        DEVELOPMENT_FILE
    )

    print(
        f"\nDevelopment rows: {len(df):,}"
    )

    df = build_weak_labels(df)

    df = df[
        df["weak_intent"] != "other"
    ].copy()

    print("\nOriginal weak-label counts:")

    print(
        df["weak_intent"]
        .value_counts()
    )

    balanced_parts = []

    for intent in sorted(
        df["weak_intent"].unique()
    ):

        subset = df[
            df["weak_intent"] == intent
        ]

        sample_size = min(
            SAMPLES_PER_INTENT,
            len(subset)
        )

        sampled = subset.sample(
            n=sample_size,
            random_state=42
        )

        balanced_parts.append(
            sampled
        )

    balanced = pd.concat(
        balanced_parts,
        ignore_index=True
    )

    balanced = balanced.sample(
        frac=1.0,
        random_state=42
    ).reset_index(
        drop=True
    )

    balanced.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nBalanced training rows: "
        f"{len(balanced):,}"
    )

    print("\nBalanced distribution:")

    print(
        balanced["weak_intent"]
        .value_counts()
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()