import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


TRAINING_FILE = Path(
    "data/processed/applesupport/"
    "balanced_training_messages.csv"
)


class IntentClassifier:

    def __init__(self):
        print("Loading balanced training data...")

        df = pd.read_csv(
            TRAINING_FILE
        )

        print(
            f"Training rows: {len(df):,}"
        )

        df = df[
            df["weak_intent"] != "other"
        ].copy()

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            sublinear_tf=True,
            max_features=50000,
        )

        X = self.vectorizer.fit_transform(
            df["customer_text"].astype(str)
        )

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

        self.model.fit(
            X,
            df["weak_intent"]
        )

        print(
            f"Intent classifier trained on "
            f"{len(df):,} weakly labelled messages."
        )

    @staticmethod
    def _build_weak_labels(df):

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

    def predict(self, text):

        text = str(text).strip()

        if not text:

            return {
                "intent": "other",
                "confidence": 0.0,
            }

        X = self.vectorizer.transform(
            [text]
        )

        probabilities = (
            self.model.predict_proba(X)[0]
        )

        best_index = probabilities.argmax()

        intent = self.model.classes_[
            best_index
        ]

        confidence = float(
            probabilities[best_index]
        )

        return {
            "intent": intent,
            "confidence": round(
                confidence,
                4
            ),
        }


if __name__ == "__main__":

    classifier = IntentClassifier()

    test_messages = [
        "My iPhone battery is draining very quickly",
        "WiFi keeps disconnecting after the update",
        "I cannot sign into my Apple ID",
        "My iPhone screen is cracked",
        "iMessage is not sending",
        "My keyboard keeps autocorrecting everything",
    ]

    for message in test_messages:

        result = classifier.predict(
            message
        )

        print("\nCustomer:")
        print(message)

        print("Prediction:")
        print(result)