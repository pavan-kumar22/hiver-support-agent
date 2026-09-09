from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TRAINING_FILE = Path(
    "data/processed/applesupport/"
    "balanced_training_messages.csv"
)


class HybridIntentClassifier:

    def __init__(self):

        print("Loading balanced intent examples...")

        df = pd.read_csv(
            TRAINING_FILE
        )

        df = df[
            df["weak_intent"] != "other"
        ].copy()

        self.training = df.reset_index(
            drop=True
        )

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        )

        self.matrix = self.vectorizer.fit_transform(
            self.training["customer_text"]
            .astype(str)
        )

        self.intents = sorted(
            self.training["weak_intent"]
            .unique()
        )

        print(
            f"Loaded {len(self.training):,} "
            "training examples."
        )

    def predict(
        self,
        text,
        top_k=10
    ):

        text = str(text).strip()

        if not text:

            return {
                "intent": "other",
                "confidence": 0.0,
                "margin": 0.0,
            }

        query_vector = (
            self.vectorizer.transform(
                [text]
            )
        )

        similarities = (
            cosine_similarity(
                query_vector,
                self.matrix
            )[0]
        )

        top_indices = similarities.argsort()[
            ::-1
        ][:top_k]

        intent_scores = {}

        for index in top_indices:

            intent = self.training.iloc[index][
                "weak_intent"
            ]

            similarity = float(
                similarities[index]
            )

            if intent not in intent_scores:
                intent_scores[intent] = []

            intent_scores[intent].append(
                similarity
            )

        # Score each intent using the strongest
        # retrieved examples plus supporting examples.
        aggregated = {}

        for intent, scores in intent_scores.items():

            scores = sorted(
                scores,
                reverse=True
            )

            weighted_score = (
                scores[0]
                + 0.5 * (
                    scores[1]
                    if len(scores) > 1
                    else 0
                )
                + 0.25 * (
                    scores[2]
                    if len(scores) > 2
                    else 0
                )
            )

            aggregated[intent] = weighted_score

        if not aggregated:

            return {
                "intent": "other",
                "confidence": 0.0,
                "margin": 0.0,
            }

        ranked = sorted(
            aggregated.items(),
            key=lambda x: x[1],
            reverse=True
        )

        best_intent = ranked[0][0]
        best_score = ranked[0][1]

        second_score = (
            ranked[1][1]
            if len(ranked) > 1
            else 0.0
        )

        total_score = sum(
            aggregated.values()
        )

        confidence = (
            best_score / total_score
            if total_score > 0
            else 0.0
        )

        margin = (
            best_score - second_score
        )

        return {
            "intent": best_intent,
            "confidence": round(
                confidence,
                4
            ),
            "margin": round(
                margin,
                4
            ),
        }


if __name__ == "__main__":

    classifier = HybridIntentClassifier()

    test_messages = [

        "My iPhone battery is draining very quickly",

        "WiFi keeps disconnecting after the update",

        "I cannot sign into my Apple ID",

        "My iPhone screen is cracked",

        "iMessage is not sending",

        "My keyboard keeps autocorrecting everything",

        "I need help with an unusual problem",
    ]

    for message in test_messages:

        result = classifier.predict(
            message
        )

        print("\nCustomer:")
        print(message)

        print("Prediction:")
        print(result)