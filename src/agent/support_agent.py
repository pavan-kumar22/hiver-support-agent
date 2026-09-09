from pathlib import Path
import sys

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intents.classifier import IntentClassifier
from src.retrieval.retriever import HistoricalRetriever
from src.agent.local_responder import LocalResponseGenerator


class AppleSupportAgent:

    def __init__(
        self,
        intent_threshold=0.70,
        evidence_threshold=0.35
    ):

        print("Initializing Apple Support Agent...")

        self.classifier = IntentClassifier()
        self.retriever = HistoricalRetriever()
        self.response_generator = LocalResponseGenerator()

        self.intent_threshold = intent_threshold
        self.evidence_threshold = evidence_threshold

        # High-risk intents are always escalated.
        # These intents may involve account access, money,
        # physical damage, repairs, or cases requiring
        # human investigation.
        self.always_escalate_intents = {
            "other",
            "repair_support",
            "purchases_billing",
            "account_icloud",
            "hardware_device",
        }

        print("Agent ready.")

    def analyze(self, customer_message):

        customer_message = str(
            customer_message
        ).strip()

        if not customer_message:

            return {
                "intent": "other",
                "intent_confidence": 0.0,
                "evidence_level": "none",
                "top_similarity": 0.0,
                "action": "escalate",
                "reason": (
                    "The customer message is empty "
                    "or contains insufficient information."
                ),
                "reply": (
                    "Thanks for contacting Apple Support. "
                    "Please provide more details about "
                    "the issue you're experiencing."
                ),
                "evidence": [],
            }

        # --------------------------------------------------------
        # 1. Classify intent
        # --------------------------------------------------------

        classification = self.classifier.predict(
            customer_message
        )

        intent = classification["intent"]
        confidence = classification["confidence"]

        # --------------------------------------------------------
        # 2. Retrieve historical resolutions
        # --------------------------------------------------------

        evidence = self.retriever.evidence_summary(
            customer_message,
            top_k=3
        )

        evidence_level = evidence["evidence_level"]
        top_similarity = evidence["top_similarity"]

        # --------------------------------------------------------
        # 3. Risk-aware escalation policy
        # --------------------------------------------------------

        reasons = []

        # Rule 1: Always escalate high-risk intents.
        if intent in self.always_escalate_intents:

            reasons.append(
                f"The '{intent}' intent is designated as "
                "high-risk and requires human review."
            )

        # Rule 2: Never auto-handle unsupported/unclear intent.
        if intent == "other":

            reasons.append(
                "The issue does not map confidently "
                "to a supported intent."
            )

        # Rule 3: Confidence must be above threshold.
        if confidence < self.intent_threshold:

            reasons.append(
                "Intent classification confidence "
                "is below the safe handling threshold."
            )

        # Rule 4: Historical evidence must be strong enough.
        if top_similarity < self.evidence_threshold:

            reasons.append(
                "There is insufficient historical "
                "resolution evidence for a grounded reply."
            )

        # Rule 5:
        # Connectivity can involve account/network/carrier
        # dependencies, so require stronger evidence.
        if intent == "connectivity_calls":

            connectivity_confidence_threshold = 0.80
            connectivity_evidence_threshold = 0.45

            if confidence < connectivity_confidence_threshold:

                reasons.append(
                    "Connectivity issues require higher "
                    "classification confidence before automation."
                )

            if top_similarity < connectivity_evidence_threshold:

                reasons.append(
                    "Connectivity issues require stronger "
                    "historical evidence before automation."
                )

        # Rule 6:
        # Software updates can affect the whole device.
        # Require stronger evidence before auto-handling.
        if intent == "software_update":

            update_confidence_threshold = 0.80
            update_evidence_threshold = 0.45

            if confidence < update_confidence_threshold:

                reasons.append(
                    "Software-update issues require higher "
                    "classification confidence before automation."
                )

            if top_similarity < update_evidence_threshold:

                reasons.append(
                    "Software-update issues require stronger "
                    "historical evidence before automation."
                )

        # --------------------------------------------------------
        # Final action decision
        # --------------------------------------------------------

        if reasons:

            action = "escalate"

            reason = " ".join(reasons)

        else:

            action = "auto_handle"

            reason = (
                "The intent is considered low-risk and both "
                "classification confidence and historical "
                "evidence exceed the fixed handling thresholds."
            )

        # --------------------------------------------------------
        # 4. Build grounded draft
        # --------------------------------------------------------

        reply = self.response_generator.generate(
            customer_message=customer_message,
            intent=intent,
            evidence=evidence["results"],
            action=action
        )

        return {
            "intent": intent,
            "intent_confidence": confidence,
            "evidence_level": evidence_level,
            "top_similarity": top_similarity,
            "action": action,
            "reason": reason,
            "reply": reply,
            "evidence": evidence["results"],
        }

    @staticmethod
    def _build_grounded_reply(
        customer_message,
        intent,
        evidence
    ):

        if not evidence:

            return (
                "Thanks for contacting Apple Support. "
                "We'd be happy to help. Please send us "
                "more details about the issue."
            )

        best_response = evidence[0][
            "historical_response"
        ]

        # We intentionally do NOT copy the historical
        # response verbatim. Historical responses are
        # used as evidence for the support approach.

        return (
            "Thanks for reaching out. "
            "We understand you're experiencing an issue "
            "related to "
            f"{intent.replace('_', ' ')}. "
            "Based on similar AppleSupport cases, "
            "we'd like to look into this with you. "
            "Please provide the relevant device details "
            "and any error message you're seeing."
        )

    @staticmethod
    def _build_escalation_reply(
        customer_message,
        intent,
        evidence
    ):

        return (
            "Thanks for contacting Apple Support. "
            "We'd like to take a closer look at this issue. "
            "Please provide more details about your device "
            "and what you're experiencing so that a support "
            "specialist can assist you."
        )


if __name__ == "__main__":

    agent = AppleSupportAgent()

    test_messages = [

        "My iPhone battery is draining very quickly",

        "WiFi keeps disconnecting after the update",

        "I cannot sign into my Apple ID",

        "My iPhone screen is cracked",

        "My issue is completely unusual and I need help",
    ]

    for message in test_messages:

        print("\n" + "=" * 70)
        print("CUSTOMER")
        print("=" * 70)

        print(message)

        result = agent.analyze(
            message
        )

        print("\nINTENT:")
        print(result["intent"])

        print("\nINTENT CONFIDENCE:")
        print(result["intent_confidence"])

        print("\nEVIDENCE LEVEL:")
        print(result["evidence_level"])

        print("\nTOP SIMILARITY:")
        print(result["top_similarity"])

        print("\nACTION:")
        print(result["action"])

        print("\nREASON:")
        print(result["reason"])

        print("\nDRAFT REPLY:")
        print(result["reply"])

        print("\nTOP HISTORICAL EVIDENCE:")

        for item in result["evidence"]:

            print(
                f"\n{item['rank']}. "
                f"Similarity: "
                f"{item['similarity']}"
            )

            print(
                f"Customer: "
                f"{item['customer_text']}"
            )

            print(
                f"Response: "
                f"{item['historical_response']}"
            )

