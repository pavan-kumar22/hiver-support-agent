import re


class LocalResponseGenerator:

    def __init__(self):
        pass

    @staticmethod
    def clean_historical_response(text):
        """
        Remove Twitter usernames and URLs from historical
        AppleSupport responses before using them as evidence.
        """

        text = str(text)

        # Remove URLs
        text = re.sub(
            r"https?://\S+",
            "",
            text
        )

        # Remove @mentions
        text = re.sub(
            r"@\w+",
            "",
            text
        )

        # Remove excessive whitespace
        text = " ".join(
            text.split()
        )

        return text.strip()

    @staticmethod
    def extract_action(evidence):

        if not evidence:
            return None

        response = LocalResponseGenerator.clean_historical_response(
            evidence[0]["historical_response"]
        )

        response_lower = response.lower()

        # Historical support behavior patterns.
        if "dm" in response_lower:
            return "dm"

        if "restart" in response_lower:
            return "restart"

        if "update" in response_lower:
            return "update"

        if "repair" in response_lower:
            return "repair"

        if "country" in response_lower:
            return "country"

        if "error" in response_lower:
            return "error"

        if "device" in response_lower:
            return "device"

        return None

    def generate(
        self,
        customer_message,
        intent,
        evidence,
        action
    ):

        if action == "escalate":

            return (
                "Thanks for reaching out to Apple Support. "
                "We'd like to take a closer look at this issue. "
                "Please provide more details about what you're "
                "experiencing so we can determine the best way "
                "to assist you."
            )

        if not evidence:

            return (
                "Thanks for reaching out to Apple Support. "
                "We'd be happy to help. Please provide more "
                "details about the issue you're experiencing."
            )

        historical_action = self.extract_action(
            evidence
        )

        # ------------------------------------------------------
        # Intent-specific response construction
        # ------------------------------------------------------

        if intent == "battery_charging":

            if historical_action == "update":

                return (
                    "Thanks for reaching out. We understand "
                    "you're experiencing rapid battery drain. "
                    "Similar AppleSupport cases were addressed "
                    "by checking the iOS version and, where "
                    "appropriate, updating the device. Please "
                    "let us know which iPhone model and iOS "
                    "version you're currently using."
                )

            return (
                "Thanks for reaching out. We understand "
                "your battery is draining quickly. We'd like "
                "to look into this with you. Please let us "
                "know which iPhone model and iOS version "
                "you're using."
            )

        if intent == "connectivity_calls":

            return (
                "Thanks for reaching out. We'd be happy to "
                "help with your connectivity issue. Similar "
                "AppleSupport cases were handled by checking "
                "the device and network involved and confirming "
                "whether the issue occurs across multiple "
                "networks. Please let us know which device "
                "you're using and whether this happens on "
                "multiple networks."
            )

        if intent == "account_icloud":

            return (
                "Thanks for reaching out. We'd like to help "
                "with your Apple ID issue. Similar cases were "
                "handled by checking where the sign-in is "
                "failing and whether an error message appears. "
                "Please let us know where you're trying to sign "
                "in and the exact error message, if any."
            )

        if intent == "hardware_device":

            if historical_action == "country":

                return (
                    "We're sorry you're experiencing this "
                    "device issue. Similar AppleSupport cases "
                    "were handled by checking available service "
                    "options. Please let us know your current "
                    "country so we can determine the appropriate "
                    "support options."
                )

            return (
                "Thanks for reaching out. We'd like to look "
                "into your device issue. Please let us know "
                "which device you're using and describe the "
                "problem in a little more detail."
            )

        if intent == "messaging":

            return (
                "Thanks for reaching out. We'd be happy to "
                "help with your messaging issue. Please let "
                "us know which device you're using and whether "
                "the problem affects sending, receiving, or "
                "both."
            )

        if intent == "keyboard_input":

            return (
                "Thanks for reaching out. We'd like to help "
                "with the keyboard issue. Please let us know "
                "which device and iOS version you're using and "
                "describe what happens when you type."
            )

        if intent == "software_update":

            return (
                "Thanks for reaching out. We'd like to help "
                "with the software update issue. Similar "
                "AppleSupport cases were handled by checking "
                "the device and iOS version and understanding "
                "what happens during or after the update. "
                "Please let us know your device model and "
                "current iOS version."
            )

        if intent == "apps_services":

            return (
                "Thanks for reaching out. We'd be happy to "
                "help with the app or Apple service issue. "
                "Please let us know which app or service is "
                "affected, which device you're using, and any "
                "error message you're seeing."
            )

        if intent == "purchases_billing":

            return (
                "Thanks for reaching out. We'd like to help "
                "with the purchase or billing issue. Please "
                "provide the relevant purchase or subscription "
                "details and let us know what appears incorrect."
            )

        if intent == "repair_support":

            return (
                "Thanks for reaching out. We'd like to help "
                "with your support or service issue. Please "
                "provide the current status of the repair or "
                "service request and any relevant details."
            )

        return (
            "Thanks for contacting Apple Support. We'd be "
            "happy to help. Please provide more details about "
            "the issue you're experiencing."
        )


if __name__ == "__main__":

    generator = LocalResponseGenerator()

    test_reply = generator.generate(
        customer_message=(
            "My iPhone battery is draining very quickly"
        ),
        intent="battery_charging",
        evidence=[
            {
                "rank": 1,
                "similarity": 0.5474,
                "customer_text": (
                    "Battery draining very quickly"
                ),
                "historical_response": (
                    "Thanks for reaching out. "
                    "We're here to help. DM us so "
                    "we can look into this further."
                ),
            }
        ],
        action="auto_handle"
    )

    print("\nGenerated reply:")
    print(test_reply)