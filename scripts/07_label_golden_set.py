import pandas as pd
from pathlib import Path

INPUT_FILE = Path("evaluation/golden_labeling.csv")
OUTPUT_FILE = Path("evaluation/golden_labeling.csv")

INTENTS = [
    "software_update",
    "battery_charging",
    "messaging",
    "keyboard_input",
    "apps_services",
    "account_icloud",
    "connectivity_calls",
    "hardware_device",
    "purchases_billing",
    "repair_support",
    "other",
]

ACTIONS = [
    "auto_handle",
    "escalate",
]


def show_intents():
    print("\nINTENTS")
    print("-" * 60)
    for i, intent in enumerate(INTENTS, start=1):
        print(f"{i:2}. {intent}")


def show_actions():
    print("\nACTIONS")
    print("-" * 60)
    print("1. auto_handle")
    print("2. escalate")


def get_intent():
    while True:
        show_intents()
        value = input("\nEnter intent number: ").strip()

        if value.isdigit():
            number = int(value)
            if 1 <= number <= len(INTENTS):
                return INTENTS[number - 1]

        print("Invalid choice. Enter a number from 1 to 11.")


def get_action():
    while True:
        show_actions()
        value = input("\nEnter action number: ").strip()

        if value == "1":
            return "auto_handle"
        if value == "2":
            return "escalate"

        print("Invalid choice. Enter 1 or 2.")


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: File not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    # Convert annotation columns to string-compatible columns.
# Empty CSV cells are read by pandas as float64, so explicitly
# convert them before inserting text labels.
    annotation_columns = [
        "gold_intent",
        "gold_action",
        "gold_reason",
        "annotator_notes",
    ]

    for column in annotation_columns:
        df[column] = df[column].astype("object")

    required_columns = [
        "example_id",
        "conversation_id",
        "customer_text",
        "next_agent_response",
        "gold_intent",
        "gold_action",
        "gold_reason",
        "annotator_notes",
    ]

    for column in required_columns:
        if column not in df.columns:
            print(f"ERROR: Missing column: {column}")
            return

    print("=" * 70)
    print("APPLE SUPPORT — GOLDEN SET MANUAL LABELING")
    print("=" * 70)
    print(f"Total examples: {len(df)}")

    # Resume from first unlabeled row
    unlabeled = df[
        df["gold_intent"].isna()
        | df["gold_action"].isna()
    ]

    if len(unlabeled) == 0:
        print("\nAll examples are already labeled.")
        return

    print(f"Remaining examples: {len(unlabeled)}")

    for index in unlabeled.index:

        row = df.loc[index]

        print("\n")
        print("=" * 70)
        print(f"EXAMPLE {index + 1} / {len(df)}")
        print("=" * 70)

        print(f"\nExample ID:")
        print(row["example_id"])

        print(f"\nCustomer message:")
        print("-" * 70)
        print(row["customer_text"])

        print(f"\nHistorical AppleSupport response:")
        print("-" * 70)
        print(row["next_agent_response"])

        print("\n")

        # Allow quitting without losing previous labels
        command = input(
            "Press ENTER to label, or type 'q' to quit: "
        ).strip().lower()

        if command == "q":
            df.to_csv(OUTPUT_FILE, index=False)
            print("\nProgress saved.")
            return

        intent = get_intent()
        action = get_action()

        reason = input(
            "\nShort reason for this classification: "
        ).strip()

        notes = input(
            "Optional annotator notes (press ENTER to skip): "
        ).strip()

        df.at[index, "gold_intent"] = intent
        df.at[index, "gold_action"] = action
        df.at[index, "gold_reason"] = reason
        df.at[index, "annotator_notes"] = notes

        # Save after EVERY example
        df.to_csv(OUTPUT_FILE, index=False)

        print("\nSaved.")

    print("\n")
    print("=" * 70)
    print("LABELING COMPLETE")
    print("=" * 70)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()