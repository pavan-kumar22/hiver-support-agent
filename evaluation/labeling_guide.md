# AppleSupport Intent Labeling Guide

## Goal

Assign exactly one primary intent to each customer message.

Use the customer's current message and available preceding conversation
context when provided.

---

## Priority Rules

When multiple issues are present, choose the issue that represents the
customer's primary request.

Use the following precedence when necessary:

1. Account / authentication
2. Purchases / billing
3. Repair / support / service
4. Battery / charging
5. Connectivity / calls / network
6. Messaging
7. Keyboard / autocorrect
8. Apps / services
9. Hardware / device
10. Software / iOS update
11. Other

Do not choose an intent merely because the message contains a keyword.

---

## software_update

Use when the main problem is caused by or directly related to an
iOS/macOS/watchOS update.

Examples:

- "Since updating to iOS 11 my phone keeps freezing."
- "How do I downgrade from iOS 11?"
- "The latest update broke my apps."

---

## battery_charging

Use for:

- battery drain
- battery health
- unexpected battery percentage changes
- charging failures
- charging speed
- charging accessories
- battery-related overheating

Examples:

- "My battery drops from 80% to 20% in an hour."
- "My iPhone won't charge."
- "Why is my phone getting hot while charging?"

---

## messaging

Use for:

- iMessage
- SMS
- Messages
- sending/receiving texts
- message synchronization
- message notifications

Examples:

- "My iMessage won't activate."
- "My texts aren't being delivered."
- "Messages disappeared after the update."

---

## keyboard_input

Use for:

- autocorrect
- keyboard glitches
- typing characters incorrectly
- predictive text
- character rendering

Examples:

- "Typing I gives me a question mark."
- "Autocorrect keeps changing my words."
- "My keyboard isn't predicting words."

---

## apps_services

Use when an Apple app/service itself is the main problem.

Examples:

- App Store won't open
- Apple Music won't load
- Safari problem
- Photos problem
- Mail problem
- FaceTime problem

---

## account_icloud

Use for:

- Apple ID
- iCloud
- password
- authentication
- verification
- account lock
- backups
- suspicious account activity

Examples:

- "My Apple ID is locked."
- "I can't access my iCloud backup."
- "Is this Apple ID email real?"

---

## connectivity_calls

Use for:

- Wi-Fi
- Bluetooth
- cellular network
- SIM
- calls
- activation
- network reception

Examples:

- "My phone can't connect to Wi-Fi."
- "Calls keep dropping."
- "Bluetooth won't connect."

---

## hardware_device

Use for physical/device functionality problems.

Examples:

- broken screen
- broken port
- speaker failure
- camera failure
- damaged device
- physical overheating
- button failure

---

## purchases_billing

Use for:

- payment
- billing
- purchases
- refunds
- subscriptions
- unauthorized purchases
- restoring purchases

Examples:

- "I was charged twice."
- "How do I get a refund?"
- "My payment method isn't working."

---

## repair_support

Use for:

- repair status
- Apple Store service
- service center
- replacement
- warranty service
- unresolved support interaction
- explicit human-support escalation

Examples:

- "My phone has been at the repair center for two weeks."
- "Apple Support hasn't responded."
- "Can I get a replacement?"

---

## other

Use when:

- insufficient information
- unrelated question
- unclear intent
- multiple unrelated issues where no primary issue can be established
- purely conversational message
- genuinely novel issue