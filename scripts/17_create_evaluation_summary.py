from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[1]

agent_path = ROOT / "evaluation" / "agent_results.csv"
reply_path = ROOT / "evaluation" / "reply_evaluation.csv"
evidence_path = ROOT / "evaluation" / "evidence_review.csv"

agent = pd.read_csv(agent_path)
reply = pd.read_csv(reply_path)
evidence = pd.read_csv(evidence_path)

lines = []

lines.append("=" * 70)
lines.append("APPLE SUPPORT AGENT - FINAL EVALUATION SUMMARY")
lines.append("=" * 70)
lines.append("")


# ==================================================================
# 1. AGENT EVALUATION
# ==================================================================

lines.append("1. AGENT EVALUATION")
lines.append("-" * 70)

# Intent metrics
if "gold_intent" in agent.columns and "model_intent" in agent.columns:

    y_true = agent["gold_intent"]
    y_pred = agent["model_intent"]

    intent_accuracy = accuracy_score(y_true, y_pred)

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    lines.append(
        f"Intent accuracy: {intent_accuracy:.4f}"
    )

    lines.append(
        f"Macro intent precision: {macro_precision:.4f}"
    )

    lines.append(
        f"Macro intent recall: {macro_recall:.4f}"
    )

    lines.append(
        f"Macro intent F1: {macro_f1:.4f}"
    )


# Action metrics
if "gold_action" in agent.columns and "model_action" in agent.columns:

    gold_action = agent["gold_action"]
    model_action = agent["model_action"]

    action_accuracy = accuracy_score(
        gold_action,
        model_action,
    )

    lines.append(
        f"Action accuracy: {action_accuracy:.4f}"
    )

    predicted_auto = model_action == "auto_handle"
    gold_auto = gold_action == "auto_handle"

    predicted_auto_count = predicted_auto.sum()
    gold_auto_count = gold_auto.sum()

    correct_auto_count = (
        predicted_auto & gold_auto
    ).sum()

    false_auto_handles = (
        predicted_auto & ~gold_auto
    ).sum()

    if predicted_auto_count > 0:
        auto_precision = (
            correct_auto_count / predicted_auto_count
        )
    else:
        auto_precision = 0.0

    if gold_auto_count > 0:
        auto_recall = (
            correct_auto_count / gold_auto_count
        )
    else:
        auto_recall = 0.0

    auto_f1 = (
        2 * auto_precision * auto_recall
        / (auto_precision + auto_recall)
        if (auto_precision + auto_recall) > 0
        else 0.0
    )

    automation_coverage = (
        predicted_auto_count / len(agent)
        if len(agent) > 0
        else 0.0
    )

    lines.append(
        f"Auto-handle precision: {auto_precision:.4f}"
    )

    lines.append(
        f"Auto-handle recall: {auto_recall:.4f}"
    )

    lines.append(
        f"Auto-handle F1: {auto_f1:.4f}"
    )

    lines.append(
        f"Automation coverage: {automation_coverage:.4f}"
    )

    lines.append(
        f"False auto-handles: {false_auto_handles}"
    )


# Confidence
if "model_confidence" in agent.columns:

    lines.append(
        f"Average intent confidence: "
        f"{agent['model_confidence'].mean():.4f}"
    )


# Retrieval similarity
if "top_similarity" in agent.columns:

    lines.append(
        f"Average retrieval similarity: "
        f"{agent['top_similarity'].mean():.4f}"
    )


# Action distribution
if "model_action" in agent.columns:

    action_counts = agent["model_action"].value_counts()

    lines.append(
        f"Escalations: "
        f"{action_counts.get('escalate', 0)}"
    )

    lines.append(
        f"Auto-handled: "
        f"{action_counts.get('auto_handle', 0)}"
    )


# Model intent distribution
if "model_intent" in agent.columns:

    lines.append("")
    lines.append("Model intent distribution:")

    intent_counts = agent["model_intent"].value_counts()

    for intent, count in intent_counts.items():
        lines.append(
            f"  {intent}: {count}"
        )


lines.append("")


# ==================================================================
# 2. RETRIEVAL / EVIDENCE
# ==================================================================

lines.append("2. RETRIEVAL / EVIDENCE")
lines.append("-" * 70)

if "evidence_level" in evidence.columns:

    counts = evidence["evidence_level"].value_counts()

    for level in [
        "strong",
        "moderate",
        "weak",
        "none",
    ]:
        lines.append(
            f"{level.capitalize()} evidence: "
            f"{counts.get(level, 0)}"
        )


if "evidence_exists" in evidence.columns:

    lines.append(
        f"Evidence exists rate: "
        f"{evidence['evidence_exists'].mean():.4f}"
    )


if "intent_grounded" in evidence.columns:

    lines.append(
        f"Intent-grounded rate: "
        f"{evidence['intent_grounded'].mean():.4f}"
    )


if "similarity" in evidence.columns:

    lines.append(
        f"Average evidence similarity: "
        f"{evidence['similarity'].mean():.4f}"
    )

elif "top_similarity" in agent.columns:

    lines.append(
        f"Average retrieval similarity: "
        f"{agent['top_similarity'].mean():.4f}"
    )


lines.append("")


# ==================================================================
# 3. REPLY EVALUATION
# ==================================================================

lines.append("3. REPLY EVALUATION - OFFLINE RUBRIC")
lines.append("-" * 70)

reply_metrics = [
    "non_empty",
    "professional",
    "helpful_request",
    "intent_grounded",
    "escalation_appropriate",
    "historical_grounding",
    "no_unsupported_claims",
]

for column in reply_metrics:

    if column in reply.columns:

        lines.append(
            f"{column}: "
            f"{reply[column].mean():.4f}"
        )


if "quality_score" in reply.columns:

    lines.append(
        f"Average offline quality score: "
        f"{reply['quality_score'].mean():.4f}"
    )


lines.append("")

lines.append(
    "NOTE: The reply-quality score above is a deterministic "
    "offline rubric, not an LLM-as-judge score."
)

lines.append("")


# ==================================================================
# 4. HUMAN / AUTOMATED EVIDENCE REVIEW
# ==================================================================

lines.append("4. HUMAN / AUTOMATED EVIDENCE REVIEW")
lines.append("-" * 70)

lines.append(
    "A 56-example high-priority subset was prepared "
    "for evidence review."
)

lines.append(
    "The current review file was created through an "
    "AI-assisted prefill workflow and has not been "
    "independently confirmed as human annotation."
)

lines.append(
    "Therefore, no independent human-agreement metric "
    "is reported as a final evaluation result."
)

lines.append("")

lines.append("Current AI-assisted prefill:")

lines.append(
    "Examples prepared for review: 56"
)

lines.append(
    "AI-assisted relevance = Yes: 13"
)

lines.append(
    "AI-assisted relevance = No: 43"
)

lines.append(
    "Independent human confirmation: Pending"
)

lines.append("")


# ==================================================================
# 5. METHODOLOGICAL CAVEATS
# ==================================================================

lines.append("5. METHODOLOGICAL CAVEATS")
lines.append("-" * 70)

lines.append(
    "- The 200-example golden set was initially "
    "machine-assisted."
)

lines.append(
    "- The 56-example evidence review contains "
    "AI-assisted draft labels and should not be represented "
    "as fully independent human annotation unless a human "
    "reviewer independently confirmed/corrected them."
)

lines.append(
    "- OpenAI API-based LLM judging was unavailable because "
    "the API account had insufficient quota; therefore no "
    "fabricated LLM-judge result is reported."
)

lines.append(
    "- Raw action accuracy is affected by the imbalance "
    "between escalation and auto-handle decisions."
)

lines.append(
    "- The final risk-aware policy intentionally favors "
    "conservative escalation and therefore has low "
    "automation coverage."
)

lines.append("")

lines.append("=" * 70)


# ==================================================================
# WRITE OUTPUT
# ==================================================================

output = (
    ROOT
    / "evaluation"
    / "final_evaluation_summary.txt"
)

output.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print(f"Saved: {output}")