# AppleSupport AI Support Agent

An end-to-end AI support-agent prototype built for the Hiver SDE Intern take-home assignment.

The system:

1. classifies incoming customer messages into operational support intents,
2. retrieves historically similar AppleSupport interactions,
3. drafts a response grounded in historical support patterns,
4. decides whether to auto-handle or escalate,
5. evaluates classification, retrieval, response quality, and escalation behavior.

The pipeline is designed to run locally without requiring a paid LLM API. The submitted evaluation uses a deterministic local responder; an OpenAI-based responder is included as an optional component but was not required for the reported results.


---

## 1. Project Structure

```text
hiver-support-agent/
│
├── data/
│   ├── raw/
│   │   └── twcs.csv
│   └── processed/
│       └── applesupport/
│
├── src/
│   ├── intents/
│   ├── retrieval/
│   └── agent/
│
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_build_conversations.py
│   ├── 03_analyze_intents.py
│   ├── 04_discover_intents.py
│   ├── 05_semantic_intent_discovery.py
│   ├── 06_create_golden_set.py
│   ├── 07_label_golden_set.py
│   ├── 08_baseline_majority.py
│   ├── 09_baseline_tfidf.py
│   ├── 10_build_retrieval_index.py
│   ├── 11_evaluate_agent.py
│   ├── 12_build_balanced_training.py
│   ├── 13_analyze_failures.py
│   ├── 14_evaluate_replies.py
│   ├── 14_create_evidence_review.py
│   ├── 15_create_human_review.py
│   ├── 16_evaluate_human_agreement.py
│   └── 17_create_evaluation_summary.py
│
├── evaluation/
│   ├── golden_set_v1.csv
│   ├── agent_results.csv
│   ├── reply_evaluation.csv
│   ├── evidence_review.csv
│   ├── human_evidence_review.csv
│   ├── human_agreement_results.txt
│   └── final_evaluation_summary.txt
│
├── report/
│   ├── final_report.md
│   └── decision_log.md
│
├── requirements.txt
└── README.md
```

---

## 2. Environment Setup

Python 3.10+ is recommended.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## 3. Dataset

The project uses the Kaggle **Customer Support on Twitter** dataset.

The raw file is expected at:

```text
data/raw/twcs.csv
```

The implementation focuses on the `AppleSupport` brand.

The extraction pipeline links AppleSupport responses to their parent customer messages using `in_response_to_tweet_id`.

The resulting dataset contains approximately:

* 106,646 customer-agent pairs
* 106,623 unique customer messages
* 76,365 unique customers

The complete raw dataset is not required to be processed during every evaluation run once the processed artifacts have been generated.

---

## 4. Intent Taxonomy

The final taxonomy contains 11 operational intents:

| ID                   | Intent                              |
| -------------------- | ----------------------------------- |
| `software_update`    | Software / iOS Update Issue         |
| `battery_charging`   | Battery / Charging                  |
| `messaging`          | Messaging / iMessage / SMS          |
| `keyboard_input`     | Keyboard / Autocorrect              |
| `apps_services`      | Apps / Apple Services               |
| `account_icloud`     | Apple ID / iCloud / Account         |
| `connectivity_calls` | Connectivity / Calls / Network      |
| `hardware_device`    | Hardware / Device Problem           |
| `purchases_billing`  | Purchases / Billing / Subscriptions |
| `repair_support`     | Repair / Apple Support / Service    |
| `other`              | Other / Unclear                     |

The taxonomy is stored in:

```text
data/processed/applesupport/intent_taxonomy.yaml
```

---

## 5. Run the Pipeline

If starting from the raw dataset, the main preparation stages are:

```powershell
python scripts/01_prepare_data.py
python scripts/02_build_conversations.py
python scripts/03_analyze_intents.py
python scripts/05_semantic_intent_discovery.py
python scripts/06_create_golden_set.py
python scripts/12_build_balanced_training.py
python scripts/10_build_retrieval_index.py
```

The final evaluation can then be run with:

```powershell
python scripts/11_evaluate_agent.py
```

---

## 6. Baselines

Run the majority baseline:

```powershell
python scripts/08_baseline_majority.py
```

Run the TF-IDF + Logistic Regression baseline:

```powershell
python scripts/09_baseline_tfidf.py
```

The baselines are evaluated against the frozen golden set.

---

## 7. Evaluation

Run failure analysis:

```powershell
python scripts/13_analyze_failures.py
```

Run the deterministic reply-quality evaluation:

```powershell
python scripts/14_evaluate_replies.py
```

Create the evidence-review set:

```powershell
python scripts/14_create_evidence_review.py
```

Create the human-review file:

```powershell
python scripts/15_create_human_review.py
```

After human review, evaluate agreement:

```powershell
python scripts/16_evaluate_human_agreement.py
```

Finally create the consolidated evaluation summary:

```powershell
python scripts/17_create_evaluation_summary.py
```

---

## 8. Headline Results

Evaluation uses a frozen 200-example golden set.

### Baselines

| System                            | Intent Accuracy | Action Accuracy |
| --------------------------------- | --------------: | --------------: |
| Majority intent + always escalate |       **25.0%** |       **78.0%** |
| TF-IDF + Logistic Regression      |       **44.5%** |       **42.5%** |
| Final risk-aware agent            |       **44.5%** |       **74.5%** |

The TF-IDF baseline uses weak development labels and does not use the final risk-aware escalation policy.

### Final Agent

| Metric                       |    Result |
| ---------------------------- | --------: |
| Intent accuracy              | **44.5%** |
| Macro precision              | **49.2%** |
| Macro recall                 | **40.0%** |
| Macro F1                     | **40.4%** |
| Action accuracy              | **74.5%** |
| Auto-handle precision        | **29.4%** |
| Auto-handle recall           | **11.4%** |
| Strong evidence              | **52.5%** |
| Moderate/strong evidence     | **72.0%** |
| Average retrieval similarity | **0.615** |

The final policy produced:

```text
183 escalations
17 auto-handled cases
```

---

## 9. Important Evaluation Caveat

Action accuracy is not sufficient to judge this system.

The golden set contains substantially more escalation examples than auto-handle examples.

As a result, an always-escalate policy achieves **78.0% action accuracy**, which is higher than the final agent's 74.5%.

For this reason the project also reports:

* auto-handle precision,
* auto-handle recall,
* false auto-handles,
* automation coverage,
* evidence quality,
* and failure modes.

The final risk-aware policy reduced false auto-handles from **39 to 12**, but also reduced automation coverage.

The system is therefore intentionally conservative.

---

## 10. Reply Evaluation

The local reply evaluator reports:

```text
Non-empty:              100.0%
Professional:           100.0%
Helpful request:        100.0%
Intent grounded:         61.5%
Escalation appropriate: 100.0%
Historical grounding:   30.5%
No unsupported claims:  100.0%

Average offline quality: 84.6%
```

The 84.6% score is a **deterministic offline rubric**, not an LLM-as-judge score.

An API-based LLM judge was not run because the available API account had insufficient quota. No fabricated LLM evaluation is included.

---

## 11. Evidence Evaluation

Across the 200 golden examples:

```text
Strong evidence:       105
Moderate evidence:      39
Weak evidence:          48
No evidence:             8
```

Evidence existed for **72.0%** of examples.

The system performs best when the historical dataset contains a close analogue to the incoming customer issue.

The primary retrieval limitation is that TF-IDF similarity is lexical rather than fully semantic: two messages can share words without sharing the same underlying resolution.

---

## 12. Human / Automated Evidence Review

A 56-example high-priority subset was prepared for evidence review.

The review workflow was designed to compare automated evidence judgments with human judgments and to measure evaluator agreement using accuracy and Cohen's kappa.

However, the current review file was generated through an **AI-assisted prefill workflow and has not been independently confirmed as human annotation**. Therefore, the current agreement statistics are treated as preliminary and are **not claimed as valid independent human agreement**.

The current AI-assisted prefill contains:

```text
Examples prepared for review:       56
AI-assisted relevance = Yes:        13
AI-assisted relevance = No:         43
Independent human confirmation:     Pending
```

No independent human-agreement conclusion is used as a headline result.

A genuinely hand-labelled review should replace or independently confirm these fields before reporting Cohen's kappa or human-vs-automated agreement as final evaluation evidence.

---

## 13. Known Failure Modes

The five main failure modes are:

1. `other` messages being absorbed into `hardware_device`
2. app/service problems being confused with hardware failures
3. update-related issues being confused with hardware problems
4. weak retrieval producing superficially similar evidence
5. conservative escalation resulting in low automation coverage

Detailed examples and hypotheses are documented in:

```text
report/final_report.md
```

---

## 14. Reproducibility

For the fastest evaluation after the processed artifacts have been generated:

```powershell
python scripts/11_evaluate_agent.py
python scripts/13_analyze_failures.py
python scripts/14_evaluate_replies.py
python scripts/17_create_evaluation_summary.py
```

The evaluation operates on the frozen golden set rather than requiring the evaluator to process the full multi-million-row raw dataset.

The retrieval index is also precomputed from a deterministic 30,000-example subset.

---

## 15. Reports

Final report:

```text
report/final_report.md
```

Decision log:

```text
report/decision_log.md
```

Evaluation summary:

```text
evaluation/final_evaluation_summary.txt
```

---

## 16. Limitations

The current prototype has several important limitations:

* the golden-set labels were initially machine-assisted;
* the human evidence review was AI-assisted and requires independent human confirmation to qualify as independent annotation;
* the intent classifier achieves only 44.5% accuracy;
* `other`, `apps_services`, and `repair_support` remain difficult classes;
* TF-IDF retrieval can return lexically similar but weak evidence;
* the deterministic responder is less flexible than a generative LLM;
* the LLM-as-judge evaluation could not be completed because of API quota limitations;
* automation coverage is intentionally low under the conservative escalation policy.

These limitations are explicitly reported rather than hidden behind aggregate metrics.

---

## 17. Conclusion

The project demonstrates a complete local support-agent pipeline combining intent classification, historical retrieval, response drafting, escalation, evaluation, and failure analysis.

The current system is best viewed as a **conservative evidence-aware support triage and drafting assistant**, not as a fully autonomous support agent.

The next improvements should focus on genuinely human-labelled training data, better semantic retrieval, confidence calibration, independent LLM/human reply evaluation, and increasing automation coverage without increasing unsafe auto-handling.
