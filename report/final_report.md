# AI Support Agent for AppleSupport

## 1. Problem Framing

The objective is to build an AI support agent that can take an incoming customer-support message, identify the customer's intent, retrieve historically similar AppleSupport interactions, draft a grounded response, and decide whether the case should be automatically handled or escalated.

The system was developed using the Kaggle Customer Support on Twitter dataset, focusing on the **AppleSupport** brand. The extracted AppleSupport interaction data contains approximately **106K customer-agent pairs**.

The design prioritizes safe escalation over aggressive automation. Cases involving account access, purchases/billing, hardware problems, repairs, and unclear issues are treated as higher risk and are preferentially escalated.

---

## 2. Data and Intent Taxonomy

The AppleSupport response tweets were linked to their parent customer tweets using the dataset's `in_response_to_tweet_id` relationship.

The resulting interaction set contains:

* 106,646 matched customer-agent pairs
* 106,623 unique customer messages
* 76,365 unique customers

A compact 11-intent taxonomy was developed from the customer-message distribution and semantic inspection:

1. Software / iOS Update Issue
2. Battery / Charging
3. Messaging / iMessage / SMS
4. Keyboard / Autocorrect
5. Apps / Apple Services
6. Apple ID / iCloud / Account
7. Connectivity / Calls / Network
8. Hardware / Device Problem
9. Purchases / Billing / Subscriptions
10. Repair / Apple Support / Service
11. Other / Unclear

The taxonomy deliberately favors operationally useful support categories rather than attempting to reproduce every topic appearing in the raw dataset.

---

## 3. System Design

The pipeline consists of four primary stages:

**Incoming customer message**

→ **Intent classifier**

→ **Historical retrieval**

→ **Response generation + escalation policy**

The intent classifier uses TF-IDF features with a Logistic Regression classifier trained using a balanced development set.

The retrieval component uses TF-IDF similarity over a deterministic 30,000-example subset of historical AppleSupport interactions.

The response generator is an API-free deterministic responder that extracts useful patterns from historically retrieved AppleSupport responses and combines them with intent-specific support language.

The escalation policy combines:

* intent confidence,
* retrieval similarity,
* predicted intent risk,
* and predefined high-risk categories.

High-risk categories such as account, billing, repair, hardware, and unclear cases are escalated rather than automatically handled.

---

## 4. Evaluation Setup

The evaluation uses a frozen 200-example golden set.

Two baselines were implemented:

### Baseline 1 — Majority Intent + Always Escalate

The classifier always predicts the frozen majority intent and always escalates.

Results:

| Metric          |    Result |
| --------------- | --------: |
| Intent accuracy | **25.0%** |
| Action accuracy | **78.0%** |

The high action score is largely produced by the fact that escalation is the dominant gold action.

### Baseline 2 — TF-IDF + Logistic Regression

A TF-IDF + Logistic Regression classifier was trained using weak development labels, with the golden set excluded from training.

Results:

| Metric          |    Result |
| --------------- | --------: |
| Intent accuracy | **44.5%** |
| Macro F1        | **41.0%** |
| Action accuracy | **42.5%** |

This demonstrates that simple lexical classification provides useful signal, but weak-label noise and class imbalance limit performance.

---

## 5. Agent Results

The final risk-aware agent achieved:

| Metric                       |    Result |
| ---------------------------- | --------: |
| Intent accuracy              | **44.5%** |
| Macro precision              | **49.2%** |
| Macro recall                 | **40.0%** |
| Macro F1                     | **40.4%** |
| Action accuracy              | **74.5%** |
| Auto-handle precision        | **29.4%** |
| Auto-handle recall           | **11.4%** |
| Strong retrieval evidence    | **52.5%** |
| Moderate/strong evidence     | **72.0%** |
| Average retrieval similarity | **0.615** |

The risk-aware policy produced:

* **183 escalations**
* **17 auto-handled cases**

Compared with the earlier policy, risk-aware escalation reduced false auto-handles from **39 to 12** and reduced action errors from **67 to 51**.

This came at the cost of very low automation coverage.

---

## 6. Retrieval and Evidence Evaluation

Across the 200 golden examples:

| Evidence level | Examples |
| -------------- | -------: |
| Strong         |      105 |
| Moderate       |       39 |
| Weak           |       48 |
| None           |        8 |

Overall:

* Evidence existed for **72.0%** of examples.
* **59.0%** were judged intent-grounded by the automated evidence rubric.
* Average retrieval similarity was **0.615**.

Exact or near-exact historical examples work particularly well. However, generic or unusual customer messages can produce superficially similar historical tweets that do not actually provide useful resolution evidence.

---

## 7. Reply Evaluation

An offline deterministic rubric was used to evaluate generated replies.

| Criterion               |     Score |
| ----------------------- | --------: |
| Non-empty               |    100.0% |
| Professional            |    100.0% |
| Helpful request         |    100.0% |
| Intent grounded         |     61.5% |
| Escalation appropriate  |    100.0% |
| Historical grounding    |     30.5% |
| No unsupported claims   |    100.0% |
| Average offline quality | **84.6%** |

The **84.6% score should not be treated as an LLM-judge score**. It is a deterministic rubric implemented locally.

An API-based LLM judge could not be run because the available OpenAI API account had insufficient quota. No fabricated LLM-judge result is reported.

The low historical-grounding score is particularly important: the responder can produce professional and safe-looking replies even when the retrieved evidence does not strongly support the response.

---

## 8. Human / Automated Evidence Review

A 56-example high-priority evidence subset was prepared for manual review.

The intended workflow was to compare automated evidence judgments with independent human judgments and measure agreement using accuracy and Cohen's kappa.

However, the current review file was created through an **AI-assisted prefill workflow and has not been independently confirmed as human annotation**. Therefore, the current agreement statistics are **preliminary and are not treated as valid independent human-evaluation results**.

The current AI-assisted prefill contains:

```text
Examples prepared for review:       56
AI-assisted relevance = Yes:        13
AI-assisted relevance = No:         43
Independent human confirmation:     Pending
```

For this reason, no human-agreement metric is used as a headline result or as evidence of independent evaluator reliability.

A final submission claiming human agreement should replace or independently confirm these labels through genuine human review before reporting Cohen's kappa or human-vs-automated agreement as final evaluation evidence.

This limitation is explicitly disclosed rather than presenting AI-assisted labels as independent human annotation.


---

## 9. Top Five Failure Modes

### 1. `other` messages collapse into hardware

This was the largest confusion:

**19 examples:** gold `other` → predicted `hardware_device`.

The classifier has strong representation for hardware because hardware-related messages dominate the development data. Short, vague, or novel messages therefore frequently get absorbed into this class.

**Hypothesis:** The `other` class is too heterogeneous and underrepresented, while hardware has much stronger lexical coverage.

**Potential improvement:** Add a novelty/uncertainty detector and require stronger evidence before assigning a specific operational intent.

---

### 2. Apps / services are confused with hardware

There were **18 apps → hardware** errors.

Examples include application crashes, service failures, and other software-level problems where the classifier interprets generic malfunction language as a physical device issue.

**Hypothesis:** Words such as "crash", "broken", "not working", and "freeze" occur in both hardware and software contexts.

**Potential improvement:** Add app/service-specific lexical features and use a hierarchical classifier separating software/service problems from physical-device problems.

---

### 3. Software-update messages are confused with hardware

There were **15 software_update → hardware** errors.

Update-related messages frequently describe freezing, overheating, slowness, or general malfunction.

**Hypothesis:** The classifier relies too heavily on symptom words rather than identifying the causal phrase indicating that the problem began after an update.

**Potential improvement:** Explicitly weight temporal/causal expressions such as "after update", "since iOS", and "after upgrading".

---

### 4. Weak retrieval can look superficially relevant

Several weak-evidence examples had moderate lexical similarity without providing useful resolution evidence.

For example, one battery complaint retrieved another battery complaint with an update-related context, while some other cases retrieved generic support messages.

**Hypothesis:** TF-IDF captures shared words well but does not understand whether two messages describe the same underlying problem or resolution.

**Potential improvement:** Replace or augment TF-IDF retrieval with dense embeddings plus cross-encoder reranking, and impose a stricter evidence threshold.

---

### 5. Safe escalation dramatically reduces automation coverage

The final policy produces only **17 auto-handled cases out of 200**.

This is intentional: the system prioritizes avoiding unsafe automatic responses. However, it means that the agent currently behaves more like an **evidence-aware triage assistant** than a high-coverage autonomous support agent.

**Hypothesis:** The classifier's confidence is not sufficiently calibrated to support broad automation.

**Potential improvement:** Calibrate confidence on a separate validation set and establish action thresholds based on the acceptable false-auto-handle rate rather than maximizing action accuracy.

---

## 10. Avoid External LLM Dependency for the Core Pipeline

**Decision:** Use an API-free deterministic response generator for the submitted pipeline, while retaining the OpenAI-based responder as an optional component.

**Why:** The core evaluation should be reproducible locally without requiring a paid external API. During development, the available OpenAI API account also had insufficient quota to run the intended LLM-based responder and judge reliably.

**Trade-off:** The local responder is less flexible and natural than a capable generative LLM and can produce more templated responses. The submitted evaluation therefore does not claim LLM-generated response quality.


---

## 11. Key Findings

Three conclusions emerge from the evaluation.

**First, retrieval works best when the historical dataset contains a close analogue.** Exact or highly similar historical customer issues provide useful grounding, while generic or unusual messages remain difficult.

**Second, intent classification is currently the main bottleneck.** The final agent's intent accuracy is 44.5%, with particularly poor performance on `other`, `apps_services`, and `repair_support`.

**Third, safety and automation are currently in tension.** Risk-aware escalation improves the safety profile by reducing false auto-handles, but the resulting automation coverage is only 8.5%.

The current system should therefore be positioned as a **conservative support triage and drafting agent**, rather than claiming fully autonomous customer support.

---

## 12. What I Would Do Next Week

### Priority 1 — Improve intent classification

Create a genuinely human-labelled training/validation set covering all 11 intents, especially `other`, `apps_services`, and `repair_support`.

Then compare:

* TF-IDF + Logistic Regression
* sentence embeddings + linear classifier
* calibrated classifier
* hierarchical intent classification

### Priority 2 — Improve retrieval

Move beyond pure TF-IDF retrieval by testing dense embeddings and reranking.

Evaluate retrieval using human judgments of whether the historical interaction actually supports the proposed resolution, rather than relying primarily on similarity.

### Priority 3 — Calibrate escalation

Create a validation set specifically for the auto-handle decision.

Optimize for a target false-auto-handle rate, for example:

> "Auto-handle only when the estimated probability of an unsafe automatic response is below the agreed threshold."

This is preferable to optimizing raw action accuracy.

### Priority 4 — Add a real LLM judge

Run the reply evaluation using an available LLM judge and compare its judgments against independently human-reviewed examples.

Measure:

* reply helpfulness
* factual support
* historical grounding
* intent correctness
* escalation appropriateness

### Priority 5 — Increase automation coverage safely

Once classification, retrieval, and calibration improve, gradually lower the escalation thresholds and measure the resulting safety/coverage trade-off.

The goal should not be maximum automation. The goal should be **maximum safe automation supported by strong historical evidence**.

---

## 13. Overall Conclusion

The project demonstrates an end-to-end support-agent pipeline over real customer-support interactions: data extraction, intent taxonomy, classification, historical retrieval, response drafting, escalation, evaluation, failure analysis, and evidence review.

The strongest result is not the 74.5% action accuracy or 84.6% offline reply score. The more important finding is that **safe escalation substantially reduces false auto-handling, while retrieval quality and intent classification remain the primary limitations**.

The system is therefore a useful prototype for evidence-aware customer-support triage, but further human-labelled data, better retrieval, calibrated confidence, and independent LLM/human reply evaluation are needed before treating it as a high-coverage autonomous support system.
