# Decision Log — AppleSupport AI Support Agent

## 1. Brand Selection — AppleSupport

**Decision:** Use AppleSupport as the target brand.

**Why:** AppleSupport has a large number of customer-agent interactions in the dataset, providing enough historical examples for intent discovery, retrieval, and response drafting.

**Trade-off:** Apple-specific issues and terminology may limit generalization to other brands.

---

## 2. Extract Customer-Agent Pairs Through Response IDs

**Decision:** Link AppleSupport response tweets to their parent customer tweets using `in_response_to_tweet_id`.

**Why:** This creates explicit customer → agent interaction pairs rather than treating all tweets as independent text examples.

**Trade-off:** Conversations that cannot be linked through the available response metadata are excluded.

---

## 3. Use a Compact Operational Intent Taxonomy

**Decision:** Use 11 operational intents rather than a large unconstrained topic taxonomy.

**Why:** The agent needs actionable categories that can influence retrieval, response drafting, and escalation.

**Trade-off:** Some nuanced customer problems are forced into broader categories.

---

## 4. Keep `other` as an Explicit Intent

**Decision:** Retain `other / unclear` rather than forcing every message into a known support category.

**Why:** Real support traffic contains vague, conversational, and novel requests.

**Trade-off:** The `other` category is difficult to learn because it contains heterogeneous language. It became one of the largest sources of classification errors.

---

## 5. Reject the Initial Unbalanced Classifier

**Decision:** Do not use the first classifier trained directly on the naturally imbalanced development distribution.

**Why:** The model collapsed toward the dominant hardware/software categories.

**Trade-off:** Balancing the training data changes the effective class distribution and does not perfectly represent production traffic.

---

## 6. Use Balanced Training for the Final Classifier

**Decision:** Build a balanced training subset with up to 3,000 examples per major intent.

**Why:** This gives minority intents substantially more representation and prevents the classifier from simply following the dominant classes.

**Trade-off:** The balanced distribution differs from the real dataset distribution.

---

## 7. Use TF-IDF Retrieval as the Initial Retrieval System

**Decision:** Use TF-IDF cosine similarity over historical customer-agent interactions.

**Why:** It is deterministic, fast, explainable, and can run locally without external APIs.

**Trade-off:** Lexical similarity does not guarantee semantic or resolution-level similarity.

---

## 8. Use a Deterministic 30K Retrieval Index

**Decision:** Build the retrieval index from a deterministic 30,000-example sample using a fixed random seed.

**Why:** The complete dataset is unnecessarily expensive for the evaluation workflow, while 30K examples provide substantial historical coverage.

**Trade-off:** Relevant examples outside the sampled index cannot be retrieved.

---

## 9. Do Not Use the Hybrid Intent Classifier

**Decision:** Keep the experimental hybrid classifier out of the final pipeline.

**Why:** Its sanity checks did not consistently outperform the simpler classifier. In particular, it produced incorrect predictions for some clearly identifiable hardware/software cases.

**Trade-off:** The final system uses a simpler model with known limitations.

---

## 10. Avoid External LLM Dependency for the Core Pipeline

**Decision:** Use an API-free deterministic response generator for the submitted pipeline.

**Why:** The available OpenAI API account did not have sufficient quota, and the assignment should remain reproducible without requiring paid API usage.

**Trade-off:** The local responder is less flexible than a capable generative LLM and can produce templated responses.

---

## 11. Separate Response Generation from Escalation

**Decision:** Treat response drafting and auto-handle/escalate as separate decisions.

**Why:** A useful draft can still be inappropriate to send automatically. High-risk cases may benefit from a drafted response while requiring human review.

**Trade-off:** This increases the number of escalated cases.

---

## 12. Prefer Conservative Escalation

**Decision:** Escalate high-risk intents and cases with low classifier confidence or weak retrieval evidence.

**Why:** An incorrect automatic support response can be more harmful than asking a human agent to review the case.

**Trade-off:** Automation coverage became very low: only 17 of 200 golden examples were auto-handled.

---

## 13. Do Not Optimize Escalation on the Golden Set

**Decision:** Freeze the escalation thresholds after the development experiments.

**Why:** Further threshold tuning on the 200-example golden set would leak evaluation information into the policy and make the reported results less trustworthy.

**Trade-off:** The final thresholds may not be optimal.

---

## 14. Report Safety Metrics Alongside Action Accuracy

**Decision:** Report auto-handle precision, auto-handle recall, false auto-handles, and coverage rather than relying only on action accuracy.

**Why:** The golden set is dominated by escalation cases, making raw action accuracy misleading. The always-escalate baseline actually achieved 78.0% action accuracy.

**Trade-off:** These metrics make the current low automation coverage visible rather than hiding it behind a strong-looking accuracy number.

---

## 15. Do Not Fabricate LLM-as-Judge Results

**Decision:** Report the deterministic offline reply rubric separately and explicitly state that an LLM-as-judge evaluation could not be completed because of API quota limitations.

**Why:** Producing an invented LLM score would undermine the validity of the evaluation.

**Trade-off:** The submission does not fully satisfy the ideal LLM-as-judge component and must transparently document this limitation.
