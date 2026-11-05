# LLM Evaluations For Testers

- **Course Name**: LLM Evaluations For Testers
- **Course Code**: LLM-EV-04
- **Price**: 320 USD
- **Duration**: 8 weeks
- **Date**: 15th Jan 2027
- **Time**: 7:00 to 9:30 PM IST
- **Mode**: Live online sessions on Zoom, recordings shared within 24 hours
- **Batch size**: Maximum 20 learners per batch
- **Instructor**: Mukesh Otwani
- **Prerequisites**: Comfortable with Python basics. No machine-learning background needed.

## Who this course is for

QA engineers and test leads who have been handed an LLM or RAG feature and asked
"how do we test this?". The course is about evaluation discipline, not about
training models.

## Why LLM apps break normal testing

Traditional software is deterministic: the same input always produces the same output,
so a test can assert one exact expected value. LLM applications are probabilistic —
the same prompt can produce different wordings, and most questions have many
acceptable answers rather than one correct string. Exact-match assertions therefore
stop working. Instead of asserting equality, LLM evaluation scores *qualities* of the
output, usually against a golden dataset and often with another LLM acting as judge.

## Vibe testing and why it is not enough

Vibe testing means casually trying an application with a handful of prompts and judging
it by feel. It is informal, subjective and not repeatable: the five questions you ask
today are not the five you will ask next month. It is fine for a personal project and
unacceptable as the only gate before production.

## The vocabulary

- **Golden dataset** — a fixed, human-reviewed set of inputs with expected outputs or
  expected contexts. It is the answer key an offline eval scores against.
- **Reference-based eval** — needs a ground-truth answer to compare against. Correctness
  is the classic example.
- **Reference-free eval** — needs no ground truth and instead judges an intrinsic property
  of the output, such as faithfulness, toxicity or answer relevancy.
- **Offline eval** — runs before deployment on a fixed golden dataset in a controlled
  setting. It gates releases.
- **Online eval** — runs after deployment on live production traffic, without an answer
  key, using signals such as thumbs up/down, latency and cost. It monitors real behaviour.
  Offline and online evals are complementary, not rivals.
- **Drift** — the slow change in production inputs or model behaviour that makes a system
  that passed its offline gate start failing in the field. It is the reason online eval
  never ends.
- **LLM as a judge** — using a capable model, with an explicit rubric, to score outputs
  at a scale humans cannot reach. The judge model should be pinned to a specific version,
  because changing the judge changes the scores.

## The three ways to run an eval

1. **Programmatic (deterministic)** — plain code and fixed rules. Cheap, fast, perfectly
   repeatable, but only usable when correctness can be expressed in code.
2. **LLM as a judge** — a model scores the output against a rubric. Handles semantic and
   subjective judgments, costs money and tokens, and carries its own variance.
3. **Human evaluation** — people score outputs. The most trustworthy for nuanced quality,
   and the slowest and most expensive. Used to calibrate the other two.

## The RAG triad

A RAG answer can fail in three different places, so it is measured in three places:

- **Contextual relevancy** — of the chunks the retriever returned, how many were actually
  relevant to the question? In other words, how much of what came back was worth
  retrieving at all.
- **Faithfulness** — of the claims in the generated answer, how many are supported by the
  retrieved context? A low score means the generator hallucinated, which is a generator
  fault, not a retrieval fault.
- **Answer relevancy** — does the answer actually address the question that was asked?
  An answer can be perfectly faithful to the context and still not answer the question.

Retrieval quality is measured separately with **contextual recall** (did the retriever miss
a chunk that was needed?) and **contextual precision** (were the relevant chunks ranked
above the irrelevant ones?).

## Safety and operational metrics

- **Toxicity** — does the output insult, demean, harass or use hateful language? Lower is
  better, so a toxicity test passes when the score stays *below* the threshold.
- **Prompt leakage** — does the bot reveal its hidden system prompt or internal rules?
- **PII leakage** — does the bot repeat API keys, credentials, emails or IDs it should not?
- **Scope adherence** — does the bot stay inside its job, or can a jailbreak turn a course
  assistant into a general-purpose chatbot?
- **Latency** — measured, not judged. Run the pipeline many times, discard warm-up runs,
  and report percentiles. The p95 tail matters far more than the mean, and time-to-first-token
  is what a streaming UI actually feels like.
- **Cost** — derived rather than measured: cost equals tokens times price. It is stable
  run-to-run, which is why it can be estimated offline.
- **Reliability** — success rate, error rate and retry rate. A system can eventually
  succeed on every request and still be unacceptably flaky on the first attempt.

## Syllabus

1. Week 1 — Why LLM testing is different; vibe testing; the eval vocabulary.
2. Week 2 — Building and reviewing a golden dataset; synthetic goldens and their dangers.
3. Week 3 — DeepEval basics: test cases, metrics, thresholds, `evaluate` and `assert_test`.
4. Week 4 — Evaluating the retriever: contextual recall and precision, chunking experiments.
5. Week 5 — Evaluating the generator: faithfulness and answer relevancy in isolation.
6. Week 6 — Application-level quality: correctness, completeness and style with G-Eval rubrics.
7. Week 7 — Safety: toxicity, prompt leakage, PII leakage, scope adherence, red teaming.
8. Week 8 — Operations: latency percentiles and SLOs, cost projection, reliability,
   regression gates in CI, and online monitoring after release.

## Tooling used

Python 3.11, DeepEval, LangChain, Chroma, OpenAI API, pytest, GitHub Actions.

## Projects

- **Project 1**: Author and review a 15-question golden dataset for a supplied RAG chatbot.
- **Project 2**: Build the full RAG triad eval suite and tune chunk size against it.
- **Capstone**: A CI regression gate that blocks a pull request when faithfulness,
  correctness, toxicity or p95 latency crosses its threshold.

## What is included

- 24 live sessions of 2.5 hours each
- Lifetime access to session recordings
- A reference evaluation repository you keep
- Weekly assignments with written feedback
- Private Slack channel for doubts, answered within one business day
- Course completion certificate
