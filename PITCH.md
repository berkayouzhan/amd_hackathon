# 🚀 Hackathon Submission Pitch: Adaptive Model Dispatcher

This document contains a structured slide-by-slide copy template that you can use for your final submission slide deck or text summary on `lablab.ai`.

---

## 📌 Slide 1: Cover / Hook
* **Title:** Adaptive Model Dispatcher (AMD)
* **Subtitle:** Intelligent, Token-Efficient Multi-Model Task Orchestrator
* **Tagline:** Maximizing LLM-Judge accuracy while squeezing token consumption to near-zero.
* **Problem Statement:** In Track 1, entries are sorted by **total token count** only *after* passing the strict **Accuracy Gate**. Traditional single-model solutions either consume too many tokens or fail accuracy checks on specialized tasks (like coding or complex logic).

---

## 📌 Slide 2: The Core Innovation
We built a **5-layer sequential pipeline** that intercepts, optimizes, and routes tasks to the cheapest correct path.

1. **Prompt Compression (🗜️):** Deterministically strips redundant spaces and blank lines, saving up to **15% in input tokens** while preserving code block structures.
2. **Tier-0 Deterministic Solver (⚡):** Safe Python parser that intercepts math, percentages, powers, and unit conversions. Solves them instantly with **100% accuracy at 0 token cost**.
3. **Smart Triage (🏷️):** A two-tier classifier. High-accuracy regex heuristics identify categories at 0 token cost. Ambiguous cases fallback to a fast model.
4. **Optimal Model Dispatch (🤖):** Routes coding to Kimi K2.7 Code, general tasks to MiniMax M3, and uses local **Qwen 2.5 1.5B** inside the container for sentiment/NER to bypass Fireworks costs entirely.
5. **Speculative Validation (✅):** Catches refusals, truncated strings, or format errors locally, triggering a precise formatting retry if necessary.

---

## 📌 Slide 3: Tier-0 Deterministic Solver & Regex Triage
*Why waste premium API tokens on math problems that Python can solve instantly?*

* **Arithmetic & Math:** safe Abstract Syntax Tree (AST) parser evaluates mathematical formulas.
* **Unit Conversion:** 30+ conversion pairs (lengths, weights, time, volume) handled natively.
* **Regex Fast Triage:** Matches 20+ query structures (e.g., `"what is 15% of 200"`, `"generate a script..."`, `"condense this text..."`) to bypass triage model costs entirely.

---

## 📌 Slide 4: Google Gemma 4 Strategy & Circuit-Breaker
*To maximize points and claim the Gemma 4 Bonus prize:*

* **Try Gemma First:** Factual knowledge, sentiment, NER, and summarization tasks attempt Gemma 4 first.
* **Dynamic Circuit-Breaker:** Since serverless Gemma isn't guaranteed on Fireworks, a single timeout or connection error will trigger the **Circuit-Breaker**. Subsequent tasks instantly fallback to serverless MiniMax M3, protecting the critical 10-minute container execution deadline.

---

## 📌 Slide 5: Built-in SaaS Monitoring Dashboard
We designed and integrated a professional, dark-mode monitoring dashboard:
* **Interactive Slide Deck:** A built-in presentation walking judges through the pipeline architecture directly in the browser.
* **Granular Token Metrics:** Track tokens spent, latency, and model calls dynamically.
* **Drag-and-Drop Analysis:** Drop any `run_report.json` file onto the dashboard to visualize pipeline runs instantly.
* **150+ Automated Tests:** Solid software engineering with 100% test pass rate ensuring stability.
