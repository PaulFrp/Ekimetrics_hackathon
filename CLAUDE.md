# CLAUDE.md — Hallucination Tester

## Project Overview

This project is a **tester** for a hackathon. The context:
- Our team generates a JSON file containing financial data with **15–25% hallucinated entries** (corrupted/false values)
- The opposing team builds an agent (using `openai/gpt-oss-120b`) to detect those hallucinations
- **Our goal**: evaluate how well their model detects our hallucinations, and use that insight to make them harder to detect
- Our score = `1 - F1(opposing team)`, so we want to **minimize their F1**

---

## What This Codebase Does

This repo contains a **hallucination detection tester** that:
1. Takes our biased JSON as input
2. Sends each entry to `openai/gpt-oss-120b` via the OpenAI API
3. Compares GPT's predictions against our ground truth labels
4. Computes **F1 score, precision, recall**
5. Outputs a **summary of missed hallucinations** (false negatives) to identify which types of corruption are hardest to catch

---

## File Structure

```
.
├── CLAUDE.md                  # This file
├── data/
│   └── biased.json            # Input: our hallucinated financial JSON (provided by our team)
├── src/
│   ├── tester.py              # Main script: calls GPT, computes metrics
│   ├── prompt.py              # Prompt engineering for the detection task
│   └── metrics.py             # F1, precision, recall + miss analysis
├── results/
│   └── report.json            # Output: scores + summary of missed hallucinations
└── requirements.txt
```

---

## Input Format

The input `data/biased.json` must follow this structure:

```json
[
  {
    "id": "entry_001",
    "is_hallucinated": true,
    "data": {
      "company": "Acme Corp",
      "revenue": 999999999,
      "..."
    }
  },
  ...
]
```

- `is_hallucinated`: ground truth label (boolean) — **this is what GPT must predict**
- `data`: the financial fields (whatever our team puts in)

---

## Model

```python
_DEFAULT_MODEL = "openai/gpt-oss-120b"
```

Called via the **OpenAI Python SDK** (`openai` package), using the standard `/v1/chat/completions` endpoint.

Set your API key in a `.env` file:
```
OPENAI_API_KEY=sk-...
```

---

## Task Given to GPT

GPT receives each financial entry and is asked:
> *"Is any value in this financial record factually incorrect, implausible, or inconsistent with the other fields? Answer YES or NO, then briefly explain."*

The prompt is in `src/prompt.py` and should be tuned iteratively.

---

## Output

### Console
```
F1 Score:   0.61
Precision:  0.74
Recall:     0.52

Missed hallucinations (false negatives): 14/27
```

### `results/report.json`
```json
{
  "f1": 0.61,
  "precision": 0.74,
  "recall": 0.52,
  "false_negatives": [
    {
      "id": "entry_042",
      "data": { ... },
      "gpt_response": "NO — values appear consistent."
    },
    ...
  ]
}
```

---

## How to Run

```bash
pip install -r requirements.txt
python src/tester.py --input data/biased.json --output results/report.json
```

---

## Key Implementation Notes for Claude Code

- Use `openai` Python SDK, **not** `requests` directly
- Load API key from `.env` via `python-dotenv`
- Process entries **sequentially** (or with a small concurrency limit) to avoid rate limits
- The prompt must **not leak** the `is_hallucinated` field to the model — strip it before sending
- F1 is computed with `sklearn.metrics.f1_score` (binary, positive label = hallucinated)
- Log GPT raw responses to help debug prompt quality

---

## Strategic Goal

Analyze the **false negatives** (hallucinations GPT misses) to understand which corruption patterns are stealthiest. Feed this back to our team to craft harder hallucinations in the next iteration.