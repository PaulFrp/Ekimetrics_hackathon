<!-- team-banner -->
# Équipe Jarvis

<img src="images/jarvis.webp" alt="Équipe Jarvis" width="200">

---
<!-- /team-banner -->

# Hackathon template — Red vs Blue

You implement either the **red** agent (extracts KPIs from a document and
smuggles hallucinations) or the **blue** agent (judges each KPI as correct
or hallucinated). The orchestrator runs everyone against everyone.

## Setup

```bash
pip install -e ".[dev,llm]"
cp .env.example .env  # add your Groq key (provided on the day)
pytest                # smoke test the baseline pipeline
```

## What to change

- **Red side:** edit `red/submission.py`. Subclass `RedAgent` and bind
  `agent` to your instance. Look at `red/baseline.py` for the contract.
- **Blue side:** edit `blue/submission.py`. Subclass `BlueAgent`.

You can split your code across as many files as you want inside `red/` or
`blue/`. The orchestrator only imports `agent` from `submission.py`.

Shared types and the scorer live in `shared/`. Don't modify them.

## Run a local match

```bash
python scripts/run_match.py examples/finance_short/example_01
```

This runs your red against your blue on a synthetic example and prints the
score breakdown. Use it to debug before pushing.

## The contract

- **Red** outputs a `RedExtraction`: a list of `KPI`s plus a private list of
  `hallucinated_ids`. Quota: 15% to 25% of extracted KPIs must be
  hallucinated. The blue never sees your hallucination labels.
- **Blue** outputs a `BlueJudgment`: a binary verdict (`correct` or
  `hallucinated`) for every KPI id the red emitted.
- **Scoring:** for each verdict that disagrees with red's true label, blue
  loses 1 point. Final blue score = `1 − errors / kpis`. Red score =
  `1 − blue_score`. Soft penalties on red for quota violations or lying
  about its labels (see `shared/scoring.py`).

## Hallucination taxonomy

Red must label each fake KPI with one of:

| Type         | Example                                                       |
| ------------ | ------------------------------------------------------------- |
| `value`      | same identity, wrong number — Revenue 4.2 B€ instead of 3.8 B€ |
| `metadata`   | right value, wrong context — wrong unit (450 M$ vs 450 M€), scope (Europe vs Group), or period (2022 vs 2023) |
| `fabricated` | a KPI that doesn't exist in the document                      |
