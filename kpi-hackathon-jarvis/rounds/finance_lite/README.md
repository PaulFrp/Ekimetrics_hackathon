# Round `finance_lite` — train set

- **Dataset:** `finance` (`lite`)
- **Train documents:** 4

Each `train/train_NN/` folder is one document:

- `document.txt` — the parsed document text your **red** agent receives.
- `ground_truth.json` — the clean KPIs (`{"kpis": [...]}`), the same
  shape red is handed at run time. The scorer derives each KPI's truth
  from this; use it to practise extraction (blue) and tampering (red).

`manifest.json` maps each folder to its source document id.

> This is **train** data only. The round is scored on a held-out **test**
> document you never see. Don't hard-code answers — generalise.
