import json
import os
import re
from dotenv import load_dotenv

from openai import OpenAI

_DEFAULT_MODEL = "openai/gpt-oss-120b"
load_dotenv("./..")
# === CONFIGURE PATHS HERE ===
ORIGINAL_PATH = "kpi-hackathon-jarvis/examples/finance_short/example_01/ground_truth.json"
BIASED_PATH = "data/biased.json"
# ============================

SYSTEM_PROMPT = (
    "You are a financial data auditor. You will receive a JSON array of KPI entries. "
    "Each entry has an 'id' and financial fields (name, value, unit, period, scope). "
    "Your task: identify which entries contain factually incorrect, implausible, or internally inconsistent values. "
    "Respond ONLY with a valid JSON object in this exact format: "
    '{"hallucinated_ids": [0, 3, 7, ...]} '
    "Include only the IDs you are confident are hallucinated. No explanation, no markdown, just the JSON."
)


def load_kpis(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["kpis"]


def compute_ground_truth(original: list, biased: list) -> dict[int, bool]:
    """Compare original vs biased entry by entry. Any differing field → hallucinated."""
    original_by_id = {e["id"]: e for e in original}
    labels = {}
    for entry in biased:
        eid = entry["id"]
        orig = original_by_id.get(eid)
        if orig is None:
            # Entry not in original → treat as hallucinated
            labels[eid] = True
        else:
            labels[eid] = any(entry.get(k) != orig.get(k) for k in set(entry) | set(orig) if k != "id")
    return labels


def call_gpt(client: OpenAI, biased: list) -> list:
    user_message = json.dumps(biased, ensure_ascii=False)
    response = client.chat.completions.create(
        model=_DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content
    print(f"\n[GPT raw response]\n{raw}\n")

    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        print("WARNING: Could not parse GPT response as JSON. Assuming no hallucinations detected.")
        return []
    parsed = json.loads(match.group())
    return parsed.get("hallucinated_ids", [])


def compute_metrics(biased: list, ground_truth: dict, detected_ids: list):
    detected_set = set(detected_ids)
    tp_entries, fp_entries, fn_entries = [], [], []

    for entry in biased:
        eid = entry["id"]
        is_hall = ground_truth.get(eid, False)
        detected = eid in detected_set
        if is_hall and detected:
            tp_entries.append(entry)
        elif not is_hall and detected:
            fp_entries.append(entry)
        elif is_hall and not detected:
            fn_entries.append(entry)

    tp, fp, fn = len(tp_entries), len(fp_entries), len(fn_entries)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return tp, fp, fn, precision, recall, f1, fn_entries


def main():
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("No API key found. Set LLM_API_KEY or OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)

    original = load_kpis(ORIGINAL_PATH)
    biased = load_kpis(BIASED_PATH)

    ground_truth = compute_ground_truth(original, biased)
    total_hallucinated = sum(ground_truth.values())
    total = len(biased)

    print(f"Loaded {total} entries. Ground truth: {total_hallucinated} hallucinated, {total - total_hallucinated} correct.")

    # Show which entries are hallucinated according to ground truth
    hall_ids = [eid for eid, v in ground_truth.items() if v]
    print(f"Hallucinated IDs (ground truth): {hall_ids}")

    detected_ids = call_gpt(client, biased)

    tp, fp, fn, precision, recall, f1, fn_entries = compute_metrics(biased, ground_truth, detected_ids)

    print(f"GPT a détecté {tp + fp} entrées comme hallucinated (dont {tp} correctes).")
    print(f"TP: {tp}  |  FP: {fp}  |  FN: {fn}")
    print(f"F1: {f1:.2f}  |  Precision: {precision:.2f}  |  Recall: {recall:.2f}")

    if fn_entries:
        print(f"\nHallucinations manquées (FN) — {len(fn_entries)}/{total_hallucinated}:")
        for e in fn_entries:
            print(f"  - id={e['id']} : {json.dumps(e, ensure_ascii=False)}")
    else:
        print("\nAucune hallucination manquée.")


if __name__ == "__main__":
    main()
