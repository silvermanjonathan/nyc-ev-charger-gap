"""Grade the classifier's dev-set labels against the answer key.

Usage (repo folder, venv active):
    python evaluate.py results/dev_v1.csv                              # one run
    python evaluate.py results/dev_v1.csv results/dev_v2.csv           # two runs, plus what changed
    python evaluate.py results/dev_v2.csv results/dev_v2_nocache.csv   # stability check
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEY_PATH = ROOT / "labels" / "dev_set_labels.csv"
LABELS = ["ev_charging", "parking_or_curb", "clean_air", "other"]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["dev_id"]: row for row in csv.DictReader(f)}


def evaluate(key, pred):
    """Return a report as a list of lines. A row with an error counts as wrong."""
    ids = sorted(key)
    missing = [i for i in ids if i not in pred]
    n = len(ids)
    got = {i: pred[i]["pred_label"] if i in pred else "" for i in ids}
    want = {i: key[i]["final_label"] for i in ids}
    got_flag = {i: pred[i]["pred_parking_concern"] if i in pred else "" for i in ids}
    want_flag = {i: key[i]["final_parking_concern"] for i in ids}

    labels_right = sum(got[i] == want[i] for i in ids)
    flags_right = sum(got_flag[i] == want_flag[i] for i in ids)
    lines = [f"Labels right: {labels_right} of {n} ({labels_right / n:.0%})",
             f"Parking flags right: {flags_right} of {n} ({flags_right / n:.0%})"]

    # The go/no-go counts ev_charging, so check both ways it can go wrong.
    tp = sum(got[i] == want[i] == "ev_charging" for i in ids)
    fp = sum(got[i] == "ev_charging" != want[i] for i in ids)
    fn = sum(want[i] == "ev_charging" != got[i] for i in ids)
    lines.append(f"ev_charging: found {tp} of {tp + fn} in the key (missed {fn}); "
                 f"{fp} labeled ev_charging that aren't")

    lines += ["", "Answer key (rows) vs. model (columns)",
              f"{'':17}" + "".join(f"{c[:11]:>12}" for c in LABELS + ['(error)'])]
    for r in LABELS:
        counts = [sum(want[i] == r and got[i] == c for i in ids) for c in LABELS]
        counts.append(sum(want[i] == r and got[i] not in LABELS for i in ids))
        lines.append(f"{r:17}" + "".join(f"{c:>12}" for c in counts))

    wrong = [i for i in ids if got[i] != want[i] or got_flag[i] != want_flag[i]]
    lines += ["", f"Disagreements ({len(wrong)})"]
    for i in wrong:
        p = pred.get(i, {})
        lines.append(f"{i}  key {want[i]}/{want_flag[i]}  model {got[i] or '-'}/{got_flag[i] or '-'}")
        lines.append(f"     {p.get('error') or p.get('reason', 'missing from predictions')}")
    if missing:
        lines.append(f"Missing from predictions: {', '.join(missing)}")
    return lines


def changes(first, second):
    """Rows whose label or flag differs between two runs, as report lines."""
    lines = []
    for i in sorted(set(first) | set(second)):
        a, b = first.get(i, {}), second.get(i, {})
        before = (a.get("pred_label", "-"), a.get("pred_parking_concern", "-"))
        after = (b.get("pred_label", "-"), b.get("pred_parking_concern", "-"))
        if before != after:
            lines.append(f"{i}  {'/'.join(before)}  ->  {'/'.join(after)}")
            lines.append(f"     second run: {b.get('error') or b.get('reason', '')}")
    return [f"Rows that changed between the two runs ({len(lines) // 2})"] + lines


def main():
    paths = sys.argv[1:]
    if len(paths) not in (1, 2):
        sys.exit(__doc__)
    key = read_rows(KEY_PATH)
    for path in paths:
        print(f"=== {path}")
        print("\n".join(evaluate(key, read_rows(path))))
        print()
    if len(paths) == 2:
        print("\n".join(changes(read_rows(paths[0]), read_rows(paths[1]))))


if __name__ == "__main__":
    main()
