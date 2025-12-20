# baseline_unconditional.py
import argparse
import pandas as pd
from collections import defaultdict, Counter

LABELS = ["increase", "decrease", "remain constant"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--min_support", type=int, default=20)
    ap.add_argument("--top_rules", type=int, default=30)
    args = ap.parse_args()

    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)

    # Count P(label | feature) on TRAIN (using binary presence: any act => present)
    feat_label = defaultdict(Counter)     # feature_id -> Counter(label)
    feat_count = Counter()

    for (fid, lab), g in train.groupby(["feature_id", "label"]):
        c = g["sample_id"].nunique()
        feat_label[int(fid)][str(lab)] += int(c)

    for fid, g in train.groupby("feature_id"):
        feat_count[int(fid)] = g["sample_id"].nunique()

    # Build rules: f -> argmax_y P(y|f)
    rules = []
    for fid, total in feat_count.items():
        if total < args.min_support:
            continue
        counts = feat_label[fid]
        best = max(LABELS, key=lambda y: counts[y])
        conf = counts[best] / total
        rules.append((fid, best, conf, total))

    rules.sort(key=lambda x: (x[2], x[3]), reverse=True)

    print(f"[Train] rules kept={len(rules)} (min_support={args.min_support})")
    print("[Train] top rules (feature_id -> label, conf, support):")
    for r in rules[:args.top_rules]:
        print("  ", r)

    # Predict on TEST by voting with all present features that have rules
    rule_map = {fid: (lab, conf) for fid, lab, conf, supp in rules}

    # Build per-sample predictions
    # vote score = sum(conf) for predicted label among present features
    pred = {}
    true = {}

    for sid, g in test.groupby("sample_id"):
        lab_true = g["label"].iloc[0]
        true[int(sid)] = lab_true

        scores = {y: 0.0 for y in LABELS}
        for fid in g["feature_id"].unique():
            fid = int(fid)
            if fid in rule_map:
                lab, conf = rule_map[fid]
                scores[lab] += conf

        # fallback: if no rule fires, predict majority class on train
        if all(v == 0.0 for v in scores.values()):
            pred_lab = train["label"].mode().iloc[0]
        else:
            pred_lab = max(LABELS, key=lambda y: scores[y])

        pred[int(sid)] = pred_lab

    # Report accuracy and CF-specific error
    total = len(true)
    correct = sum(pred[sid] == true[sid] for sid in true)

    # Counterfactual split metrics
    meta = test.groupby("sample_id")[["is_counterfactual"]].first()
    cf_ids = [int(sid) for sid, row in meta.iterrows() if bool(row["is_counterfactual"])]
    tr_ids = [int(sid) for sid, row in meta.iterrows() if not bool(row["is_counterfactual"])]

    def acc(ids):
        if not ids:
            return 0.0
        return sum(pred[sid] == true[sid] for sid in ids) / len(ids)

    print(f"[Test] accuracy={correct/total:.3f} (n={total})")
    print(f"[Test] accuracy_true={acc(tr_ids):.3f} (n={len(tr_ids)})")
    print(f"[Test] accuracy_cf  ={acc(cf_ids):.3f} (n={len(cf_ids)})")

if __name__ == "__main__":
    main()