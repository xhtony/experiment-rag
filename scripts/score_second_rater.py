"""Score the second-rater workbooks against the researcher key.

Works with a combined key (both phases) or a per-round key (one phase).

Usage:
    python score_second_rater.py --key RESEARCHER_KEY_do_not_send.xlsx \
        --part-a SecondRater_PartA_blind_<name>.xlsx \
        [--part-b SecondRater_PartB_goldcheck_<name>.xlsx] \
        --out second_rater_report.xlsx

Requires pandas, numpy and openpyxl. Reports, separately for each phase:
  * raw agreement and Cohen's kappa (95% bootstrap CI) on the three-way action
    (ANSWER / CLARIFY / REFUSE);
  * Phase 1 also on the binary answer-vs-abstain label, because the Phase 1 gold
    label is binary (expect_refusal): CLARIFY and REFUSE both count as abstain;
  * the same excluding designed-ambiguous items, plus agreement by qtype;
  * confusion matrices;
  * evidence coverage: on items both raters label ANSWER, the share of gold
    evidence groups covered by the rater's chunk ids (a group is covered when
    any of its ids is cited);
  * Part B: the rater's verdict on the researcher's labels;
  * a disagreement list with an empty column for the adjudication decision.
"""

import argparse
import json
import re

import numpy as np
import pandas as pd

ACTIONS = ["ANSWER", "CLARIFY", "REFUSE"]
QTYPES = ["single_doc", "multi_doc", "conditional", "ambiguous", "unanswerable"]


def kappa(a, b, labels):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return None if pe == 1 else (po - pe) / (1 - pe)


def boot_ci(a, b, labels, reps=2000, seed=7):
    rng = np.random.default_rng(seed)
    a, b = np.array(a), np.array(b)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, len(a), len(a))
        k = kappa(a[idx], b[idx], labels)
        if k is not None:
            vals.append(k)
    return (round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)) if vals else (None, None)


def agree_row(name, d, gcol, rcol, labels):
    if d.empty:
        return dict(subset=name, n=0)
    k = kappa(d[gcol], d[rcol], labels)
    lo, hi = boot_ci(d[gcol], d[rcol], labels)
    return dict(subset=name, n=len(d), agreement=round(float((d[gcol] == d[rcol]).mean()), 3),
                kappa=None if k is None else round(k, 3), kappa_ci95_low=lo, kappa_ci95_high=hi)


def read_part_a(path):
    df = pd.read_excel(path, sheet_name="Rating")
    df = df.rename(columns={"Item": "item", "Action": "rater_action", "Evidence chunk ids": "rater_evidence",
                            "Short answer / missing information / reason": "rater_text",
                            "Confidence": "rater_confidence", "Notes": "rater_notes"})
    df["rater_action"] = df["rater_action"].astype(str).str.strip().str.upper().replace({"NAN": np.nan})
    bad = df[df.rater_action.notna() & ~df.rater_action.isin(ACTIONS)]
    if len(bad):
        raise SystemExit(f"Unrecognised actions in Part A: {bad[['item', 'rater_action']].to_dict('records')}")
    return df[["item", "rater_action", "rater_evidence", "rater_text", "rater_confidence", "rater_notes"]]


def coverage(groups_json, evidence):
    groups = json.loads(groups_json) if isinstance(groups_json, str) else []
    if not groups:
        return np.nan
    cited = set(re.findall(r"[a-z0-9-]+::\d{4}", str(evidence).lower()))
    return sum(any(c in cited for c in g) for g in groups) / len(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--part-a", required=True)
    ap.add_argument("--part-b")
    ap.add_argument("--out", default="second_rater_report.xlsx")
    args = ap.parse_args()

    key = pd.read_excel(args.key, sheet_name="Key")
    d = key.merge(read_part_a(args.part_a), on="item", how="left")
    missing = d[d.rater_action.isna()]
    if len(missing):
        print(f"WARNING: {len(missing)} items without a rater action are excluded: {', '.join(missing.item)}")
    d = d[d.rater_action.notna()].copy()
    d["rater_abstain"] = (d.rater_action != "ANSWER").astype(int)
    d["agree_action"] = (d.gold_action == d.rater_action).astype(int)
    d["agree_binary"] = (d.gold_abstain == d.rater_abstain).astype(int)
    phases = [p for p in ["Phase 2", "Phase 1"] if p in set(key.phase)]   # one or both phases per key
    d["evidence_coverage"] = [coverage(g, e) if (ga == "ANSWER" and ra == "ANSWER") else np.nan
                              for g, e, ga, ra in zip(d.gold_evidence_groups, d.rater_evidence, d.gold_action, d.rater_action)]

    summary = []
    for ph in phases:
        x = d[d.phase == ph]
        na = x[x.qtype != "ambiguous"]
        summary.append({"phase": ph, **agree_row("three-way action, all items", x, "gold_action", "rater_action", ACTIONS)})
        summary.append({"phase": ph, **agree_row("three-way action, excluding ambiguous", na, "gold_action", "rater_action", ACTIONS)})
        if ph == "Phase 1":
            summary.append({"phase": ph, **agree_row("binary answer vs abstain (Phase 1 gold label), all items", x, "gold_abstain", "rater_abstain", [0, 1])})
            summary.append({"phase": ph, **agree_row("binary answer vs abstain, excluding ambiguous", na, "gold_abstain", "rater_abstain", [0, 1])})
        cov = x.evidence_coverage.dropna()
        summary.append({"phase": ph, "subset": "evidence coverage of gold groups (both ANSWER)", "n": int(len(cov)),
                        "agreement": round(float(cov.mean()), 3) if len(cov) else None})
    S = pd.DataFrame(summary)

    by_q = []
    for ph in phases:
        for q in QTYPES:
            x = d[(d.phase == ph) & (d.qtype == q)]
            if len(x):
                by_q.append(dict(phase=ph, qtype=q, n=len(x), gold_action=x.gold_action.iloc[0],
                                 action_agreement=round(x.agree_action.mean(), 3),
                                 binary_agreement=round(x.agree_binary.mean(), 3) if ph == "Phase 1" else None,
                                 rater_actions={k: int(v) for k, v in x.rater_action.value_counts().items()}))
    BQ = pd.DataFrame(by_q)
    BQ["rater_actions"] = BQ.rater_actions.astype(str)

    cms = []
    for ph in phases:
        x = d[d.phase == ph]
        cm = pd.crosstab(x.gold_action, x.rater_action).reindex(index=ACTIONS, columns=ACTIONS, fill_value=0)
        cm.index = [f"{ph} gold {i}" for i in cm.index]
        cms.append(cm)
    CM = pd.concat(cms)
    CM.columns = [f"rater {c}" for c in CM.columns]
    CM.index.name = "gold \\ rater"

    dis = d[(d.agree_action == 0) | (d.evidence_coverage < 1)].copy()
    dis["disagreement"] = np.where(dis.agree_action == 0, "action", "evidence incomplete")
    dis["adjudication_decision"] = ""
    dis["adjudication_note"] = ""
    DIS = dis[["item", "phase", "question_id", "qtype", "disagreement", "gold_action", "rater_action", "gold_chunk_ids",
               "rater_evidence", "evidence_coverage", "rater_text", "rater_confidence", "rater_notes", "gold_answer",
               "adjudication_decision", "adjudication_note"]].sort_values(["phase", "disagreement", "question_id"])

    sheets = {"Summary": S, "By_qtype": BQ, "Confusion": CM, "Disagreements": DIS, "Item_level": d}
    if args.part_b:
        b = pd.read_excel(args.part_b, sheet_name="Gold_Check").rename(columns={
            "Item": "item", "1 Action correct?": "b_action", "2 Answer supported?": "b_answer",
            "3 Evidence complete?": "b_evidence", "Comment": "b_comment"})
        b = key[["item", "phase", "question_id", "qtype"]].merge(b[["item", "b_action", "b_answer", "b_evidence", "b_comment"]], on="item")
        rows = []
        for ph in phases:
            x = b[b.phase == ph]
            for col, lab in (("b_action", "intended action correct"), ("b_answer", "reference answer supported"), ("b_evidence", "evidence complete")):
                vc = x[col].astype(str).str.strip().str.lower().value_counts()
                rows.append(dict(phase=ph, check=lab, n_rated=int(x[col].notna().sum()), **{k: int(v) for k, v in vc.items() if k != "nan"}))
        sheets["PartB_summary"] = pd.DataFrame(rows).fillna(0)
        sheets["PartB_flags"] = b[b[["b_action", "b_answer", "b_evidence"]].astype(str).apply(
            lambda s: s.str.lower().isin(["no", "debatable", "partly"])).any(axis=1)]

    with pd.ExcelWriter(args.out) as w:
        for name, frame in sheets.items():
            frame.to_excel(w, sheet_name=name, index=(name == "Confusion"))
    pd.set_option("display.width", 200)
    print(S.to_string(index=False))
    print(f"\n{len(DIS)} items to adjudicate. Report written to {args.out}")


if __name__ == "__main__":
    main()
