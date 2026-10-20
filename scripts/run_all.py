"""Regenerate every number of the PIMA evaluation (Section VI-A, Table IV and the checks in the text).

Usage:  python scripts/run_all.py            (about 15-30 min, DE tuning dominates)
Outputs: results/*.csv, results/*.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import t2d_models as M  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
MODELS = ["Majority", "LR", "Fuzzy-expert", "Fuzzy-tuned", "Hybrid"]
KEYS = ["acc", "prec", "rec", "f1", "auc"]


def splits(X, y):
    return list(RepeatedStratifiedKFold(n_splits=M.N_SPLITS, n_repeats=M.N_REPEATS,
                                        random_state=M.SEED).split(X, y))


def main_cv():
    X, y, _ = M.load_pima(M.FEAT4)
    n = len(y)
    res = {m: [] for m in MODELS}
    cms = {m: np.zeros((2, 2)) for m in MODELS}
    for k, (tr, te) in enumerate(splits(X, y)):
        Xtr, Xte = M.impute_median(X[tr], X[te])
        p = {"Majority": np.full(len(te), float(y[tr].mean() > 0.5)),
             "LR": M.fit_lr(Xtr, y[tr])(Xte),
             "Fuzzy-expert": M.expert_fuzzy(Xte),
             "Fuzzy-tuned": M.TunedTSK(M.SEED + k).fit(Xtr, y[tr]).predict_proba(Xte)}
        p["Hybrid"] = 0.5 * (p["LR"] + p["Fuzzy-tuned"])
        for m in MODELS:
            mt, cm = M.metrics(y[te], p[m])
            res[m].append(mt)
            cms[m] += cm
        print(f"split {k + 1}/50 done", flush=True)
    res = {m: pd.DataFrame(v) for m, v in res.items()}
    for m, d in res.items():
        d.to_csv(RES / f"splits_{m}.csv", index=False)
    rows = []
    for m in MODELS:
        r = {"model": m}
        for key in KEYS:
            mu, sd, lo, hi = M.corrected_ci(res[m][key].values, n)
            r[key] = f"{100 * mu:.1f} ± {100 * sd:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
        tn, fp, fn, tp = (cms[m] / M.N_REPEATS).ravel()
        r["TN/FP/FN/TP"] = f"{tn:.0f}/{fp:.0f}/{fn:.0f}/{tp:.0f}"
        rows.append(r)
    pd.DataFrame(rows).to_csv(RES / "table_IV_pima.csv", index=False)
    pairs = [("LR", "Majority"), ("Fuzzy-expert", "Majority"), ("Fuzzy-tuned", "Majority"),
             ("Hybrid", "Majority"), ("LR", "Fuzzy-expert"), ("LR", "Fuzzy-tuned"),
             ("Hybrid", "LR"), ("Hybrid", "Fuzzy-tuned")]
    pv = {f"{a} vs {b}": {k: M.corrected_ttest(res[a][k].values, res[b][k].values, n)
                          for k in ["acc", "f1", "auc"]} for a, b in pairs}
    eq = {f"{k}_pm{int(100 * mg)}": M.tost(res["Fuzzy-tuned"][k].values, res["LR"][k].values, n, mg)
          for k in ["acc", "auc"] for mg in (0.03, 0.05)}
    return res, pv, eq


def lr_only(features, labels=None):
    X, y, _ = M.load_pima(features)
    if labels is not None:
        y = labels
    out = []
    for tr, te in splits(X, y):
        Xtr, Xte = M.impute_median(X[tr], X[te])
        out.append(M.metrics(y[te], M.fit_lr(Xtr, y[tr])(Xte))[0])
    return pd.DataFrame(out)


if __name__ == "__main__":
    X, y, df = M.load_pima(M.FEAT4)
    n = len(y)
    res, pv, eq = main_cv()
    lr4 = res["LR"]
    lr8 = lr_only(M.FEAT8)
    noglu = lr_only(["BloodPressure", "BMI", "Age"])
    Ximp, _ = M.impute_median(X, X)
    y_ada = (Ximp[:, 0] >= 126).astype(int)  # leakage check: label = ADA glucose rule
    ada = lr_only(M.FEAT4, labels=y_ada)

    def summ(d):
        return {k: [round(100 * v, 1) for v in M.corrected_ci(d[k].values, n)] for k in KEYS}

    out = {
        "p_values": pv,
        "tost_tuned_fuzzy_minus_lr": eq,
        "lr_8_features": summ(lr8),
        "lr8_vs_lr4_p": {k: M.corrected_ttest(lr8[k].values, lr4[k].values, n) for k in ["acc", "auc"]},
        "lr_without_glucose": summ(noglu),
        "leakage_check_ada_labels_lr": summ(ada),
        "ada_label_prevalence": float(y_ada.mean()),
    }
    (RES / "summary.json").write_text(json.dumps(out, indent=1))
    print(pd.read_csv(RES / "table_IV_pima.csv").to_string(index=False))
    print(json.dumps(out, indent=1))
