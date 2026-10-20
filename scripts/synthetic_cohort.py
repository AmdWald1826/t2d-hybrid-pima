"""Synthetic ADA-rule cohort (illustration of label leakage) and the
illustrative cost-risk comparison of three decision rules (Table V, Fig. 7).

Cohort (N = 1,000, NumPy default_rng, seed 2026), independent uniform inputs:
    glucose  ~ U(60, 220) mg/dL      diastolic BP ~ U(50, 110) mmHg
    BMI      ~ U(15, 45) kg/m2        age          ~ U(20, 80) years
Label: ADA rule, glucose >= 126 mg/dL, then 5% of labels flipped at random
(label noise, so that the rule is not perfectly recoverable). Glucose is also
an input, hence label leakage: results on this cohort are illustrative only.
No noise is injected into any model prediction.

Evaluation: the same 10 x 5 stratified repeated cross-validation as for PIMA
(seed 2026). Every model or rule is applied to the test folds; trainable parts
(LR, naive Bayes bins and probabilities) are fitted on the training folds only.

Decision rules of Table V
-------------------------
* Deterministic: k = number of criteria met among glucose >= 126 mg/dL,
  diastolic BP >= 90 mmHg, BMI >= 30 kg/m2, age >= 45 years; score = k/4;
  positive if k >= 2.
* Bayesian: naive Bayes on the four inputs discretized into tertiles of the
  training fold, Laplace smoothing, prior = training-fold prevalence;
  score = posterior P(T2D); positive if score >= 0.5.
* Dempster-Shafer: frame {T2D, non-T2D}; each input i is a source with
  m_i({T2D}) = h_i, m_i({non-T2D}) = l_i and m_i(frame) = 1 - h_i - l_i, where
  h_i and l_i are the "high" and "low" memberships of the expert fuzzy system;
  the four sources are combined by Dempster's rule (conflict renormalized);
  score = pignistic P(T2D) = m({T2D}) + m(frame)/2; positive if score >= 0.5.
* Logistic regression (reference), as for PIMA.

Risk level = mean score over the test fold (a mean predicted score, not a
clinical risk). Illustrative annual cost per screened person (scenario
assumptions, not published estimates): a positive prediction triggers a
prevention programme of $750 (medication $420 + monitoring $180 +
consultations $150); a missed case (false negative) incurs one expected
admission of $3,200; a true negative costs $0. Mean cost per person over the
test fold = (750 (TP + FP) + 3200 FN) / n.

Usage:  python scripts/synthetic_cohort.py      (about 1 minute)
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

RES, FIG = ROOT / "results", ROOT / "figures"
N, SEED, FLIP = 1000, 2026, 0.05
C_PREV, C_ADM = 750.0, 3200.0


def make_cohort(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    g = rng.uniform(60, 220, n)
    bp = rng.uniform(50, 110, n)
    bmi = rng.uniform(15, 45, n)
    age = rng.uniform(20, 80, n)
    y = (g >= 126).astype(int)
    flip = rng.choice(n, int(FLIP * n), replace=False)
    y[flip] = 1 - y[flip]
    X = np.column_stack([g, bp, bmi, age])  # same column order as M.FEAT4
    return X, y


def deterministic(X):
    g, bp, bmi, age = X.T
    k = (g >= 126).astype(int) + (bp >= 90) + (bmi >= 30) + (age >= 45)
    return k / 4, (k >= 2).astype(int)


class NaiveBayesTertiles:
    def fit(self, X, y):
        self.cuts = [np.quantile(X[:, j], [1 / 3, 2 / 3]) for j in range(X.shape[1])]
        B = self._bins(X)
        self.prior = y.mean()
        self.lik = {c: [(np.bincount(B[y == c, j], minlength=3) + 1) / ((y == c).sum() + 3)
                        for j in range(X.shape[1])] for c in (0, 1)}
        return self

    def _bins(self, X):
        return np.column_stack([np.digitize(X[:, j], self.cuts[j]) for j in range(X.shape[1])])

    def score(self, X):
        B = self._bins(X)
        l1 = np.log(self.prior) + sum(np.log(self.lik[1][j][B[:, j]]) for j in range(B.shape[1]))
        l0 = np.log(1 - self.prior) + sum(np.log(self.lik[0][j][B[:, j]]) for j in range(B.shape[1]))
        return 1 / (1 + np.exp(l0 - l1))


def dempster_shafer(X):
    t = np.ones(len(X)) * 0.0
    n = np.zeros(len(X))
    th = np.ones(len(X))  # vacuous start: all mass on the frame
    for v, f in zip(X.T, M.FEAT4):
        h, l = M.tri(v, *M.EXPERT_MF[f]["high"]), M.tri(v, *M.EXPERT_MF[f]["low"])
        s = np.maximum(h + l, 1.0)
        h, l = h / s, l / s
        u = 1 - h - l
        conflict = t * l + n * h
        t, n, th = (t * h + t * u + th * h), (n * l + n * u + th * l), th * u
        norm = 1 - conflict
        t, n, th = t / norm, n / norm, th / norm
    return t + th / 2


def cost(y, yhat):
    tp = ((yhat == 1) & (y == 1)).sum()
    fp = ((yhat == 1) & (y == 0)).sum()
    fn = ((yhat == 0) & (y == 1)).sum()
    return (C_PREV * (tp + fp) + C_ADM * fn) / len(y)


def main():
    X, y = make_cohort()
    pd.DataFrame(np.column_stack([X, y]), columns=M.FEAT4 + ["Label_ADA"]).round(2) \
        .to_csv(RES / "synthetic_cohort.csv", index=False)
    rskf = RepeatedStratifiedKFold(n_splits=M.N_SPLITS, n_repeats=M.N_REPEATS, random_state=SEED)
    tab4 = {m: [] for m in ["Majority", "LR", "Fuzzy-expert"]}
    cms = {m: np.zeros((2, 2)) for m in tab4}
    tab5 = {m: [] for m in ["Deterministic", "Bayesian", "Dempster-Shafer", "LR"]}
    oof = pd.DataFrame(index=range(N), columns=["LR", "Fuzzy-expert"], dtype=float)
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        p_lr = M.fit_lr(Xtr, ytr)(Xte)
        p = {"Majority": np.full(len(te), float(ytr.mean() > 0.5)), "LR": p_lr,
             "Fuzzy-expert": M.expert_fuzzy(Xte)}
        for m in tab4:
            mt, cm = M.metrics(yte, p[m])
            tab4[m].append(mt)
            cms[m] += cm
        if k < M.N_SPLITS:
            oof.loc[te, "LR"] = p_lr
            oof.loc[te, "Fuzzy-expert"] = p["Fuzzy-expert"]
        s_det, yh_det = deterministic(Xte)
        s_nb = NaiveBayesTertiles().fit(Xtr, ytr).score(Xte)
        s_ds = dempster_shafer(Xte)
        for m, s, yh in [("Deterministic", s_det, yh_det), ("Bayesian", s_nb, (s_nb >= 0.5).astype(int)),
                         ("Dempster-Shafer", s_ds, (s_ds >= 0.5).astype(int)),
                         ("LR", p_lr, (p_lr >= 0.5).astype(int))]:
            tab5[m].append({"risk": s.mean(), "cost": cost(yte, yh), "acc": (yh == yte).mean()})
    n = len(y)
    rows = []
    for m, v in tab4.items():
        d = pd.DataFrame(v)
        r = {"model": m}
        for key in ["acc", "prec", "rec", "f1", "auc"]:
            mu, sd, lo, hi = M.corrected_ci(d[key].values, n)
            r[key] = f"{100 * mu:.1f} ± {100 * sd:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
        tn, fp, fn, tp = (cms[m] / M.N_REPEATS).ravel()
        r["TN/FP/FN/TP"] = f"{tn:.0f}/{fp:.0f}/{fn:.0f}/{tp:.0f}"
        rows.append(r)
    pd.DataFrame(rows).to_csv(RES / "table_IV_synthetic.csv", index=False)
    t5 = {m: pd.DataFrame(v) for m, v in tab5.items()}
    for m, d in t5.items():
        d.to_csv(RES / f"tableV_splits_{m}.csv", index=False)
    summ = {m: {k: [float(d[k].mean()), float(d[k].std(ddof=1))] for k in d} for m, d in t5.items()}
    pv = {f"{a} vs {b}": M.corrected_ttest(t5[a]["acc"].values, t5[b]["acc"].values, n)
          for a, b in [("Dempster-Shafer", "Bayesian"), ("Dempster-Shafer", "Deterministic"),
                       ("Bayesian", "Deterministic"), ("LR", "Dempster-Shafer")]}
    out = {"prevalence": float(y.mean()), "table_V_mean_sd": summ, "accuracy_p_values": pv}
    (RES / "synthetic_tableV_summary.json").write_text(json.dumps(out, indent=1))
    figures(X, y, oof, t5)
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(out, indent=1))


def figures(X, y, oof, t5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g, bmi = X[:, 0], X[:, 2]
    for name, c, lab in [("fig4_synth_reference", y, "Reference label (ADA rule, 5% noise)"),
                         ("fig5_synth_lr", oof["LR"], "Logistic regression: out-of-fold P(T2D)"),
                         ("fig6_synth_fuzzy", oof["Fuzzy-expert"], "Expert fuzzy system: P(T2D)")]:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        sc = ax.scatter(g, bmi, c=np.asarray(c, float), cmap="coolwarm", vmin=0, vmax=1, s=12,
                        edgecolors="k", linewidths=0.2)
        fig.colorbar(sc, ax=ax).set_label(lab, fontsize=8)
        ax.axvline(126, ls="--", lw=0.8, c="grey")
        ax.set_xlabel("Plasma glucose (mg/dL)")
        ax.set_ylabel("BMI (kg/m$^2$)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        for ext in ("pdf", "png", "eps"):
            fig.savefig(FIG / f"{name}.{ext}", dpi=300)
        plt.close(fig)
    names = ["Deterministic", "Bayesian", "Dempster-Shafer", "LR"]
    labels = ["Determ.", "Bayesian", "DS", "LR (ref.)"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, key, yl, t in zip(axes, ["risk", "cost", "acc"],
                              ["Mean score (risk level)", "Mean cost per person ($)", "Accuracy"],
                              ["(a)", "(b)", "(c)"]):
        ax.boxplot([t5[m][key].values for m in names], tick_labels=labels)
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png", "eps"):
        fig.savefig(FIG / f"fig7_decision_rules.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
