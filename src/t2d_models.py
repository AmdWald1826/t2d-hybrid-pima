"""Data loading, models and statistics for the PIMA evaluation.

Models
------
* Majority-class baseline.
* Logistic regression (LR): L2 penalty, C = 1, L-BFGS, inputs standardized
  on the training fold.
* Expert (untuned) fuzzy system: triangular "low"/"high" membership functions
  at clinical cut-points and six Mamdani-type rules (see ``expert_fuzzy``).
* Tuned fuzzy system: zero-order Takagi-Sugeno-Kang (ANFIS-type) with two
  Gaussian membership functions per input (16 rules); its 32 parameters are
  fitted by differential evolution on the training fold only.
* Hybrid: mean of the LR and tuned-fuzzy probabilities.

Statistics
----------
* Nadeau-Bengio corrected variance for repeated cross-validation, used for the
  95% confidence intervals, the corrected resampled t-test and a two one-sided
  tests (TOST) equivalence analysis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

SEED = 2026
COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
        "BMI", "DiabetesPedigree", "Age", "Outcome"]
FEAT4 = ["Glucose", "BloodPressure", "BMI", "Age"]
FEAT8 = COLS[:8]
# A value of 0 is physiologically impossible for these attributes -> missing.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
N_SPLITS, N_REPEATS = 5, 10

DATA = Path(__file__).resolve().parents[1] / "data" / "pima-indians-diabetes.csv"


def load_pima(features=FEAT4, path=DATA):
    """Return X (float, NaN for missing) and y for the requested features."""
    df = pd.read_csv(path, header=None, names=COLS)
    X = df[list(features)].astype(float).copy()
    for c in ZERO_AS_MISSING:
        if c in X:
            X.loc[X[c] == 0, c] = np.nan
    return X.values, df["Outcome"].values.astype(int), df


def impute_median(Xtr, Xte):
    """Median imputation fitted on the training fold only."""
    med = np.nanmedian(Xtr, axis=0)
    A, B = Xtr.copy(), Xte.copy()
    for j in range(A.shape[1]):
        A[np.isnan(A[:, j]), j] = med[j]
        B[np.isnan(B[:, j]), j] = med[j]
    return A, B


# ----------------------------------------------------------------- models
def fit_lr(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)  # L2 penalty (default)
    lr.fit(sc.transform(Xtr), ytr)
    return lambda X: lr.predict_proba(sc.transform(X))[:, 1]


def tri(x, a, b, c):
    """Triangular membership function with feet a, c and peak b."""
    return np.clip(np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)), 0, 1)


# Expert rule base: (low, high) triangular sets for each input of FEAT4.
EXPERT_MF = {
    "Glucose":       {"low": (40, 80, 125), "high": (110, 160, 250)},   # mg/dL
    "BloodPressure": {"low": (30, 60, 90),  "high": (85, 100, 130)},    # mmHg (diastolic in PIMA)
    "BMI":           {"low": (10, 20, 29),  "high": (28, 38, 70)},      # kg/m2
    "Age":           {"low": (15, 25, 42),  "high": (40, 60, 90)},      # years
}


def expert_fuzzy(X):
    """Untuned expert system on FEAT4 (Glucose, BloodPressure, BMI, Age).

    Diabetic rules (max-min):   G_high & BMI_high;  G_high & Age_high;
                                G_high & BP_high & BMI_high.
    Non-diabetic rules:         G_low;  BMI_low & Age_low;  BP_low & BMI_low.
    Output: firing strength of the diabetic rules normalized by the total
    firing strength (centroid of two singletons at 0 and 1).
    """
    g, bp, bmi, age = X.T
    m = {k: {s: tri(v, *EXPERT_MF[k][s]) for s in ("low", "high")}
         for k, v in zip(FEAT4, (g, bp, bmi, age))}
    diab = np.maximum.reduce([
        np.minimum(m["Glucose"]["high"], m["BMI"]["high"]),
        np.minimum(m["Glucose"]["high"], m["Age"]["high"]),
        np.minimum.reduce([m["Glucose"]["high"], m["BloodPressure"]["high"], m["BMI"]["high"]]),
    ])
    norm = np.maximum.reduce([
        m["Glucose"]["low"],
        np.minimum(m["BMI"]["low"], m["Age"]["low"]),
        np.minimum(m["BloodPressure"]["low"], m["BMI"]["low"]),
    ])
    return diab / (diab + norm + 1e-9)


class TunedTSK:
    """Zero-order TSK fuzzy system, 2 Gaussian MFs per input (16 rules).

    Parameters (32): 8 centres, 8 log-widths (standardized inputs) and 16
    rule consequents (sigmoid-mapped to [0, 1]). They minimize the binary
    cross-entropy by differential evolution (population 12 x 32, at most 80
    generations, L-BFGS-B polishing), on the training fold only.
    """

    def __init__(self, seed: int):
        self.seed = seed

    @staticmethod
    def _out(th, Z):
        c = th[:8].reshape(4, 2)
        s = np.exp(th[8:16]).reshape(4, 2)
        w = 1 / (1 + np.exp(-th[16:32]))
        mu = np.exp(-0.5 * ((Z[:, :, None] - c[None]) / s[None]) ** 2)
        idx = np.array(np.meshgrid(*[[0, 1]] * 4, indexing="ij")).reshape(4, -1).T
        fire = np.prod(mu[:, np.arange(4)[None, :], idx], axis=2)
        return (fire @ w) / (fire.sum(1) + 1e-12)

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        lo, hi = [(-1.5, -0.2)] * 4, [(0.2, 1.5)] * 4
        bounds = [b for pair in zip(lo, hi) for b in pair] + [(-1.5, 1.0)] * 8 + [(-6, 6)] * 16

        def loss(th):
            p = np.clip(self._out(th, Z), 1e-6, 1 - 1e-6)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        r = differential_evolution(loss, bounds, seed=self.seed, maxiter=80,
                                   popsize=12, tol=1e-6, polish=True)
        self.th = r.x
        return self

    def predict_proba(self, X):
        return self._out(self.th, self.sc.transform(X))


# ----------------------------------------------------------------- metrics
def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    out = dict(acc=accuracy_score(y, yh), prec=precision_score(y, yh, zero_division=0),
               rec=recall_score(y, yh), f1=f1_score(y, yh, zero_division=0),
               auc=roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5)
    return out, confusion_matrix(y, yh, labels=[0, 1])


def _var_factor(n):
    n_te = n / N_SPLITS
    n_tr = n - n_te
    return 1 / (N_SPLITS * N_REPEATS) + n_te / n_tr


def corrected_ci(x, n, level=0.95):
    """Mean, SD and Nadeau-Bengio corrected confidence interval."""
    x = np.asarray(x)
    J = len(x)
    se = np.sqrt(_var_factor(n) * x.var(ddof=1))
    t = stats.t.ppf(0.5 + level / 2, J - 1)
    return x.mean(), x.std(ddof=1), x.mean() - t * se, x.mean() + t * se


def corrected_ttest(a, b, n):
    """Two-sided corrected resampled t-test on paired split-level scores."""
    d = np.asarray(a) - np.asarray(b)
    v = d.var(ddof=1)
    if v == 0:
        return np.nan
    t = d.mean() / np.sqrt(_var_factor(n) * v)
    return 2 * stats.t.sf(abs(t), len(d) - 1)


def tost(a, b, n, margin):
    """Two one-sided tests for equivalence of mean(a-b) within +/- margin."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    se = np.sqrt(_var_factor(n) * d.var(ddof=1))
    mu = d.mean()
    p = max(stats.t.sf((mu + margin) / se, J - 1), stats.t.cdf((mu - margin) / se, J - 1))
    h = stats.t.ppf(0.95, J - 1) * se
    return dict(diff=mu, ci90_low=mu - h, ci90_high=mu + h, margin=margin, p=p)
