"""The five validation steps, each implemented once.

Every verdict, the ablation and every figure call these functions, so a rule
cannot differ between analyses.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import brier_score_loss
from imblearn.over_sampling import SMOTE

import config as C


# ------------------------------------------------------------- data and split
def load():
    df = pd.read_csv(C.DATA)
    enc = df.copy()
    labels = {}
    for c in ['Gender', 'Pathogen_Present'] + C.TARGETS:
        le = LabelEncoder()
        enc[c] = le.fit_transform(enc[c])
        labels[c] = list(le.classes_)
    # Every feature is recorded as an integer and the matrix is kept integer-typed,
    # so SMOTE's synthetic values (Eq. 5) are rounded down to integers.
    X = enc[C.FEATURES].values
    assert X.dtype.kind == 'i'
    Y = {t: enc[t].values for t in C.TARGETS}                       # native targets
    B = {t: (df[t] == C.POSITIVE[t]).astype(int).values for t in C.TARGETS}  # decision contrasts
    assert labels['Gender'] == ['Female', 'Male'] and labels['Pathogen_Present'] == ['No', 'Yes']
    return df, X, Y, B, labels


def split(y_outcome):
    idx = np.arange(len(y_outcome))
    tr, te = train_test_split(idx, test_size=C.TEST_SIZE, random_state=C.SEED,
                              stratify=y_outcome)
    tr, va = train_test_split(tr, test_size=C.VAL_SIZE / (1 - C.TEST_SIZE),
                              random_state=C.SEED, stratify=y_outcome[tr])
    return tr, va, te


def smote_if_binary(X, y):
    if len(np.unique(y)) == 2 and np.bincount(y).min() > 5:
        return SMOTE(k_neighbors=5, random_state=C.SEED).fit_resample(X, y)
    return X, y


# ------------------------------------------------------------------- Step 1
def dependence(X, y):
    """Pearson r, Spearman rho and mutual information of every feature with y."""
    r = np.array([pearsonr(X[:, j], y)[0] for j in range(X.shape[1])])
    rho = np.array([spearmanr(X[:, j], y)[0] for j in range(X.shape[1])])
    mi = mutual_info_classif(X, y, random_state=C.SEED)
    return r, rho, mi


def passes_screen(abs_r, mi):
    """The Step 1 rule, Eq. (6). Used by Step 1 and by Step 4."""
    return bool(abs_r > C.TAU or mi > C.MI_THRESHOLD)


def step1(X_tr, y_tr):
    r, rho, mi = dependence(X_tr, y_tr)
    a = np.abs(r)
    return {'max_abs_r': float(a.max()), 'argmax_r': C.FEATURES[int(a.argmax())],
            'max_abs_rho': float(np.abs(rho).max()),
            'max_mi': float(mi.max()), 'argmax_mi': C.FEATURES[int(mi.argmax())],
            'pass': passes_screen(a.max(), mi.max())}


# ------------------------------------------------------------------- Step 2
def brier_parts(y, p, n_bins=C.N_BINS):
    q = np.unique(np.percentile(p, np.linspace(0, 100, n_bins + 1)))
    b = np.digitize(p, q[1:-1])
    ybar = y.mean()
    rel = res = 0.0
    for k in np.unique(b):
        m = b == k
        rel += m.sum() * (p[m].mean() - y[m].mean()) ** 2
        res += m.sum() * (y[m].mean() - ybar) ** 2
    bs = brier_score_loss(y, p)
    unc = ybar * (1 - ybar)
    return {'BS': float(bs), 'REL': float(rel / len(y)), 'RES': float(res / len(y)),
            'UNC': float(unc), 'BS_minus_UNC': float(bs - unc), 'BSS': float(1 - bs / unc)}


def step2(y, p_cal):
    d = brier_parts(y, p_cal)
    d['pass'] = bool(d['BS'] < d['UNC'])
    return d


# ------------------------------------------------------------------- Step 3
def net_benefit(y, p, pt):
    pred = p >= pt
    n = len(y)
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return tp / n - pt / (1 - pt) * fp / n


def nb_treat_all(y, pt):
    prev = y.mean()
    return prev - pt / (1 - pt) * (1 - prev)


def thresholds():
    lo, hi = C.DCA_RANGE
    return np.round(np.arange(lo, hi + 1e-9, C.DCA_STEP), 2)


def step3(y, p_cal):
    ts = thresholds()
    nb = np.array([net_benefit(y, p_cal, t) for t in ts])
    ref = np.maximum(np.array([nb_treat_all(y, t) for t in ts]), 0.0)
    excess = nb - ref
    k = int(np.argmax(excess))
    return {'max_excess_nb': float(excess[k]), 'at_threshold': float(ts[k]),
            'pass': bool(excess[k] >= C.NB_MARGIN)}


# ------------------------------------------------------------------- Step 4
def mean_abs_shap(model, X):
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = np.abs(np.array(sv))
    if sv.ndim == 3:                     # (classes, n, p) or (n, p, classes)
        sv = sv.mean(axis=0) if sv.shape[0] != X.shape[0] else sv.mean(axis=2)
    return sv.mean(axis=0)


def step4(importance, X_tr, yb_tr):
    r, _, mi = dependence(X_tr, yb_tr)
    j = int(np.argmax(importance))
    return {'top_feature': C.FEATURES[j], 'top_abs_r': float(abs(r[j])),
            'top_mi': float(mi[j]), 'pass': passes_screen(abs(r[j]), mi[j])}


# ------------------------------------------------------------------- Step 5
def fit_audit_lr(X_tr, yb_tr):
    sc = StandardScaler().fit(X_tr)
    lr = C.make_model('LR').fit(sc.transform(X_tr), yb_tr)
    return sc, lr


def sign_audit(coef, reference):
    rows, agree = [], 0
    for f, d in reference.items():
        s = int(np.sign(coef[C.FEATURES.index(f)]))
        ok = s == d
        agree += ok
        rows.append({'feature': f, 'expected': d, 'fitted_coef': float(coef[C.FEATURES.index(f)]),
                     'agrees': bool(ok)})
    return rows, agree, len(reference)


def step5(coef, target):
    rows, agree, n = sign_audit(coef, C.S5_REFERENCE[target])
    ref_all = {**C.S5_REFERENCE[target], **C.S5_SINGLE_SOURCE[target]}
    _, agree_all, n_all = sign_audit(coef, ref_all)
    return {'agree': agree, 'scored': n, 'share': agree / n,
            'pass': bool(agree / n >= C.S5_MIN_SHARE), 'audit': rows,
            'coef': {f: float(c) for f, c in zip(C.FEATURES, coef)},
            'sensitivity_single_source': {'agree': agree_all, 'scored': n_all,
                                          'pass': bool(agree_all / n_all >= C.S5_MIN_SHARE)}}


# ----------------------------------------------- integer score (Step 5 support)
def integer_score(coef, X, X_tr):
    """Integer weights w_j = round(K beta_j / beta*), applied to indicators that are
    x_j above its training-fold 75th percentile (continuous) or x_j = 1 (binary)."""
    bstar = np.abs(coef).max()
    w = np.round(C.SCORE_K * coef / bstar).astype(int)
    cut = np.percentile(X_tr, C.SCORE_PERCENTILE, axis=0)
    ind = np.zeros_like(X, dtype=int)
    for j, f in enumerate(C.FEATURES):
        ind[:, j] = (X[:, j] == 1) if f in C.BINARY_FEATURES else (X[:, j] > cut[j])
    bound = len(coef) * bstar / (8 * C.SCORE_K)     # Corollary 5, probability scale
    return (ind * w).sum(axis=1), w, cut, float(bstar), float(bound)


def verdict(flags):
    return 'passes' if all(flags.values()) else 'does not pass'
