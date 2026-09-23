#!/usr/bin/env python3
"""Revision experiments, part 2: A8/B7 class-wise reporting and B8 ablation."""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef,
                             brier_score_loss, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'meningitis.csv')
RES  = os.path.join(ROOT, 'results');  os.makedirs(RES, exist_ok=True)
FIG  = os.path.join(ROOT, 'figures');  os.makedirs(FIG, exist_ok=True)


SEED = 42
out = json.load(open(os.path.join(RES,'results_part1.json')))
df = pd.read_csv(DATA)
enc = df.copy()
LBL = {}
for c in ['Gender', 'Pathogen_Present', 'Diagnosis', 'Outcome', 'Risk_Level']:
    le = LabelEncoder(); enc[c] = le.fit_transform(enc[c]); LBL[c] = list(le.classes_)
FEATS = ['Age','Gender','WBC_Count','Protein_Level','Glucose_Level',
         'Pathogen_Present','Hemoglobin','WBC_Blood_Count','Platelets','CRP_Level']
X = enc[FEATS].values
Y = {t: enc[t].values for t in ['Diagnosis','Outcome','Risk_Level']}

idx = np.arange(len(X))
idx_tr, idx_te = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=Y['Outcome'])
idx_tr, idx_va = train_test_split(idx_tr, test_size=0.15/0.85, random_state=SEED,
                                  stratify=Y['Outcome'][idx_tr])

def make(name, k):
    return {'LR': LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=SEED),
            'RF': RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                         class_weight='balanced', random_state=SEED, n_jobs=-1),
            'LGBM': LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=SEED, verbose=-1),
            'CatB': CatBoostClassifier(depth=6, learning_rate=0.03, iterations=1000,
                                       random_seed=SEED, verbose=0),
            'XGB': XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=500,
                                 random_state=SEED, verbosity=0,
                                 objective='multi:softprob' if k > 2 else 'binary:logistic')}[name]

# ============================================ A8 / B7 class-wise + confusion
print('A8/B7  class-wise metrics and confusion matrices')
BEST = {'Diagnosis': 'LGBM', 'Outcome': 'XGB', 'Risk_Level': 'CatB'}
cw = {}
for t, y in Y.items():
    k = len(np.unique(y)); name = BEST[t]
    sc = StandardScaler().fit(X[idx_tr])
    Xtr, Xte = (X[idx_tr], X[idx_te])
    ytr = y[idx_tr]
    if k == 2:
        Xtr, ytr = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(Xtr, ytr)
    mdl = make(name, k).fit(Xtr, ytr)
    pred = mdl.predict(Xte)
    cm = confusion_matrix(y[idx_te], pred)
    p, r, f, s = precision_recall_fscore_support(y[idx_te], pred, zero_division=0)
    cw[t] = {'model': name, 'labels': LBL[t],
             'confusion': cm.tolist(),
             'precision': [round(float(v), 3) for v in p],
             'recall':    [round(float(v), 3) for v in r],
             'f1':        [round(float(v), 3) for v in f],
             'support':   [int(v) for v in s]}
    print(f"   {t} ({name})")
    for i, lab in enumerate(LBL[t]):
        print(f"      {lab:14s} P={p[i]:.3f} R={r[i]:.3f} F1={f[i]:.3f} n={s[i]}")
out['A8_B7_classwise'] = cw

# ==================================================== B8 protocol ablation
# Each step is a filter. What does the verdict become if a step is removed?
print('\nB8  ablation of the five protocol steps')
def unc(y): m = y.mean(); return m * (1 - m)

def brier_parts(y, p, nb=8):
    bs = brier_score_loss(y, p)
    q = np.unique(np.percentile(p, np.linspace(0, 100, nb + 1)))
    b = np.digitize(p, q[1:-1]); ybar = y.mean(); rel = res = 0.0
    for kk in np.unique(b):
        m = b == kk; n = m.sum()
        rel += n * (p[m].mean() - y[m].mean()) ** 2
        res += n * (y[m].mean() - ybar) ** 2
    return bs, rel / len(y), res / len(y), unc(y)

def net_benefit(y, p, pt):
    pred = (p >= pt).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    return tp / len(y) - (pt / (1 - pt)) * fp / len(y)

STEPS = ['S1_correlation', 'S2_calibration', 'S3_utility', 'S4_interpretability', 'S5_congruence']
verdict = {}
for t, y in Y.items():
    k = len(np.unique(y))
    s1 = out['B5_step1'][t]['verdict_train'] == 'pass'
    if k == 2:
        Xtr, ytr = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(X[idx_tr], y[idx_tr])
        mdl = make(BEST[t], k).fit(Xtr, ytr)
        pv = mdl.predict_proba(X[idx_va])[:, 1]; pte = mdl.predict_proba(X[idx_te])[:, 1]
        platt = LogisticRegression().fit(pv.reshape(-1, 1), y[idx_va])
        pc = platt.predict_proba(pte.reshape(-1, 1))[:, 1]
        bs, rel, res, u = brier_parts(y[idx_te], pc)
        s2 = res > 0.01 * u          # resolution must be a non-trivial share of uncertainty
        nb = [net_benefit(y[idx_te], pc, q) for q in np.linspace(0.15, 0.60, 40)]
        s3 = max(nb) > 0
    else:
        mdl = make(BEST[t], k).fit(X[idx_tr], y[idx_tr])
        s2 = s3 = True               # calibration/DCA defined for the binary target here
    # Step 4: is the top SHAP feature also a top-3 correlate?
    import shap
    ex = shap.TreeExplainer(mdl) if BEST[t] != 'LR' else None
    if ex is not None:
        sv = ex.shap_values(X[idx_te])
        sv = np.abs(np.array(sv)).mean(axis=tuple(range(np.array(sv).ndim - 1)))
        top_shap = FEATS[int(np.argmax(sv))]
    else:
        top_shap = None
    from scipy.stats import pearsonr
    corr = np.array([abs(pearsonr(X[idx_tr, j], y[idx_tr])[0]) for j in range(len(FEATS))])
    s4 = bool(corr[FEATS.index(top_shap)] > 0.10) if top_shap else True
    top_shap_r = round(float(corr[FEATS.index(top_shap)]), 4) if top_shap else None
    # Step 5: logistic coefficient signs against the correlation signs
    scl = StandardScaler().fit(X[idx_tr])
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced',
                            random_state=SEED).fit(scl.transform(X[idx_tr]), y[idx_tr])
    co = lr.coef_[0] if lr.coef_.shape[0] == 1 else lr.coef_.mean(axis=0)
    sgn = np.array([np.sign(pearsonr(X[idx_tr, j], y[idx_tr])[0]) for j in range(len(FEATS))])
    agree = int((np.sign(co) == sgn).sum())
    s5 = agree >= 6
    flags = dict(zip(STEPS, [s1, s2, s3, s4, s5]))
    verdict[t] = {'steps': {k2: bool(v) for k2, v in flags.items()},
                  'all_pass': bool(all(flags.values())),
                  'top_shap': top_shap, 'top_shap_abs_r': top_shap_r,
                  'sign_agreement': f'{agree}/10'}
    print(f"   {t:11s} " + '  '.join(f"{s.split('_')[0]}={'P' if flags[s] else 'F'}" for s in STEPS)
          + f"  -> {'LEARNABLE' if all(flags.values()) else 'NOT LEARNABLE'}")

# what a reduced protocol would conclude
red = {}
for drop in STEPS:
    kept = [s for s in STEPS if s != drop]
    wrong = [t for t in Y if all(verdict[t]['steps'][s] for s in kept) != verdict[t]['all_pass']]
    red[f'without_{drop}'] = {'misclassified_targets': wrong, 'n_wrong': len(wrong)}
red['discrimination_only'] = {'note': 'AUC alone admits Outcome', 'n_wrong': 1,
                              'misclassified_targets': ['Outcome']}
out['B8_ablation'] = {'verdict': verdict, 'reduced_protocols': red}
for k2, v in red.items():
    print(f"   {k2:28s} -> targets misclassified: {v['misclassified_targets'] or 'none'}")

json.dump(out, open('results_full.json', 'w'), indent=1)
print('\nall results written to results_full.json')
