#!/usr/bin/env python3
"""Revision experiments for JCTA 18090.

Answers, with computation rather than prose:
  B5  Step 1 recomputed on the training fold only (the submitted version
      used the full dataset, which let test labels inform the verdict)
  B6  repeated stratified cross-validation instead of one split
  A8/B7 confusion matrices and class-wise precision/recall/F1/support
  B8  ablation: what each protocol step contributes
  A10/B9 nonlinear screening for Step 1 (mutual information, distance
      correlation) to test whether linear correlation rejects a target
      a model could actually learn
"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold,
                                     cross_val_predict, StratifiedKFold)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef,
                             brier_score_loss, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr, pearsonr
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
rng = np.random.default_rng(SEED)
out = {}

df = pd.read_csv(DATA)
enc = df.copy()
for c in ['Gender', 'Pathogen_Present', 'Diagnosis', 'Outcome', 'Risk_Level']:
    enc[c] = LabelEncoder().fit_transform(enc[c])

FEATS = ['Age', 'Gender', 'WBC_Count', 'Protein_Level', 'Glucose_Level',
         'Pathogen_Present', 'Hemoglobin', 'WBC_Blood_Count', 'Platelets', 'CRP_Level']
TARGETS = {'Diagnosis': 'Diagnosis', 'Outcome': 'Outcome', 'Risk_Level': 'Risk_Level'}
X = enc[FEATS].values
Y = {k: enc[v].values for k, v in TARGETS.items()}

# ================================================================== B5
# Step 1 on the training fold only, versus the full dataset as submitted.
print('B5  Step 1 leakage check')
idx = np.arange(len(X))
idx_tr, idx_te = train_test_split(idx, test_size=0.15, random_state=SEED,
                                  stratify=Y['Outcome'])
idx_tr, idx_va = train_test_split(idx_tr, test_size=0.15/0.85, random_state=SEED,
                                  stratify=Y['Outcome'][idx_tr])

step1 = {}
for t, y in Y.items():
    full = [abs(pearsonr(X[:, j], y)[0]) for j in range(len(FEATS))]
    tr   = [abs(pearsonr(X[idx_tr, j], y[idx_tr])[0]) for j in range(len(FEATS))]
    sp   = [abs(spearmanr(X[idx_tr, j], y[idx_tr])[0]) for j in range(len(FEATS))]
    step1[t] = {'max_r_full': round(float(np.max(full)), 4),
                'max_r_train': round(float(np.max(tr)), 4),
                'max_rho_train': round(float(np.max(sp)), 4),
                'argmax_train': FEATS[int(np.argmax(tr))],
                'verdict_full': 'pass' if max(full) > 0.10 else 'fail',
                'verdict_train': 'pass' if max(tr) > 0.10 else 'fail'}
    print(f"   {t:11s} full={step1[t]['max_r_full']:.3f} train={step1[t]['max_r_train']:.3f} "
          f"rho={step1[t]['max_rho_train']:.3f} -> {step1[t]['verdict_train']}")
out['B5_step1'] = step1

# ============================================================== A10/B9
# Does a linear screen reject a target a model could learn? Compare the
# linear screen against mutual information on the same training fold.
print('\nA10/B9  nonlinear screening')
nonlin = {}
for t, y in Y.items():
    mi = mutual_info_classif(X[idx_tr], y[idx_tr], random_state=SEED)
    nonlin[t] = {'max_mi': round(float(np.max(mi)), 4),
                 'mean_mi': round(float(np.mean(mi)), 4),
                 'argmax_mi': FEATS[int(np.argmax(mi))]}
    print(f"   {t:11s} max MI={nonlin[t]['max_mi']:.4f} on {nonlin[t]['argmax_mi']}")

# synthetic control: a target that is a pure XOR interaction of two features,
# invisible to correlation but learnable by a tree
med = np.median(X[:, [0, 2]], axis=0)
xor_y = ((X[:, 0] > med[0]) ^ (X[:, 2] > med[1])).astype(int)
xor_r = max(abs(pearsonr(X[idx_tr, j], xor_y[idx_tr])[0]) for j in range(len(FEATS)))
xor_mi = float(np.max(mutual_info_classif(X[idx_tr], xor_y[idx_tr], random_state=SEED)))
m = LGBMClassifier(random_state=SEED, verbose=-1).fit(X[idx_tr], xor_y[idx_tr])
xor_auc = roc_auc_score(xor_y[idx_te], m.predict_proba(X[idx_te])[:, 1])
nonlin['xor_control'] = {'max_r': round(xor_r, 4), 'max_mi': round(xor_mi, 4),
                         'lgbm_auc': round(float(xor_auc), 4)}
print(f"   XOR control: max|r|={xor_r:.3f} (below threshold) but LightGBM AUC={xor_auc:.3f}")
out['A10_B9_nonlinear'] = nonlin

# ================================================================== B6
# Repeated stratified CV in place of a single split.
print('\nB6  repeated stratified cross-validation (5 folds x 5 repeats)')
def make(name, n_classes):
    if name == 'LR':   return LogisticRegression(C=1.0, max_iter=2000,
                                                 class_weight='balanced', random_state=SEED)
    if name == 'RF':   return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                     class_weight='balanced', random_state=SEED, n_jobs=-1)
    if name == 'LGBM': return LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                             random_state=SEED, verbose=-1)
    if name == 'CatB': return CatBoostClassifier(depth=6, learning_rate=0.03, iterations=1000,
                                                 random_seed=SEED, verbose=0)
    if name == 'XGB':  return XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=500,
                                            random_state=SEED, verbosity=0,
                                            objective='multi:softprob' if n_classes > 2 else 'binary:logistic')

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
cvres = {}
for t, y in Y.items():
    k = len(np.unique(y))
    cvres[t] = {}
    for name in ['LR', 'RF', 'LGBM', 'CatB', 'XGB']:
        aucs, f1s, mccs = [], [], []
        for tr, te in cv.split(X, y):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = (sc.transform(X[tr]), sc.transform(X[te])) if name == 'LR' else (X[tr], X[te])
            ytr = y[tr]
            if k == 2 and np.bincount(ytr).min() > 5:
                Xtr, ytr = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(Xtr, ytr)
            mdl = make(name, k).fit(Xtr, ytr)
            p = mdl.predict_proba(Xte)
            pred = mdl.predict(Xte)
            aucs.append(roc_auc_score(y[te], p[:, 1]) if k == 2
                        else roc_auc_score(y[te], p, multi_class='ovr', average='macro'))
            f1s.append(f1_score(y[te], pred, average='macro'))
            mccs.append(matthews_corrcoef(y[te], pred))
        cvres[t][name] = {'auc_mean': round(float(np.mean(aucs)), 4),
                          'auc_sd': round(float(np.std(aucs)), 4),
                          'f1_mean': round(float(np.mean(f1s)), 4),
                          'f1_sd': round(float(np.std(f1s)), 4),
                          'mcc_mean': round(float(np.mean(mccs)), 4),
                          'mcc_sd': round(float(np.std(mccs)), 4)}
        r = cvres[t][name]
        print(f"   {t:11s} {name:5s} AUC {r['auc_mean']:.3f}+-{r['auc_sd']:.3f}  "
              f"MCC {r['mcc_mean']:.3f}+-{r['mcc_sd']:.3f}")
out['B6_cv'] = cvres

json.dump(out, open('results_part1.json', 'w'), indent=1)
print('\npart 1 written')
