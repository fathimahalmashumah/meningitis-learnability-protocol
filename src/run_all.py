"""Runs every analysis reported in the manuscript and writes results/results.json.

    python src/run_all.py        # analyses (about 5 minutes)
    python src/figures.py        # Figures 2-9 from the saved results
"""
import json, warnings
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix,
                             precision_recall_fscore_support, roc_curve)
from sklearn.feature_selection import mutual_info_classif
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')

import config as C
import protocol as P

R = {'config': {'seed': C.SEED, 'tau': C.TAU, 'mi_threshold': C.MI_THRESHOLD,
                'dca_range': C.DCA_RANGE, 'nb_margin': C.NB_MARGIN,
                's5_min_share': C.S5_MIN_SHARE, 'positive': C.POSITIVE,
                'representative': C.REPRESENTATIVE,
                's5_reference': C.S5_REFERENCE}}
df, X, Y, B, LBL = P.load()
tr, va, te = P.split(Y['Outcome'])
R['split'] = {'train': len(tr), 'val': len(va), 'test': len(te),
              'test_deceased': int(B['Outcome'][te].sum())}
SAVE = {'tr': tr, 'va': va, 'te': te}

# --------------------------------------------- Step 1 on the native targets
R['step1'] = {t: P.step1(X[tr], Y[t][tr]) for t in C.TARGETS}

# exclusive-or control: a target a pairwise screen can nearly miss
med = np.median(X[:, [0, 2]], axis=0)
xor = ((X[:, 0] > med[0]) ^ (X[:, 2] > med[1])).astype(int)
r_x, _, mi_x = P.dependence(X[tr], xor[tr])
auc_x = roc_auc_score(xor[te], LGBMClassifier(random_state=C.SEED, verbose=-1)
                      .fit(X[tr], xor[tr]).predict_proba(X[te])[:, 1])
R['xor_control'] = {'max_abs_r': float(np.abs(r_x).max()), 'max_mi': float(mi_x.max()),
                    'lgbm_test_auc': float(auc_x)}
R['xor_control']['step1_pass'] = P.passes_screen(np.abs(r_x).max(), mi_x.max())
# second control from two nearly independent features (age above its training median, male sex):
# interaction-only structure with no usable univariate association
xor2 = ((X[:, 0] > np.median(X[tr, 0])) ^ (X[:, 1] == 1)).astype(int)
r_2, _, mi_2 = P.dependence(X[tr], xor2[tr])
auc_2 = roc_auc_score(xor2[te], LGBMClassifier(random_state=C.SEED, verbose=-1)
                      .fit(X[tr], xor2[tr]).predict_proba(X[te])[:, 1])
R['xor_control_age_sex'] = {'max_abs_r': float(np.abs(r_2).max()), 'max_mi': float(mi_2.max()),
                            'step1_pass': P.passes_screen(np.abs(r_2).max(), mi_2.max()),
                            'lgbm_test_auc': float(auc_2)}

# --------------------------------------- Table 1: repeated stratified CV
cv = RepeatedStratifiedKFold(n_splits=C.CV_SPLITS, n_repeats=C.CV_REPEATS, random_state=C.SEED)
R['cv'] = {}
for t in C.TARGETS:
    y = Y[t]; k = len(np.unique(y)); R['cv'][t] = {}
    for m in C.MODELS:
        a, f, c = [], [], []
        for i_tr, i_te in cv.split(X, y):
            Xtr, Xte, ytr = X[i_tr], X[i_te], y[i_tr]
            if m == 'LR':
                sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            Xtr, ytr = P.smote_if_binary(Xtr, ytr)
            mo = C.make_model(m, k).fit(Xtr, ytr)
            p = mo.predict_proba(Xte); pr = mo.predict(Xte).ravel()
            a.append(roc_auc_score(y[i_te], p[:, 1]) if k == 2
                     else roc_auc_score(y[i_te], p, multi_class='ovr', average='macro'))
            f.append(f1_score(y[i_te], pr, average='macro'))
            c.append(matthews_corrcoef(y[i_te], pr))
        R['cv'][t][m] = {'auc': float(np.mean(a)), 'auc_sd': float(np.std(a)),
                         'f1': float(np.mean(f)), 'f1_sd': float(np.std(f)),
                         'mcc': float(np.mean(c)), 'mcc_sd': float(np.std(c))}
    print('CV done', t)

# -------------- Table 2 and Fig. 3: class-wise results, representative model
R['classwise'] = {}
for t in C.TARGETS:
    y = Y[t]; k = len(np.unique(y))
    Xtr, ytr = P.smote_if_binary(X[tr], y[tr])
    pr = C.make_model(C.REPRESENTATIVE, k).fit(Xtr, ytr).predict(X[te]).ravel()
    p_, r_, f_, s_ = precision_recall_fscore_support(y[te], pr, zero_division=0)
    R['classwise'][t] = {'model': C.REPRESENTATIVE, 'labels': LBL[t],
                         'confusion': confusion_matrix(y[te], pr).tolist(),
                         'precision': p_.tolist(), 'recall': r_.tolist(),
                         'f1': f_.tolist(), 'support': s_.tolist()}

# ----- Table 3, Figs. 5-6: every model on the mortality contrast (Deceased = 1)
yb = B['Outcome']
Xs, ys = P.smote_if_binary(X[tr], yb[tr])
sc_o = StandardScaler().fit(Xs)
prob = {}
fitted = {}
for m in C.MODELS:
    mo = C.make_model(m)
    if m == 'LR':
        mo.fit(sc_o.transform(Xs), ys); prob[m] = mo.predict_proba(sc_o.transform(X[te]))[:, 1]
    else:
        mo.fit(Xs, ys); prob[m] = mo.predict_proba(X[te])[:, 1]
    fitted[m] = mo
lg = fitted['LightGBM']
platt = LogisticRegression().fit(lg.predict_proba(X[va])[:, 1].reshape(-1, 1), yb[va])
prob['LGBM-Cal'] = platt.predict_proba(prob['LightGBM'].reshape(-1, 1))[:, 1]
R['brier_mortality'] = {m: P.brier_parts(yb[te], p) for m, p in prob.items()}
R['lgbm_cal_range'] = [float(prob['LGBM-Cal'].min()), float(prob['LGBM-Cal'].max())]
ts_plot = np.round(np.arange(0.01, 0.801, 0.005), 3)
R['dca_mortality'] = {m: {'max_excess_in_range': P.step3(yb[te], p)['max_excess_nb'],
                          'min_nb_in_range': float(min(P.net_benefit(yb[te], p, t)
                                                       for t in P.thresholds()))}
                      for m, p in prob.items()}
SAVE.update({f'prob_{m}': p for m, p in prob.items()})
SAVE['y_mort_te'] = yb[te]

# ------------------------------------ Steps 2-5 on the decision contrasts
R['steps'] = {}
for t in C.TARGETS:
    yb = B[t]
    Xs, ys = P.smote_if_binary(X[tr], yb[tr])
    mo = C.make_model(C.REPRESENTATIVE).fit(Xs, ys)
    pl = LogisticRegression().fit(mo.predict_proba(X[va])[:, 1].reshape(-1, 1), yb[va])
    pc = pl.predict_proba(mo.predict_proba(X[te])[:, 1].reshape(-1, 1))[:, 1]
    imp = P.mean_abs_shap(mo, X[te])
    sc, lr = P.fit_audit_lr(X[tr], yb[tr])
    s = {'S1': R['step1'][t], 'S2': P.step2(yb[te], pc), 'S3': P.step3(yb[te], pc),
         'S4': P.step4(imp, X[tr], yb[tr]), 'S5': P.step5(lr.coef_[0], t)}
    s['S4']['importance'] = {f: float(v) for f, v in zip(C.FEATURES, imp)}
    flags = {k: s[k]['pass'] for k in ['S1', 'S2', 'S3', 'S4', 'S5']}
    R['steps'][t] = {'contrast': C.CONTRAST[t], 'flags': flags,
                     'verdict': P.verdict(flags), **s}
    SAVE[f'shap_imp_{t}'] = imp
    if t == 'Outcome':
        import shap
        sv = np.array(shap.TreeExplainer(mo).shap_values(X[te]))
        if sv.ndim == 3: sv = sv[1] if sv.shape[0] == 2 else sv[..., 1]
        p_raw = mo.predict_proba(X[te])[:, 1]
        dec = np.where(yb[te] == 1)[0]
        SAVE.update({'shap_mort': sv, 'p_raw_mort': p_raw,
                     'caseA': dec[np.argmax(p_raw[dec])], 'caseB': dec[np.argmin(p_raw[dec])],
                     'X_te': X[te]})
        R['waterfall'] = {'caseA_p': float(p_raw[SAVE['caseA']]),
                          'caseB_p': float(p_raw[SAVE['caseB']])}
        # integer score on the same audit regression
        S, w, cut, bstar, bound = P.integer_score(lr.coef_[0], X, X[tr])
        St = S[te]; yt = yb[te]
        fpr, tpr, th = roc_curve(yt, St); j = int(np.argmax(tpr - fpr))
        p_lr = lr.predict_proba(sc.transform(X[te]))[:, 1]
        R['integer_score'] = {'weights': dict(zip(C.FEATURES, map(int, w))),
                              'beta_star': bstar, 'sum_abs_beta': float(np.abs(lr.coef_[0]).sum()),
                              'rounding_bound_prob': bound,
                              'auc': float(roc_auc_score(yt, St)), 'cutoff': float(th[j]),
                              'sens': float(tpr[j]), 'spec': float(1 - fpr[j]),
                              'range': [int(St.min()), int(St.max())],
                              'auc_lr_prob': float(roc_auc_score(yt, p_lr)),
                              'auc_lgbm': float(roc_auc_score(yt, p_raw))}
        SAVE.update({'score_te': St, 'p_lr_te': p_lr})
    print('steps done', t, flags)

# ------------------------------------------------------------- ablation
steps = ['S1', 'S2', 'S3', 'S4', 'S5']
full = {t: R['steps'][t]['verdict'] == 'passes' for t in C.TARGETS}
abl = {}
for d in steps:
    keep = [s for s in steps if s != d]
    abl[f'without_{d}'] = [t for t in C.TARGETS
                           if all(R['steps'][t]['flags'][s] for s in keep) != full[t]]
for s in steps:
    abl[f'only_{s}'] = [t for t in C.TARGETS if R['steps'][t]['flags'][s] != full[t]]
abl['discrimination_only'] = [t for t in C.TARGETS
                              if (R['cv'][t][C.REPRESENTATIVE]['auc'] > 0.5) != full[t]]
abl['discrimination_only_any_model'] = [t for t in C.TARGETS
                                        if any(R['cv'][t][m]['auc'] > 0.5 for m in C.MODELS) != full[t]]
R['ablation'] = abl

json.dump(R, open(f'{C.RES}/results.json', 'w'), indent=1)
np.savez(f'{C.RES}/predictions.npz', **SAVE)
print(json.dumps({t: (R['steps'][t]['flags'], R['steps'][t]['verdict']) for t in C.TARGETS}, indent=1))
print('ablation', abl)
