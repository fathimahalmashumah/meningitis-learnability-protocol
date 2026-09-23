#!/usr/bin/env python3
"""Definitive revision run for JCTA 18090.

Design decision taken here: Steps 2, 3 and 5 (Brier decomposition, net
benefit, integer score and coefficient signs) are defined for a binary
decision. Rather than leave them undefined for the two three-class targets,
each target is reduced to the binary contrast that carries the clinical
decision:

    Diagnosis   -> Bacterial vs. non-bacterial   (start antibiotics or not)
    Risk Level  -> High risk vs. not high risk   (escalate care or not)
    Outcome     -> Deceased vs. recovered        (already binary)

Multiclass discrimination is still reported for the two three-class targets,
so nothing is lost; the binary contrast is used only where the step requires
one. This makes all five steps well defined for all three targets.
"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef,
                             brier_score_loss, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr, spearmanr
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import shap
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'meningitis.csv')
RES  = os.path.join(ROOT, 'results');  os.makedirs(RES, exist_ok=True)
FIG  = os.path.join(ROOT, 'figures');  os.makedirs(FIG, exist_ok=True)


SEED = 42
TAU  = 0.10
R = {}

df  = pd.read_csv(DATA)
enc = df.copy(); LBL = {}
for c in ['Gender','Pathogen_Present','Diagnosis','Outcome','Risk_Level']:
    le = LabelEncoder(); enc[c] = le.fit_transform(enc[c]); LBL[c] = list(le.classes_)
FEATS = ['Age','Gender','WBC_Count','Protein_Level','Glucose_Level',
         'Pathogen_Present','Hemoglobin','WBC_Blood_Count','Platelets','CRP_Level']
X = enc[FEATS].values
Y = {t: enc[t].values for t in ['Diagnosis','Outcome','Risk_Level']}

# binary decision contrast for each target
BIN = {'Diagnosis':  (Y['Diagnosis']  == LBL['Diagnosis'].index('Bacterial')).astype(int),
       'Outcome':    (Y['Outcome']    == LBL['Outcome'].index('Deceased')).astype(int),
       'Risk_Level': (Y['Risk_Level'] == LBL['Risk_Level'].index('High Risk')).astype(int)}
CONTRAST = {'Diagnosis':'Bacterial vs. non-bacterial',
            'Outcome':'Deceased vs. recovered',
            'Risk_Level':'High risk vs. not high risk'}

idx = np.arange(len(X))
i_tr, i_te = train_test_split(idx, test_size=.15, random_state=SEED, stratify=Y['Outcome'])
i_tr, i_va = train_test_split(i_tr, test_size=.15/.85, random_state=SEED, stratify=Y['Outcome'][i_tr])

def mk(n, k):
    return {'LR':  LogisticRegression(C=1., max_iter=2000, class_weight='balanced', random_state=SEED),
            'RF':  RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                          class_weight='balanced', random_state=SEED, n_jobs=-1),
            'LGBM':LGBMClassifier(n_estimators=500, learning_rate=.05, random_state=SEED, verbose=-1),
            'CatB':CatBoostClassifier(depth=6, learning_rate=.03, iterations=1000,
                                      random_seed=SEED, verbose=0),
            'XGB': XGBClassifier(max_depth=6, learning_rate=.05, n_estimators=500,
                                 random_state=SEED, verbosity=0,
                                 objective='multi:softprob' if k>2 else 'binary:logistic')}[n]

# ---------------------------------------------- STEP 1 (training fold only)
s1 = {}
for t in Y:
    y = Y[t]
    r_full = max(abs(pearsonr(X[:,j], y)[0]) for j in range(len(FEATS)))
    rs  = [abs(pearsonr(X[i_tr,j], y[i_tr])[0]) for j in range(len(FEATS))]
    rho = [abs(spearmanr(X[i_tr,j], y[i_tr])[0]) for j in range(len(FEATS))]
    mi  = mutual_info_classif(X[i_tr], y[i_tr], random_state=SEED)
    s1[t] = {'max_r_full_dataset': round(r_full,4),
             'max_r_train': round(float(max(rs)),4), 'argmax_r': FEATS[int(np.argmax(rs))],
             'max_rho_train': round(float(max(rho)),4),
             'max_mi_train': round(float(max(mi)),4), 'argmax_mi': FEATS[int(np.argmax(mi))],
             'pass': bool(max(rs) > TAU or max(mi) > 0.05)}
R['step1'] = s1

# XOR control: correlation-invisible but learnable
med = np.median(X[:,[0,2]], axis=0)
xor = ((X[:,0]>med[0]) ^ (X[:,2]>med[1])).astype(int)
R['xor_control'] = {
 'max_abs_r': round(max(abs(pearsonr(X[i_tr,j], xor[i_tr])[0]) for j in range(len(FEATS))),4),
 'max_mi':    round(float(max(mutual_info_classif(X[i_tr], xor[i_tr], random_state=SEED))),4),
 'lgbm_test_auc': round(float(roc_auc_score(xor[i_te],
     LGBMClassifier(random_state=SEED, verbose=-1).fit(X[i_tr], xor[i_tr])
     .predict_proba(X[i_te])[:,1])),4)}

# ------------------------------------------------- repeated stratified CV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
cvr = {}
for t, y in Y.items():
    k = len(np.unique(y)); cvr[t] = {}
    for n in ['LR','RF','LGBM','CatB','XGB']:
        a,f,m = [],[],[]
        for tr,te in cv.split(X,y):
            Xtr,Xte = (X[tr],X[te]); ytr = y[tr]
            if n=='LR':
                sc = StandardScaler().fit(X[tr]); Xtr,Xte = sc.transform(X[tr]), sc.transform(X[te])
            if k==2 and np.bincount(ytr).min()>5:
                Xtr,ytr = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(Xtr,ytr)
            mo = mk(n,k).fit(Xtr,ytr); p = mo.predict_proba(Xte); pr = mo.predict(Xte)
            a.append(roc_auc_score(y[te], p[:,1]) if k==2
                     else roc_auc_score(y[te], p, multi_class='ovr', average='macro'))
            f.append(f1_score(y[te], pr, average='macro')); m.append(matthews_corrcoef(y[te], pr))
        cvr[t][n] = {k2: round(float(v),4) for k2,v in
                     [('auc',np.mean(a)),('auc_sd',np.std(a)),('f1',np.mean(f)),
                      ('f1_sd',np.std(f)),('mcc',np.mean(m)),('mcc_sd',np.std(m))]}
R['cv'] = cvr

# ------------------------------------------- class-wise + confusion matrix
BEST = {'Diagnosis':'LGBM','Outcome':'XGB','Risk_Level':'CatB'}
cw = {}
for t,y in Y.items():
    k = len(np.unique(y)); Xtr,ytr = X[i_tr], y[i_tr]
    if k==2: Xtr,ytr = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(Xtr,ytr)
    mo = mk(BEST[t],k).fit(Xtr,ytr); pr = mo.predict(X[i_te])
    p,rc,f,s = precision_recall_fscore_support(y[i_te], pr, zero_division=0)
    cw[t] = {'model':BEST[t],'labels':LBL[t],'confusion':confusion_matrix(y[i_te],pr).tolist(),
             'precision':[round(float(v),3) for v in p],'recall':[round(float(v),3) for v in rc],
             'f1':[round(float(v),3) for v in f],'support':[int(v) for v in s]}
R['classwise'] = cw

# ------------------------- Steps 2-5 on the binary decision contrast
def brier_parts(y,p,nb=8):
    q = np.unique(np.percentile(p, np.linspace(0,100,nb+1))); b = np.digitize(p,q[1:-1])
    yb = y.mean(); rel=res=0.
    for kk in np.unique(b):
        m=b==kk; n=m.sum()
        rel += n*(p[m].mean()-y[m].mean())**2; res += n*(y[m].mean()-yb)**2
    return brier_score_loss(y,p), rel/len(y), res/len(y), yb*(1-yb)
def nb_curve(y,p,ts): 
    o=[]
    for pt in ts:
        pd_=(p>=pt).astype(int)
        tp=((pd_==1)&(y==1)).sum(); fp=((pd_==1)&(y==0)).sum()
        o.append(tp/len(y)-(pt/(1-pt))*fp/len(y))
    return np.array(o)

steps = {}
for t in Y:
    yb = BIN[t]
    Xtr,ytr = X[i_tr], yb[i_tr]
    if np.bincount(ytr).min()>5:
        Xtr,ytr = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(Xtr,ytr)
    mo = LGBMClassifier(n_estimators=500, learning_rate=.05, random_state=SEED, verbose=-1).fit(Xtr,ytr)
    pv, pt_ = mo.predict_proba(X[i_va])[:,1], mo.predict_proba(X[i_te])[:,1]
    pc = LogisticRegression().fit(pv.reshape(-1,1), yb[i_va]).predict_proba(pt_.reshape(-1,1))[:,1]
    bs, rel, res, unc = brier_parts(yb[i_te], pc)
    ts = np.linspace(.15,.60,40); nb = nb_curve(yb[i_te], pc, ts)
    # Step 4: top SHAP feature must itself clear the Step 1 screen
    sv = np.abs(shap.TreeExplainer(mo).shap_values(X[i_te]))
    while sv.ndim > 2: sv = sv.mean(axis=-1)
    top = FEATS[int(np.argmax(sv.mean(axis=0)))]
    rj  = np.array([abs(pearsonr(X[i_tr,j], yb[i_tr])[0]) for j in range(len(FEATS))])
    # Step 5: sign congruence on the same binary contrast
    sc = StandardScaler().fit(X[i_tr])
    lr = LogisticRegression(C=1., max_iter=2000, class_weight='balanced',
                            random_state=SEED).fit(sc.transform(X[i_tr]), yb[i_tr])
    sg = np.array([np.sign(pearsonr(X[i_tr,j], yb[i_tr])[0]) for j in range(len(FEATS))])
    agree = int((np.sign(lr.coef_[0])==sg).sum())
    st = {'S1': s1[t]['pass'],
          'S2': bool(res > .01*unc),
          'S3': bool(nb.max() > 0),
          'S4': bool(rj[FEATS.index(top)] > TAU),
          'S5': bool(agree >= 7)}
    steps[t] = {'contrast':CONTRAST[t],'flags':st,'all_pass':bool(all(st.values())),
                'brier':round(float(bs),4),'REL':round(float(rel),4),
                'RES':round(float(res),4),'UNC':round(float(unc),4),
                'max_net_benefit':round(float(nb.max()),4),
                'top_shap':top,'top_shap_abs_r':round(float(rj[FEATS.index(top)]),4),
                'sign_agreement':f'{agree}/10'}
R['five_steps'] = steps

# ----------------------------------------------------------- ablation
abl = {}
for drop in ['S1','S2','S3','S4','S5']:
    keep = [s for s in ['S1','S2','S3','S4','S5'] if s!=drop]
    wrong = [t for t in Y if all(steps[t]['flags'][s] for s in keep) != steps[t]['all_pass']]
    abl[f'without_{drop}'] = wrong
abl['discrimination_only'] = [t for t in Y
    if (cvr[t][BEST[t]]['auc'] > .5) != steps[t]['all_pass']]
R['ablation'] = abl

json.dump(R, open(os.path.join(RES,'final_results.json'),'w'), indent=1)

print('STEP 1 (training fold only)')
for t,v in s1.items():
    print(f"  {t:11s} r={v['max_r_train']:.3f} ({v['argmax_r']})  MI={v['max_mi_train']:.3f}  "
          f"full-data r={v['max_r_full_dataset']:.3f}  -> {'pass' if v['pass'] else 'FAIL'}")
print(f"\nXOR control: |r|={R['xor_control']['max_abs_r']:.3f}  "
      f"MI={R['xor_control']['max_mi']:.3f}  LightGBM AUC={R['xor_control']['lgbm_test_auc']:.3f}")
print('\nFIVE STEPS on the binary decision contrast')
for t,v in steps.items():
    print(f"  {t:11s} [{v['contrast']}]")
    print(f"     {v['flags']}  -> {'LEARNABLE' if v['all_pass'] else 'NOT LEARNABLE'}")
    print(f"     RES={v['RES']:.4f} UNC={v['UNC']:.4f}  maxNB={v['max_net_benefit']:.4f}  "
          f"topSHAP={v['top_shap']}(r={v['top_shap_abs_r']:.3f})  signs={v['sign_agreement']}")
print('\nABLATION (targets misclassified when a step is removed)')
for k2,v in abl.items(): print(f"  {k2:24s} {v or 'none'}")
