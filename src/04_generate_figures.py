"""Regenerate Fig. 4 (repeated CV), Fig. 5 (matching Table 3), Fig. 9 (binary, -5 cutoff)."""
import json, warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'meningitis.csv')
RES  = os.path.join(ROOT, 'results');  os.makedirs(RES, exist_ok=True)
FIG  = os.path.join(ROOT, 'figures');  os.makedirs(FIG, exist_ok=True)

SEED=42
CV=json.load(open(os.path.join(RES,'final_results.json')))['cv']
df=pd.read_csv(DATA); enc=df.copy(); LBL={}
for c in ['Gender','Pathogen_Present','Diagnosis','Outcome','Risk_Level']:
    le=LabelEncoder(); enc[c]=le.fit_transform(enc[c]); LBL[c]=list(le.classes_)
FE=['Age','Gender','WBC_Count','Protein_Level','Glucose_Level','Pathogen_Present',
    'Hemoglobin','WBC_Blood_Count','Platelets','CRP_Level']
X=enc[FE].values; y=enc['Outcome'].values
i=np.arange(len(X))
i_tr,i_te=train_test_split(i,test_size=.15,random_state=SEED,stratify=y)
i_tr,i_va=train_test_split(i_tr,test_size=.15/.85,random_state=SEED,stratify=y[i_tr])

# ---------- Fig. 4: repeated-CV discrimination with error bars ----------
MODS=['LR','RF','LGBM','CatB','XGB']; NM=['LR','RF','LGBM','CatB','XGB']
TG=[('Diagnosis','Diagnosis'),('Outcome','Mortality'),('Risk_Level','Risk level')]
fig,axes=plt.subplots(1,3,figsize=(14,4.2))
for ax,(k,lab) in zip(axes,TG):
    a=[CV[k][m]['auc'] for m in MODS]; ae=[CV[k][m]['auc_sd'] for m in MODS]
    f=[CV[k][m]['f1'] for m in MODS];  fe=[CV[k][m]['f1_sd'] for m in MODS]
    xx=np.arange(len(MODS)); w=.36
    ax.bar(xx-w/2,a,w,yerr=ae,capsize=3,label='AUC',color='#3B6E8F')
    ax.bar(xx+w/2,f,w,yerr=fe,capsize=3,label='macro F1',color='#D96B4A')
    ax.set_xticks(xx); ax.set_xticklabels(NM,fontsize=9)
    ax.set_ylim(0,1.05) if k=='Outcome' else ax.set_ylim(.5,1.05)
    ax.set_title(f'Target: {lab}',fontweight='bold',fontsize=11)
    ax.set_ylabel('Score'); ax.grid(axis='y',alpha=.3); ax.legend(fontsize=8)
fig.suptitle('Discrimination under repeated stratified cross-validation (5 folds x 5 repeats)',
             fontweight='bold',fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig4_cv_performance.png'),dpi=150,bbox_inches='tight'); plt.close()
print('Fig. 4 regenerated from repeated CV')

# ---------- Fig. 5: calibration matching Table 3 exactly ----------
Xtr,ytr=SMOTE(k_neighbors=5,random_state=SEED).fit_resample(X[i_tr],y[i_tr])
MOD={'LR':LogisticRegression(C=1.,max_iter=2000,class_weight='balanced',random_state=SEED),
     'RF':RandomForestClassifier(n_estimators=300,min_samples_leaf=2,random_state=SEED,n_jobs=-1),
     'LightGBM':LGBMClassifier(n_estimators=500,learning_rate=.05,random_state=SEED,verbose=-1),
     'CatBoost':CatBoostClassifier(depth=6,learning_rate=.03,iterations=1000,random_seed=SEED,verbose=0),
     'XGBoost':XGBClassifier(max_depth=6,learning_rate=.05,n_estimators=500,random_state=SEED,verbosity=0)}
P={}
for n_,m in MOD.items(): m.fit(Xtr,ytr); P[n_]=m.predict_proba(X[i_te])[:,1]
lg=MOD['LightGBM']; pv=lg.predict_proba(X[i_va])[:,1]
P['LGBM-Cal']=LogisticRegression().fit(pv.reshape(-1,1),y[i_va])\
    .predict_proba(lg.predict_proba(X[i_te])[:,1].reshape(-1,1))[:,1]
fig,axes=plt.subplots(2,3,figsize=(13,7))
for ax,(n_,p) in zip(axes.ravel(),P.items()):
    bs=brier_score_loss(y[i_te],p)
    ft,mp=calibration_curve(y[i_te],p,n_bins=8,strategy='quantile')
    ax.plot([0,1],[0,1],'--',c='grey',lw=1,label='Perfect calibration')
    ax.plot(mp,ft,'o-',lw=1.8,label=n_)
    ax.set_title(f'{n_}  (Brier = {bs:.4f})',fontweight='bold',fontsize=10)
    ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Fraction of positives')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(fontsize=7); ax.grid(alpha=.3)
fig.suptitle('Reliability diagrams for the mortality target (held-out fold)',
             fontweight='bold',fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig5_calibration.png'),dpi=150,bbox_inches='tight'); plt.close()
print('Fig. 5 regenerated; Brier values now match Table 3:',
      {k:round(brier_score_loss(y[i_te],v),4) for k,v in P.items()})

# ---------- Fig. 9: integer risk score, binary outcome, Youden cutoff ----------
sc=StandardScaler().fit(X[i_tr])
lr=LogisticRegression(C=1.,max_iter=2000,class_weight='balanced',random_state=SEED)\
   .fit(sc.transform(X[i_tr]),y[i_tr])
b=lr.coef_[0]; K=5; w=np.round(K*b/np.abs(b).max()).astype(int)
cut=np.percentile(X[i_tr],75,axis=0)
S=((X> cut).astype(int)*w).sum(axis=1)
Ste=S[i_te]; yte=y[i_te]
fpr,tpr,th=roc_curve(yte,Ste); j=np.argmax(tpr-fpr); yo=th[j]
auc_s=roc_auc_score(yte,Ste)
fig,axes=plt.subplots(1,2,figsize=(13,4.6))
ax=axes[0]
bins=np.arange(Ste.min()-.5,Ste.max()+1.5)
ax.hist(Ste[yte==0],bins=bins,alpha=.65,label=LBL['Outcome'][1],color='#3B6E8F',density=True)
ax.hist(Ste[yte==1],bins=bins,alpha=.65,label=LBL['Outcome'][0],color='#D96B4A',density=True)
ax.axvline(yo,ls='--',c='crimson',lw=2,label=f'Youden-optimal cut-off ({yo:.0f})')
ax.set_xlabel('Integer risk score'); ax.set_ylabel('Density')
ax.set_title('Score distribution by mortality outcome',fontweight='bold',fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=.3)
ax=axes[1]
ax.plot(fpr,tpr,lw=2,c='#1B7A6E',label=f'Integer score (AUC = {auc_s:.3f})')
ax.plot([0,1],[0,1],':',c='grey',lw=1)
ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity')
ax.set_title('ROC for the integer score',fontweight='bold',fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=.3)
fig.suptitle('Integer risk score for the mortality target (held-out fold)',fontweight='bold',fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig9_riskscore.png'),dpi=150,bbox_inches='tight'); plt.close()
sens=tpr[j]; spec=1-fpr[j]
print(f'Fig. 9 regenerated: AUC={auc_s:.3f}, Youden cut-off={yo:.0f}, sens={sens:.3f}, spec={spec:.3f}')
json.dump({'score_auc':round(float(auc_s),3),'cutoff':int(yo),
           'sens':round(float(sens),3),'spec':round(float(spec),3),
           'range':[int(Ste.min()),int(Ste.max())]},open(os.path.join(RES,'fig9_stats.json'),'w'),indent=1)
