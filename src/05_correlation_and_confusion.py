"""Training-fold-only correlation matrix, confusion matrices, Brier table."""
import json, warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, brier_score_loss
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
df=pd.read_csv(DATA); enc=df.copy(); LBL={}
for c in ['Gender','Pathogen_Present','Diagnosis','Outcome','Risk_Level']:
    le=LabelEncoder(); enc[c]=le.fit_transform(enc[c]); LBL[c]=list(le.classes_)
COLS=['Age','Gender','WBC_Count','Protein_Level','Glucose_Level','Pathogen_Present',
      'Diagnosis','Outcome','Hemoglobin','WBC_Blood_Count','Platelets','CRP_Level','Risk_Level']
FE=[c for c in COLS if c not in ('Diagnosis','Outcome','Risk_Level')]
idx=np.arange(len(enc))
i_tr,i_te=train_test_split(idx,test_size=.15,random_state=SEED,stratify=enc['Outcome'])
i_tr,i_va=train_test_split(i_tr,test_size=.15/.85,random_state=SEED,stratify=enc['Outcome'].values[i_tr])

# ---- Fig 2: correlation on the TRAINING FOLD only ----
sub=enc.iloc[i_tr][COLS]
corr=sub.corr()
fig,ax=plt.subplots(figsize=(11,9))
sns.heatmap(corr,mask=np.triu(np.ones_like(corr,dtype=bool)),annot=True,fmt='.2f',
            cmap='coolwarm',center=0,ax=ax,annot_kws={'size':8},cbar_kws={'shrink':.8})
ax.set_title('Feature Correlation Matrix (training fold only, n = 840)',fontweight='bold',fontsize=13)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_correlation_trainfold.png'),dpi=150,bbox_inches='tight'); plt.close()
key={}
for t in ['Diagnosis','Outcome','Risk_Level']:
    s=corr[t].drop(['Diagnosis','Outcome','Risk_Level']).sort_values(key=abs,ascending=False)
    key[t]={k:round(float(v),2) for k,v in s.head(5).items()}
print('training-fold correlations (top 5 per target):')
for t,v in key.items(): print(' ',t,v)

# ---- Fig: confusion matrices ----
X=enc[FE].values
BEST={'Diagnosis':('LightGBM',LGBMClassifier(n_estimators=500,learning_rate=.05,random_state=SEED,verbose=-1)),
      'Outcome':('XGBoost',XGBClassifier(max_depth=6,learning_rate=.05,n_estimators=500,random_state=SEED,verbosity=0)),
      'Risk_Level':('CatBoost',CatBoostClassifier(depth=6,learning_rate=.03,iterations=1000,random_seed=SEED,verbose=0))}
fig,axes=plt.subplots(1,3,figsize=(15,4.2))
cms={}
for ax,(t,(nm,mdl)) in zip(axes,BEST.items()):
    y=enc[t].values; Xtr,ytr=X[i_tr],y[i_tr]
    if len(np.unique(y))==2: Xtr,ytr=SMOTE(k_neighbors=5,random_state=SEED).fit_resample(Xtr,ytr)
    mdl.fit(Xtr,ytr); cm=confusion_matrix(y[i_te],mdl.predict(X[i_te])); cms[t]=cm.tolist()
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,cbar=False,
                xticklabels=LBL[t],yticklabels=LBL[t],annot_kws={'size':11})
    ax.set_title(f'{t.replace("_"," ")} ({nm})',fontweight='bold',fontsize=11)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.tick_params(axis='x',rotation=20); ax.tick_params(axis='y',rotation=0)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_confusion.png'),dpi=150,bbox_inches='tight'); plt.close()
print('\nconfusion matrices:',{k:v for k,v in cms.items()})

# ---- Brier decomposition on the mortality target ----
def parts(y,p,nb=8):
    q=np.unique(np.percentile(p,np.linspace(0,100,nb+1))); b=np.digitize(p,q[1:-1])
    yb=y.mean(); rel=res=0.
    for kk in np.unique(b):
        m=b==kk; n=m.sum(); rel+=n*(p[m].mean()-y[m].mean())**2; res+=n*(y[m].mean()-yb)**2
    return brier_score_loss(y,p),rel/len(y),res/len(y),yb*(1-yb)
y=enc['Outcome'].values
Xtr,ytr=SMOTE(k_neighbors=5,random_state=SEED).fit_resample(X[i_tr],y[i_tr])
rows={}
MOD={'LR':LogisticRegression(C=1.,max_iter=2000,class_weight='balanced',random_state=SEED),
     'RF':RandomForestClassifier(n_estimators=300,min_samples_leaf=2,random_state=SEED,n_jobs=-1),
     'LightGBM':LGBMClassifier(n_estimators=500,learning_rate=.05,random_state=SEED,verbose=-1),
     'CatBoost':CatBoostClassifier(depth=6,learning_rate=.03,iterations=1000,random_seed=SEED,verbose=0),
     'XGBoost':XGBClassifier(max_depth=6,learning_rate=.05,n_estimators=500,random_state=SEED,verbosity=0)}
for nm,m in MOD.items():
    m.fit(Xtr,ytr); p=m.predict_proba(X[i_te])[:,1]
    bs,rel,res,u=parts(y[i_te],p); rows[nm]=[round(v,4) for v in (bs,rel,res,u,bs-u)]
lg=MOD['LightGBM']; pv=lg.predict_proba(X[i_va])[:,1]; pt=lg.predict_proba(X[i_te])[:,1]
pc=LogisticRegression().fit(pv.reshape(-1,1),y[i_va]).predict_proba(pt.reshape(-1,1))[:,1]
bs,rel,res,u=parts(y[i_te],pc); rows['LGBM-Cal']=[round(v,4) for v in (bs,rel,res,u,bs-u)]
print('\nBrier decomposition (mortality):')
for k2,v in rows.items(): print(f'  {k2:9s} BS={v[0]:.4f} REL={v[1]:.4f} RES={v[2]:.4f} UNC={v[3]:.4f} BS-UNC={v[4]:+.4f}')
json.dump({'corr_trainfold':key,'confusion':cms,'brier':rows},open(os.path.join(RES,'new_assets.json'),'w'),indent=1)
