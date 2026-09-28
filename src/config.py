"""Single source of truth for the five-step learnability validation protocol.

Every script, table and figure reads its rules from this module. Nothing
that decides a verdict is defined anywhere else.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'meningitis.csv')
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEED = 42

# ------------------------------------------------------------------ data
FEATURES = ['Age', 'Gender', 'WBC_Count', 'Protein_Level', 'Glucose_Level',
            'Pathogen_Present', 'Hemoglobin', 'WBC_Blood_Count', 'Platelets',
            'CRP_Level']
BINARY_FEATURES = ['Gender', 'Pathogen_Present']          # Male = 1, Yes = 1
TARGETS = ['Diagnosis', 'Outcome', 'Risk_Level']

# display names used in every figure and table (match the manuscript text)
LABEL = {'Age': 'Age', 'Gender': 'Gender (male)', 'WBC_Count': 'CSF WBC count',
         'Protein_Level': 'CSF protein', 'Glucose_Level': 'CSF glucose',
         'Pathogen_Present': 'Pathogen present', 'Hemoglobin': 'Hemoglobin',
         'WBC_Blood_Count': 'Blood WBC count', 'Platelets': 'Platelets',
         'CRP_Level': 'CRP', 'Diagnosis': 'Diagnosis', 'Outcome': 'Mortality',
         'Risk_Level': 'Risk level'}
TARGET_NAME = {'Diagnosis': 'Diagnosis', 'Outcome': 'Mortality', 'Risk_Level': 'Risk level'}

# Positive class of the binary decision contrast used by Steps 2-5.
# The clinically actionable class is always coded 1.
POSITIVE = {'Diagnosis': 'Bacterial', 'Outcome': 'Deceased', 'Risk_Level': 'High Risk'}
CONTRAST = {'Diagnosis': 'Bacterial vs. non-bacterial',
            'Outcome': 'Deceased vs. recovered',
            'Risk_Level': 'High risk vs. not high risk'}

# stratified 70/15/15 partition, stratified on the mortality label
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# ---------------------------------------------------------- protocol rules
TAU = 0.10            # Step 1: absolute Pearson correlation threshold
MI_THRESHOLD = 0.05   # Step 1: mutual information threshold (nats)
# Step 2: the Platt-scaled reference must beat the prevalence model, BS < UNC
N_BINS = 8            # equal-frequency bins for the Brier decomposition
# Step 3: net benefit must exceed max(treat-all, treat-none) by NB_MARGIN at
# some threshold inside DCA_RANGE (1:19 to 1:1 false positives per true positive)
DCA_RANGE = (0.05, 0.50)
DCA_STEP = 0.01
NB_MARGIN = 0.01
# Step 5: share of scored reference directions the logistic-regression
# coefficient signs must reproduce
S5_MIN_SHARE = 0.70

# Step 5 reference directions (+1 higher value -> positive class, -1 lower
# value -> positive class). A direction is scored only when at least two
# independent published sources report it; everything else is not scored.
# Presentation-time reference (bacterial likelihood / triage) is used for the
# two decisions taken at presentation, diagnosis and risk level; the
# prognostic reference is used for mortality.
REF_PRESENTATION = {'WBC_Count': +1,        # Spanos 1989; Nigrovic 2007; Alnomasy 2021
                    'Protein_Level': +1,    # Spanos 1989; Nigrovic 2007; Alnomasy 2021
                    'Glucose_Level': -1,    # Spanos 1989; Alnomasy 2021
                    'WBC_Blood_Count': +1,  # Nigrovic 2007; Alnomasy 2021
                    'CRP_Level': +1}        # Gerdes 1998; Singh 2025
REF_PROGNOSTIC = {'Age': +1,                # van de Beek 2004; Bijlsma 2016; Tubiana 2020; Drost 2025; Chekrouni 2023; Zhou 2025
                  'WBC_Count': -1,          # van de Beek 2004; Bijlsma 2016; Tubiana 2020; Drost 2025
                  'Glucose_Level': -1,      # Tubiana 2020; Chekrouni 2023
                  'Pathogen_Present': +1,   # positive culture: van de Beek 2004; Bijlsma 2016
                  'CRP_Level': +1}          # Bijlsma 2016; Chekrouni 2023; Zhou 2025
S5_REFERENCE = {'Diagnosis': REF_PRESENTATION, 'Risk_Level': REF_PRESENTATION,
                'Outcome': REF_PROGNOSTIC}
# single-source directions, used only for the sensitivity analysis
S5_SINGLE_SOURCE = {'Diagnosis': {'Pathogen_Present': +1},             # Nigrovic 2007 (Gram stain)
                    'Risk_Level': {'Pathogen_Present': +1},
                    'Outcome': {'Protein_Level': +1,                   # Tubiana 2020
                                'Platelets': -1,                       # van de Beek 2004
                                'WBC_Blood_Count': -1,                 # Chekrouni 2023
                                'Gender': +1}}                         # Tubiana 2020 (male)

# integer risk score (supporting diagnostic of Step 5)
SCORE_K = 5
SCORE_PERCENTILE = 75

# ------------------------------------------------------------------ models
MODELS = ['LR', 'RF', 'LightGBM', 'CatBoost', 'XGBoost']
REPRESENTATIVE = 'LightGBM'   # instrument for Steps 2-4 and single-fold analyses
CV_SPLITS, CV_REPEATS = 5, 5


def make_model(name, n_classes=2):
    """The one model factory. Identical settings in every analysis."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    if name == 'LR':
        return LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced',
                                  random_state=SEED)
    if name == 'RF':
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      class_weight='balanced', random_state=SEED, n_jobs=-1)
    if name == 'LightGBM':
        return LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=SEED,
                              verbose=-1)
    if name == 'CatBoost':
        return CatBoostClassifier(depth=6, learning_rate=0.03, iterations=1000,
                                  random_seed=SEED, verbose=0)
    if name == 'XGBoost':
        return XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=500,
                             random_state=SEED, verbosity=0,
                             objective='multi:softprob' if n_classes > 2 else 'binary:logistic')
    raise ValueError(name)
