"""Consistency checks: one rule set, one positive class, one verdict.

    python -m pytest tests            # after python src/run_all.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import config as C
import protocol as P

R = json.load(open(os.path.join(C.RES, 'results.json')))


def test_positive_classes():
    df, X, Y, B, L = P.load()
    for t in C.TARGETS:
        assert set(np.unique(B[t])) == {0, 1}
        assert (df[t][B[t] == 1] == C.POSITIVE[t]).all()
    assert C.POSITIVE['Outcome'] == 'Deceased'


def test_single_thresholds():
    assert R['config']['tau'] == C.TAU == 0.10
    assert R['config']['mi_threshold'] == C.MI_THRESHOLD == 0.05
    assert R['config']['s5_min_share'] == C.S5_MIN_SHARE == 0.70
    assert tuple(R['config']['dca_range']) == C.DCA_RANGE


def test_step1_rule_includes_mutual_information():
    assert P.passes_screen(0.05, 0.06) and P.passes_screen(0.11, 0.0)
    assert not P.passes_screen(0.10, 0.05)


def test_flags_recompute():
    for t, s in R['steps'].items():
        assert s['S1']['pass'] == P.passes_screen(s['S1']['max_abs_r'], s['S1']['max_mi'])
        assert s['S2']['pass'] == (s['S2']['BS'] < s['S2']['UNC'])
        assert s['S3']['pass'] == (s['S3']['max_excess_nb'] >= C.NB_MARGIN)
        assert s['S4']['pass'] == P.passes_screen(s['S4']['top_abs_r'], s['S4']['top_mi'])
        assert s['S5']['pass'] == (s['S5']['agree'] / s['S5']['scored'] >= C.S5_MIN_SHARE)
        assert s['verdict'] == ('passes' if all(s['flags'].values()) else 'does not pass')


def test_ablation_uses_final_flags():
    full = {t: R['steps'][t]['verdict'] == 'passes' for t in C.TARGETS}
    for k in range(1, 6):
        keep = [f'S{j}' for j in range(1, 6) if j != k]
        wrong = [t for t in C.TARGETS if all(R['steps'][t]['flags'][s] for s in keep) != full[t]]
        assert wrong == R['ablation'][f'without_S{k}']


def test_reported_verdicts():
    v = {t: R['steps'][t]['verdict'] for t in C.TARGETS}
    assert v == {'Diagnosis': 'passes', 'Outcome': 'does not pass', 'Risk_Level': 'passes'}


def test_xor_controls():
    assert R['xor_control']['step1_pass'] is True
    assert R['xor_control_age_sex']['step1_pass'] is False
    assert R['xor_control_age_sex']['lgbm_test_auc'] > 0.99
