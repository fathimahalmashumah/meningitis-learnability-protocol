"""Figures 1-9 at their final print size (JCTA A4 layout).

Full-width figures are drawn at 18.4 cm and text-column figures at 13.9 cm, with
8-pt text, so they are inserted at 100 % and never rescaled.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve

import config as C
import protocol as P

CM = 1 / 2.54
FULL, COL = 16.0 * CM, 12.0 * CM
DPI = 600
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5,
                     'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'legend.fontsize': 7.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
                     'savefig.dpi': DPI})
INK, INK2, GRID = '#0b0b0b', '#52514e', '#c9c8c2'
S = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
MODEL_COLOR = dict(zip(['LR', 'RF', 'LightGBM', 'CatBoost', 'XGBoost', 'LGBM-Cal'], S))
TC = {'Diagnosis': S[0], 'Outcome': S[1], 'Risk_Level': S[2]}

R = json.load(open(f'{C.RES}/results.json'))
D = np.load(f'{C.RES}/predictions.npz')


def save(fig, name):
    fig.savefig(f'{C.FIG}/{name}.png', bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


# ------------------------------------------------------------ Fig. 1 workflow
def fig1():
    W, H = 10.6 * CM, 15.6 * CM
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 115); ax.set_ylim(-5, 168); ax.axis('off')
    SC = 0.88
    NAVY, TEAL, GOLD = '#1f3b5a', '#127a6e', '#a8780a'

    def box(x, y, w, h, title, body='', fc='white', ec=NAVY, tc=NAVY, lw=0.9, fs=8, bs=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.6',
                                    fc=fc, ec=ec, lw=lw))
        if body:
            nl = title.count('\n')
            ax.text(x + w / 2, y + h - 3.6 - 1.8 * nl, title, ha='center', va='center', fontsize=fs * SC,
                    fontweight='bold', color=tc)
            ax.text(x + w / 2, y + (h - 6.2 - 3.6 * nl) / 2, body, ha='center', va='center',
                    fontsize=bs * SC, color=INK2, linespacing=1.25)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=fs * SC,
                    fontweight='bold', color=tc, linespacing=1.25)

    def diamond(cx, cy, w, h, text, fs=7.2):
        ax.add_patch(Polygon([[cx - w / 2, cy], [cx, cy + h / 2], [cx + w / 2, cy], [cx, cy - h / 2]],
                             closed=True, fc='#fdf6e3', ec=GOLD, lw=0.9))
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fs * SC, color='#6b4e00',
                fontweight='bold', linespacing=1.25)

    def arrow(x1, y1, x2, y2, c=NAVY, ls='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=0.9, ls=ls,
                                    mutation_scale=8, shrinkA=0, shrinkB=0))

    cx = 57.5
    box(22, 156, 71, 11, 'Meningitis dataset',
        'n = 1,200 · 10 clinical predictors · 3 targets', fc='#eef2f7')
    arrow(cx, 156, cx, 151.5)
    box(22, 139.5, 71, 12, 'Stage A: partition',
        'stratified 70/15/15 split before any estimation\nfit: training · Platt scaling: validation · evaluation: test')
    arrow(cx, 139.5, cx, 135)
    diamond(cx, 125, 60, 20, 'Step 1: dependence screen\nmax |r| > 0.10  or  max MI > 0.05 ?')
    ax.text(cx + 2, 113.3, 'yes', fontsize=7 * SC, color=NAVY, fontweight='bold')
    # warning branch: rejoins the flow
    ax.plot([cx + 30, 104.5], [125, 125], color=GOLD, lw=0.9)
    ax.text(89, 126.4, 'no', fontsize=7 * SC, color=GOLD, fontweight='bold')
    arrow(104.5, 125, 104.5, 120, c=GOLD)
    box(94.5, 107, 20, 13, 'Warning', 'recorded;\ncontinue',
        fc='#fdf6e3', ec=GOLD, tc='#6b4e00', fs=7.2, bs=6.6)
    ax.plot([104.5, 104.5], [107, 98.75], color=GOLD, lw=0.9)
    arrow(104.5, 98.75, 93, 98.75, c=GOLD)
    arrow(cx, 115, cx, 105.5)
    box(22, 92, 71, 13.5, 'Stage B: preparation',
        'standardisation (LR) · SMOTE for binary targets\ntraining fold only · Eqs. (5), (9), (10)')
    arrow(cx, 92, cx, 87.5)
    box(22, 74, 71, 13.5, 'Stage C: training',
        'LR · RF · LightGBM · CatBoost · XGBoost\nLightGBM Platt-scaled on validation fold (LGBM-Cal)')
    arrow(cx, 74, cx, 69.5)
    # steps 2-5 panel
    ax.add_patch(FancyBboxPatch((1.5, 30), 112, 39.5, boxstyle='round,pad=0,rounding_size=1.8',
                                fc='#eaf4f2', ec=TEAL, lw=0.9))
    ax.text(cx, 65.3, 'Steps 2–5 on each binary decision contrast (positive class: Bacterial,\n'
            'High risk, Deceased; held-out test fold)', ha='center', va='center', fontsize=7.4 * SC,
            fontweight='bold', color=TEAL, linespacing=1.25)
    steps = [('Step 2', 'Brier\ndecomposition\nBS < UNC\nEqs. (2), (25), (26)'),
             ('Step 3', 'Decision curve\nNB above both\ndefaults by ≥ 0.01\npₜ 0.05–0.50\nEq. (3)'),
             ('Step 4', 'SHAP\nplausibility\ntop feature passes\nStep 1 rule · Eq. (1)'),
             ('Step 5', 'Sign audit vs.\nliterature\n≥ 70 % agree\nEqs. (4), (11)–(13)')]
    for i, (t, b) in enumerate(steps):
        box(3.5 + i * 27.3, 34.5, 25.3, 24.5, t, b, fs=7.6, bs=6.5, ec=NAVY)
    ax.text(cx, 32, 'Steps 2–4 on LGBM-Cal · Step 5 on logistic regression',
            ha='center', va='center', fontsize=6.8 * SC, style='italic', color=TEAL)
    arrow(cx, 30, cx, 25.5)
    diamond(cx, 17, 46, 17, 'Stage D: verdict\nall five steps pass?')
    ax.plot([cx - 23, 13], [17, 17], color=INK2, lw=0.9); arrow(13, 17, 13, 11.5, c=INK2)
    ax.plot([cx + 23, 102], [17, 17], color=TEAL, lw=0.9); arrow(102, 17, 102, 11.5, c=TEAL)
    ax.text(26, 18.4, 'no', fontsize=7 * SC, color=INK2, fontweight='bold')
    ax.text(86, 18.4, 'yes', fontsize=7 * SC, color=TEAL, fontweight='bold')
    box(0.5, -4.5, 44, 16, 'DOES NOT PASS', 'report the failing steps\nand any Step 1 warning',
        fc='#f2f2f0', ec=INK2, tc=INK2, fs=7.4, bs=6.6)
    box(70.5, -4.5, 44, 16, 'PASSES THE FIVE-STEP\nVALIDATION',
        'eligible for further (external)\nvalidation', fc='#eaf4f2', ec=TEAL, tc=TEAL, fs=6.9, bs=6.6)
    save(fig, 'fig1_workflow')


# ------------------------------------------------ Fig. 2 correlation matrix
def fig2():
    df, X, Y, B, L = P.load()
    tr, _, _ = P.split(Y['Outcome'])
    cols = C.FEATURES + C.TARGETS
    M = pd.DataFrame(np.column_stack([X[tr]] + [Y[t][tr] for t in C.TARGETS]),
                     columns=[C.LABEL[c] for c in cols]).astype(float).corr()
    fig, ax = plt.subplots(figsize=(COL, COL * 0.86))
    ax.grid(False)
    n = len(cols)
    mask = np.triu(np.ones((n, n), bool), 1)
    Z = np.ma.array(M.values, mask=mask)
    im = ax.imshow(Z, cmap='RdBu_r', vmin=-1, vmax=1)
    for i in range(n):
        for j in range(i + 1):
            v = M.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.6,
                    color='white' if abs(v) > 0.55 else INK)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(M.columns, rotation=45, ha='right'); ax.set_yticklabels(M.columns)
    for s in ax.spines.values(): s.set_visible(False)
    for k in range(n - 3, n):
        ax.get_yticklabels()[k].set_fontweight('bold'); ax.get_xticklabels()[k].set_fontweight('bold')
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02); cb.ax.tick_params(labelsize=7)
    cb.set_label('Pearson r (training fold, n = 840)')
    save(fig, 'fig2_correlation')


# ------------------------------------------------- Fig. 3 confusion matrices
def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 5.8 * CM), gridspec_kw={'wspace': 0.75})
    names = {'Bacterial': 'Bacterial', 'Unknown': 'Unknown', 'Viral': 'Viral',
             'Deceased': 'Deceased', 'Recovered': 'Recovered', 'High Risk': 'High',
             'Low Risk': 'Low', 'Moderate Risk': 'Moderate'}
    for ax, t in zip(axes, C.TARGETS):
        cw = R['classwise'][t]; cm = np.array(cw['confusion']); labs = [names[l] for l in cw['labels']]
        ax.grid(False)
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
        for i in range(len(labs)):
            for j in range(len(labs)):
                frac = cm[i, j] / cm[i].sum()
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8,
                        color='white' if frac > 0.6 else INK)
        ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=25, ha='right'); ax.set_yticklabels(labs)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(C.TARGET_NAME[t])
        for s in ax.spines.values(): s.set_visible(False)
    save(fig, 'fig4_confusion')


# ------------------------------------------------- Fig. 4 repeated CV
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 5.6 * CM), gridspec_kw={'wspace': 0.3})
    xx = np.arange(len(C.MODELS)); w = 0.36
    for ax, t in zip(axes, C.TARGETS):
        cv = R['cv'][t]
        a = [cv[m]['auc'] for m in C.MODELS]; ae = [cv[m]['auc_sd'] for m in C.MODELS]
        c = [cv[m]['mcc'] for m in C.MODELS]; ce = [cv[m]['mcc_sd'] for m in C.MODELS]
        ax.bar(xx - w / 2 - 0.01, a, w, yerr=ae, capsize=2, color=S[0], label='AUC',
               error_kw={'lw': 0.7, 'capthick': 0.7})
        ax.bar(xx + w / 2 + 0.01, c, w, yerr=ce, capsize=2, color=S[1], label='MCC',
               error_kw={'lw': 0.7, 'capthick': 0.7})
        ax.axhline(0, color=INK2, lw=0.6)
        if t == 'Outcome':
            ax.axhline(0.5, color=INK2, lw=0.6, ls=':')
            ax.text(-0.45, 0.62, 'AUC = 0.5 (dotted)', fontsize=6.5, color=INK2, ha='left')
        ax.set_ylim(-0.1 if t == 'Outcome' else 0, 1.05)
        ax.set_xticks(xx); ax.set_xticklabels(['LR', 'RF', 'LightGBM', 'CatBoost', 'XGBoost'],
                                              rotation=30, ha='right')
        ax.set_title(C.TARGET_NAME[t]); ax.grid(axis='x', visible=False)
        if t == 'Diagnosis':
            ax.set_ylabel('Mean ± SD over 25 fits')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.07))
    save(fig, 'fig3_cv')


# ------------------------------------------------- Fig. 5 reliability diagrams
def fig5():
    y = D['y_mort_te']
    order = ['LR', 'RF', 'LightGBM', 'CatBoost', 'XGBoost', 'LGBM-Cal']
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 9.6 * CM),
                             gridspec_kw={'wspace': 0.25, 'hspace': 0.4})
    for ax, m in zip(axes.ravel(), order):
        p = D[f'prob_{m}']; b = R['brier_mortality'][m]
        ft, mp = calibration_curve(y, p, n_bins=C.N_BINS, strategy='quantile')
        ax.plot([0, 1], [0, 1], ls='--', color=INK2, lw=0.7)
        ax.axhline(y.mean(), color=GRID, lw=0.8)
        ax.plot(mp, ft, '-o', color=MODEL_COLOR[m], lw=1.4, ms=3.5, mec='white', mew=0.6)
        ax.plot(p, np.full_like(p, -0.035), '|', color=MODEL_COLOR[m], ms=5, alpha=0.5, mew=0.6)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.07, 1.02)
        ax.set_title(f'{m} (BS = {b["BS"]:.4f})')
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    fig.supxlabel('Mean predicted P(deceased)', fontsize=8, y=0.01)
    for ax in axes[:, 0]:
        ax.set_ylabel('Observed fraction deceased')
    save(fig, 'fig5_calibration')


# ------------------------------------------------- Fig. 6 decision curves
def fig6():
    y = D['y_mort_te']
    ts = np.round(np.arange(0.01, 0.801, 0.005), 3)
    fig, ax = plt.subplots(figsize=(COL, 8.2 * CM))
    ax.axvspan(*C.DCA_RANGE, color='#f1efe8', zorder=0, lw=0)
    ax.text(np.mean(C.DCA_RANGE), -0.335, 'Step 3 threshold range', ha='center', fontsize=7,
            color=INK2)
    for m in ['LR', 'RF', 'LightGBM', 'CatBoost', 'XGBoost', 'LGBM-Cal']:
        nb = [P.net_benefit(y, D[f'prob_{m}'], t) for t in ts]
        ax.plot(ts, nb, color=MODEL_COLOR[m], lw=1.4 if m == 'LGBM-Cal' else 1.1, label=m)
    ax.plot(ts, [P.nb_treat_all(y, t) for t in ts], color=INK, ls='--', lw=1, label='Treat all')
    ax.axhline(0, color=INK2, ls=':', lw=1, label='Treat none')
    ax.set_xlim(0, 0.8); ax.set_ylim(-0.36, 0.13)
    ax.set_xlabel('Threshold probability $p_t$'); ax.set_ylabel('Net benefit')
    ax.legend(ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    save(fig, 'fig6_dca')


# ------------------------------------------------- Fig. 7 global SHAP
def fig7():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 6.8 * CM), gridspec_kw={'wspace': 0.95})
    for ax, t in zip(axes, C.TARGETS):
        imp = R['steps'][t]['S4']['importance']; top = R['steps'][t]['S4']['top_feature']
        o = sorted(C.FEATURES, key=lambda f: imp[f])
        cols = [TC[t] if f == top else '#b8c4d0' for f in o]
        ax.barh([C.LABEL[f] for f in o], [imp[f] for f in o], color=cols, height=0.7)
        s4 = R['steps'][t]['S4']
        ax.set_title(f'{C.TARGET_NAME[t]}\n(top: {C.LABEL[top]}, |r| = {s4["top_abs_r"]:.3f})')
        ax.set_xlabel('Mean |SHAP| (log-odds)'); ax.grid(axis='y', visible=False)
    save(fig, 'fig7_shap_global')


# ------------------------------------------------- Fig. 8 local SHAP
def fig8():
    sv, Xt, p = D['shap_mort'], D['X_te'], D['p_raw_mort']
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 6.8 * CM), gridspec_kw={'wspace': 0.75})
    for ax, key, title in [(axes[0], 'caseA', 'Case A: deceased, highest predicted risk'),
                           (axes[1], 'caseB', 'Case B: deceased, lowest predicted risk')]:
        i = int(D[key]); row = sv[i]; o = np.argsort(np.abs(row))[::-1][:8][::-1]
        lab = [f'{C.LABEL[C.FEATURES[j]]} = {Xt[i, j]:,.0f}' for j in o]
        ax.barh(lab, row[o], color=[S[1] if v > 0 else S[0] for v in row[o]], height=0.7)
        ax.axvline(0, color=INK, lw=0.7); ax.grid(axis='y', visible=False)
        ax.set_xlabel('SHAP value (log-odds of death)')
        ax.set_title(f'{title}\nP(deceased) = {p[i]:.2f}')
    save(fig, 'fig8_shap_local')


# ------------------------------------------------- Fig. 9 integer score
def fig9():
    y = D['y_mort_te']; s = D['score_te']; sc = R['integer_score']
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 6.4 * CM), gridspec_kw={'wspace': 0.3})
    ax = axes[0]
    bins = np.arange(s.min() - 0.5, s.max() + 1.5)
    ax.hist(s[y == 0], bins=bins, density=True, color=S[0], alpha=0.75, label='Recovered',
            edgecolor='white', lw=0.6)
    ax.hist(s[y == 1], bins=bins, density=True, histtype='step', color=S[1], lw=1.6,
            label='Deceased')
    ax.axvline(sc['cutoff'] - 0.5, color=INK, ls='--', lw=0.9)
    ax.text(sc['cutoff'] - 0.4, ax.get_ylim()[1] * 0.93, 'Youden cut-off\n(score ≥ ' + f'{sc["cutoff"]:.0f}'.replace('-', '−') + ')',
            fontsize=6.8, color=INK)
    ax.set_xlabel('Integer risk score'); ax.set_ylabel('Density'); ax.legend(frameon=False, loc='upper left')
    ax.set_title('Score distribution by outcome')
    ax = axes[1]
    for lab, v, c, ls in [(f'Integer score (AUC = {sc["auc"]:.3f})', s, S[2], '-'),
                          (f'LR probability (AUC = {sc["auc_lr_prob"]:.3f})', D['p_lr_te'], S[0], '--'),
                          (f'LightGBM (AUC = {sc["auc_lgbm"]:.3f})', D['p_raw_mort'], S[1], '-.')]:
        f, t_, _ = roc_curve(y, v); ax.plot(f, t_, color=c, ls=ls, lw=1.3, label=lab)
    ax.plot([0, 1], [0, 1], ':', color=INK2, lw=0.8)
    ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity')
    ax.legend(frameon=False, loc='upper left'); ax.set_title('ROC curves, held-out fold')
    save(fig, 'fig9_score')


if __name__ == '__main__':
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]:
        f(); print('done', f.__name__)
