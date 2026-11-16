# make_figures_tables.py: Figures 1-10, the graphical abstract, Tables 1-6 and S1-S8 and the p-value audit of Paper 2.
# Everything is read from the saved per-person result files (no new analysis).
# Repository version of the script used for the submitted manuscript (make_final3.py, 13 Nov 2026). Changes for the
# repository: paths come from p2style.py (P2_RESULTS, P2_FIG_OUT, P2_TAB_OUT), the one-off code inspection and the Colab
# zip download were removed, and temporary files go to the system temp folder. The figure and table code is unchanged.
import sys, os, io, glob, json, shutil, zipfile, importlib, traceback, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2style; importlib.reload(p2style)
from p2style import *
import numpy as np, pandas as pd
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, Arc
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

def rd(path, **kw):
    """CSV read with exact (round-trip) floats, so that equal accuracies stay exactly equal."""
    kw.setdefault('float_precision', 'round_trip')
    return pd.read_csv(path, **kw)

def wil_robust(d, alternative='two-sided'):
    """Wilcoxon signed-rank test. Differences rounded to 1e-6 points so that equal differences count as ties; exact null
    distribution when no differences are tied or zero, otherwise the normal approximation with tie correction."""
    d = np.round(np.asarray(d, float), 6); d = d[~np.isnan(d)]; nz = d[d != 0]
    if len(nz) == 0:
        return 1.0, 'all zero'
    if len(np.unique(np.abs(nz))) == len(nz) and len(nz) == len(d):
        return float(stats.wilcoxon(d, alternative=alternative, method='exact').pvalue), 'exact'
    try:
        p = stats.wilcoxon(d, alternative=alternative, method='asymptotic').pvalue
    except ValueError:                                   # older scipy: 'approx'
        p = stats.wilcoxon(d, alternative=alternative, method='approx').pvalue
    return float(p), 'normal approximation'

class _P:
    def __init__(self, p):
        self.pvalue = p
def wilp(d, alternative='two-sided'):
    return _P(wil_robust(d, alternative)[0])

def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(np.argsort(ps)):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj

def fig1():
    # Figure 1 (v3, after R.2): study design. (a) four cells; (b) data use for each new person. Double column (190 mm). No data needed.
    from matplotlib.patches import FancyBboxPatch

    def blank(ax):
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    def box(ax, x, y, w, h, text='', fc=C['fill'], ec=C['axis'], lw=0.6, fs=FS, color=C['ink'], weight='normal'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.0', fc=fc, ec=ec, lw=lw))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color, weight=weight,
                    linespacing=1.2)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', lw=0.7, color=C['ink2'], mutation_scale=7, shrinkA=0, shrinkB=0))

    fig = plt.figure(figsize=(W2, 72 * MM))

    # (a) four cells
    axa = fig.add_axes([0.0, 0.0, 0.46, 0.975]); blank(axa)
    letter(fig, 0.0, 0.995, 'a', 'Four-cell transfer design')
    for x, t in ((44.5, 'Same person'), (81.5, 'Other 14 people')):
        axa.text(x, 89, t, ha='center', va='bottom', fontsize=7.5, weight='bold')
    for y, t in ((69, 'Imagery\n(MI)'), (39, 'Execution\n(ME)')):
        axa.text(11, y, t, ha='center', va='center', fontsize=7.5, weight='bold', linespacing=1.15)
    cells = [(27, 55, '1  Own MI', 'runs 1–5, labelled\n(own calibration)', False),
             (64, 55, '2  Others’ MI', 'zero calibration', False),
             (27, 25, '3  Own ME', 'same person’s\nexecution session', False),
             (64, 25, '4  Others’ ME', 'zero calibration\n(main transfer)', True)]
    for x, y, t, d, main in cells:
        box(axa, x, y, 35, 28, ec=C['ink'] if main else C['axis'], lw=1.2 if main else 0.6)
        axa.text(x + 17.5, y + 24.5, t, ha='center', va='top', fontsize=7.5, weight='bold')
        axa.text(x + 17.5, y + 17, d, ha='center', va='top', fontsize=FS, color=C['ink2'], linespacing=1.2)
    box(axa, 27, 2, 72, 14, fc=C['ink2'], ec=C['ink2'])
    axa.text(63, 9, 'Test set for every cell:\nthe new person’s MI runs 6–10', ha='center', va='center',
             fontsize=7.5, color='white', linespacing=1.2)
    for x in (44.5, 81.5):
        arrow(axa, x, 25, x, 16.5)

    # (b) data use for each new person
    axb = fig.add_axes([0.53, 0.0, 0.47, 0.975]); blank(axb)
    letter(fig, 0.53, 0.995, 'b', 'Data use for each new person (leave-one-subject-out)')
    axb.text(0, 78.5, 'Other 14\npeople', va='center', ha='left', fontsize=7.5, weight='bold', linespacing=1.15)
    box(axb, 24, 81, 75, 8, 'ME runs 1–10: training the classifier (library)')
    box(axb, 24, 71, 75, 8, 'MI runs 6–10: model selection only', fc='white')
    axb.text(24, 66.3, 'sites, early stopping (2 people), thresholds, window', ha='left',
             va='center', fontsize=FS, color=C['muted'])
    axb.text(0, 57, 'New\nperson', va='center', ha='left', fontsize=7.5, weight='bold', linespacing=1.15)
    box(axb, 24, 52, 42, 10, 'MI runs 1–5')
    box(axb, 67, 52, 32, 10, 'MI runs 6–10: test', fc=C['ink2'], ec=C['ink2'], color='white')
    for N, lab in ((10, '10'), (20, '20'), (30, '30'), (45, '45'), (60, '60'), (90, 'all')):
        x = 24 + 42 * N / 90
        axb.plot([x, x], [49, 52], color=C['ink2'], lw=0.6)
        axb.text(x, 47.8, lab, ha='center', va='top', fontsize=FS, color=C['ink2'])
    axb.text(24, 39, 'budget N = the first N trials (≈ 90 in runs 1–5)', ha='left', va='center', fontsize=FS, color=C['muted'])
    axb.text(0, 31, 'The first N trials are used:', ha='left', va='center', fontsize=7.5, weight='bold')
    for i, t in enumerate(['•  not at all: zero calibration',
                           '•  without labels: recentering (TS+LDA), alignment (deep)',
                           '•  with labels: own calibration, or hybrid (TS+LDA)']):
        axb.text(2, 24 - 7 * i, t, ha='left', va='center', fontsize=FS)
    axb.text(0, 2.5, 'The new person’s runs 6–10 never inform any choice (sites, model, settings).', ha='left', va='center',
             fontsize=FS, color=C['muted'], style='italic')

    save(fig, 'fig1_design')
    plt.close("all")

def fig2():
    # Figure 2 (v2): trial timing and decoders. (a) timing and windows; (b) TS+LDA and ShallowConvNet pipelines. 190 mm. No data needed.
    from matplotlib.patches import FancyBboxPatch, Rectangle

    def blank(ax):
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    def box(ax, x, y, w, h, text='', fc=C['fill'], ec=C['axis'], lw=0.6, fs=FS, color=C['ink']):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.0', fc=fc, ec=ec, lw=lw))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color, linespacing=1.2)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', lw=0.7, color=C['ink2'], mutation_scale=7, shrinkA=0, shrinkB=0))

    fig = plt.figure(figsize=(W2, 72 * MM))

    # (a) trial timing
    axa = fig.add_axes([0.0, 0.0, 0.46, 0.975]); blank(axa)
    letter(fig, 0.0, 0.995, 'a', 'Trial timing (relative to the cue)')
    tx = lambda t: 6 + (t + 2.5) * 92 / 6.5
    axa.add_patch(Rectangle((tx(-2), 24), tx(0) - tx(-2), 56, fc=C['fill'], ec='none'))
    axa.text((tx(-2) + tx(0)) / 2, 77, 'pre-cue\ncontrol', ha='center', va='top', fontsize=FS, color=C['ink2'], linespacing=1.15)
    axa.plot([tx(0)] * 2, [21, 84], color=C['ink'], lw=0.8)
    axa.text(tx(0), 85, 'cue: picture of hand open,\nhand close or rest', ha='center', va='bottom', fontsize=FS, linespacing=1.15)
    for k in range(21):
        axa.plot([tx(0.1 * k), tx(0.1 * k + 1)], [28 + 2.2 * k] * 2, color=C['ts'], lw=0.9, solid_capstyle='butt')
    axa.text(tx(0) + 2, 78, 'analysis window 0–3 s', ha='left', va='center', fontsize=FS, color=C['ink2'])
    axa.plot([tx(0), tx(3)], [24.5, 24.5], color=C['deep'], lw=2.2, solid_capstyle='butt')
    axa.plot([tx(-2.5), tx(4)], [21, 21], color=C['ink2'], lw=0.6)
    for t in range(-2, 5):
        axa.plot([tx(t)] * 2, [20, 21], color=C['ink2'], lw=0.6)
        axa.text(tx(t), 19, str(t).replace('-', '−') + (' s' if t == 4 else ''), ha='center', va='top', fontsize=FS, color=C['ink2'])
    axa.plot([6, 11], [9, 9], color=C['ts'], lw=1.2)
    axa.text(12.5, 9, 'TS+LDA: 21 sliding 1-s windows, decision = mean', ha='left', va='center', fontsize=FS)
    axa.plot([6, 11], [3, 3], color=C['deep'], lw=2.2, solid_capstyle='butt')
    axa.text(12.5, 3, 'ShallowConvNet: one 0–3 s epoch', ha='left', va='center', fontsize=FS)

    # (b) decoders
    axb = fig.add_axes([0.53, 0.0, 0.47, 0.975]); blank(axb)
    letter(fig, 0.53, 0.995, 'b', 'Decoders (sites chosen on training people only)')
    def flow(y, h, items, gap=3.5):
        x = 3
        for i, (w, t) in enumerate(items):
            box(axb, x, y, w, h, t)
            if i < len(items) - 1:
                arrow(axb, x + w + 0.3, y + h / 2, x + w + gap - 0.3, y + h / 2)
            x += w + gap
    axb.plot([3, 9], [88, 88], color=C['ts'], lw=1.6)
    axb.text(10.5, 88, 'TS+LDA (primary)', ha='left', va='center', fontsize=7.5, weight='bold')
    flow(66, 15, [(12, '8 sites'), (22, 'causal filters\n0.3–3, 8–30 Hz'), (8, 'CAR'), (21, 'covariance,\ntangent space'),
                  (14, 'shrinkage\nLDA')])
    axb.text(3, 59, '+ recentering on the new person’s unlabelled trials', ha='left', va='center', fontsize=FS, color=C['ink2'])
    axb.plot([3, 9], [46, 46], color=C['deep'], lw=1.6)
    axb.text(10.5, 46, 'ShallowConvNet', ha='left', va='center', fontsize=7.5, weight='bold')
    flow(24, 15, [(12, '8 sites'), (19, 'causal filter\n0.3–40 Hz'), (8, 'CAR'), (26, 'temporal + spatial\nconvolution'),
                  (17, 'log-power,\ndense layer')], gap=3.2)
    axb.text(3, 17, '+ Euclidean alignment on the new person’s unlabelled trials', ha='left', va='center', fontsize=FS, color=C['ink2'])
    axb.text(3, 10.5, '5 seeds; early stopping on 2 training people', ha='left', va='center', fontsize=FS, color=C['ink2'])
    axb.text(3, 3.5, 'Most-selected sites: Cz, P4, CP3, C1, P3, C2, Pz, C6', ha='left', va='center', fontsize=FS, color=C['muted'])

    save(fig, 'fig2_timing_decoders')
    plt.close("all")

def fig3():
    # Figure 3 (v3, after R.2): electrode study (A.2b; selection bias from A.4a). (a) accuracy vs number of sites; (b) selection frequency map.
    # TS+LDA, others' ME -> new person's MI runs 6-10, strict zero calibration. 190 mm.
    import numpy as np, pandas as pd
    from scipy import stats
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Circle, Polygon, Arc

    A2B, A4 = f'{ROOT}/03_A2b_electrodes', f'{ROOT}/05_A4_four_cells'
    out = rd(f'{A2B}/A2b2_outer.csv'); out = out[out.analysis == 'ME->MI T1 strict']
    ana = rd(f'{A2B}/A2b2_anatomical.csv')
    ref = rd(f'{A2B}/A2b_stage1_references.csv')
    ref = ref[(ref.cell == 'ME others -> MI') & (ref.bands == 'low+mu_beta') & (ref['mode'] == 'strict')]
    a4 = rd(f'{A4}/A4a_results.csv'); a4 = a4[a4.cell == '4 ME others']
    cons = json.load(open(f'{A2B}/A2b2_consensus.json'))

    def mci(v):
        v = np.asarray(v, float); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return v.mean(), v.mean() - h, v.mean() + h

    POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
            'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
    def pos(ch):                       # schematic 10-10 grid: odd = left, even = right, z = midline
        pre = ch.rstrip('z0123456789'); suf = ch[len(pre):]
        x = 0 if suf == 'z' else (-(int(suf) + 1) // 2 if int(suf) % 2 else int(suf) // 2)
        return x, {'F': 2, 'FC': 1, 'C': 0, 'CP': -1, 'P': -2}[pre]

    fig = plt.figure(figsize=(W2, 78 * MM))

    # (a) accuracy vs number of sites
    axa = fig.add_axes([0.065, 0.15, 0.50, 0.72])
    YL = (45, 80)
    axa.axhline(50, color=C['chance'], lw=0.7, ls=(0, (1.5, 1.5)), label='Chance')
    sizes = sorted(out['size'].unique())
    M = np.array([mci(out[out['size'] == k].bal_acc) for k in sizes])
    axa.fill_between(sizes, M[:, 1], M[:, 2], color=light(C['ts'], 0.82), lw=0)
    axa.plot(sizes, M[:, 0], '-o', color=C['ts'], ms=3.2, lw=1.3, label='Selected without the test person')
    asz = sorted(ana['size'].unique())
    A = np.array([mci(ana[ana['size'] == k].bal_acc) for k in asz])
    axa.errorbar(np.array(asz) + 0.35, A[:, 0], yerr=[A[:, 0] - A[:, 1], A[:, 2] - A[:, 0]], fmt='s-', color=C['ink2'],
                 ms=3, lw=1.0, elinewidth=0.6, capsize=0, label='Nearest sites to C3')
    fold8 = mci(a4[a4.analysis == 'T1 fold8 (primary)'].sort_values('subject').bal_acc)
    opt8 = mci(a4[a4.analysis == 'T1 cons8 (optimistic)'].sort_values('subject').bal_acc)
    axa.errorbar([8.45], [opt8[0]], yerr=[[opt8[0] - opt8[1]], [opt8[2] - opt8[0]]], fmt='^', color=C['ink'], mfc='white',
                 ms=4.5, mew=0.9, elinewidth=0.6, capsize=0, label='Same 8 chosen with the test people')
    axa.annotate('', xy=(7.55, opt8[0] - 0.4), xytext=(7.55, fold8[0] + 0.4),
                 arrowprops=dict(arrowstyle='-|>', lw=0.7, color=C['ink2'], mutation_scale=6, shrinkA=0, shrinkB=0))
    axa.text(7.3, (opt8[0] + fold8[0]) / 2, f'+{opt8[0] - fold8[0]:.1f}', ha='right', va='center', fontsize=FS,
             color=C['ink2'], bbox=dict(fc='white', ec='none', pad=0.6))
    for k, (x, mont) in enumerate(((18, 'standard31'), (19.5, 'all61'))):
        m, lo, hi = mci(ref[ref.montage == mont].sort_values('subject').bal_acc)
        axa.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt='D', color=C['ink'], mfc='white', ms=3, mew=0.8,
                     elinewidth=0.6, capsize=0, label='All 31 or 61 sites' if k == 0 else None)
    axa.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 19.5])
    axa.set_xticklabels(['2', '4', '6', '8', '10', '12', '14', '16', '31', '61'])
    for xb in (16.85, 17.15):
        axa.plot([xb - 0.12, xb + 0.12], [YL[0] - 0.7, YL[0] + 0.7], color=C['axis'], lw=0.6, clip_on=False)
    axa.set_xlim(1.2, 20.3); axa.set_ylim(*YL)
    axa.set_xlabel('Number of sites'); axa.set_ylabel('Balanced accuracy (%)')
    h, l = axa.get_legend_handles_labels()
    order = [l.index(x) for x in ('Selected without the test person', 'Nearest sites to C3', 'Same 8 chosen with the test people',
                                  'All 31 or 61 sites', 'Chance')]
    axa.legend([h[i] for i in order], [l[i] for i in order], loc='upper center', ncol=2, handlelength=1.8, columnspacing=1.2,
               labelspacing=0.35, borderaxespad=0.2, frameon=True, facecolor='white', edgecolor='none', framealpha=1)
    letter(fig, 0.0, 0.995, 'a', 'Accuracy vs number of sites (zero calibration)')

    # (b) selection frequency on the scalp
    axb = fig.add_axes([0.60, 0.13, 0.38, 0.76]); axb.set_aspect('equal'); axb.axis('off')
    freq = {s: int(cons['freq'].get(s, 0)) for s in POOL}
    cmap = LinearSegmentedColormap.from_list('seqblue', ['#9cc3f2', '#5f9de9', '#3987e5', '#1c5cab', '#0d366b'])
    norm = Normalize(1, 15)
    R = 3.75
    for patch in (Circle((0, 0), R, fc='none', ec=C['axis'], lw=0.8),
                  Polygon([[-0.4, R - 0.05], [0, R + 0.45], [0.4, R - 0.05]], closed=False, fc='none', ec=C['axis'], lw=0.8),
                  Arc((R, 0), 0.6, 1.4, theta1=-90, theta2=90, ec=C['axis'], lw=0.8),
                  Arc((-R, 0), 0.6, 1.4, theta1=90, theta2=270, ec=C['axis'], lw=0.8)):
        axb.add_patch(patch)
    for s in POOL:
        x, y = pos(s); f = freq[s]; dep = s in cons['sites']
        axb.add_patch(Circle((x, y), 0.43, fc=cmap(norm(f)) if f > 0 else 'white',
                             ec=C['ink'] if dep else (C['axis'] if f == 0 else 'none'), lw=1.2 if dep else 0.5))
        axb.text(x, y, s, ha='center', va='center', fontsize=6.5, color='white' if f >= 8 else C['ink'])
    axb.text(-4.4, 4.5, 'black ring: the 8 most\nfrequently selected sites', ha='left', va='top', fontsize=FS, color=C['ink2'], linespacing=1.15)
    axb.set_xlim(-4.4, 4.4); axb.set_ylim(-4.0, 4.6)
    cax = fig.add_axes([0.67, 0.075, 0.24, 0.025])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', ticks=[1, 5, 10, 15])
    cb.outline.set_linewidth(0.4); cax.tick_params(labelsize=FS, length=2, width=0.5)
    cb.set_label('folds (of 15) with the site among the first 8', fontsize=FS, labelpad=2)
    letter(fig, 0.58, 0.995, 'b', 'Selection frequency per site')

    save(fig, 'fig3_electrodes')
    print('\nchecks (step files): selected 8 sites %.1f (61.0); nearest-C3 8 sites %.1f (55.4); 31 sites %.1f (60.8); 61 sites %.1f (58.3);'
          % (M[sizes.index(8), 0], A[asz.index(8), 0], mci(ref[ref.montage == 'standard31'].bal_acc)[0], mci(ref[ref.montage == 'all61'].bal_acc)[0]))
    print('  A.4a cell 4: fold-specific 8 %.1f (61.0), same 8 chosen with the test people %.1f (66.9)' % (fold8[0], opt8[0]))
    plt.close("all")

def fig4():
    # Figure 4 (v4, after R.2; v2 header follows): four cells and per-person results. (a) T1 four cells, TS+LDA and ShallowConvNet (strict);
    # (b) T2 (open vs close), TS+LDA; (c) per person: best own calibration (N1) vs zero calibration (strict, recentered, deep aligned). 190 mm.
    # v3: deep cell 2 drawn open (it failed its shuffled-label control, A.4b-1c); panel c uses the best own calibration (N1, nested choice).
    import numpy as np, pandas as pd
    from scipy import stats
    from matplotlib.lines import Line2D

    A4, A4B, A9, A10 = (f'{ROOT}/05_A4_four_cells', f'{ROOT}/07_A4b_deep', f'{ROOT}/11_A9_robustness',
                        f'{ROOT}/12_A10_deep_calibration')
    a4 = rd(f'{A4}/A4a_results.csv')
    b1 = rd(f'{A4B}/A4b1_results.csv')
    b1 = b1[(b1.model == 'ShallowConvNet') & (b1['mode'] == 'strict')]
    a9 = rd(f'{A9}/A9_results.csv'); a9 = a9[(a9.montage == 'fold-specific 8') & (a9.t_end == 3.0)]
    a10 = rd(f'{A10}/A10b_results.csv')
    CELLS = ['1 MI self', '2 MI others', '3 ME self', '4 ME others']

    def mci(v):
        v = np.asarray(v, float); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return v.mean(), v.mean() - h, v.mean() + h
    def per_person(df, col='bal_acc'):
        return df.groupby('subject')[col].mean().sort_index()

    ts = {c: per_person(a4[(a4.analysis == 'T1 fold8 (primary)') & (a4.cell == c)]) for c in CELLS}
    t2 = {c: per_person(a4[(a4.analysis == 'T2 fold8') & (a4.cell == c)]) for c in CELLS}
    deep = {c: per_person(b1[b1.cell == c]) for c in CELLS[:3]}
    deep[CELLS[3]] = per_person(a10[a10.method == 'strict'])          # cell 4: 5 seeds (A.10b), as in the rest of the paper

    fig = plt.figure(figsize=(W2, 85 * MM))
    YL = (35, 95)
    rng = np.random.default_rng(1)

    # (a) four cells
    axa = fig.add_axes([0.065, 0.20, 0.365, 0.68])
    axa.axhline(50, color=C['chance'], lw=0.7, ls=(0, (1.5, 1.5)))
    for i, c in enumerate(CELLS):
        for off, d, col in ((-0.18, ts, C['ts']), (0.18, deep, C['deep'])):
            v = d[c].values; m, lo, hi = mci(v)
            axa.scatter(i + off + rng.uniform(-0.05, 0.05, len(v)), v, s=6, color=light(col, 0.45), lw=0, zorder=2)
            failed = (d is deep and c == '2 MI others')
            axa.errorbar([i + off], [m], yerr=[[m - lo], [hi - m]], fmt='o', color=col, ms=4.5, elinewidth=1.0, capsize=0, zorder=3,
                         mfc='white' if failed else col, mew=1.0)
    axa.set_xticks(range(4)); axa.set_xticklabels(['1\nOwn MI', '2\nOthers’ MI', '3\nOwn ME', '4\nOthers’ ME'])
    axa.tick_params(axis='x', length=0); axa.grid(axis='x', visible=False)
    axa.set_xlim(-0.5, 3.5); axa.set_ylim(*YL); axa.set_ylabel('Balanced accuracy (%)')
    axa.legend([Line2D([], [], marker='o', ls='', color=C['ts'], ms=4.5), Line2D([], [], marker='o', ls='', color=C['deep'], ms=4.5),
                Line2D([], [], color=C['chance'], lw=0.7, ls=(0, (1.5, 1.5)))],
               ['TS+LDA', 'ShallowConvNet', 'Chance'], loc='upper center', ncol=3, handlelength=1.5, columnspacing=1.0,
               borderaxespad=0.2, frameon=True, facecolor='white', edgecolor='none', framealpha=1)
    letter(fig, 0.0, 0.995, 'a', 'Hand movement vs rest')

    # (b) T2, TS+LDA
    axb = fig.add_axes([0.495, 0.20, 0.095, 0.68])
    axb.axhline(50, color=C['chance'], lw=0.7, ls=(0, (1.5, 1.5)))
    for i, c in enumerate(CELLS):
        v = t2[c].values; m, lo, hi = mci(v)
        axb.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=6, color=light(C['ts'], 0.45), lw=0, zorder=2)
        axb.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt='o', color=C['ts'], ms=4.5, elinewidth=1.0, capsize=0, zorder=3)
    axb.set_xticks(range(4)); axb.set_xticklabels(['1', '2', '3', '4']); axb.set_xlabel('Cell (TS+LDA)')
    axb.tick_params(axis='x', length=0); axb.grid(axis='x', visible=False)
    axb.set_xlim(-0.6, 3.6); axb.set_ylim(*YL)
    letter(fig, 0.465, 0.995, 'b', 'Open vs close')

    # (c) per person
    axc = fig.add_axes([0.685, 0.20, 0.305, 0.68])
    own = rd(f'{ROOT}/15_R1_followup/N1N3_per_subject.csv').set_index('subject').own_nested.sort_index()
    strict = per_person(a9[a9.method == 'cell 4 others ME (strict)'], 'acc')
    rec = per_person(a9[a9.method == 'recentered 60 trials'], 'acc')
    dal = per_person(a10[a10.method == 'aligned 60'])
    order = own.sort_values().index.tolist()
    for r, s in enumerate(order):
        vals = [own[s], strict[s], rec[s], dal[s]]
        axc.plot([min(vals), max(vals)], [r, r], color=C['grid'], lw=1.6, solid_capstyle='round', zorder=1)
        axc.scatter(strict[s], r, s=16, facecolor='white', edgecolor=C['ts'], lw=0.9, zorder=3)
        axc.scatter(rec[s], r, s=16, color=C['ts'], lw=0, zorder=3)
        axc.scatter(dal[s], r, s=18, marker='D', color=C['deep'], lw=0, zorder=4)
        axc.scatter(own[s], r, s=16, color=C['ink'], lw=0, zorder=5)
    axc.axvline(50, color=C['chance'], lw=0.7, ls=(0, (1.5, 1.5)))
    axc.axvline(60, color=C['muted'], lw=0.6, ls=(0, (4, 2)))
    axc.set_yticks(range(len(order))); axc.set_yticklabels([f'S{s}' for s in order]); axc.tick_params(axis='y', length=0)
    axc.grid(axis='y', visible=False)
    axc.set_xlim(40, 95); axc.set_ylim(-0.7, len(order) - 0.3); axc.set_xlabel('Balanced accuracy (%)')
    axc.legend([Line2D([], [], marker='o', ls='', color=C['ink'], ms=4), Line2D([], [], marker='o', ls='', mfc='white', mec=C['ts'], ms=4),
                Line2D([], [], marker='o', ls='', color=C['ts'], ms=4), Line2D([], [], marker='D', ls='', color=C['deep'], ms=3.8)],
               ['Selected own calibration', 'Zero calibration', 'Recentered, 60 trials', 'ShallowConvNet, aligned'], loc='upper center',
               bbox_to_anchor=(0.42, -0.2), ncol=2, handlelength=1.0, columnspacing=1.0, labelspacing=0.3, borderaxespad=0)
    letter(fig, 0.625, 0.995, 'c', 'Per person, sorted by selected own calibration')

    save(fig, 'fig4_cells_people')
    print('\nchecks: TS+LDA cells %s (67.1/62.0/62.7/61.0)' % ' / '.join(f'{ts[c].mean():.1f}' for c in CELLS))
    print('  ShallowConvNet cells %s (61.4/68.5/64.6 A.4b-1; 66.3 A.10b)' % ' / '.join(f'{deep[c].mean():.1f}' for c in CELLS))
    print('  T2 %s (51.1/49.8/52.3/50.8)' % ' / '.join(f'{t2[c].mean():.1f}' for c in CELLS))
    print('  per person means: selected own calibration %.1f (73.1), strict %.1f (61.0), recentered %.1f (65.4), deep aligned 60 %.1f (68.1)'
          % (own.mean(), strict.mean(), rec.mean(), dal.mean()))
    plt.close("all")

def fig5():
    # Figure 5 (v4, after R.2): (a) accuracy vs trials from the new person (TS+LDA from 1.0 s: own labelled = strongest own-calibration
    # pipeline, recentering, hybrid; ShallowConvNet aligned; 0-3 s own calibration dashed), with 95% CIs; (b) difference of each mode from
    # the selected own calibration (N1 nested choice), 90% CI; rows from 1.0 s are descriptive. 190 mm.
    import numpy as np, pandas as pd
    from scipy import stats
    from matplotlib.lines import Line2D

    F15 = f'{ROOT}/15_R1_followup'
    n5 = rd(f'{F15}/N5_curves.csv'); n5['budget'] = n5['budget'].astype(str)
    best = rd(f'{F15}/N1N3_per_subject.csv').set_index('subject').own_nested.sort_index()
    a10 = rd(f'{ROOT}/12_A10_deep_calibration/A10b_results.csv')
    BUD = ['10', '20', '30', '45', '60', 'all']; XN = {'0': 0, '10': 10, '20': 20, '30': 30, '45': 45, '60': 60, 'all': 90}

    def cv(method, budget, win):
        return n5[(n5.method == method) & (n5.budget == budget) & (n5.window == win)].groupby('subject').acc.mean().sort_index()
    def dp(m):
        return a10[a10.method == m].groupby('subject').bal_acc.mean().sort_index()
    def ci95(v):
        v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v); h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
        return v.mean(), h
    def ci90(d):
        d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
        lo, hi = stats.t.interval(0.90, n - 1, loc=m, scale=se)
        return m, lo, hi

    L = 'from 1.0 s'
    fig = plt.figure(figsize=(W2, 90 * MM))

    # (a) accuracy vs trials from the new person
    axa = fig.add_axes([0.07, 0.33, 0.40, 0.58])
    axa.axhline(best.mean(), color=C['ink2'], lw=0.8, ls=(0, (4, 2)), zorder=1, label=f'Selected own calibration ({best.mean():.1f}%)')
    series = [('Own calibration (1.0 s)', C['ink'], 'o', '-', -1.2, [(XN[b], cv('own labelled', b, L)) for b in BUD]),
              ('Own calibration, 0–3 s', C['muted'], 'o', '--', None, [(XN[b], cv('own labelled', b, '0-3 s')) for b in BUD]),
              ('Library + recentering (1.0 s)', C['ts'], 'o', '-', -0.4,
               [(0, cv('others ME strict', '0', L))] + [(XN[b], cv('others ME + recentering', b, L)) for b in BUD]),
              ('Hybrid (1.0 s)', C['hybrid'], 's', '-', 0.4, [(XN[b], cv('hybrid', b, L)) for b in BUD]),
              ('ShallowConvNet + alignment', C['deep'], 'D', '-', 1.2, [(0, dp('strict'))] + [(XN[b], dp(f'aligned {b}')) for b in BUD])]
    checks = []
    for name, col, mk, ls, off, pts in series:
        x = np.array([p[0] for p in pts], float); st = [ci95(p[1].values) for p in pts]
        y = np.array([s[0] for s in st]); h = np.array([s[1] for s in st])
        if off is not None:                                   # 95% CI as thin bars, slightly offset so they do not overlap
            axa.errorbar(x + off, y, yerr=h, fmt='none', ecolor=light(col, 0.35), elinewidth=0.7, capsize=0, zorder=2)
        axa.plot(x, y, ls, marker=mk, color=col, ms=3.6 if mk != 'D' else 3.2, lw=1.2 if ls == '-' else 0.96, label=name,
                 mfc='white' if ls == '--' else col, mew=0.9, zorder=3)
        checks.append(f'{name[:40]:40s} ' + ' '.join(f'{int(a)}:{v:.1f}' for a, v in zip(x, y)))
    p0 = cv('others ME strict', '0', '0-3 s').mean()
    axa.plot([0], [p0], 'o', ms=3.6, mfc='white', mec=C['ts'], mew=0.9, ls='', label='Zero cal. TS+LDA, 0–3 s', zorder=4)
    xt = [0, 10, 20, 30, 45, 60, 90]
    axa.set_xticks(xt); axa.set_xticklabels([f'{n}\n{n * 7.5 / 60:.1f}' for n in xt])
    axa.set_xlim(-4, 94); axa.set_ylim(48, 84); axa.set_yticks([50, 55, 60, 65, 70, 75, 80])
    axa.set_xlabel('Trials from the new person (minutes of a hand-vs-rest session)'); axa.set_ylabel('Balanced accuracy (%)')
    h_, l_ = axa.get_legend_handles_labels()
    fig.legend(h_, l_, loc='lower left', bbox_to_anchor=(0.01, 0.0), ncol=3, handlelength=2.0, columnspacing=1.5, labelspacing=0.35,
               borderaxespad=0.2)
    letter(fig, 0.0, 0.995, 'a', 'Accuracy vs trials from the new person (mean, 95% CI)')

    # (b) difference from the selected own calibration
    rows = [('TS+LDA, 0–3 s', None, None),
            ('Zero calibration', cv('others ME strict', '0', '0-3 s'), C['ts']),
            ('Recentered, 60 trials', cv('others ME + recentering', '60', '0-3 s'), C['ts']),
            ('Hybrid, 60 trials', cv('hybrid', '60', '0-3 s'), C['hybrid']),
            ('TS+LDA, from 1.0 s (descriptive)', None, None),
            ('Zero calibration', cv('others ME strict', '0', L), C['ts']),
            ('Recentered, 10 trials', cv('others ME + recentering', '10', L), C['ts']),
            ('Recentered, 60 trials', cv('others ME + recentering', '60', L), C['ts']),
            ('Hybrid, 10 trials', cv('hybrid', '10', L), C['hybrid']),
            ('Hybrid, 60 trials', cv('hybrid', '60', L), C['hybrid']),
            ('ShallowConvNet', None, None),
            ('Zero calibration', dp('strict'), C['deep']),
            ('Aligned, 10 trials', dp('aligned 10'), C['deep']),
            ('Aligned, 60 trials', dp('aligned 60'), C['deep'])]
    axb = fig.add_axes([0.715, 0.33, 0.275, 0.58])
    axb.axvspan(-5, 5, color=C['fill'], lw=0, zorder=0)
    axb.axvline(0, color=C['ink2'], lw=0.6)
    n = len(rows); res = []
    for i, (lab, v, col) in enumerate(rows):
        y = n - 1 - i
        if v is None:
            continue
        m, lo, hi = ci90((v - best.reindex(v.index)).values)
        res.append((lab, m, lo, hi))
        axb.plot([lo, hi], [y, y], color=col, lw=1.2, solid_capstyle='butt')
        axb.plot([m], [y], 'o', ms=4.2, color=col, mec='white', mew=0.5)
    axb.set_yticks(range(n)); axb.set_yticklabels([r[0] for r in rows][::-1])
    for t, (lab, v, col) in zip(axb.get_yticklabels(), rows[::-1]):
        if v is None:
            t.set_fontweight('bold'); t.set_color(C['ink'])
    axb.tick_params(axis='y', length=0); axb.grid(axis='y', visible=False)
    axb.set_xlim(-16, 6); axb.set_ylim(-0.7, n - 0.3); axb.set_xticks([-15, -10, -5, 0, 5])
    axb.set_xlabel('Difference from the selected own\ncalibration (points)')
    letter(fig, 0.50, 0.995, 'b', 'Difference from the selected own calibration (90% CI)')

    save(fig, 'fig5_calibration')
    print('\nchecks N5 (mean %):')
    for c in checks:
        print(' ', c)
    print(f'  zero calibration TS+LDA 0-3 s {p0:.1f} (61.0); selected own calibration {best.mean():.1f} (73.1)')
    print('  documented: own (1.0 s) 58.2/65.6/68.3/72.9/73.8/75.8; own 0-3 s 56.5/59.7/61.3/63.4/64.4/67.1; '
          'recentering (1.0 s) 64.6 | 68.6/68.9/68.6/68.9/68.6/69.4; hybrid (1.0 s) 70.1/71.6/73.1/73.4/74.2/75.2; deep 66.3 | 67.4 ... 67.5')
    print('checks difference from the selected own calibration (diff [90% CI]):')
    for lab, m, lo, hi in res:
        print(f'  {lab:24s} {m:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]')
    print('  documented (N1 family D, 0-3 s and deep): zero cal -12.1; rec60 -7.7; hybrid 60 -6.7; deep zero -6.8; aligned 10 -5.7; aligned 60 -5.0')
    plt.close("all")

def fig6():
    # Figure 6 (v3, after R.2): library-size curves (A.11, pre-registered). (a) TS+LDA; (b) ShallowConvNet. 140 mm (1.5 column).
    import numpy as np, pandas as pd
    from scipy import stats

    D = f'{ROOT}/14_A11_13'
    tsl = rd(f'{D}/A1113_ts_library.csv').rename(columns={'acc': 'v'})
    dl = rd(f'{D}/A1113_deep_library.csv').rename(columns={'bal_acc': 'v'})

    def curve(df, mode, Ks):
        """Per-person means over subsets, group mean and 95% CI per K, slope per doubling with a bootstrap CI, and the
        pre-registered verdict (same rule and seed as a11_13_cpu.py)."""
        per = df[df['mode'] == mode].groupby(['subject', 'K']).v.mean().unstack()[Ks]
        x = np.log2(np.array(Ks, float))
        b = np.polyfit(np.tile(x, len(per)), per.values.ravel(), 1)[0]
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            idx = rng.integers(0, len(per), len(per))
            bs.append(np.polyfit(np.tile(x, len(idx)), per.values[idx].ravel(), 1)[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        gain = per[Ks[-1]].mean() - per[Ks[-2]].mean()
        verdict = 'still rising' if (gain >= 1.0 and (lo > 0 or hi < 0)) else ('plateau' if gain < 0.5 else 'inconclusive')
        m = per.mean().values; h = stats.t.ppf(0.975, len(per) - 1) * per.std(ddof=1).values / np.sqrt(len(per))
        return np.array(Ks, float), m, h, b, lo, hi, verdict

    fig = plt.figure(figsize=(W15, 80 * MM))
    YL = (50, 76)
    panels = [('a', 'TS+LDA', tsl, [2, 4, 6, 8, 10, 12, 14], (('strict', 'Zero calibration'), ('recentered 60', 'Recentered, 60 trials')), C['ts']),
              ('b', 'ShallowConvNet', dl, [3, 6, 9, 12], (('strict', 'Zero calibration'), ('aligned 60', 'Aligned, 60 trials')), C['deep'])]
    checks = []
    for k, (L, title, df, Ks, modes, col) in enumerate(panels):
        ax = fig.add_axes([0.10 + 0.47 * k, 0.34, 0.38, 0.55])
        ax.axhline(50, color=C['chance'], lw=0.7, ls=(0, (1.5, 1.5)))
        for j, (mode, lab) in enumerate(modes):
            x, m, h, b, lo, hi, verdict = curve(df, mode, Ks)
            xs = x * (0.97 if j == 0 else 1.03)
            ax.errorbar(xs, m, yerr=h, fmt='-o', color=col, mfc='white' if j == 0 else col, mew=1.0, ms=3.8, lw=1.2,
                        elinewidth=0.6, capsize=0, label=f'{lab}\n{b:+.2f} per doubling: {verdict}')
            checks.append(f'{title} {mode}: ' + ' '.join(f'{int(a)}:{v:.1f}' for a, v in zip(x, m)) +
                          f' | b {b:+.2f} [{lo:+.2f}, {hi:+.2f}] {verdict}')
        ax.set_xscale('log', base=2); ax.set_xticks(Ks); ax.set_xticklabels([str(K) for K in Ks]); ax.minorticks_off()
        ax.set_xlim(Ks[0] / 1.2, Ks[-1] * 1.2); ax.set_ylim(*YL)
        ax.set_xlabel('People in the library (log scale)')
        if k == 0:
            ax.set_ylabel('Balanced accuracy (%)')
        ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.24), handlelength=1.6, labelspacing=0.5, borderaxespad=0)
        letter(fig, 0.0 + 0.47 * k, 0.995, L, title)

    save(fig, 'fig6_library_size')
    print('\nchecks (a11_13_cpu log):')
    for c in checks:
        print(' ', c)
    print('  documented: TS strict 58.0 ... 61.0, b +1.01 plateau; TS recentered 61.7 ... 65.4, b +1.23 inconclusive;')
    print('              deep strict 62.2/63.4/64.0/66.3, b +1.85 still rising; deep aligned 64.4/65.2/65.4/68.1, b +1.56 still rising')
    plt.close("all")

def fig7():
    # Figure 7 (v3, after R.2): detection behaviour (A.12, pre-registered). (a) mean ROC curves with operating points (default rule, and
    # thresholds set on other people for 20% and 10% false-alarm targets); (b) achieved false-alarm rate per person at the 20% target.
    import numpy as np, pandas as pd
    from matplotlib.lines import Line2D

    D = f'{ROOT}/14_A11_13'
    main = rd(f'{D}/A1113_main.csv')
    roc = rd(f'{D}/A1113_roc_mean.csv')

    fig = plt.figure(figsize=(W2, 82 * MM))

    # (a) ROC
    axa = fig.add_axes([0.075, 0.17, 0.30, 0.72]); axa.set_aspect('equal')
    axa.plot([0, 100], [0, 100], color=C['chance'], lw=0.7, ls=(0, (1.5, 1.5)))
    CFG = [('TS own calibration', 'TS+LDA own cal.', C['ink']),
           ('TS recentered 60', 'TS+LDA recentered', C['ts']),
           ('deep aligned 60', 'ShallowConvNet aligned', C['deep'])]
    for cfg, lab, col in CFG:
        r = roc[roc.config == cfg].sort_values('fpr')
        g = main[main.config == cfg]
        axa.plot(100 * r.fpr, 100 * r.tpr, color=col, lw=1.2, label=f'{lab}, AUC {g.auc.mean() / 100:.3f}')
        for (h, f), mk in (((g.hit_default, g.fa_default), 'o'), ((g.hit_20, g.fa_20), '^'), ((g.hit_10, g.fa_10), 's')):
            axa.plot([f.mean()], [h.mean()], mk, color=col, ms=4.2, mec='white', mew=0.6, zorder=4)
    axa.set_xlim(0, 100); axa.set_ylim(0, 100); axa.set_xticks(range(0, 101, 20)); axa.set_yticks(range(0, 101, 20))
    axa.set_xlabel('False-alarm rate (%)'); axa.set_ylabel('Hit rate (%)')
    h1, l1 = axa.get_legend_handles_labels()
    mk = [Line2D([], [], marker=m, ls='', color=C['ink2'], ms=4.2) for m in ('o', '^', 's')]
    axa.legend(h1 + mk, l1 + ['default rule', '20% target', '10% target'], loc='lower right',
               handlelength=1.4, labelspacing=0.3, borderaxespad=0.3, frameon=True, facecolor='white', edgecolor='none', framealpha=1)
    letter(fig, 0.0, 0.995, 'a', 'Detection: ROC and operating points')

    # (b) achieved false-alarm rate per person at the 20% target
    axb = fig.add_axes([0.53, 0.28, 0.455, 0.61])
    CB = [('TS strict', 'TS+LDA\nzero cal.', C['ts'], False), ('TS recentered 60', 'TS+LDA\nrecentered', C['ts'], True),
          ('TS own calibration', 'TS+LDA\nown cal.', C['ink'], True), ('deep strict', 'Shallow-\nConvNet\nzero cal.', C['deep'], False),
          ('deep aligned 60', 'Shallow-\nConvNet\naligned', C['deep'], True)]
    axb.axhspan(0, 20, color=C['fill'], lw=0, zorder=0)
    axb.axhline(20, color=C['ink2'], lw=0.7)
    axb.axhline(30, color=C['muted'], lw=0.6, ls=(0, (4, 2)))
    rng = np.random.default_rng(2)
    for i, (cfg, lab, col, filled) in enumerate(CB):
        v = main[main.config == cfg].fa_20.values
        axb.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=9, color=light(col, 0.45), lw=0, zorder=2)
        axb.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=col, lw=1.6, solid_capstyle='butt', zorder=3)
        CB[i] = (cfg, f'{lab}\n{int((v > 30).sum())} above 30%', col, filled)
    axb.set_xticks(range(len(CB))); axb.set_xticklabels([c[1] for c in CB]); axb.tick_params(axis='x', length=0)
    axb.grid(axis='x', visible=False); axb.set_xlim(-0.6, len(CB) - 0.4); axb.set_ylim(0, 100)
    axb.set_ylabel('Achieved false-alarm rate (%)')
    letter(fig, 0.455, 0.995, 'b', 'False alarms per person at the 20% target')

    save(fig, 'fig7_detection')
    print('\nchecks (A.12): config | AUC | default hit/FA | 20%: hit/FA/n>30 | 10%: hit/FA')
    for cfg in ('TS strict', 'TS recentered 60', 'TS own calibration', 'deep strict', 'deep aligned 60'):
        g = main[main.config == cfg]
        print(f'  {cfg:20s} {g.auc.mean():5.1f} | {g.hit_default.mean():5.1f}/{g.fa_default.mean():5.1f} | '
              f'{g.hit_20.mean():5.1f}/{g.fa_20.mean():5.1f}/{int((g.fa_20 > 30).sum())} | {g.hit_10.mean():5.1f}/{g.fa_10.mean():5.1f}')
    print('  documented: TS strict 71.2 | 64.2/42.2 | 45.3/28.2/5 | 30.0/17.1; TS rec 71.5 | 75.7/44.9 | 49.4/21.6/3 | 32.9/11.1;')
    print('              own 73.6 | 77.1/42.9 | 53.2/20.4/3 | 38.9/12.4; deep strict 74.3 | 66.2/33.8 | 58.4/27.0/5 | 43.1/16.1;')
    print('              deep aligned 75.6 | 67.9/32.0 | 60.5/26.3/6 | 43.5/16.6')
    plt.close("all")

def fig8():
    # Figure 8 (v3, after R.2): decision time (A.9). TS+LDA accuracy when the trial decision uses only the windows that end by time T,
    # with 95% CIs; filled symbols = above 50% (one-sided Wilcoxon, p < 0.05, uncorrected), open = not significant. 90 mm.
    import numpy as np, pandas as pd
    from scipy import stats
    from matplotlib.lines import Line2D

    a9 = rd(f'{ROOT}/11_A9_robustness/A9_results.csv'); a9 = a9[a9.montage == 'fold-specific 8']
    TE = [1.0, 1.5, 2.0, 2.5, 3.0]
    S = [('cell 1 own MI (labelled, ~27 min)', 'Own calibration (0–3 s)', C['ink'], 'o', -0.045),
         ('cell 4 others ME (strict)', 'Zero calibration', C['ts'], '^', -0.015),
         ('recentered 60 trials', 'Recentered, 60 trials', C['ts'], 'o', 0.015),
         ('hybrid 60 trials', 'Hybrid, 60 trials', C['hybrid'], 's', 0.045)]

    fig = plt.figure(figsize=(W1, 84 * MM))
    ax = fig.add_axes([0.15, 0.40, 0.82, 0.52])
    ax.axhline(50, color=C['chance'], lw=0.7, ls=(0, (1.5, 1.5)))
    checks = []
    for m, lab, col, mk, off in S:
        g = a9[a9.method == m]
        v = [g[g.t_end == t].acc.values for t in TE]
        y = np.array([x.mean() for x in v])
        h = np.array([stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) for x in v])
        p = [wilp(x - 50, alternative='greater').pvalue for x in v]
        xs = np.array(TE) + off
        ax.errorbar(xs, y, yerr=h, fmt='none', ecolor=light(col, 0.35), elinewidth=0.7, capsize=0, zorder=2)
        ax.plot(xs, y, '-', color=col, lw=1.1, zorder=3, label=lab)
        for xi, yi, q in zip(xs, y, p):
            ax.plot([xi], [yi], mk, color=col, mfc=col if q < 0.05 else 'white', mew=1.0, ms=4.0 if mk != '^' else 4.4, zorder=4)
        checks.append(f'{lab:24s} ' + ' '.join(f'{a:.1f}{"*" if q < 0.05 else " "}' for a, q in zip(y, p)))
    ax.set_xticks(TE); ax.set_xlim(0.85, 3.15); ax.set_ylim(40, 75)
    ax.set_xlabel('Decision time after the cue (s)'); ax.set_ylabel('Balanced accuracy (%)')
    hs = [Line2D([], [], color=col, lw=1.1, marker=mk, mfc=col, ms=4.0) for _, _, col, mk, _ in S]
    hs += [Line2D([], [], color=C['ink2'], lw=0, marker='o', mfc=C['ink2'], ms=4.0),
           Line2D([], [], color=C['ink2'], lw=0, marker='o', mfc='white', ms=4.0)]
    ax.legend(hs, [s[1] for s in S] + ['above 50% (p < 0.05)', 'not significant'], loc='upper center', bbox_to_anchor=(0.45, -0.22),
              ncol=2, handlelength=1.8, columnspacing=1.0, labelspacing=0.35, borderaxespad=0)
    fig.text(0.0, 0.995, 'Accuracy vs decision time (TS+LDA; mean, 95% CI)', fontsize=FS_TITLE, va='top', ha='left')

    save(fig, 'fig8_decision_time')
    print('\nchecks (* = above 50%, one-sided Wilcoxon p < 0.05):')
    for c in checks:
        print(' ', c)
    print('  documented: own 48.7/48.9/52.7/60.8*/67.1*; strict 50.8/51.6/53.9*/59.4*/61.0*; recentered 51.3/53.3*/55.6*/61.1*/65.4*;'
          ' hybrid 50.9/51.8/55.4*/61.7*/66.3*')
    plt.close("all")

def fig9():
    # Figure 9 (v2): controls for the main transfer (cell 4: others' ME -> new person's MI, strict). TS+LDA (A.4a) and
    # ShallowConvNet (A.4b-1 / A.4b-1c, seeds 0-1 for the reference and the controls alike). 140 mm.
    import numpy as np, pandas as pd
    from scipy import stats
    from matplotlib.lines import Line2D

    a4 = rd(f'{ROOT}/05_A4_four_cells/A4a_results.csv'); a4 = a4[a4.cell == '4 ME others']
    b1 = rd(f'{ROOT}/07_A4b_deep/A4b1_results.csv')
    b1 = b1[(b1.cell == '4 ME others') & (b1.model == 'ShallowConvNet') & (b1['mode'] == 'strict')]
    bc = rd(f'{ROOT}/07_A4b_deep/A4b1c_results.csv')
    bc = bc[(bc.cell == '4 ME others') & (bc.model == 'ShallowConvNet')]

    def mci(v):
        v = np.asarray(v, float); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return v.mean(), v.mean() - h, v.mean() + h
    def ts(analysis, col='bal_acc'):
        return a4[a4.analysis == analysis].groupby('subject')[col].mean()
    def dp(control):
        return bc[bc.control == control].groupby('subject').bal_acc.mean()
    n4 = rd(f'{ROOT}/15_R1_followup/N4_deep_controls.csv')
    def dp4(control):                                   # N4: same code, seeds 0-1
        return n4[n4.control == control].groupby('subject').bal_acc.mean()

    ROWS = [('Full model (0–3 s)', ts('T1 fold8 (primary)'), b1.groupby('subject').bal_acc.mean()),
            ('Without the first 0.5 s', ts('control: fold8 without first 0.5 s'), dp('without first 0.5 s')),
            ('Pre-cue window (−2 to 0 s)', ts('control: fold8 pre-cue (-2 to 0 s)'), dp('pre-cue (-2 to 0 s)')),
            ('Shuffled training labels', ts('T1 fold8 (primary)', 'perm_null_mean'), dp('shuffled training labels')),
            ('Central 5 sites only', ts('control: central5'), dp('central5 only')),
            ('Parietal 3 sites only', ts('control: parietal3'), dp('parietal3 only')),
            ('Nearest 8 sites to C3', ts('control: nearC3_8'), None),
            ('EOG channels only', ts('control: eog3'), dp4('EOG 3 only')),
            ('Frontal 5 sites only', ts('control: frontal5'), dp4('frontal 5 only'))]

    fig = plt.figure(figsize=(W15, 74 * MM))
    ax = fig.add_axes([0.30, 0.22, 0.66, 0.72])
    ax.axvline(50, color=C['chance'], lw=0.7, ls=(0, (1.5, 1.5)))
    n = len(ROWS); checks = []
    for i, (lab, t, d) in enumerate(ROWS):
        y = n - 1 - i
        for v, col, off in ((t, C['ts'], 0.14), (d, C['deep'], -0.14)):
            if v is None:
                continue
            m, lo, hi = mci(v.values)
            ax.plot([lo, hi], [y + off] * 2, color=col, lw=1.1, solid_capstyle='butt')
            ax.plot([m], [y + off], 'o', color=col, ms=4.2, mec='white', mew=0.5)
        checks.append(f'{lab:28s} TS {mci(t.values)[0]:5.1f}' + (f' | deep {mci(d.values)[0]:5.1f}' if d is not None else ''))
        if i == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=C['fill'], lw=0, zorder=0)
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in ROWS][::-1]); ax.tick_params(axis='y', length=0)
    ax.grid(axis='y', visible=False); ax.set_ylim(-0.6, n - 0.4); ax.set_xlim(40, 75)
    ax.set_xlabel('Balanced accuracy (%)')
    ax.legend([Line2D([], [], marker='o', color=C['ts'], ms=4.2, lw=1.1), Line2D([], [], marker='o', color=C['deep'], ms=4.2, lw=1.1)],
              ['TS+LDA', 'ShallowConvNet'], loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, handlelength=1.8,
              columnspacing=1.5, borderaxespad=0)
    fig.text(0.0, 0.995, 'Controls for the main transfer (others’ ME to the new person’s MI, zero calibration)', fontsize=FS_TITLE, va='top', ha='left')

    save(fig, 'fig9_controls')
    print('\nchecks (means):')
    for c in checks:
        print(' ', c)
    print('  documented TS: 61.0, 62.6, 50.2, 50.0, 55.6, 51.6, 55.4, 50.7, 52.1; deep: 66.5, 65.3, 49.5, 50.3, 63.0, 60.9, EOG 51.0, frontal 51.9 (N4)')
    plt.close("all")

def fig10():
    # Figure 10 (v2): PhysioNet EEGMMIDB (106 people), zero calibration, others' ME -> new person's MI. ShallowConvNet (primary)
    # and TS+LDA. (a) left vs right fist: eye-region test (A.7-1); (b) right fist vs rest: montage and visual-confound checks (A.7-2a).
    import numpy as np, pandas as pd
    from scipy import stats
    from matplotlib.lines import Line2D

    r1 = rd(f'{ROOT}/08_A7_physionet/A7_1_results.csv')
    r2 = rd(f'{ROOT}/09_A7_2a_physionet_rest/A72a_results.csv')

    def mci(v):
        v = np.asarray(v, float); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return v.mean(), v.mean() - h, v.mean() + h
    def get(df, model, conds):
        d = df[(df.model == model) & (df.condition.isin(conds))]
        return None if d.empty else d.groupby('subject').bal_acc.mean()     # random removal: mean of the 3 draws per person

    RAND = ['rand25_0 post', 'rand25_1 post', 'rand25_2 post']
    A = [('All 64 sites', ['all64 full']), ('All 64, without first 0.5 s', ['all64 post']),
         ('No frontal/temporal sites, without first 0.5 s', ['noeye39 post']), ('25 random sites removed, without first 0.5 s', RAND),
         ('25 frontal/temporal sites only', ['eye25 full'])]
    B = [('Our 8 sites', ['cons8 strict']), ('Our 8, without first 0.5 s', ['cons8 strict post']),
         ('Shuffled training labels', ['cons8 strict shuffled']), ('Central 5 sites only', ['central5 strict']),
         ('Parietal 3 sites only', ['parietal3 strict']), ('Occipital 5 sites only (visual)', ['occ5 strict']),
         ('Nearest 8 sites to C3', ['nearC3_8 strict']), ('All 64 sites', ['all64 strict'])]

    fig = plt.figure(figsize=(W2, 118 * MM))
    checks = []
    def panel(rect, df, rows, ref, reflab, highlight=None):
        ax = fig.add_axes(rect)
        ax.axvline(50, color=C['chance'], lw=0.7, ls=(0, (1.5, 1.5)))
        ax.axvline(ref, color=C['ink2'], lw=0.7, ls=(0, (4, 2)))
        n = len(rows)
        ax.text(ref - 0.4, n - 0.35, reflab, ha='right', va='center', fontsize=FS, color=C['ink2'])
        for i, (lab, conds) in enumerate(rows):
            y = n - 1 - i
            if i == highlight:
                ax.axhspan(y - 0.5, y + 0.5, color=C['fill'], lw=0, zorder=0)
            line = f'{lab:46s}'
            for model, col, off in (('ShallowConvNet', C['deep'], 0.14), ('TS+LDA', C['ts'], -0.14)):
                v = get(df, model, conds)
                if v is None:
                    line += f' | {model} --'; continue
                m, lo, hi = mci(v.values)
                ax.plot([lo, hi], [y + off] * 2, color=col, lw=1.1, solid_capstyle='butt')
                ax.plot([m], [y + off], 'o', color=col, ms=4.2, mec='white', mew=0.5)
                line += f' | {model} {m:5.1f} (n = {len(v)})'
            checks.append(line)
        ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1]); ax.tick_params(axis='y', length=0)
        ax.grid(axis='y', visible=False); ax.set_ylim(-0.6, n - 0.05); ax.set_xlim(45, 90)
        return ax

    axa = panel([0.30, 0.62, 0.68, 0.31], r1, A, 86.2, 'published (EEGSym, LOSO): 86.2')
    letter(fig, 0.0, 0.995, 'a', 'Left vs right fist (106 people): frontal and temporal sites')
    axb = panel([0.30, 0.10, 0.68, 0.41], r2, B, 81.0, 'left vs right, all 64 (panel a): 81.0', highlight=5)
    axb.set_xlabel('Balanced accuracy (%)')
    letter(fig, 0.0, 0.565, 'b', 'Right fist vs rest (same people)')
    fig.legend([Line2D([], [], marker='o', color=C['deep'], ms=4.2, lw=1.1), Line2D([], [], marker='o', color=C['ts'], ms=4.2, lw=1.1)],
               ['ShallowConvNet (primary)', 'TS+LDA'], loc='upper right', bbox_to_anchor=(0.985, 1.0), ncol=2, handlelength=1.8,
               columnspacing=1.5, borderaxespad=0)

    # pre-specified statistics, recomputed from the per-person results
    sh = lambda conds: get(r1, 'ShallowConvNet', conds)
    full = sh(['all64 full'])
    d_eye = (full - sh(['noeye39 post']).reindex(full.index)); d_rand = (full - sh(RAND).reindex(full.index))
    p_eye = wilp(d_eye - d_rand).pvalue
    c5, o5 = get(r2, 'ShallowConvNet', ['central5 strict']), get(r2, 'ShallowConvNet', ['occ5 strict'])
    p_occ = wilp(c5 - o5.reindex(c5.index), alternative='greater').pvalue   # one-sided, as pre-specified in A.7-2a

    save(fig, 'fig10_physionet')
    print('\nchecks (means):')
    for c in checks:
        print(' ', c)
    print(f'  A.7-1 rule: drop no-eye + post {d_eye.mean():.1f}, drop random + post {d_rand.mean():.1f}, difference '
          f'{(d_eye - d_rand).mean():.1f} (p = {p_eye:.3f})   documented 5.6 / 3.1 / 2.6 (p = 0.012)')
    print(f'  A.7-2a: central 5 - occipital 5 = {(c5 - o5.reindex(c5.index)).mean():+.1f} (p = {p_occ:.3f})   documented +2.5 (p = 0.080)')
    print('  documented A.7-1 Shallow 81.0/77.5/75.3/77.9/78.7, TS 63.4/61.5/62.1/61.5/58.6;')
    print('             A.7-2a Shallow 69.7/68.3/51.0/68.4/58.3/65.9/65.4/72.5, TS 62.2/-/50.9/-/-/-/59.3/64.9')
    plt.close("all")

def graphical_abstract():
    # Graphical abstract for Paper 2 (Elsevier: at least 1328 x 531 px at 300 dpi, 5:2 ratio, read left to right, no heading,
    # legible when reduced to 200 px high; TIFF/EPS/PDF). Numbers are read from the result CSVs, not typed. Version 4 (after R.2):
    # rows 2-3 compare own calibration and the hybrid with the same first 10 labelled trials (same 13 people); row 4 = selected own calibration.
    import pandas as pd
    from matplotlib.patches import Circle, Polygon, Arc, Ellipse, FancyBboxPatch, FancyArrowPatch, Rectangle
    from PIL import Image

    N5 = rd(f'{ROOT}/15_R1_followup/N5_curves.csv', float_precision='round_trip'); N5['budget'] = N5['budget'].astype(str)
    A10B = rd(f'{ROOT}/12_A10_deep_calibration/A10b_results.csv', float_precision='round_trip')
    def n5(method, budget, win):
        return N5[(N5.method == method) & (N5.budget == budget) & (N5.window == win)].groupby('subject').acc.mean().mean()
    d0 = A10B[A10B.method == 'strict'].groupby('subject').bal_acc.mean().mean()
    ts0 = n5('others ME strict', '0', '0-3 s')
    SB = rd(f'{ROOT}/15_R1_followup/N5_same_budget.csv', float_precision='round_trip'); SB['budget'] = SB['budget'].astype(str)
    sb10 = SB[(SB.window == 'from 1.0 s') & (SB.budget == '10')].iloc[0]
    own10, hyb10, n10 = sb10.own_mean, sb10.hybrid_mean, int(sb10.n)
    sel = rd(f'{ROOT}/15_R1_followup/N1N3_per_subject.csv', float_precision='round_trip').own_nested.mean()
    A10C = rd(f'{ROOT}/12_A10_deep_calibration/A10c_results.csv', float_precision='round_trip')
    fa_deep = A10C[A10C.method == 'strict'].groupby('subject').false_alarm.mean().mean()
    A9 = rd(f'{ROOT}/11_A9_robustness/A9_results.csv', float_precision='round_trip')
    fa_ts = A9[(A9.method == 'cell 4 others ME (strict)') & (A9.montage == 'fold-specific 8') & (A9.t_end == 3.0)].false_alarm.mean()
    print(f'checks: deep strict {d0:.1f} (66.3) | TS+LDA strict {ts0:.1f} (61.0) | own 10 {own10:.1f} (58.2, n = {n10}) | '
          f'hybrid 10 same people {hyb10:.1f} (71.2) | selected own {sel:.1f} (73.1) | false alarms deep {fa_deep:.1f} (33.1), TS {fa_ts:.1f} (42.2)')

    FH, FB, FV, FN = 15, 12, 14, 10.5          # header, body, value, note (pt)
    fig = plt.figure(figsize=(200 * MM, 80 * MM), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 250); ax.set_ylim(0, 100); ax.axis('off')

    SITES = [(0, 0), (-1, 0), (1, 0), (3, 0), (-2, -1), (-2, -2), (0, -2), (2, -2)]   # Cz C1 C2 C6 CP3 P3 Pz P4 (top view, nose up)
    def head(cx, cy, r, lw=1.5):
        ax.add_patch(Polygon([[cx - 0.18 * r, cy + 0.97 * r], [cx, cy + 1.18 * r], [cx + 0.18 * r, cy + 0.97 * r]], closed=False,
                             fc='none', ec=C['ink2'], lw=lw, zorder=2))
        for x0, t1, t2 in ((cx + r, -90, 90), (cx - r, 90, 270)):
            ax.add_patch(Arc((x0, cy), 0.25 * r, 0.5 * r, theta1=t1, theta2=t2, ec=C['ink2'], lw=lw, zorder=2))
        ax.add_patch(Circle((cx, cy), r, fc='white', ec=C['ink2'], lw=lw, zorder=3))
        for x, y in SITES:
            ax.add_patch(Circle((cx + x * r / 4.2, cy + y * r / 4.2), 0.09 * r, fc=C['ts'], ec='none', zorder=4))
    def arrow(x0, x1, y=55):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=18, lw=2.2, color=C['ink2'], zorder=5))
    def text(x, y, s, size=FB, **kw):
        ax.text(x, y, s, fontsize=size, ha=kw.pop('ha', 'center'), va=kw.pop('va', 'center'), color=kw.pop('color', C['ink']),
                linespacing=1.15, **kw)

    YH, YC = 91, 19                              # header row, caption row

    # 1) healthy library: other people's executed movement
    text(24, YH, 'Healthy library', FH, weight='bold')
    for cx, cy in ((12, 53), (36, 53), (24, 58)):
        head(cx, cy, 9)
    text(26, YC, '14 people,\nreal hand movement', 11.5)
    arrow(47, 57)

    # 2) detector: 8 standard sites, two decoders
    text(87, YH, '8-site detector', FH, weight='bold')
    ax.add_patch(FancyBboxPatch((62, 32), 50, 48, boxstyle='round,pad=0,rounding_size=4', fc=C['fill'], ec=C['axis'], lw=1.0, zorder=1))
    head(87, 54, 15)
    text(87, YC, 'TS+LDA or\nShallowConvNet')
    arrow(115, 124)

    # 3) new user imagines the movement, no calibration
    text(147, YH, 'New user', FH, weight='bold')
    head(137, 46, 10)
    ax.add_patch(Ellipse((153, 72), 44, 18, fc='white', ec=C['ink2'], lw=1.3, ls=(0, (3, 2)), zorder=3))
    for (x, y), rr in (((145.0, 58.5), 1.1), ((147.0, 62.0), 1.6)):
        ax.add_patch(Circle((x, y), rr, fc='white', ec=C['ink2'], lw=1.1, zorder=3))
    text(153, 72, 'imagines\nhand movement', 11.5, zorder=4)
    text(147, YC, 'imagined movement,\nno calibration')
    arrow(170, 178)

    # 4) result: balanced accuracy, movement vs rest (track = 0-100%, dashed mark = chance 50%)
    text(213, YH, 'Movement vs rest', FH, weight='bold')
    X0, L = 179, 50
    for y, lab, v, col in ((79, 'No calibration', d0, C['deep']), (66, 'Own, 10 labelled trials', own10, C['ink']),
                           (53, 'Same 10 + library', hyb10, C['hybrid']), (40, 'Selected own calibration', sel, C['ink'])):
        text(X0, y + 4.6, lab, 11, ha='left')
        ax.add_patch(Rectangle((X0, y - 3.2), L, 4.4, fc=C['grid'], ec='none', zorder=2))
        ax.add_patch(Rectangle((X0, y - 3.2), L * v / 100, 4.4, fc=col, ec='none', zorder=3))
        ax.plot([X0 + L / 2] * 2, [y - 4.6, y + 2.6], color=C['chance'], lw=1.0, ls=(0, (2, 1.5)), zorder=4)
        text(X0 + L + 1.8, y - 1.0, f'{v:.0f}%', FV, ha='left', weight='bold', color=C['ink'])
    text(X0 + L / 2, 32.5, 'chance', FN, color=C['muted'])
    text(X0, 15, f'15 healthy people, offline.\nNo calibration: ShallowConvNet\n(TS+LDA {ts0:.0f}%). False alarms\n'
         f'{fa_deep:.0f}–{fa_ts:.0f}%. Rows 2–3: {n10} people.', FN, ha='left', color=C['ink2'])

    # save without cropping (exact 200 x 80 mm = 2362 x 945 px at 300 dpi): PDF (vector), TIFF (RGB, LZW), PNG preview
    with mpl.rc_context({'savefig.bbox': 'standard'}):            # p2style crops ('tight'); the abstract keeps its exact size
        fig.savefig(f'{FIG}/graphical_abstract.pdf', facecolor='white')
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=300, facecolor='white'); buf.seek(0)
        im = Image.open(buf).convert('RGB'); im.save(f'{FIG}/graphical_abstract.tif', compression='tiff_lzw', dpi=(300, 300))
        fig.savefig(f'{FIG}/graphical_abstract_preview.png', dpi=150, facecolor='white')
    print(f'TIFF size: {im.size[0]} x {im.size[1]} px (needs at least 1328 x 531; ratio {im.size[0] / im.size[1]:.2f}, needs 2.50)')
    im.resize((round(200 * im.size[0] / im.size[1]), 200), Image.LANCZOS).save(f'{FIG}/graphical_abstract_200px.png')
    plt.close("all")

def tables():
    # make_tables.py v5 (after the R.2 review): Tables 1-6 in the text (renumbered by first citation) and Table S8, built from the saved
    # result files only. New order: 1 methods, 2 10-20 site sets, 3 accuracy, 4 own-calibration decoders, 5 detection, 6 prior work.
    # No new analysis: every number is read from a results CSV and checked against the step documents.
    import os, shutil
    import numpy as np, pandas as pd
    from scipy import stats

    OUT = TAB
    os.makedirs(OUT, exist_ok=True)

    A9 = rd(f'{ROOT}/11_A9_robustness/A9_results.csv', float_precision='round_trip')
    A10B = rd(f'{ROOT}/12_A10_deep_calibration/A10b_results.csv', float_precision='round_trip')
    A10C = rd(f'{ROOT}/12_A10_deep_calibration/A10c_results.csv', float_precision='round_trip')
    D = f'{ROOT}/14_A11_13'
    MAIN = rd(f'{D}/A1113_main.csv', float_precision='round_trip')
    CH = rd(f'{D}/A1113_window_choice.csv', float_precision='round_trip')
    ENS = rd(f'{D}/A1113_ensemble.csv', float_precision='round_trip')
    for df in (A9, A10B, A10C, MAIN, CH, ENS):
        df['subject'] = df['subject'].astype(int)

    # ---------- helpers ----------
    def mci(v):
        v = np.asarray(v, float); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return v.mean(), v.mean() - h, v.mean() + h
    def tost(d, delta=5.0):
        d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
        p = max(1 - stats.t.cdf((m + delta) / se, n - 1), stats.t.cdf((m - delta) / se, n - 1))
        lo, hi = stats.t.interval(0.90, n - 1, loc=m, scale=se)
        return m, lo, hi, p
    RAW, DIAG = {}, []
    def wil(d, label=None, doc=None):
        """Paired Wilcoxon signed-rank test, two-sided. Differences are rounded to 1e-6 points so that equal differences count as ties
        (balanced accuracies lie on a discrete grid; floating-point noise of 1e-14 otherwise decides whether two people tie). Exact null
        distribution when there are no ties and no zero differences, otherwise the normal approximation with tie correction (scipy).
        The p of the earlier code (no rounding, scipy 'auto') is kept for the diagnostic print only."""
        d = np.asarray(d, float)
        raw = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        r = np.round(d, 6); nz = r[r != 0]
        if len(nz) == 0:
            rob, meth = 1.0, 'all zero'
        elif len(np.unique(np.abs(nz))) == len(nz) and len(nz) == len(r):
            rob, meth = float(stats.wilcoxon(r, method='exact').pvalue), 'exact'
        else:
            try:
                rob = float(stats.wilcoxon(r, method='asymptotic').pvalue)
            except ValueError:                      # older scipy names the normal approximation 'approx'
                rob = float(stats.wilcoxon(r, method='approx').pvalue)
            meth = f'normal approx. ({len(nz) - len(np.unique(np.abs(nz)))} tied, {len(r) - len(nz)} zero)'
        if label:
            RAW[label] = raw; DIAG.append((label, doc, raw, rob, meth))
        return rob
    def holm2(p):
        p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(2)
        adj[o[0]] = min(1, 2 * p[o[0]]); adj[o[1]] = max(adj[o[0]], p[o[1]])
        return adj
    MINUS = '−'
    def n1(x):
        return f'{x:.1f}'.replace('-', MINUS)
    def s1(x):
        return f'{x:+.1f}'.replace('-', MINUS)
    def fp(p):
        if p < 0.001:
            return '< 0.001'
        return f'{p:.4f}' if 0.0495 <= p < 0.0505 else f'{p:.3f}'
    def ci(m, lo, hi, signed=False):
        f = s1 if signed else n1
        return f'{f(m)} [{n1(lo)}, {n1(hi)}]'

    CHECKS = []
    def chk(label, got, doc, tol=0.051):
        CHECKS.append(f'  {label:58s} {got:8.3f}  documented {doc:8.3f}  {"OK" if abs(got - doc) <= tol else "DIFFERENT"}')

    def ts(method, what='acc', montage='fold-specific 8'):
        g = A9[(A9.method == method) & (A9.montage == montage) & (A9.t_end == 3.0)]
        return g.set_index('subject')[what].sort_index()
    def deep_ba(method):
        return A10B[A10B.method == method].groupby('subject').bal_acc.mean().sort_index()
    def deep_hf(method, what):
        return A10C[A10C.method == method].groupby('subject')[what].mean().sort_index()
    def a12(cfg):
        return MAIN[MAIN.config == cfg].set_index('subject').sort_index()

    OWN_M = 'cell 1 own MI (labelled, ~27 min)'
    OWN = ts(OWN_M)
    md = []

    # ---------- R.1 follow-up results (N1-N5) ----------
    F15 = f'{ROOT}/15_R1_followup'
    N13 = rd(f'{F15}/N1N3_per_subject.csv', float_precision='round_trip')
    N13['subject'] = N13['subject'].astype(int); N13 = N13.set_index('subject').sort_index()
    N2S = rd(f'{F15}/N2_specificity.csv', float_precision='round_trip'); N2S['subject'] = N2S['subject'].astype(int)
    N3T = rd(f'{F15}/N3_tost.csv', float_precision='round_trip')
    N5C = rd(f'{F15}/N5_curves.csv', float_precision='round_trip')
    N5C['subject'] = N5C['subject'].astype(int); N5C['budget'] = N5C['budget'].astype(str)
    SB = rd(f'{F15}/N5_same_budget.csv', float_precision='round_trip'); SB['budget'] = SB['budget'].astype(str)
    def pf(p):
        return 'p < 0.001' if p < 0.001 else f'p = {fp(p)}'
    def n5(method, budget, win):
        return N5C[(N5C.method == method) & (N5C.budget == budget) & (N5C.window == win)].set_index('subject').acc.sort_index()
    LATE, BUDS = 'from 1.0 s', ['10', '20', '30', '45', '60', 'all']
    BEST = N13.own_nested
    OWNL = {b: n5('own labelled', b, LATE) for b in BUDS}
    def matched(v):
        """Budget interval at which the strongest own-calibration pipeline's group mean first reaches the mode (descriptive)."""
        prev = None
        for b in BUDS:
            if OWNL[b].mean() >= v.mean():
                lab = 'all (≈ 90)' if b == 'all' else b
                return f'≤ {lab}' if prev is None else f'{prev}–{lab}'
            prev = b
        return 'not reached in runs 1–5'
    def dci90(v, ref):
        d = (v - ref.reindex(v.index)).dropna().values; n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
        lo, hi = stats.t.interval(0.90, n - 1, loc=m, scale=se)
        return m, lo, hi
    MIN = lambda n: f'{n * 7.5 / 60:.1f} min'
    OWN_M = 'cell 1 own MI (labelled, ~27 min)'
    md = []

    # ---------- Table 1 (methods summary; text) ----------
    md += ['**Table 1.** Data, evaluation design and pipelines.', '',
           '| Item | Setting |', '|---|---|',
           '| Data | Ofner et al. [21], BNCI Horizon 2020 data set 001-2017 (Zenodo 834976): 15 healthy adults; 61 EEG and 3 EOG channels at 512 Hz; '
           'one executed-movement (ME) and one imagined-movement (MI) session per person, 10 runs each; every run has 6 cued trials of each of 7 classes '
           '(6 right-arm and hand movements, and rest) |',
           '| Task | Hand movement (open or close) vs rest, cue-paced; rest is a cued class inside the same runs. Detection of the other four imagined arm '
           'movements is also reported (Table 5). Open vs close is a secondary task |',
           '| New person (test set) | Leave-one-subject-out over the 15 people. Test set in every analysis: the new person’s MI runs 6–10, all '
           'hand and rest trials (60 movement, 30 rest) |',
           '| Library (training data) | The other 14 people’s ME sessions (trials free of artifacts). Model selection used other training people’s '
           'labelled MI runs 6–10 (site selection, early stopping, thresholds, window choice); nothing from the new person. ShallowConvNet: 12 people '
           'for training and 2 for early stopping |',
           '| Data from the new person | Zero calibration: none. Adaptation: the first N trials of MI runs 1–5 of a hand-vs-rest cued protocol, '
           'without labels (recentering, alignment) or with the cue labels (hybrid). Own calibration: labelled MI runs 1–5 (≈ 90 trials recorded, '
           '79 artifact-free on average). Minutes are minutes of a hand-vs-rest session (7.5 s per trial) |',
           '| Sites | 8 standard 10-10 sites, selected inside each fold without the new person (forward selection from 31 frontal, central and '
           'parietal sites, scored by inner leave-one-subject-out over the 14 training people; the first eight sites of each fold). Most frequently '
           'selected: Cz, P4, CP3, C1, P3, C2, Pz, C6 |',
           '| TS+LDA (primary pipeline) | Causal band-pass 0.3–3 Hz and 8–30 Hz (fourth-order Butterworth); common average reference within the montage '
           '(exact projection to n − 1 virtual channels); one-second windows '
           'starting every 0.1 s from 0 to 2.0 s after the cue (21 windows, decision at 3 s), or only those starting at or after 1.0 s (11 windows); '
           'per-window covariance mapped to a log-Euclidean tangent vector (matrix logarithm); the vectors are averaged, the two bands concatenated, '
           'and the result classified by shrinkage LDA with equal '
           'priors. Recentering: each training session and the new person are centred on their own mean tangent vector. Hybrid: class means '
           'updated with the cue labels of the same trials (prior weight 20 trials per class) |',
           '| ShallowConvNet | Schirrmeister et al. architecture [28] (40 temporal and 40 spatial filters, dropout 0.5; kernel sizes scaled to 128 Hz: '
           'temporal kernel 13, pooling 38, stride 8); causal 0.3–40 Hz; '
           'cue to 3 s; common average reference within the montage; Adam (learning rate 10⁻³), batch 64, class-weighted loss, early stopping '
           '(patience 20, at most 150 epochs); 5 seeds. Alignment: Euclidean alignment on the first N unlabelled trials |',
           '| Own-calibration references | Eight pre-specified labelled decoders (Table 4). Selected own calibration (primary reference): per person, '
           'the decoder with the highest leave-one-run-out accuracy within runs 1–5, then tested on runs 6–10. Strongest own-calibration pipeline: '
           'the decoder with the highest mean test accuracy (TS+LDA, windows from 1.0 s), identified with the test runs |',
           '| Measures | Balanced accuracy; hit rate (movement trials detected); false-alarm rate (rest trials detected as movement); AUC |',
           '| Statistics | Mean and t-based 95% CI over the 15 people; paired Wilcoxon signed-rank tests (two-sided; exact without tied or zero '
           'differences, otherwise the normal approximation with tie correction); TOST with a ±5-point margin fixed in advance (90% CI); '
           '"equivalent" only where the Holm-adjusted TOST p < 0.05 within its pre-specified family; other p-values are uncorrected unless Holm is '
           'stated |', '']
    T1 = md; md = []

    # ---------- Table 2 (accuracy for a new person) ----------
    R2 = [('TS+LDA, windows 0–3 s (primary pipeline)', None, None, None),
          ('Zero calibration', 'None', ts('cell 4 others ME (strict)'), 61.0),
          ('Recentered, 10 trials', f'10 unlabelled ({MIN(10)})', ts('recentered 10 trials'), 63.9),
          ('Recentered, 60 trials', f'60 unlabelled ({MIN(60)})', ts('recentered 60 trials'), 65.4),
          ('Hybrid, 60 trials', f'60 with cue labels ({MIN(60)})', ts('hybrid 60 trials'), 66.3),
          ('Own calibration (earlier reference)', f'≈ 90 labelled ({MIN(90)})', ts(OWN_M), 67.1),
          ('TS+LDA, windows from 1.0 s', None, None, None),
          ('Zero calibration', 'None', n5('others ME strict', '0', LATE), 64.6),
          ('Recentered, 10 trials', f'10 unlabelled ({MIN(10)})', n5('others ME + recentering', '10', LATE), 68.6),
          ('Recentered, 60 trials', f'60 unlabelled ({MIN(60)})', n5('others ME + recentering', '60', LATE), 68.6),
          ('Hybrid, 10 trials', f'10 with cue labels ({MIN(10)})', n5('hybrid', '10', LATE), 70.1),
          ('Hybrid, 60 trials', f'60 with cue labels ({MIN(60)})', n5('hybrid', '60', LATE), 74.2),
          ('Own calibration (strongest pipeline)', f'≈ 90 labelled ({MIN(90)})', n5('own labelled', 'all', LATE), 75.8),
          ('ShallowConvNet (0–3 s)', None, None, None),
          ('Zero calibration', 'None', deep_ba('strict'), 66.3),
          ('Aligned, 10 trials', f'10 unlabelled ({MIN(10)})', deep_ba('aligned 10'), 67.4),
          ('Aligned, 60 trials', f'60 unlabelled ({MIN(60)})', deep_ba('aligned 60'), 68.1),
          ('Reference', None, None, None),
          ('Selected own calibration (chosen on runs 1–5)', f'≈ 90 labelled ({MIN(90)})', BEST, 73.1)]
    md += ['**Table 3.** Accuracy for a new user: zero calibration from other people’s executed movements, adaptation and own calibration '
           '(mean of 15 people).', '',
           '| Pipeline and mode | Data from the new person | Balanced accuracy, % [95% CI] | Difference from the selected own calibration, points '
           '[90% CI] | Labelled trials at which the strongest own-calibration pipeline reaches this level |', '|---|---|---|---|---|']
    csvA = []; grp = ''
    for lab, data, v, doc in R2:
        if v is None:
            md.append(f'| **{lab}** | | | | |'); grp = lab; continue
        m, lo, hi = mci(v.values)
        chk(f'T2 {grp[:22]} {lab}: balanced accuracy', m, doc)
        is_best = v is BEST
        own_row = lab.startswith('Own calibration')
        dtxt = '–' if is_best else ci(*dci90(v, BEST), signed=True)
        mtxt = '–' if (is_best or (own_row and grp.startswith('TS+LDA, windows from'))) else matched(v)
        md.append(f'| {lab} | {data} | {ci(m, lo, hi)} | {dtxt} | {mtxt} |')
        csvA.append(dict(group=grp, mode=lab, data=data, ba=m, ba_lo=lo, ba_hi=hi,
                         diff_best=np.nan if is_best else dci90(v, BEST)[0], matched=mtxt))
    sb = SB[SB.window == LATE].set_index('budget')
    sb_txt = '; '.join(f'{b}: {s1(-sb.loc[b, "own_minus_hybrid"])} [{n1(-sb.loc[b, "ci95_hi"])}, {n1(-sb.loc[b, "ci95_lo"])}]'
                       + (f' (n = {int(sb.loc[b, "n"])})' if int(sb.loc[b, 'n']) < 15 else '') for b in ('10', '20', '30'))
    md += ['',
           'Notes. Test set: the new person’s MI runs 6–10. Strongest own-calibration pipeline: TS+LDA with windows from 1.0 s, labelled trials of runs 1–5 '
           f'(its curve: {", ".join(f"{b}: {OWNL[b].mean():.1f}" for b in BUDS)}%; n = {OWNL["10"].notna().sum()} at 10 trials, where two people '
           'lacked both classes). The last column gives the budget interval in which its group mean first reaches the mode (group means at budgets '
           '10, 20, 30, 45, 60 and all; descriptive). Selected own calibration: Table 4. Differences from it are paired, person by person; none of the '
           'modes in the pre-specified equivalence family (TS+LDA 0–3 s; ShallowConvNet) is within ±5 points after Holm correction; rows from 1.0 s '
           'are descriptive. Hybrid minus own calibration with the same labelled trials (windows from 1.0 s), points [95% CI]: ' + sb_txt +
           ' (Table S4). ShallowConvNet: mean of 5 seeds per person.', '']
    T3new = md; md = []

    # ---------- Table 3 (own-calibration references, N1) ----------
    CAND = [('P0', 'TS+LDA', '8', 'windows 0–3 s (earlier reference)'), ('P1', 'TS+LDA', '8', 'windows from 1.0 s (strongest pipeline)'),
            ('P2', 'TS+LDA', '31', 'windows 0–3 s'), ('P3', 'TS+LDA', '31', 'windows from 1.0 s'),
            ('P4', 'Low-frequency amplitudes + LDA', '61', 'causal 0.3–3 Hz, 15 samples from 0.2 to 3.0 s'),
            ('P5', 'Low-frequency amplitudes + LDA', '31', 'as P4'),
            ('P6', 'FBCSP + LDA', '31', '7 bands 4–32 Hz, 2 CSP pairs each, 0–3 s'), ('P7', 'FBCSP + LDA', '31', 'as P6, 1–3 s')]
    DOCN1 = {'P0': 67.1, 'P1': 75.8, 'P2': 65.6, 'P3': 70.1, 'P4': 62.9, 'P5': 63.0, 'P6': 59.3, 'P7': 68.4}
    cnt = N13.nested_choice.value_counts()
    md += ['**Table 4.** Own labelled calibration on the same split (MI runs 1–5 → runs 6–10): eight pre-specified decoders and the selected own '
           'calibration (mean of 15 people).', '',
           '| # | Decoder | Sites | Features | Test accuracy, % [95% CI] | Difference from P0, points (p) | Leave-one-run-out in runs 1–5, % | '
           'Chosen for (people) |', '|---|---|---|---|---|---|---|---|']
    csvB = []
    for p, dec, sites, feat in CAND:
        v = N13[f'own_{p}']; m, lo, hi = mci(v.values); chk(f'T3 {p}', m, DOCN1[p])
        dd = (v - N13.own_P0).values
        ptxt = '–' if p == 'P0' else f'{s1(dd.mean())} ({fp(wil(dd))})'
        md.append(f'| {p} | {dec} | {sites} | {feat} | {ci(m, lo, hi)} | {ptxt} | {n1(N13[f"loro_{p}"].mean())} | {int(cnt.get(p, 0))} |')
        csvB.append(dict(candidate=p, decoder=dec, sites=sites, features=feat, test=m, test_lo=lo, test_hi=hi, loro=N13[f'loro_{p}'].mean(),
                         chosen=int(cnt.get(p, 0))))
    m, lo, hi = mci(BEST.values); dd = (BEST - N13.own_P0).values; chk('T3 best own calibration', m, 73.1)
    md.append(f'| – | **Selected own calibration**: per person, the decoder with the best leave-one-run-out accuracy | | | {ci(m, lo, hi)} | '
              f'{s1(dd.mean())} ({fp(wil(dd))}) | – | 15 |')
    md += ['', 'Notes. All decoders were fixed before running. Own calibration uses the artifact-free labelled trials of runs 1–5. '
           'The highest test accuracy (P1, 75.8%) is an optimistic upper bound because it is chosen on the test runs; the selected own calibration '
           '(73.1%) is chosen on runs 1–5 only. p: paired Wilcoxon signed-rank test, two-sided, uncorrected.', '']
    T4new = md; md = []

    # ---------- Table 4 (detection behaviour: A.12 operating points and N2 specificity) ----------
    spec = {m: g.set_index('subject') for m, g in N2S.groupby('mode')}
    CFG4 = [('TS+LDA zero calibration', 'TS strict', ts('cell 4 others ME (strict)', 'hit'), ts('cell 4 others ME (strict)', 'false_alarm'), 'strict',
             (71.2, 64.2, 42.2, 45.3, 28.2, 5, 61.3)),
            ('TS+LDA recentered, 60 trials', 'TS recentered 60', ts('recentered 60 trials', 'hit'), ts('recentered 60 trials', 'false_alarm'), 'rec60',
             (71.5, 75.7, 44.9, 49.4, 21.6, 3, 70.0)),
            ('TS+LDA own calibration (0–3 s)', 'TS own calibration', ts(OWN_M, 'hit'), ts(OWN_M, 'false_alarm'), None,
             (73.6, 77.1, 42.9, 53.2, 20.4, 3, None)),
            ('ShallowConvNet zero calibration', 'deep strict', deep_hf('strict', 'hit'), deep_hf('strict', 'false_alarm'), None,
             (74.3, 66.3, 33.1, 58.4, 27.0, 5, None)),
            ('ShallowConvNet aligned, 60 trials', 'deep aligned 60', deep_hf('aligned 60', 'hit'), deep_hf('aligned 60', 'false_alarm'), None,
             (75.6, 68.1, 31.7, 60.5, 26.3, 6, None))]
    md += ['**Table 5.** Detection behaviour (mean of 15 people; rates in %).', '',
           '| Detector | AUC | Hit / false alarm (default rule) | Hit / false alarm (20% target); users above 30% | Hit / false alarm (10% target) | '
           'Other imagined arm movements detected |', '|---|---|---|---|---|---|']
    csvC = []
    for lab, cfg, h, f, smode, doc in CFG4:
        g = a12(cfg); n30 = int((g.fa_20 > 30).sum())
        for k, (nm, val) in enumerate((('AUC', g.auc.mean()), ('hit', h.mean()), ('false alarm', f.mean()), ('hit 20', g.hit_20.mean()),
                                       ('false alarm 20', g.fa_20.mean()), ('users > 30%', n30))):
            chk(f'T4 {lab}: {nm}', val, doc[k], 0 if nm == 'users > 30%' else 0.051)
        arm = '–'
        if smode:
            sg = spec[smode]; arm = n1(sg.other_arm.mean()); chk(f'T4 {lab}: other arm detected', sg.other_arm.mean(), doc[6])
        md.append(f'| {lab} | {g.auc.mean() / 100:.3f} | {n1(h.mean())} / {n1(f.mean())} | {n1(g.hit_20.mean())} / {n1(g.fa_20.mean())}; {n30} | '
                  f'{n1(g.hit_10.mean())} / {n1(g.fa_10.mean())} | {arm} |')
        csvC.append(dict(detector=lab, auc=g.auc.mean(), hit=h.mean(), fa=f.mean(), hit_20=g.hit_20.mean(), fa_20=g.fa_20.mean(), n_fa20_above30=n30,
                         hit_10=g.hit_10.mean(), fa_10=g.fa_10.mean(), other_arm=np.nan if arm == '–' else spec[smode].other_arm.mean()))
    sp_txt = []
    for smode, lab in (('strict', 'zero calibration'), ('rec60', 'recentered')):
        sg = spec[smode]; cls = [c for c in ('elbow_flexion', 'elbow_extension', 'supination', 'pronation') if c in sg.columns]
        sp_txt.append(f'{lab}: ' + ', '.join(f'{c.replace("_", " ")} {n1(sg[c].mean())}' for c in cls) +
                      f'; other arm − hand {s1((sg.other_arm - sg.hand_hit).mean())} points ({pf(wil((sg.other_arm - sg.hand_hit).values))}), '
                      f'other arm − rest {s1((sg.other_arm - sg.rest_fa).mean())} ({pf(wil((sg.other_arm - sg.rest_fa).values))})')
    md += ['', 'Notes. Default rule: posterior probability above 0.5. Targets: threshold set without the new person (TS+LDA: on the 14 training '
           'people; own calibration: leave-one-run-out within runs 1–5; ShallowConvNet: on the 2 early-stopping people). ShallowConvNet: mean of 5 '
           'seeds (default rule from the first five-seed run; AUC and targets from a repeat run with identical code). Other imagined arm movements: '
           'the new person’s MI trials of '
           'elbow flexion and extension, supination and pronation in runs 6–10 (120 trials), passed through the same detector; '
           + '; '.join(sp_txt) + ' (uncorrected). Pre-specified verdict in both modes: partly specific (below hand, closer to hand than to rest).', '']
    T5new = md; md = []

    # ---------- Table 5 (rule-defined 10-20 sets) ----------
    famC = N3T[N3T.family.str.startswith('C:')].set_index('comparison')
    METH5 = [('strict', 'Zero calibration', 'cell 4 others ME (strict)'), ('rec60', 'Recentered, 60 trials', 'recentered 60 trials'),
             ('own', 'Own calibration (0–3 s)', OWN_M)]
    DOC5 = {('1020_6', 'strict'): (61.2, 0.98), ('1020_6', 'rec60'): (63.7, 0.18), ('1020_6', 'own'): (66.3, 0.65),
            ('1020_9', 'strict'): (60.0, 0.95), ('1020_9', 'rec60'): (62.9, 0.21), ('1020_9', 'own'): (68.2, 0.43)}
    md += ['**Table 2.** Rule-defined 10-20 site sets vs the selected sites (TS+LDA, 0–3 s; mean of 15 people).', '',
           '| Site set | Mode | Balanced accuracy, % | Difference from the selected sites, points [90% CI] | Wilcoxon p | TOST p (Holm) | '
           'Equivalent within ±5 |', '|---|---|---|---|---|---|---|']
    for mk, ml, meth in METH5:
        md.append(f'| Selected 8 sites (primary) | {ml} | {n1(ts(meth).mean())} | – | – | – | – |')
    csvD = []
    for key, name in (('1020_6', '10-20, 6 sites: C3, Cz, C4, P3, Pz, P4'), ('1020_9', '10-20, 9 sites: the 6 + F3, Fz, F4')):
        for mk, ml, meth in METH5:
            x = ts(meth, montage=key); d = (x - ts(meth).reindex(x.index)).values
            doc = DOC5[(key, mk)]; chk(f'T5 {key} {mk}: accuracy', x.mean(), doc[0])
            pw_ = wil(d, f'T5 {key} {ml}', doc[1])
            r = famC.loc[f'{key} {mk} - selected 8']
            chk(f'T5 {key} {mk}: difference (N3 file)', r['diff'], d.mean(), 0.001)
            eq = 'yes' if r['p_holm'] < 0.05 else 'not shown'
            md.append(f'| {name} | {ml} | {n1(x.mean())} | {s1(r["diff"])} [{n1(r["ci90_lo"])}, {n1(r["ci90_hi"])}] | {fp(pw_)} | '
                      f'{fp(r["p"])} ({fp(r["p_holm"])}) | {eq} |')
            csvD.append(dict(site_set=name, mode=ml, ba=x.mean(), diff=r['diff'], ci90_lo=r['ci90_lo'], ci90_hi=r['ci90_hi'], p_wilcoxon=pw_,
                             p_tost=r['p'], p_tost_holm=r['p_holm']))
    md += ['', 'Notes. The selected sites are chosen inside each fold without the new person; the 10-20 sets are fixed by rule. TOST: margin ±5 '
           'points, Holm over the six comparisons (pre-specified). Wilcoxon p: two-sided, uncorrected.', '']
    T2new = md; md = []

    # ---------- Table 6 (prior work; literature rows checked against the original papers on 7 and 12 Nov 2026) ----------
    ts0, dp0 = ts('cell 4 others ME (strict)').mean(), deep_ba('strict').mean()
    ts60, dp60 = ts('recentered 60 trials').mean(), deep_ba('aligned 60').mean()
    md += ['**Table 6.** Studies that used the Ofner et al. [21] data, and the closest calibration-free studies on other data.', '',
           '| Study | Data (people) | Task | Trained on → tested on | Across people | Labelled data needed from the new person | Reported result |',
           '|---|---|---|---|---|---|---|',
           '| *Ofner et al. [21] data* | | | | | | |',
           '| Ofner et al. [21] | Ofner 2017 (15) | 6 arm and hand movements; movement vs rest | Own ME → own ME; own MI → own MI | No '
           '| Yes (within-person cross-validation) | Movement vs rest: 87% (ME), 73% (MI) |',
           '| Mammone et al. [22] | Ofner 2017 (15) | Pre-movement vs rest; movement pairs | Own ME → own ME | No | Yes (70% of own trials) '
           '| Pre-movement vs rest: 90.3% |',
           '| Lee et al. [10] | Ofner 2017 | 4 classes (elbow extension, supination, hand open, hand close) | Own ME + own MI → own MI | No '
           '| Yes (80% of own MI; trials split at random) | 69% (chance 25%) |',
           '| Guetschel and Tangermann [23], preprint | Ofner 2017 and other MI data sets | 7 classes (with rest) | Other data set → Ofner 2017, '
           'and the reverse | Across data sets | Yes (16 trials per class, linear read-out) | Near chance in both directions |',
           '| Manivannan and Lakshminarayan [24], preprint | Ofner 2017 | 6 classes (imagery) | Others → new person | Yes | None '
           '| 17.1–18.6% across 7 decoders (chance 16.7%) |',
           f'| **This study** | Ofner 2017 (15) | Hand (open or close) vs rest; also detects other imagined arm movements | **Others’ ME → new person’s MI** | Yes '
           f'| **None** (zero calibration); 10–60 unlabelled trials (adapted) | Zero calibration: {n1(ts0)}% (TS+LDA), {n1(dp0)}% (ShallowConvNet), the '
           f'level the strongest own-calibration pipeline reaches with {matched(ts("cell 4 others ME (strict)"))} and '
           f'{matched(deep_ba("strict"))} labelled trials; selected own calibration {n1(BEST.mean())}% |',
           '| *Closest calibration-free studies on other data* | | | | | | |',
           '| Pérez-Velasco et al. [14] | PhysioNet EEGMMIDB (109) | Left vs right fist; also left vs right vs rest | Others’ ME → new person’s MI | '
           'Yes | None | 86.2% (others’ MI → MI: 86.5%); three classes with rest from a separate baseline run: 78.1% |',
           '| Wimpff et al. [8], preprint | Dreyer2023 (79); Lee2019 (54) | Left vs right hand imagery | Others’ MI → new person’s MI | Yes '
           '| None (unlabelled online adaptation) | 68.3% and 72.4% without adaptation; 71.2% and 76.9% with adaptation |',
           '| Nagarajan et al. [18] | OpenBMI (54 healthy) → stroke (71) | Imagery vs rest | Healthy people’s MI → patient’s MI | Yes '
           '| None, or about half of the patient’s trials for adaptation | 53.0% without adaptation; 71.2% with adaptation |',
           '',
           'Notes. Results are as reported by each study; tasks, number of classes, chance levels and evaluation differ, so the numbers are not '
           'comparable across rows. ME, motor execution (executed movement); MI, motor imagery. Selected studies from searches on 7, 12 and 13 November '
           '2026; preprints are marked. Reference numbers as in the main text.', '']
    T6 = md; md = []
    chk('T6 this study: TS+LDA strict', ts0, 61.0); chk('T6 this study: ShallowConvNet strict', dp0, 66.3)
    chk('T6 this study: TS+LDA recentered 60', ts60, 65.4); chk('T6 this study: ShallowConvNet aligned 60', dp60, 68.1)


    # ---------- Table S7 (A.13, pre-registered accuracy attempts; supplementary) ----------
    md += ['**Table S8.** Two pre-specified attempts to raise accuracy (balanced accuracy, %, mean of 15 people).', '',
           '| Attempt | Mode | Reference | Reference, % | Attempt, % | Difference, points [95% CI] | p (Holm) | Pre-specified verdict |',
           '|---|---|---|---|---|---|---|---|']
    csv4 = []
    pw = []
    DOC4A = {'strict': (61.0, 64.2, 3.2, 0.013, 0.026), 'recentered 60': (65.4, 68.6, 3.2, 0.050, 0.050)}
    for mode, lab in (('strict', 'Zero calibration'), ('recentered 60', 'Recentered, 60 trials')):
        g = a12(f'TS {mode}'); d = g.acc_nested_window - g.acc_default
        pw.append((mode, lab, g.acc_default.mean(), g.acc_nested_window.mean(), mci(d.values), wil(d.values, f'T4 window {mode}', DOC4A[mode][3]),
                   CH[CH['mode'] == mode].start_s.value_counts().sort_index(ascending=False)))
    adj = holm2([x[5] for x in pw])
    HOLM = [(f'T4 window {x[0]}', DOC4A[x[0]][4], r, a) for x, r, a in zip(pw, holm2([RAW[f'T4 window {x[0]}'] for x in pw]), adj)]
    for (mode, lab, a0, a1, dci, p, cnt), pj in zip(pw, adj):
        verdict = 'improves' if (pj < 0.05 and dci[0] > 0) else 'no improvement shown'
        starts = ', '.join(f'{s:.1f} s in {int(c)}' for s, c in cnt.items())
        md.append(f'| Analysis window chosen on the training people (TS+LDA) | {lab} | Fixed 0–3 s window | {n1(a0)} | {n1(a1)} | '
                  f'{ci(*dci, signed=True)} | {fp(p)} ({fp(pj)}) | {verdict} |')
        csv4.append(dict(attempt='window', mode=mode, reference=a0, attempt_acc=a1, diff=dci[0], diff_lo=dci[1], diff_hi=dci[2], p=p,
                         p_holm=pj, verdict=verdict, chosen_starts=starts))
        doc = DOC4A[mode]
        for k, (lbl, v) in enumerate((('0-3 s', a0), ('nested', a1), ('difference', dci[0]))):
            chk(f'T4 window {mode}: {lbl}', v, doc[k])
    pe = []
    DOC4B = {'strict': (67.7, 66.7, -1.0, 0.272, 0.479, 5.7, 0.009, 77.3, 69.6, 36.2),
             'adapted': (69.2, 70.7, 1.5, 0.239, 0.479, 5.3, 0.009, 77.5, 74.4, 33.1)}
    for name, lab in (('strict', 'Zero calibration'), ('adapted', 'Adapted, 60 trials')):
        g = ENS[ENS.config == name]; d = (g.acc_ens - g.acc_deep_seedavg).values
        pe.append((name, lab, g, mci(d), wil(d, f'T4 ensemble {name}', DOC4B[name][3]), (g.acc_ens - g.acc_ts).mean(),
                   wil((g.acc_ens - g.acc_ts).values, f'T4 ensemble {name} vs TS+LDA', DOC4B[name][6])))
    adj = holm2([x[4] for x in pe])
    HOLM += [(f'T4 ensemble {x[0]}', DOC4B[x[0]][4], r, a) for x, r, a in zip(pe, holm2([RAW[f'T4 ensemble {x[0]}'] for x in pe]), adj)]
    ens_note = []
    for (name, lab, g, dci, p, dts, pts), pj in zip(pe, adj):
        verdict = 'improves' if (pj < 0.05 and dci[0] > 0) else 'no improvement shown'
        md.append(f'| Fixed ensemble of TS+LDA and ShallowConvNet | {lab} | ShallowConvNet alone | {n1(g.acc_deep_seedavg.mean())} | '
                  f'{n1(g.acc_ens.mean())} | {ci(*dci, signed=True)} | {fp(p)} ({fp(pj)}) | {verdict} |')
        csv4.append(dict(attempt='ensemble', mode=name, reference=g.acc_deep_seedavg.mean(), attempt_acc=g.acc_ens.mean(), diff=dci[0],
                         diff_lo=dci[1], diff_hi=dci[2], p=p, p_holm=pj, verdict=verdict, ens_minus_ts=dts, p_ens_minus_ts=pts,
                         auc_ens=g.auc_ens.mean(), hit_ens=g.hit_ens.mean(), fa_ens=g.fa_ens.mean()))
        ens_note.append(f'{lab.lower()}: ensemble − TS+LDA {s1(dts)} points ({pf(pts)}); ensemble AUC {g.auc_ens.mean() / 100:.3f}, '
                        f'hit {n1(g.hit_ens.mean())}%, false alarm {n1(g.fa_ens.mean())}%')
        doc = DOC4B[name]
        for k, (lbl, v) in enumerate((('deep seed-averaged', g.acc_deep_seedavg.mean()), ('ensemble', g.acc_ens.mean()), ('difference', dci[0]))):
            chk(f'T4 ensemble {name}: {lbl}', v, doc[k])
        chk(f'T4 ensemble {name}: ensemble - TS+LDA', dts, doc[5])
        for k, v in zip((7, 8, 9), (g.auc_ens.mean(), g.hit_ens.mean(), g.fa_ens.mean())):
            chk(f'T4 ensemble {name}: AUC / hit / false alarm [{k - 6}]', v, doc[k])
    win_starts = '; '.join(f'{x[1].lower()}: {", ".join(f"{s:.1f} s in {int(c)} folds" for s, c in x[6].items())}' for x in pw)
    md += ['',
           'Notes. Both attempts, their candidates and the verdict rule (improves only if the Holm-adjusted p < 0.05 and the difference is positive) '
           'were fixed before the analyses were run. Window choice: the trial decision uses the windows starting at or after 0.0, 0.5 or 1.0 s, '
           'chosen per fold by inner leave-one-subject-out over the 14 training people; the decision is still made at 3 s. '
           f'Chosen start ({win_starts}). Ensemble: mean of the two posterior probabilities, threshold 0.5, nothing tuned; ShallowConvNet = decision '
           'from the mean of its 5 seeds’ probabilities, the same input the ensemble uses (this seed averaging was not pre-specified and its own '
           'gain is not claimed). Adapted = TS+LDA recentered and ShallowConvNet aligned on the first 60 unlabelled trials. '
           + ('; '.join(ens_note))[:1].upper() + ('; '.join(ens_note))[1:] + '. p: paired Wilcoxon signed-rank test, two-sided (exact when there are no tied or zero differences, otherwise '
           'the normal approximation with tie correction); Holm over the 2 modes of each attempt.', '']


    # ---------- save (new order by first citation) ----------
    S8 = md; md = T1 + T2new + T3new + T4new + T5new + T6 + S8
    text = '\n'.join(md)
    for name, obj in (('P2_tables.md', text), ('Table3_accuracy.csv', pd.DataFrame(csvA)), ('Table4_own_references.csv', pd.DataFrame(csvB)),
                      ('Table5_detection.csv', pd.DataFrame(csvC)), ('Table2_1020.csv', pd.DataFrame(csvD)), ('TableS8_attempts.csv', pd.DataFrame(csv4))):
        tmp = os.path.join(tempfile.gettempdir(), name)
        if isinstance(obj, str):
            open(tmp, 'w', encoding='utf-8').write(obj)
        else:
            obj.to_csv(tmp, index=False)
        shutil.copyfile(tmp, f'{OUT}/{name}.part'); os.replace(f'{OUT}/{name}.part', f'{OUT}/{name}')
    print('\n=== tables: checks against the step documents (only differences are listed) ===')
    print('\n'.join(c for c in CHECKS if 'DIFFERENT' in c))
    print(f'\n=== Wilcoxon p: documented | earlier code (no rounding) | tie-robust (used in the tables)   [scipy {__import__("scipy").__version__}] ===')
    for label, doc, raw, rob, meth in DIAG:
        print(f'  {label:40s} {doc:7.3f} | {raw:7.4f} | {rob:7.4f}  ({meth})')
    print('  Holm-adjusted:')
    for label, doc, raw, rob in HOLM:
        print(f'  {label:40s} {doc:7.3f} | {raw:7.4f} | {rob:7.4f}  tie-robust Holm p < 0.05: {"yes" if rob < 0.05 else "no"}')
    n_bad = sum('DIFFERENT' in c for c in CHECKS)
    print(f'\n{len(CHECKS) - n_bad}/{len(CHECKS)} checks OK' + ('' if n_bad == 0 else f'; {n_bad} DIFFERENT: send me these lines'))
    print('saved:', sorted(os.listdir(OUT)))
    plt.close("all")

def supplementary():
    """Supplementary tables S1-S7 (S8 is in P2_tables.md), renumbered by first citation after the R.2 review. Markdown + one long CSV."""
    OUT = TAB
    A9 = rd(f'{ROOT}/11_A9_robustness/A9_results.csv'); A9 = A9[(A9.montage == 'fold-specific 8') & (A9.t_end == 3.0)]
    A10B = rd(f'{ROOT}/12_A10_deep_calibration/A10b_results.csv')
    A10C = rd(f'{ROOT}/12_A10_deep_calibration/A10c_results.csv')
    D = f'{ROOT}/14_A11_13'
    MAIN, CH, ENS = rd(f'{D}/A1113_main.csv'), rd(f'{D}/A1113_window_choice.csv'), rd(f'{D}/A1113_ensemble.csv')
    for df in (A9, A10B, A10C, MAIN, CH, ENS):
        df['subject'] = df['subject'].astype(int)
    SUBJ = sorted(A9.subject.unique())
    F15 = f'{ROOT}/15_R1_followup'
    N13 = rd(f'{F15}/N1N3_per_subject.csv'); N13['subject'] = N13['subject'].astype(int); N13 = N13.set_index('subject').sort_index()
    md, long = [], []

    def table(num, title, cols, notes):
        """cols: list of (header, per-person Series, format). Adds a mean row."""
        md.extend([f'**Table S{num}.** {title}', '', '| Person | ' + ' | '.join(h for h, _, _ in cols) + ' |',
                   '|---|' + '---|' * len(cols)])
        for s in SUBJ:
            md.append(f'| S{s} | ' + ' | '.join(f(v.get(s, np.nan)) for _, v, f in cols) + ' |')
        md.append('| **Mean** | ' + ' | '.join(f(v.mean()) for _, v, f in cols) + ' |')
        md.extend(['', notes, ''])
        for h, v, _ in cols:
            for s in SUBJ:
                long.append(dict(table=f'S{num}', column=h, subject=s, value=v.get(s, np.nan)))
    f1 = lambda x: '–' if pd.isna(x) else f'{x:.1f}'
    f3 = lambda x: '–' if pd.isna(x) else f'{x / 100:.3f}'
    pp = lambda df, col: df.groupby('subject')[col].mean()

    ts = lambda m, col='acc': A9[A9.method == m].set_index('subject')[col]
    OWN = 'cell 1 own MI (labelled, ~27 min)'
    table(1, 'TS+LDA balanced accuracy per person (%, fold-specific 8 sites; windows 0–3 s unless stated).',
          [('Selected own calibration', N13.own_nested, f1), ('Strongest pipeline (own, from 1.0 s)', N13.own_P1, f1),
           ('Own calibration, 0–3 s (cell 1)', ts(OWN), f1), ('Others’ MI (cell 2)', ts('cell 2 others MI'), f1),
           ('Own ME (cell 3)', ts('cell 3 own ME'), f1), ('Zero calibration (cell 4)', ts('cell 4 others ME (strict)'), f1),
           ('Recentered, 10', ts('recentered 10 trials'), f1), ('Recentered, 60', ts('recentered 60 trials'), f1),
           ('Recentered, all', ts('recentered all'), f1), ('Hybrid, 60', ts('hybrid 60 trials'), f1)],
          'Test set: the person’s MI runs 6–10. Selected own calibration: per person, the best of eight pre-specified decoders by '
          'leave-one-run-out accuracy within runs 1–5. Recentered and hybrid use the library (cell 4) and the first N trials of MI runs 1–5.')
    dmeth = ['strict', 'aligned 10', 'aligned 20', 'aligned 30', 'aligned 45', 'aligned 60', 'aligned all']
    table(2, 'ShallowConvNet balanced accuracy per person (%, mean of 5 seeds; library = cell 4).',
          [(m.replace('strict', 'Zero calibration').replace('aligned', 'Aligned,'), pp(A10B[A10B.method == m], 'bal_acc'), f1)
           for m in dmeth],
          'First five-seed run. Aligned N: Euclidean alignment on the first N unlabelled trials of MI runs 1–5.')
    seeds = sorted(A10B.seed.unique())
    md.extend(['**Table S3.** ShallowConvNet group mean per seed (%, 15 people).', '',
               '| Mode | ' + ' | '.join(f'Seed {int(k)}' for k in seeds) + ' | Mean | SD across seeds |', '|---|' + '---|' * (len(seeds) + 2)])
    for m in dmeth:
        g = A10B[A10B.method == m].groupby('seed').bal_acc.mean().reindex(seeds)
        md.append(f'| {m} | ' + ' | '.join(f1(v) for v in g.values) + f' | {f1(g.mean())} | {g.std(ddof=1):.2f} |')
        for k, v in g.items():
            long.append(dict(table='S3', column=m, subject=f'seed {int(k)}', value=v))
    md.extend(['', 'First five-seed run. The group mean changes little between seeds.', ''])
    # S4: calibration curves (TS+LDA): group means, same-budget hybrid - own differences, per-person values from 1.0 s
    N5 = rd(f'{F15}/N5_curves.csv'); N5['budget'] = N5['budget'].astype(str); N5['subject'] = N5['subject'].astype(int)
    SB = rd(f'{F15}/N5_same_budget.csv'); SB['budget'] = SB['budget'].astype(str)
    BB = ['0', '10', '20', '30', '45', '60', 'all']
    md.extend(['**Table S4.** TS+LDA calibration curves: balanced accuracy vs the first N trials of MI runs 1–5 (%).', '',
               '(a) Group means (15 people; own labelled at 10 trials: 13 people).', '',
               '| Windows | Method | ' + ' | '.join(BB) + ' |', '|---|---|' + '---|' * len(BB)])
    for win in ('0-3 s', 'from 1.0 s'):
        for meth, lab in (('others ME strict', 'Zero calibration'), ('others ME + recentering', 'Library + recentering'),
                          ('hybrid', 'Hybrid'), ('own labelled', 'Own labelled')):
            cells = []
            for b in BB:
                g = N5[(N5.window == win) & (N5.method == meth) & (N5.budget == b)]
                cells.append('–' if g.empty else f1(g.groupby('subject').acc.mean().mean()))
            md.append(f'| {win.replace("0-3 s", "0–3 s")} | {lab} | ' + ' | '.join(cells) + ' |')
    md.extend(['', '(b) Hybrid minus own labelled calibration with the same labelled trials (paired; people with an own calibration at '
               'that budget).', '', '| Windows | Budget | n | Own | Hybrid | Hybrid − own, points [95% CI] | Wilcoxon p |',
               '|---|---|---|---|---|---|---|'])
    for _, r in SB.iterrows():
        pv = '< 0.001' if r.wilcoxon_p < 0.001 else f'{r.wilcoxon_p:.3f}'
        sg = lambda x: f'{x:+.1f}'.replace('-', '−')
        md.append(f'| {r.window.replace("0-3 s", "0–3 s")} | {r.budget} | {int(r.n)} | {f1(r.own_mean)} | {f1(r.hybrid_mean)} | '
                  f'{sg(-r.own_minus_hybrid)} [{sg(-r.ci95_hi)}, {sg(-r.ci95_lo)}] | {pv} |')
    md.extend(['', 'Wilcoxon p: two-sided, tie-robust, uncorrected; descriptive.', ''])
    pc = lambda meth, b: N5[(N5.window == 'from 1.0 s') & (N5.method == meth) & (N5.budget == b)].set_index('subject').acc
    cols4 = [(f'Own {b}', pc('own labelled', b), f1) for b in ('10', '20', '30', 'all')] + \
            [(f'Hybrid {b}', pc('hybrid', b), f1) for b in ('10', '20', '30', 'all')] + \
            [(f'Recentered {b}', pc('others ME + recentering', b), f1) for b in ('10', '60')]
    table('4, continued', '(c) TS+LDA calibration curves per person, windows from 1.0 s (%).', cols4,
          'Own labelled: – where a class had fewer than two artifact-free trials. All budgets per person are in the long CSV.')
    for meth in ('own labelled', 'hybrid', 'others ME + recentering'):
        for b in BB[1:]:
            for s_, v_ in pc(meth, b).items():
                long.append(dict(table='S4', column=f'{meth} {b} (from 1.0 s)', subject=s_, value=v_))
    cfg = [('TS strict', 'TS+LDA zero cal.'), ('TS recentered 60', 'TS+LDA recentered 60'), ('TS own calibration', 'TS+LDA own cal.'),
           ('deep strict', 'ShallowConvNet zero cal.'), ('deep aligned 60', 'ShallowConvNet aligned 60')]
    g12 = lambda c, col: MAIN[MAIN.config == c].set_index('subject')[col]
    table(5, 'AUC per person (movement vs rest).', [(lab, g12(c, 'auc'), f3) for c, lab in cfg],
          'ShallowConvNet: mean of the AUCs of 5 seeds (repeat run with identical code).')
    fa = lambda c: pd.Series({s: f'{g12(c, "fa_default")[s]:.1f} | {g12(c, "fa_20")[s]:.1f}' for s in SUBJ})
    md.extend(['**Table S6.** False-alarm rate per person (%, rest trials detected as movement), at the default rule and with a threshold '
               'for a 20% target set without the person.', '',
               '| Person | ' + ' | '.join(f'{lab}: default | {lab}: 20% target' for _, lab in cfg) + ' |', '|---|' + '---|' * (2 * len(cfg))])
    for s in SUBJ:
        md.append(f'| S{s} | ' + ' | '.join(fa(c)[s] for c, _ in cfg) + ' |')
    md.append('| **Mean** | ' + ' | '.join(f'{g12(c, "fa_default").mean():.1f} | {g12(c, "fa_20").mean():.1f}' for c, _ in cfg) + ' |')
    md.append('| **Above 30% at the 20% target** | ' + ' | '.join(f'– | {int((g12(c, "fa_20") > 30).sum())}' for c, _ in cfg) + ' |')
    md.extend(['', 'Default-rule rates of ShallowConvNet here come from the repeat run (Table 5 uses the first five-seed run).', ''])
    for c, _ in cfg:
        for col in ('fa_default', 'fa_20', 'hit_default', 'hit_20'):
            for s in SUBJ:
                long.append(dict(table='S6', column=f'{c} {col}', subject=s, value=g12(c, col)[s]))
    ch = lambda mode: CH[CH['mode'] == mode].set_index('subject').start_s
    en = lambda c, col: ENS[ENS.config == c].set_index('subject')[col]
    table(7, 'Pre-specified attempts to raise accuracy, per person (%).',
          [('Window start, zero cal. (s)', ch('strict'), lambda x: f'{x:.1f}'), ('Zero cal., 0–3 s', g12('TS strict', 'acc_default'), f1),
           ('Zero cal., chosen window', g12('TS strict', 'acc_nested_window'), f1),
           ('Window start, recentered (s)', ch('recentered 60'), lambda x: f'{x:.1f}'),
           ('Recentered, 0–3 s', g12('TS recentered 60', 'acc_default'), f1),
           ('Recentered, chosen window', g12('TS recentered 60', 'acc_nested_window'), f1),
           ('Ensemble, zero cal.', en('strict', 'acc_ens'), f1), ('ShallowConvNet seed-avg., zero cal.', en('strict', 'acc_deep_seedavg'), f1),
           ('Ensemble, adapted', en('adapted', 'acc_ens'), f1), ('ShallowConvNet seed-avg., adapted', en('adapted', 'acc_deep_seedavg'), f1)],
          'Window start chosen per fold on the 14 training people. Ensemble: mean of TS+LDA and ShallowConvNet probabilities, threshold 0.5.')
    md.extend(['Per-subset (library size), per-seed and per-trial results are in the released result files.', ''])
    text = '\n'.join(md)
    for name, obj in (('P2_supplementary.md', text), ('P2_supplementary_long.csv', pd.DataFrame(long))):
        tmp = os.path.join(tempfile.gettempdir(), name)
        if isinstance(obj, str):
            open(tmp, 'w', encoding='utf-8').write(obj)
        else:
            obj.to_csv(tmp, index=False)
        shutil.copyfile(tmp, f'{OUT}/{name}.part'); os.replace(f'{OUT}/{name}.part', f'{OUT}/{name}')
    print(f'supplementary: S1-S7 saved (P2_supplementary.md, {len(long)} values in P2_supplementary_long.csv)')
    print(f'  checks (means): S1 own {ts(OWN).mean():.1f} (67.1), zero cal. {ts("cell 4 others ME (strict)").mean():.1f} (61.0); '
          f'S2 strict {pp(A10B[A10B.method == "strict"], "bal_acc").mean():.1f} (66.3), aligned 60 '
          f'{pp(A10B[A10B.method == "aligned 60"], "bal_acc").mean():.1f} (68.1); S5 AUC deep aligned 60 {g12("deep aligned 60", "auc").mean():.1f} (75.6); '
          f'S1 selected {N13.own_nested.mean():.1f} (73.1), strongest {N13.own_P1.mean():.1f} (75.8)')


def pvalue_audit():
    """Every Wilcoxon p used in the paper's claims (outside Tables 3-4 and Figures 8 and 10, which already use the same function),
    recomputed with one tie-robust function, next to the documented value."""
    OUT = TAB
    A4 = rd(f'{ROOT}/05_A4_four_cells/A4a_results.csv')
    B1 = rd(f'{ROOT}/07_A4b_deep/A4b1_results.csv')
    A8 = rd(f'{ROOT}/10_A8_calibration/A8_results.csv'); A8['budget'] = A8['budget'].astype(str)
    A9 = rd(f'{ROOT}/11_A9_robustness/A9_results.csv'); A9 = A9[(A9.montage == 'fold-specific 8') & (A9.t_end == 3.0)]
    A10B = rd(f'{ROOT}/12_A10_deep_calibration/A10b_results.csv'); A10C = rd(f'{ROOT}/12_A10_deep_calibration/A10c_results.csv')
    MAIN = rd(f'{ROOT}/14_A11_13/A1113_main.csv')
    for df in (A4, B1, A8, A9, A10B, A10C, MAIN):
        df['subject'] = df['subject'].astype(int)
    pp = lambda df, col: df.groupby('subject')[col].mean().sort_index()
    rows = []
    def add(group, label, d, doc, holm_doc=None):
        d = pd.Series(d).dropna()
        p, meth = wil_robust(d.values)
        rows.append(dict(group=group, comparison=label, n=len(d), mean_diff=d.mean(), doc=doc, holm_doc=holm_doc, p=p, method=meth))
    def holm_group(group):
        idx = [i for i, r in enumerate(rows) if r['group'] == group]
        for i, a in zip(idx, holm([rows[i]['p'] for i in idx])):
            rows[i]['holm'] = a

    CELLS = ['1 MI self', '2 MI others', '3 ME self', '4 ME others']
    NAMES = ['own MI', 'others’ MI', 'own ME', 'others’ ME']
    tsc = {c: pp(A4[(A4.analysis == 'T1 fold8 (primary)') & (A4.cell == c)], 'bal_acc') for c in CELLS}
    DOC = {(0, 3): (0.015, 0.087), (0, 1): (0.033, 0.17), (1, 3): (0.61, None), (2, 3): (0.36, None)}
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        doc, hd = DOC.get((i, j), (None, None))
        add('A.4a TS+LDA cells (Holm over 6)', f'cell {i + 1} ({NAMES[i]}) − cell {j + 1} ({NAMES[j]})',
            tsc[CELLS[i]] - tsc[CELLS[j]].reindex(tsc[CELLS[i]].index), doc, hd)
    holm_group('A.4a TS+LDA cells (Holm over 6)')
    sh = B1[(B1.model == 'ShallowConvNet') & (B1['mode'] == 'strict')]
    for c, doc in zip(CELLS, (0.014, 0.056, 0.17, 0.002)):
        dv = pp(sh[sh.cell == c], 'bal_acc')
        add('A.4b-1 ShallowConvNet − TS+LDA, strict (seeds 0–1)', f'cell {c[0]}', dv - tsc[c].reindex(dv.index), doc)
    ts9 = lambda m, col='acc': A9[A9.method == m].set_index('subject')[col].sort_index()
    strict = ts9('cell 4 others ME (strict)')
    for m, doc in (('recentered 10 trials', 0.009), ('recentered 60 trials', 0.005), ('recentered all', 0.010)):
        add('TS+LDA recentering − strict (A.6/A.9)', m, ts9(m) - strict, doc)
    add('TS+LDA (A.8/A.9)', 'hybrid 60 − recentered 60', ts9('hybrid 60 trials') - ts9('recentered 60 trials'), None)
    add('TS+LDA (A.8/A.9)', 'own calibration − recentered all', ts9('cell 1 own MI (labelled, ~27 min)') - ts9('recentered all'), None)
    G = 'A.10 ShallowConvNet − TS+LDA at the same budget (Holm over the 6 aligned budgets)'
    add('A.10 ShallowConvNet − TS+LDA, strict', 'strict', pp(A10B[A10B.method == 'strict'], 'bal_acc') - strict, 0.006)
    for b, doc in (('10', 0.055), ('20', 0.008), ('30', 0.048), ('45', 0.004), ('60', 0.064), ('all', 0.229)):
        tsb = pp(A8[(A8.method == 'others ME + recentering') & (A8.budget == b)], 'acc')
        dv = pp(A10B[A10B.method == f'aligned {b}'], 'bal_acc')
        add(G, f'aligned {b} − recentered {b}', dv - tsb.reindex(dv.index), doc)
    holm_group(G)
    for dm, tm, lab, dh, df_ in (('strict', 'cell 4 others ME (strict)', 'strict − strict', 0.85, 0.11),
                                 ('aligned 10', 'recentered 10 trials', 'aligned 10 − recentered 10', 0.51, 0.088),
                                 ('aligned 60', 'recentered 60 trials', 'aligned 60 − recentered 60', 0.073, 0.11),
                                 ('aligned all', 'recentered all', 'aligned all − recentered all', 0.099, 0.041),
                                 ('aligned 60', 'cell 1 own MI (labelled, ~27 min)', 'aligned 60 − own calibration', 0.073, 0.078)):
        for col, tcol, doc in (('hit', 'hit', dh), ('false_alarm', 'false_alarm', df_)):
            dv = pp(A10C[A10C.method == dm], col)
            add(f'A.10c ShallowConvNet − TS+LDA, {col.replace("_", " ")}', lab, dv - ts9(tm, tcol).reindex(dv.index), doc)
    for a, b, doc, hd in (('deep strict', 'TS strict', 0.055, 0.055), ('deep aligned 60', 'TS recentered 60', 0.008, 0.017)):
        x = MAIN[MAIN.config == a].set_index('subject').auc; z = MAIN[MAIN.config == b].set_index('subject').auc.reindex(x.index)
        add('A.12 AUC (Holm over 2)', f'{a} − {b}', x - z, doc, hd)
    holm_group('A.12 AUC (Holm over 2)')
    F15 = f'{ROOT}/15_R1_followup'
    N13 = rd(f'{F15}/N1N3_per_subject.csv'); N13['subject'] = N13['subject'].astype(int); N13 = N13.set_index('subject').sort_index()
    add('N1 own calibration (Table 4)', 'P1 (from 1.0 s) − P0 (0–3 s)', N13.own_P1 - N13.own_P0, 0.001)
    add('N1 own calibration (Table 4)', 'selected own calibration − P0', N13.own_nested - N13.own_P0, 0.010)
    N2S = rd(f'{F15}/N2_specificity.csv'); N2S['subject'] = N2S['subject'].astype(int)
    for mode, lab, d1, d2 in (('strict', 'zero calibration', 0.046, 0.001), ('rec60', 'recentered 60', 0.009, 0.001)):
        g = N2S[N2S['mode'] == mode].set_index('subject').sort_index()
        add('N2 specificity (TS+LDA)', f'{lab}: other arm − hand', g.other_arm - g.hand_hit, d1)
        add('N2 specificity (TS+LDA)', f'{lab}: other arm − rest', g.other_arm - g.rest_fa, d2)
    N5C = rd(f'{F15}/N5_curves.csv'); N5C['budget'] = N5C['budget'].astype(str); N5C['subject'] = N5C['subject'].astype(int)
    n5c = lambda m, b: N5C[(N5C.method == m) & (N5C.budget == b) & (N5C.window == 'from 1.0 s')].set_index('subject').acc.sort_index()
    for b, doc in (('10', 0.003), ('20', 0.001), ('30', 0.005)):
        add('N5 same budget, from 1.0 s (descriptive)', f'hybrid {b} − own {b}', (n5c('hybrid', b) - n5c('own labelled', b)).dropna(), doc)
    r1 = rd(f'{ROOT}/08_A7_physionet/A7_1_results.csv'); r2 = rd(f'{ROOT}/09_A7_2a_physionet_rest/A72a_results.csv')
    gp = lambda df, conds: df[(df.model == 'ShallowConvNet') & (df.condition.isin(conds))].groupby('subject').bal_acc.mean()
    RANDC = ['rand25_0 post', 'rand25_1 post', 'rand25_2 post']
    full = gp(r1, ['all64 full'])
    de = full - gp(r1, ['noeye39 post']).reindex(full.index); dr = full - gp(r1, RANDC).reindex(full.index)
    add('PhysioNet (ShallowConvNet)', 'left vs right: frontal/temporal drop − random drop', de - dr, 0.012)
    c5, o5 = gp(r2, ['central5 strict']), gp(r2, ['occ5 strict'])
    p1s, m1s = wil_robust((c5 - o5.reindex(c5.index)).dropna().values, alternative='greater')
    rows.append(dict(group='PhysioNet (ShallowConvNet)', comparison='right fist vs rest: central 5 − occipital 5 (one-sided)', n=int(c5.notna().sum()),
                     mean_diff=(c5 - o5.reindex(c5.index)).mean(), doc=0.080, holm_doc=None, p=p1s, method=m1s))

    R = pd.DataFrame(rows)
    def changed(r):
        out = []
        if r['doc'] is not None and not pd.isna(r['doc']) and ((r['doc'] < 0.05) != (r['p'] < 0.05)):
            out.append('raw p crosses 0.05')
        if r.get('holm_doc') is not None and not pd.isna(r.get('holm_doc')) and not pd.isna(r.get('holm', np.nan)) \
                and ((r['holm_doc'] < 0.05) != (r['holm'] < 0.05)):
            out.append('Holm p crosses 0.05')
        return '; '.join(out) if out else ''
    R['change'] = R.apply(changed, axis=1)
    fpv = lambda x: '–' if x is None or pd.isna(x) else ('< 0.001' if x < 0.001 else f'{x:.3f}')
    md = ['**Wilcoxon p-values used in the paper: documented vs recomputed with the tie-robust function**', '',
          '| Analysis | Comparison | n | Mean difference | Documented p | Recomputed p | Recomputed Holm p | Conclusion at 0.05 changes? |',
          '|---|---|---|---|---|---|---|---|']
    for _, r in R.iterrows():
        md.append(f'| {r["group"]} | {r["comparison"]} | {r["n"]} | {r["mean_diff"]:+.1f} | {fpv(r["doc"])} | {fpv(r["p"])} | '
                  f'{fpv(r.get("holm", np.nan))} | {r["change"] or "no"} |')
    md += ['', 'Tie-robust: differences rounded to 10⁻⁶ points; exact null distribution when no differences are tied or zero, '
           'otherwise the normal approximation with tie correction. Documented p: the step documents (– = not documented as a number). '
           'Tables 2–5 and Figures 8 and 10 already use this function.', '']
    text = '\n'.join(md)
    tmp = os.path.join(tempfile.gettempdir(), 'P2_pvalues.md'); open(tmp, 'w', encoding='utf-8').write(text)
    shutil.copyfile(tmp, f'{OUT}/P2_pvalues.md.part'); os.replace(f'{OUT}/P2_pvalues.md.part', f'{OUT}/P2_pvalues.md')
    R.to_csv(f'{OUT}/P2_pvalues.csv', index=False)
    print(f'p-value audit: {len(R)} comparisons recomputed')
    for _, r in R.iterrows():
        flag = f'   <-- {r["change"]}' if r['change'] else ''
        print(f'  {r["group"][:44]:44s} {r["comparison"][:36]:36s} doc {fpv(r["doc"]):>7s} | new {fpv(r["p"]):>7s}'
              f'{" | Holm " + fpv(r.get("holm", np.nan)) if not pd.isna(r.get("holm", np.nan)) else ""}{flag}')
    n_ch = int((R['change'] != '').sum())
    print(f'  => {n_ch} conclusion(s) change at 0.05' + (': send me these lines' if n_ch else ''))


# ---------------- run everything ----------------
STEPS = [('Figure 1', fig1), ('Figure 2', fig2), ('Figure 3', fig3), ('Figure 4', fig4), ('Figure 5', fig5), ('Figure 6', fig6),
         ('Figure 7', fig7), ('Figure 8', fig8), ('Figure 9', fig9), ('Figure 10', fig10), ('Graphical abstract', graphical_abstract), ('Tables 1-6, S8', tables),
         ('Supplementary tables', supplementary), ('p-value audit', pvalue_audit)]
STATUS = []
ONLY = [x.strip() for x in os.environ.get('P2_ONLY', '').split(',') if x.strip()]   # e.g. 'Figure 5,Graphical abstract'
if ONLY:
    STEPS = [(n, f) for n, f in STEPS if n in ONLY]
    print('running only:', [n for n, _ in STEPS])
for name, f in STEPS:
    print(f'\n########## {name} ##########', flush=True)
    try:
        f(); STATUS.append((name, 'OK'))
    except Exception as e:
        traceback.print_exc(); STATUS.append((name, f'FAILED: {type(e).__name__}: {e}'))
    plt.close('all')
print('\n########## summary ##########')
for name, s in STATUS:
    print(f'  {name:22s} {s}')
print('figures:', FIG)
print('tables:', TAB)
