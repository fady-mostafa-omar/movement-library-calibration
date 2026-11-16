# Steps A.11-A.13, CPU part (pre-registered 12 Nov 2026, STEP_A11_13_PREREG.md). TS+LDA code and cache identical to A.9.
# A.11 library-size curve (TS+LDA; deep summary from the GPU run) | A.12 operating point (threshold chosen without the new
# person) and AUC | A.13a analysis-window start chosen on training people | A.13b fixed ensemble of TS+LDA and ShallowConvNet
import os, re, sys, json, time, collections, warnings
import numpy as np, pandas as pd, mne
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
A1, A2B, OUT = f'{ROOT}/01_preprocessed', f'{ROOT}/03_A2b_electrodes', f'{ROOT}/14_A11_13'
os.makedirs(OUT, exist_ok=True)
def log(msg):
    print(f'{msg} ({(time.time() - t0) / 60:.1f} min)', flush=True)
print(f'===== A.11-13 CPU started {time.strftime("%H:%M:%S")} =====', flush=True)

SUBJ, FS = list(range(1, 16)), 128
WARM = 60                                   # recentering warm-up (trials), as A.9
KS_TS, NSUB = [2, 4, 6, 8, 10, 12, 14], 10   # A.11 library sizes and random subsets per size
ALPHAS = (0.20, 0.10)                        # A.12 target false-alarm rates (primary, secondary)
J0 = (0, 5, 10)                              # A.13a first window index: starts 0.0, 0.5, 1.0 s (21 windows, step 0.1 s)
POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
CH = json.load(open(f'{A1}/ME_sub01_meta.json'))['channels']
EEG = CH[:61]

mname = 'colin27_1005' if 'colin27_1005' in mne.channels.get_builtin_montages() else 'standard_1005'
info = mne.create_info(EEG, FS, 'eeg')
info.set_montage(mne.channels.make_standard_montage(mname), match_case=False)
POS = np.array([ch['loc'][:3] for ch in info['chs']])
_, ORIGIN, _ = mne.bem.fit_sphere_to_headshape(info, dig_kinds=('eeg',), units='m', verbose=False)

def interp_matrix(names, bads):
    n = len(names); W = np.eye(n)
    b = [i for i, c in enumerate(names) if c in bads]; g = [i for i in range(n) if i not in b]
    if b and len(g) >= 3:
        P = POS[[EEG.index(c) for c in names]] - ORIGIN
        W[b] = 0.0
        W[np.ix_(b, g)] = _make_interpolation_matrix(P[g], P[b])
    return W

def logvec(C):
    m = C.shape[-1]
    tr = np.trace(C, axis1=-2, axis2=-1)[..., None, None] / m
    w, V = np.linalg.eigh(C + 1e-5 * tr * np.eye(m))
    L = (V * np.log(np.maximum(w, 1e-12))[..., None, :]) @ np.swapaxes(V, -1, -2)
    iu = np.triu_indices(m)
    return L[..., iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2))

QB = {}
def Qm(n):
    if n not in QB:
        QB[n] = np.linalg.qr(np.c_[np.ones(n), np.eye(n)[:, :n - 1]])[0][:, 1:]
    return QB[n]

DATA, BADS = {'ME': {}, 'MI': {}}, {}
for cond in ('ME', 'MI'):
    for s in SUBJ:
        fn = f'{A2B}/cache/{cond}_sub{s:02d}.npz'
        if not os.path.exists(fn):
            sys.exit(f'missing cache file: {fn}')
        DATA[cond][s] = dict(np.load(fn))
        BADS[(cond, s)] = [c for c in json.load(open(f'{A1}/{cond}_sub{s:02d}_meta.json'))['bad_channels'] if c in EEG]
log('A.2b cache loaded')

# fold-specific 8-site montages (same parser and check as A.8)
def flat(x):
    out = []
    for v in x:
        out += flat(v) if isinstance(v, (list, tuple)) else [v]
    return out
def to_idx(v):
    if isinstance(v, dict):
        v = next((v[k] for k in ('path', 'order', 'sites', 'selected', 'selection') if isinstance(v.get(k), (list, tuple))), None)
    if not isinstance(v, (list, tuple)):
        return None
    res = []
    for e in flat(v):
        if isinstance(e, str) and e in POOL:
            res.append(POOL.index(e))
        elif isinstance(e, (int, float)) and float(e).is_integer() and 0 <= int(e) < 31:
            res.append(int(e))
        else:
            return None
    return res
EXPECT = {'Cz': 13, 'P4': 13, 'CP3': 12, 'C1': 11, 'P3': 11, 'C2': 7}
def valid(m):
    if sorted(m) != SUBJ or any(v is None or len(v) < 8 for v in m.values()):
        return False
    cnt = collections.Counter(POOL[i] for v in m.values() for i in v[:8])
    return all(cnt.get(k, 0) == n for k, n in EXPECT.items())
cands = []
def walk(o, where):
    if isinstance(o, dict):
        try:
            keys = sorted(int(k) for k in o)
        except (ValueError, TypeError):
            keys = None
        if keys == SUBJ:
            cands.append((where, {int(k): to_idx(v) for k, v in o.items()}))
        for k, v in o.items():
            if isinstance(v, (dict, list, tuple)):
                walk(v, f'{where}/{k}')
    elif isinstance(o, (list, tuple)):
        if len(o) == 15 and all(isinstance(v, (list, tuple, dict)) for v in o):
            cands.append((where, {i + 1: to_idx(v) for i, v in enumerate(o)}))
        for i, v in enumerate(o):
            if isinstance(v, (dict, list, tuple)):
                walk(v, f'{where}[{i}]')
st = json.load(open(f'{A2B}/A2b2_state.json')) if os.path.exists(f'{A2B}/A2b2_state.json') else {}
walk(st, 'A2b2_state.json')
if os.path.exists(f'{A2B}/A2b2_outer.csv'):
    O = pd.read_csv(f'{A2B}/A2b2_outer.csv')
    scol = next((c for c in O.columns if c.lower() in ('subject', 'test_subject', 'outer', 'outer_subject')), None)
    for c in O.columns:
        if scol and O[c].dtype == object:
            m = {}
            for _, r in O.iterrows():
                toks = [x for x in re.split(r'[^A-Za-z0-9]+', str(r[c])) if x in POOL]
                if len(toks) == 8:
                    m[int(r[scol])] = [POOL.index(x) for x in toks]
            if len(m) == 15:
                cands.append((f'A2b2_outer.csv:{c}', m))
good = [(w, m) for w, m in cands if valid(m)]
if not good:
    sys.exit('could not identify the fold-specific montages (A.8 parser): send me this line')
MONT = {t: sorted(good[0][1][t][:8]) for t in SUBJ}

def wfeats(cond, s, S, target):
    """Per-window log-Euclidean features (n, 21, d), CAR within the montage; interpolation as in A.2b/A.4a."""
    D, parts = DATA[cond][s], []
    for b in ('low', 'mu_beta'):
        if target and f'cov31raw_{b}' in D and any(POOL[i] in BADS[(cond, s)] for i in S):
            C = D[f'cov31raw_{b}'][..., S, :][..., S].astype(np.float64)
            W = interp_matrix([POOL[i] for i in S], BADS[(cond, s)])
            C = W @ C @ W.T
        else:
            C = D[f'cov31_{b}'][..., S, :][..., S].astype(np.float64)
        Q = Qm(len(S))
        parts.append(logvec(Q.T @ C @ Q))
    return np.concatenate(parts, axis=-1)
def lda():
    return LDA(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])

from sklearn.metrics import roc_auc_score, roc_curve
bal = balanced_accuracy_score
GRID = np.linspace(0, 1, 101)

def holm2(p):
    """Holm adjustment for two p-values (returned in the same order)."""
    p = np.asarray(p, float); o = np.argsort(p)
    adj = np.empty(2); adj[o[0]] = min(1, 2 * p[o[0]]); adj[o[1]] = max(adj[o[0]], p[o[1]])
    return adj
def wil(d):
    d = np.asarray(d, float)
    return float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
def mci(v):
    v = np.asarray(v, float); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h
def roc_on_grid(y, s):
    f, tp, _ = roc_curve(y, s)
    return np.interp(GRID, f, tp)
def rates(y, pred):
    return 100 * np.mean(pred[y == 1] == 1), 100 * np.mean(pred[y == 0] == 1)

LIBR, MAIN, CHOICE, SCORES = [], [], [], []
for t in SUBJ:
    S = MONT[t]
    others = [s for s in SUBJ if s != t]
    F = {s: wfeats('ME', s, S, False) for s in others}
    FM = {s: wfeats('MI', s, S, False) for s in others}       # others' MI (interpolated from 61 channels), inner tests only
    Ft = wfeats('MI', t, S, True)                              # the new person's MI (interpolated within the montage)
    Dt = DATA['MI'][t]; te = Dt['run'] >= 6; yte = Dt['y'][te]
    cal = np.where(Dt['run'] <= 5)[0]; r4m = Dt['r4m'].astype(bool)
    clean = {s: ~DATA['ME'][s]['r4m'].astype(bool) for s in others}

    def trainset(people, j0, rec):
        Xs, ys = [], []
        for s in people:
            X = F[s][:, j0:].mean(axis=1)[clean[s]]
            Xs.append(X - X.mean(0) if rec else X); ys.append(DATA['ME'][s]['y'][clean[s]])
        return np.vstack(Xs), np.concatenate(ys)
    def target(Fx, test_mask, calidx, j0, rec):
        X = Fx[:, j0:].mean(axis=1)
        return X[test_mask] - X[calidx[:WARM]].mean(0) if rec else X[test_mask]

    for rec, mode in ((False, 'strict'), (True, 'recentered 60')):
        # A.11: library-size curve (same random subsets in both modes)
        Xte = target(Ft, te, cal, 0, rec)
        for K in KS_TS:
            rng = np.random.default_rng(1000 * t + K)
            subsets = [others] if K == 14 else [list(rng.choice(others, K, replace=False)) for _ in range(NSUB)]
            for r, ppl in enumerate(subsets):
                X, y = trainset(ppl, 0, rec)
                LIBR.append(dict(subject=t, K=K, rep=r, mode=mode, acc=100 * bal(yte, lda().fit(X, y).predict(Xte))))
        # inner leave-one-subject-out over the 14 training people: window choice (A.13a) and threshold scores (A.12)
        inner_acc = {j0: [] for j0 in J0}; rest_inner = []
        for j in others:
            Dj = DATA['MI'][j]; tj = Dj['run'] >= 6; cj = np.where(Dj['run'] <= 5)[0]; yj = Dj['y'][tj]
            for j0 in J0:
                X, y = trainset([s for s in others if s != j], j0, rec)
                m = lda().fit(X, y); Xj = target(FM[j], tj, cj, j0, rec)
                inner_acc[j0].append(100 * bal(yj, m.predict(Xj)))
                if j0 == 0:
                    rest_inner.append(m.decision_function(Xj)[yj == 0])
        best = max(J0, key=lambda j0: (np.mean(inner_acc[j0]), -j0))
        rest_inner = np.concatenate(rest_inner)
        # final models on all 14
        X, y = trainset(others, 0, rec); m0 = lda().fit(X, y); X0 = target(Ft, te, cal, 0, rec)
        d0 = m0.decision_function(X0); p0 = m0.predict_proba(X0)[:, 1]
        Xb, yb = trainset(others, best, rec); mb = lda().fit(Xb, yb)
        pb = mb.predict(target(Ft, te, cal, best, rec))
        CHOICE.append(dict(subject=t, mode=mode, start_s=best / 10,
                           **{f'inner_{j0 / 10:.1f}s': np.mean(inner_acc[j0]) for j0 in J0}))
        row = dict(subject=t, config=f'TS {mode}', acc_default=100 * bal(yte, (d0 > 0).astype(int)),
                   auc=100 * roc_auc_score(yte, d0), acc_nested_window=100 * bal(yte, pb))
        row['hit_default'], row['fa_default'] = rates(yte, (d0 > 0).astype(int))
        for a in ALPHAS:
            thr = np.quantile(rest_inner, 1 - a)
            row[f'hit_{int(a * 100)}'], row[f'fa_{int(a * 100)}'] = rates(yte, (d0 > thr).astype(int))
        MAIN.append(row)
        SCORES.append(pd.DataFrame(dict(subject=t, config=f'TS {mode}', trial=np.arange(len(yte)), y_true=yte,
                                        logodds=d0, p_move=p0)))
    # own labelled calibration (cell 1): threshold by leave-one-run-out within MI runs 1-5
    X0 = Ft.mean(axis=1); lab = cal[~r4m[cal]]
    m1 = lda().fit(X0[lab], Dt['y'][lab]); d1 = m1.decision_function(X0[te])
    rest_own = []
    for r in range(1, 6):
        trn = lab[Dt['run'][lab] != r]; tst = cal[Dt['run'][cal] == r]
        sc = lda().fit(X0[trn], Dt['y'][trn]).decision_function(X0[tst]); rest_own.append(sc[Dt['y'][tst] == 0])
    rest_own = np.concatenate(rest_own)
    row = dict(subject=t, config='TS own calibration', acc_default=100 * bal(yte, (d1 > 0).astype(int)),
               auc=100 * roc_auc_score(yte, d1))
    row['hit_default'], row['fa_default'] = rates(yte, (d1 > 0).astype(int))
    for a in ALPHAS:
        row[f'hit_{int(a * 100)}'], row[f'fa_{int(a * 100)}'] = rates(yte, (d1 > np.quantile(rest_own, 1 - a)).astype(int))
    MAIN.append(row)
    SCORES.append(pd.DataFrame(dict(subject=t, config='TS own calibration', trial=np.arange(len(yte)), y_true=yte,
                                    logodds=d1, p_move=m1.predict_proba(X0[te])[:, 1])))
    log(f'subject {t} done')

LIBR, MAIN, CHOICE = pd.DataFrame(LIBR), pd.DataFrame(MAIN), pd.DataFrame(CHOICE)
SCORES = pd.concat(SCORES, ignore_index=True)

# ShallowConvNet (GPU run): A.12 thresholds from the two early-stopping people, AUC, ensemble inputs
P = pd.read_csv(f'{OUT}/A1113_deep_test_probs.csv'); V = pd.read_csv(f'{OUT}/A1113_deep_val_probs.csv')
DL = pd.read_csv(f'{OUT}/A1113_deep_library.csv')
for c in (P, V, DL):
    c['subject'] = c['subject'].astype(int)
ROC = []
for name, meth, vmode in (('deep strict', 'strict', 'strict'), ('deep aligned 60', 'aligned 60', 'aligned')):
    for t in SUBJ:
        row = dict(subject=t, config=name); acc, auc, hd, fd = [], [], [], []; hits = {a: [] for a in ALPHAS}; fas = {a: [] for a in ALPHAS}
        for seed in sorted(P.seed.unique()):
            d = P[(P.subject == t) & (P.seed == seed) & (P.method == meth)].sort_values('trial')
            v = V[(V.subject == t) & (V.seed == seed) & (V['mode'] == vmode)]
            y, p = d.y_true.values, d.p_move.values
            acc.append(100 * bal(y, (p > 0.5).astype(int))); auc.append(100 * roc_auc_score(y, p))
            h, f = rates(y, (p > 0.5).astype(int)); hd.append(h); fd.append(f)
            for a in ALPHAS:
                h, f = rates(y, (p > np.quantile(v.p_move.values[v.y_true.values == 0], 1 - a)).astype(int))
                hits[a].append(h); fas[a].append(f)
            ROC.append(dict(config=name, subject=t, seed=seed, tpr=roc_on_grid(y, p)))
        row.update(acc_default=np.mean(acc), auc=np.mean(auc), hit_default=np.mean(hd), fa_default=np.mean(fd))
        for a in ALPHAS:
            row[f'hit_{int(a * 100)}'], row[f'fa_{int(a * 100)}'] = np.mean(hits[a]), np.mean(fas[a])
        MAIN = pd.concat([MAIN, pd.DataFrame([row])], ignore_index=True)
for cfg in ('TS strict', 'TS recentered 60', 'TS own calibration'):
    for t in SUBJ:
        d = SCORES[(SCORES.subject == t) & (SCORES.config == cfg)]
        ROC.append(dict(config=cfg, subject=t, seed=0, tpr=roc_on_grid(d.y_true.values, d.logodds.values)))

for df, fn in ((LIBR, 'A1113_ts_library.csv'), (MAIN, 'A1113_main.csv'), (CHOICE, 'A1113_window_choice.csv'),
               (SCORES, 'A1113_ts_test_scores.csv')):
    df.to_csv(f'{OUT}/{fn}', index=False)

# A.13b ensemble (first check that both caches list the same test trials in the same order)
mism = [t for t in SUBJ if not np.array_equal(
    SCORES[(SCORES.subject == t) & (SCORES.config == 'TS strict')].sort_values('trial').y_true.values,
    P[(P.subject == t) & (P.method == 'strict') & (P.seed == 0)].sort_values('trial').y_true.values)]
if mism:
    sys.exit(f'test-trial order differs between the TS+LDA and deep caches for subjects {mism}: send me this line')
ENS = []
for name, ts_cfg, meth in (('strict', 'TS strict', 'strict'), ('adapted', 'TS recentered 60', 'aligned 60')):
    for t in SUBJ:
        ts = SCORES[(SCORES.subject == t) & (SCORES.config == ts_cfg)].sort_values('trial')
        dp = P[(P.subject == t) & (P.method == meth)].groupby('trial').p_move.mean().sort_index()
        y = ts.y_true.values; pt = ts.p_move.values; pd_ = dp.values; pe = (pt + pd_) / 2
        h, f = rates(y, (pe > 0.5).astype(int))
        ENS.append(dict(subject=t, config=name, acc_ens=100 * bal(y, (pe > 0.5).astype(int)),
                        acc_deep_seedavg=100 * bal(y, (pd_ > 0.5).astype(int)), acc_ts=100 * bal(y, (pt > 0.5).astype(int)),
                        auc_ens=100 * roc_auc_score(y, pe), hit_ens=h, fa_ens=f))
        ROC.append(dict(config=f'ensemble {name}', subject=t, seed=0, tpr=roc_on_grid(y, pe)))
ENS = pd.DataFrame(ENS)

ENS.to_csv(f'{OUT}/A1113_ensemble.csv', index=False)
RC = pd.DataFrame(ROC)
roc_mean = pd.DataFrame([dict(config=c, fpr=GRID[i], tpr=np.mean(np.stack(RC[RC.config == c].tpr.values), axis=0)[i])
                         for c in RC.config.unique() for i in range(len(GRID))])
roc_mean.to_csv(f'{OUT}/A1113_roc_mean.csv', index=False)

# ===== report =====
print('\n=== anchors (TS+LDA, must match A.9 within 0.3 points) ===')
for cfg, e in (('TS strict', 61.0), ('TS recentered 60', 65.4), ('TS own calibration', 67.1)):
    v = MAIN[MAIN.config == cfg].acc_default.mean()
    print(f'  {cfg:20s} {v:5.1f} (A.9 {e})  {"OK" if abs(v - e) <= 0.3 else "DIFFERENT"}')
for mode, e in (('strict', 61.0), ('recentered 60', 65.4)):
    v = LIBR[(LIBR.K == 14) & (LIBR['mode'] == mode)].acc.mean()
    print(f'  A.11 K=14 {mode:13s} {v:5.1f} (A.9 {e})  {"OK" if abs(v - e) <= 0.3 else "DIFFERENT"}')

def curve(df, Ks, mode, col='acc', mcol='mode'):
    per = df[df[mcol] == mode].groupby(['subject', 'K'])[col].mean().unstack()[Ks]
    x = np.log2(np.tile(np.array(Ks, float), (len(per), 1))).ravel(); yv = per.values.ravel()
    b = np.polyfit(x, yv, 1)[0]
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        idx = rng.integers(0, len(per), len(per)); sub = per.values[idx]
        bs.append(np.polyfit(np.log2(np.tile(np.array(Ks, float), (len(idx), 1))).ravel(), sub.ravel(), 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    gain = per[Ks[-1]].mean() - per[Ks[-2]].mean()
    verdict = 'still rising' if (gain >= 1.0 and (lo > 0 or hi < 0)) else ('plateau' if gain < 0.5 else 'inconclusive')
    return per, b, lo, hi, gain, verdict
print('\n=== A.11 library-size curves (mean [95% CI] across 15 people) ===')
for label, df, Ks, modes, col in (('TS+LDA', LIBR, KS_TS, ('strict', 'recentered 60'), 'acc'),
                                  ('ShallowConvNet', DL, [3, 6, 9, 12], ('strict', 'aligned 60'), 'bal_acc')):
    for mode in modes:
        per, b, lo, hi, gain, verdict = curve(df, Ks, mode, col)
        txt = ' | '.join(f'K={k} {mci(per[k])[0]:.1f} [{mci(per[k])[1]:.1f}, {mci(per[k])[2]:.1f}]' for k in Ks)
        print(f'  {label} {mode}: {txt}')
        print(f'      last step +{gain:.1f}; points per doubling b = {b:+.2f} [95% CI {lo:+.2f}, {hi:+.2f}] -> {verdict.upper()}')

print('\n=== A.12 operating point and AUC (mean of 15 people; deep = mean of 5 seeds) ===')
print(f'{"configuration":20s} {"AUC":>6} | {"default: hit / FA":>18} | {"20% target: hit / FA / n FA>30%":>32} | {"10% target: hit / FA":>21}')
for cfg in ('TS strict', 'TS recentered 60', 'TS own calibration', 'deep strict', 'deep aligned 60'):
    g = MAIN[MAIN.config == cfg]
    print(f'{cfg:20s} {g.auc.mean():6.1f} | {g.hit_default.mean():8.1f} / {g.fa_default.mean():5.1f} | '
          f'{g.hit_20.mean():12.1f} / {g.fa_20.mean():5.1f} / {int((g.fa_20 > 30).sum()):2d} of 15 | {g.hit_10.mean():10.1f} / {g.fa_10.mean():5.1f}')
pa = []
for a, b in (('deep strict', 'TS strict'), ('deep aligned 60', 'TS recentered 60')):
    x = MAIN[MAIN.config == a].set_index('subject').auc; z = MAIN[MAIN.config == b].set_index('subject').auc.reindex(x.index)
    pa.append((a, b, (x - z).mean(), wil(x - z)))
adj = holm2([p for *_, p in pa])
for (a, b, d, p), pj in zip(pa, adj):
    print(f'  AUC {a} - {b}: {d:+.1f} points, p = {p:.3f}, Holm p = {pj:.3f}')

print('\n=== A.13a analysis-window start chosen on training people (TS+LDA) ===')
pw = []
for mode in ('strict', 'recentered 60'):
    g = MAIN[MAIN.config == f'TS {mode}'].set_index('subject')
    ch = CHOICE[CHOICE['mode'] == mode]
    d = g.acc_nested_window - g.acc_default
    pw.append((mode, g.acc_default.mean(), g.acc_nested_window.mean(), d.mean(), wil(d),
               dict(ch.start_s.value_counts().sort_index())))
adj = holm2([x[4] for x in pw])
for (mode, a0, a1, d, p, cnt), pj in zip(pw, adj):
    v = 'IMPROVES' if (pj < 0.05 and d > 0) else 'no improvement shown'
    print(f'  {mode:13s}: 0-3 s {a0:.1f} -> nested {a1:.1f} ({d:+.1f}), p = {p:.3f}, Holm p = {pj:.3f} -> {v}; chosen starts (s: folds) {cnt}')

print('\n=== A.13b fixed ensemble (TS+LDA + ShallowConvNet, mean of probabilities, threshold 0.5) ===')
pe = []
for name in ('strict', 'adapted'):
    g = ENS[ENS.config == name]
    pe.append((name, g, (g.acc_ens - g.acc_deep_seedavg).mean(), wil(g.acc_ens - g.acc_deep_seedavg),
               (g.acc_ens - g.acc_ts).mean(), wil(g.acc_ens - g.acc_ts)))
adj = holm2([x[3] for x in pe])
for (name, g, dd, pd1, dt, pt), pj in zip(pe, adj):
    v = 'IMPROVES' if (pj < 0.05 and dd > 0) else 'no improvement shown'
    print(f'  {name:8s}: ensemble {g.acc_ens.mean():.1f} | deep (seed-averaged) {g.acc_deep_seedavg.mean():.1f} | TS+LDA {g.acc_ts.mean():.1f}')
    print(f'            ensemble - deep {dd:+.1f} (p = {pd1:.3f}, Holm p = {pj:.3f}) -> {v}; ensemble - TS+LDA {dt:+.1f} (p = {pt:.3f})')
    print(f'            ensemble AUC {g.auc_ens.mean():.1f}; hit {g.hit_ens.mean():.1f}; false alarm {g.fa_ens.mean():.1f}')
print('  deep (seed-averaged) = decision from the mean of the 5 seeds\' probabilities, the same deep input the ensemble uses')
log('total time')
print('===== A.11-13 CPU FINISHED =====', flush=True)
