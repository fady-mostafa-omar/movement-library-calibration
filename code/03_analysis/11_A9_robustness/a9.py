# Step A.9: robustness and practical analyses on Ofner T1 (TS+LDA), approved 10 Nov 2026
# V1 equivalence (TOST, margin 5 points) | V2 who benefits | V3 decision time | V4 hit / false-alarm rates | V6 10-20 montages
import os, re, sys, json, time, collections, warnings
import numpy as np, pandas as pd, mne
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
A1, A2B, OUT = f'{ROOT}/01_preprocessed', f'{ROOT}/03_A2b_electrodes', f'{ROOT}/11_A9_robustness'
os.makedirs(OUT, exist_ok=True)
def log(msg):
    print(f'{msg} ({(time.time() - t0) / 60:.1f} min)', flush=True)
print(f'===== A.9 started {time.strftime("%H:%M:%S")} =====', flush=True)

SUBJ, FS = list(range(1, 16)), 128
DELTA, WARM, N0 = 5.0, 60, 20          # equivalence margin (points), recentering warm-up (trials, = 3 min of signal), hybrid weight
TEND = [1.0, 1.5, 2.0, 2.5, 3.0]       # decision time = end of the last 1-s window after the cue (s)
POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
FIXED = {'1020_6': ['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'],                       # rule-defined 10-20 sets (not selected on data)
         '1020_9': ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']}
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

# 1) all methods, per new person, for each montage and decision time
rows = []
def evaluate(t, S, montage, tends):
    F = {('ME', s): wfeats('ME', s, S, False) for s in SUBJ}
    F.update({('MIo', s): wfeats('MI', s, S, False) for s in SUBJ if s != t})   # others' MI: interpolated from 61 channels
    Ft = wfeats('MI', t, S, True)                                                  # the new person's MI: within the montage
    Dt = DATA['MI'][t]; te = Dt['run'] >= 6; yte = Dt['y'][te]
    cal = np.where(Dt['run'] <= 5)[0]; r4m = Dt['r4m'].astype(bool)
    for T in tends:
        k = int(round((T - 1.0) / 0.1)) + 1
        avg = lambda X: X[:, :k].mean(axis=1)
        def others(key):
            Xs, Xr, ys = [], [], []
            for s in SUBJ:
                if s == t:
                    continue
                D = DATA['ME' if key == 'ME' else 'MI'][s]; kk = ~D['r4m'].astype(bool)
                X = avg(F[(key, s)])[kk]
                Xs.append(X); Xr.append(X - X.mean(0)); ys.append(D['y'][kk])
            return np.vstack(Xs), np.vstack(Xr), np.concatenate(ys)
        Xo, Xr, yo = others('ME')
        m4, mr = lda().fit(Xo, yo), lda().fit(Xr, yo)
        Xmo, _, ymo = others('MIo')
        m2 = lda().fit(Xmo, ymo)
        Xt = avg(Ft); Xte = Xt[te]
        D3 = DATA['ME'][t]; k3 = ~D3['r4m'].astype(bool)
        m3 = lda().fit(avg(F[('ME', t)])[k3], D3['y'][k3])
        lab = cal[~r4m[cal]]
        m1 = lda().fit(Xt[lab], Dt['y'][lab])
        preds = {'cell 1 own MI (labelled, ~27 min)': m1.predict(Xte), 'cell 2 others MI': m2.predict(Xte),
                 'cell 3 own ME': m3.predict(Xte), 'cell 4 others ME (strict)': m4.predict(Xte)}
        for name, idx in (('recentered 10 trials', cal[:10]), (f'recentered {WARM} trials', cal[:WARM]), ('recentered all', cal)):
            preds[f'{name}'] = mr.predict(Xte - Xt[idx].mean(0))
        idx = cal[:WARM]; mw = Xt[idx].mean(0); lab60 = idx[~r4m[idx]]; yl = Dt['y'][lab60]
        mu, Sig = mr.means_.copy(), mr.covariance_
        for c in (0, 1):
            nc = int(np.sum(yl == c))
            if nc:
                mu[c] = (N0 * mr.means_[c] + nc * (Xt[lab60][yl == c] - mw).mean(0)) / (N0 + nc)
        w = np.linalg.solve(Sig, mu[1] - mu[0]); b0 = -0.5 * (mu[1] + mu[0]) @ w
        preds[f'hybrid {WARM} trials'] = ((Xte - mw) @ w + b0 > 0).astype(int)
        for name, p in preds.items():
            rows.append(dict(subject=t, montage=montage, t_end=T, method=name,
                             acc=100 * balanced_accuracy_score(yte, p),
                             hit=100 * np.mean(p[yte == 1] == 1), false_alarm=100 * np.mean(p[yte == 0] == 1)))

for t in SUBJ:
    evaluate(t, MONT[t], 'fold-specific 8', TEND)
    for name, sites in FIXED.items():
        evaluate(t, [POOL.index(c) for c in sites], name, [3.0])
    log(f'subject {t} done')
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/A9_results.csv', index=False)

def col(method, montage='fold-specific 8', T=3.0, what='acc'):
    g = R[(R.method == method) & (R.montage == montage) & (R.t_end == T)]
    return g.set_index('subject')[what]
def pgt(a):
    a = np.asarray(a, float)
    return float(stats.wilcoxon(a - 50, alternative='greater').pvalue) if np.any(a != 50) else 1.0

# anchors
print('\n=== anchors (fold-specific 8, 0-3 s; must match A.4a / A.6 / A.8 within 0.3 points) ===')
for m, e in [('cell 1 own MI (labelled, ~27 min)', 67.1), ('cell 2 others MI', 62.0), ('cell 3 own ME', 62.7),
             ('cell 4 others ME (strict)', 61.0), (f'recentered {WARM} trials', 65.4), ('recentered all', 65.3),
             ('recentered 10 trials', 63.9), (f'hybrid {WARM} trials', 66.3)]:
    v = col(m).mean()
    print(f'  {m:36s} {v:5.1f}  (earlier {e})  {"OK" if abs(v - e) <= 0.3 else "DIFFERENT"}')

# V1 equivalence
def tost(d, delta=DELTA):
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    p1 = 1 - stats.t.cdf((m + delta) / se, n - 1); p2 = stats.t.cdf((m - delta) / se, n - 1)
    lo, hi = stats.t.interval(0.90, n - 1, loc=m, scale=se)
    w1 = stats.wilcoxon(d + delta, alternative='greater').pvalue; w2 = stats.wilcoxon(d - delta, alternative='less').pvalue
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, n).mean() for _ in range(10000)])
    return m, lo, hi, max(p1, p2), max(w1, w2), np.percentile(boot, 2.5), np.percentile(boot, 97.5)
print(f'\n=== V1: equivalence tests (paired, 15 subjects, margin +/-{DELTA:.0f} points, fixed in advance) ===')
print(f'{"comparison":58s} {"diff":>6} {"90% CI":>15} {"p TOST":>7} {"p TOST-W":>8} {"boot 95% CI":>15} {"equivalent within":>18}')
for a, b in [(f'recentered {WARM} trials', 'cell 1 own MI (labelled, ~27 min)'), ('recentered all', 'cell 1 own MI (labelled, ~27 min)'),
             ('recentered 10 trials', 'cell 1 own MI (labelled, ~27 min)'), (f'hybrid {WARM} trials', 'cell 1 own MI (labelled, ~27 min)'),
             ('cell 4 others ME (strict)', 'cell 1 own MI (labelled, ~27 min)'), ('cell 4 others ME (strict)', 'cell 2 others MI'),
             ('cell 4 others ME (strict)', 'cell 3 own ME')]:
    x, z = col(a), col(b)
    m, lo, hi, p, pw, blo, bhi = tost((x - z.reindex(x.index)).values)
    eq = 'YES' if p < 0.05 else 'not shown'
    print(f'{a + " - " + b:58s} {m:+6.1f} [{lo:+5.1f},{hi:+5.1f}] {p:7.3f} {pw:8.3f} [{blo:+5.1f},{bhi:+5.1f}] '
          f'{max(abs(lo), abs(hi)):5.1f} pts ({eq})')
print('equivalent within X = the smallest margin that the 90% CI fits inside; TOST-W = Wilcoxon version')

# V2 who benefits
print('\n=== V2: who benefits (per-person Spearman correlations, fold-specific 8, 0-3 s) ===')
c1 = col('cell 1 own MI (labelled, ~27 min)')
for m in ['cell 4 others ME (strict)', f'recentered {WARM} trials', 'cell 3 own ME', 'cell 2 others MI']:
    x = col(m).reindex(c1.index)
    r, p = stats.spearmanr(x, c1)
    print(f'  {m:32s} vs own calibration: rho = {r:+.2f} (p = {p:.3f})')
low = lambda s: set(s[s < 60].index)
L1, L4, LR = low(c1), low(col('cell 4 others ME (strict)')), low(col(f'recentered {WARM} trials'))
print(f'  below 60% with own calibration: {sorted(L1)}')
print(f'  below 60% with strict zero calibration: {sorted(L4)}  (also low with own calibration: {sorted(L4 & L1)})')
print(f'  below 60% with recentering ({WARM} trials): {sorted(LR)}  (also low with own calibration: {sorted(LR & L1)})')
print('  per person (own calibration | strict | recentered | own ME):')
for s in SUBJ:
    print(f'    subject {s:2d}: {c1[s]:5.1f} | {col("cell 4 others ME (strict)")[s]:5.1f} | {col(f"recentered {WARM} trials")[s]:5.1f} | {col("cell 3 own ME")[s]:5.1f}')

# V3 decision time
print('\n=== V3: decision time (mean %, fold-specific 8; decision at the end of the last 1-s window after the cue) ===')
print(f'{"method":36s} ' + ' '.join(f'{T:>6.1f}s' for T in TEND))
for m in ['cell 4 others ME (strict)', f'recentered {WARM} trials', f'hybrid {WARM} trials', 'cell 1 own MI (labelled, ~27 min)']:
    vals = [col(m, T=T).mean() for T in TEND]
    ps = [pgt(col(m, T=T)) for T in TEND]
    print(f'{m:36s} ' + ' '.join(f'{v:6.1f}{"*" if p < 0.05 else " "}' for v, p in zip(vals, ps)))
print('* = above 50% (Wilcoxon, one-sided p < 0.05)')

# V4 hit and false-alarm rates
print('\n=== V4: hit rate (movement trials detected) and false-alarm rate (rest trials detected as movement), %, 0-3 s ===')
for m in ['cell 4 others ME (strict)', 'recentered 10 trials', f'recentered {WARM} trials', 'recentered all', f'hybrid {WARM} trials',
          'cell 1 own MI (labelled, ~27 min)']:
    print(f'  {m:36s} hit {col(m, what="hit").mean():5.1f}   false alarm {col(m, what="false_alarm").mean():5.1f}')

# V6 10-20 montages
print('\n=== V6: rule-defined 10-20 montages vs fold-specific 8 (mean %, 0-3 s) ===')
for m in ['cell 4 others ME (strict)', f'recentered {WARM} trials', 'cell 1 own MI (labelled, ~27 min)']:
    base = col(m)
    line = f'  {m:36s} fold-specific 8 {base.mean():5.1f}'
    for name in FIXED:
        x = col(m, montage=name)
        d = (x - base.reindex(x.index)).values
        p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        line += f' | {name} {x.mean():5.1f} ({d.mean():+.1f}, p = {p:.3f})'
    print(line)
print('  1020_6 = C3, Cz, C4, P3, Pz, P4; 1020_9 = + F3, Fz, F4 (fixed by rule, not selected on data)')
log('total time')
print('===== A.9 FINISHED =====', flush=True)
