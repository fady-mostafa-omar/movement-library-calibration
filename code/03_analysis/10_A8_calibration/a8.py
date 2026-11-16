# Step A.8: calibration curves on Ofner (T1, fold-specific 8 sites, TS+LDA)
# own labelled calibration vs others' ME + unlabelled recentering vs hybrid (approved 10 Nov 2026)
import os, re, sys, json, time, collections, warnings
import numpy as np, pandas as pd, mne
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
A1, A2B, OUT = f'{ROOT}/01_preprocessed', f'{ROOT}/03_A2b_electrodes', f'{ROOT}/10_A8_calibration'
os.makedirs(OUT, exist_ok=True)
def log(msg):
    print(f'{msg} ({(time.time() - t0) / 60:.1f} min)', flush=True)
print(f'===== A.8 started {time.strftime("%H:%M:%S")} =====', flush=True)

SUBJ, FS = list(range(1, 16)), 128
NS, N0 = [10, 20, 30, 45, 60], 20          # budgets (first N T1 trials of MI runs 1-5) and hybrid prior weight (trials per class)
POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
CH = json.load(open(f'{A1}/ME_sub01_meta.json'))['channels']
EEG = CH[:61]

# 10-05 template positions for spherical-spline interpolation (as in A.2b)
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
def car_logmean(C):
    n = C.shape[-1]
    if n not in QB:
        QB[n] = np.linalg.qr(np.c_[np.ones(n), np.eye(n)[:, :n - 1]])[0][:, 1:]
    Q = QB[n]
    return logvec(Q.T @ C @ Q).mean(axis=1)

# 1) A.2b cache and bad channels
DATA, BADS = {'ME': {}, 'MI': {}}, {}
for cond in ('ME', 'MI'):
    for s in SUBJ:
        fn = f'{A2B}/cache/{cond}_sub{s:02d}.npz'
        if not os.path.exists(fn):
            sys.exit(f'missing cache file: {fn}')
        DATA[cond][s] = dict(np.load(fn))
        BADS[(cond, s)] = [c for c in json.load(open(f'{A1}/{cond}_sub{s:02d}_meta.json'))['bad_channels'] if c in EEG]
log('A.2b cache loaded')

# typical trial period in Ofner MI runs (cue to next cue), for session-time estimates
per = []
for s in SUBJ:
    E = pd.read_csv(f'{A1}/MI_sub{s:02d}_events.csv')
    for r, g in E.groupby('run'):
        d = np.diff(np.sort(g['cue_sample_128'].values)) / FS
        per += list(d[(d > 0) & (d < 30)])
PERIOD = float(np.median(per))
print(f'median trial period (cue to next cue): {PERIOD:.2f} s', flush=True)

# 2) fold-specific 8-site montages from A.2b-2 (checked against the documented consensus counts)
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
    print('could not identify the fold-specific montages. Structure seen:', flush=True)
    print('  A2b2_state.json top level:', {k: type(v).__name__ for k, v in st.items()} if isinstance(st, dict) else type(st).__name__)
    for w, m in cands[:5]:
        print('  candidate', w, {k: (len(v) if v else None) for k, v in list(m.items())[:3]})
    if os.path.exists(f'{A2B}/A2b2_outer.csv'):
        print(pd.read_csv(f'{A2B}/A2b2_outer.csv').head(5).to_string())
    sys.exit('send me these lines')
src, M = good[0]
MONT = {t: sorted(M[t][:8]) for t in SUBJ}
print(f'fold-specific montages from {src} (consensus counts match the A.2b record):', flush=True)
for t in SUBJ:
    print(f'  subject {t:2d}: {", ".join(POOL[i] for i in MONT[t])}', flush=True)

# 3) features (as in A.2b/A.4a: training sessions interpolated from 61 channels; the new person within the montage)
def feats(cond, s, S, target):
    D, parts = DATA[cond][s], []
    for b in ('low', 'mu_beta'):
        if target and f'cov31raw_{b}' in D and any(POOL[i] in BADS[(cond, s)] for i in S):
            C = D[f'cov31raw_{b}'][..., S, :][..., S].astype(np.float64)
            W = interp_matrix([POOL[i] for i in S], BADS[(cond, s)])
            C = W @ C @ W.T
        else:
            C = D[f'cov31_{b}'][..., S, :][..., S].astype(np.float64)
        parts.append(car_logmean(C))
    return np.hstack(parts)
def lda():
    return LDA(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])

# 4) curves, one new person at a time
rows, checked = [], False
for t in SUBJ:
    S = MONT[t]
    Xo, yo, Xr = [], [], []
    for s in SUBJ:
        if s == t:
            continue
        D = DATA['ME'][s]; k = ~D['r4m'].astype(bool)
        X = feats('ME', s, S, False)[k]
        Xo.append(X); Xr.append(X - X.mean(0)); yo.append(D['y'][k])
    Xo, Xr, yo = np.vstack(Xo), np.vstack(Xr), np.concatenate(yo)
    m_str, m_rec = lda().fit(Xo, yo), lda().fit(Xr, yo)
    D = DATA['MI'][t]; F = feats('MI', t, S, True)
    te = D['run'] >= 6; Xte, yte = F[te], D['y'][te]
    cal = np.where(D['run'] <= 5)[0]                    # MI runs 1-5 in time order
    r4m = D['r4m'].astype(bool)
    def ba(pred):
        return 100 * balanced_accuracy_score(yte, pred)
    rows.append(dict(subject=t, budget='0', method='others ME strict', acc=ba(m_str.predict(Xte)), n_offered=0, n_labelled=0))
    mu_o, Sig = m_rec.means_, m_rec.covariance_
    for N in NS + ['all']:
        idx = cal if N == 'all' else cal[:N]
        mw = F[idx].mean(0)                             # unlabelled warm-up mean
        Xt = Xte - mw
        rows.append(dict(subject=t, budget=str(N), method='others ME + recentering', acc=ba(m_rec.predict(Xt)),
                         n_offered=len(idx), n_labelled=0))
        lab = idx[~r4m[idx]]; yl = D['y'][lab]
        ok = min(np.sum(yl == 0), np.sum(yl == 1)) >= 2
        a_own = ba(lda().fit(F[lab], yl).predict(Xte)) if ok else np.nan
        rows.append(dict(subject=t, budget=str(N), method='own labelled', acc=a_own, n_offered=len(idx), n_labelled=len(lab)))
        mu = mu_o.copy(); Xl = F[lab] - mw
        for c in (0, 1):
            nc = int(np.sum(yl == c))
            if nc:
                mu[c] = (N0 * mu_o[c] + nc * Xl[yl == c].mean(0)) / (N0 + nc)
        w = np.linalg.solve(Sig, mu[1] - mu[0]); b0 = -0.5 * (mu[1] + mu[0]) @ w
        rows.append(dict(subject=t, budget=str(N), method='hybrid', acc=ba((Xt @ w + b0 > 0).astype(int)),
                         n_offered=len(idx), n_labelled=len(lab)))
        if not checked:                                  # self-check: the hand-written decision rule equals sklearn
            w0 = np.linalg.solve(Sig, mu_o[1] - mu_o[0]); c0 = -0.5 * (mu_o[1] + mu_o[0]) @ w0
            same = np.mean((Xt @ w0 + c0 > 0).astype(int) == m_rec.predict(Xt))
            print(f'self-check: hand-written LDA rule vs sklearn, same decisions in {100 * same:.1f}% of trials', flush=True)
            checked = True
    log(f'subject {t} done')

# 5) summary
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/A8_results.csv', index=False)
def col(m, b):
    return R[(R.method == m) & (R.budget == b)].set_index('subject').acc
def pw(m1, m2, b):
    x, z = col(m1, b), col(m2, b)
    d = (x - z.reindex(x.index)).dropna().values
    return d.mean(), (float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
print('\n=== anchors (must match A.4a / A.6 within 0.3 points) ===')
for m, b, e in [('others ME strict', '0', 61.0), ('others ME + recentering', '10', 63.9), ('others ME + recentering', '20', 64.4),
                ('others ME + recentering', '30', 64.9), ('others ME + recentering', '60', 65.4),
                ('others ME + recentering', 'all', 65.3), ('own labelled', 'all', 67.1)]:
    v = col(m, b).mean()
    print(f'  {m:24s} @ {b:>3s}: {v:5.1f}  (earlier {e})  {"OK" if abs(v - e) <= 0.3 else "DIFFERENT"}')

print(f'\n=== A.8: T1, new person\'s MI runs 6-10, fold-specific 8 sites, mean of 15, % (strict, no data from the person: {col("others ME strict", "0").mean():.1f}) ===')
print(f'{"budget":>6} {"trials":>6} {"labelled":>8} {"sig min":>7} {"sess min":>8} | {"own labelled":>12} {"others+recenter":>15} {"hybrid":>7} | '
      f'{"hyb-rec":>8} {"p":>6} {"hyb-own":>8} {"p":>6} {"own-rec":>8} {"p":>6}')
for b in [str(n) for n in NS] + ['all']:
    g = R[(R.budget == b) & (R.method == 'own labelled')]
    nt, nl = g.n_offered.mean(), g.n_labelled.mean()
    o, r, h = col('own labelled', b), col('others ME + recentering', b), col('hybrid', b)
    d1, p1 = pw('hybrid', 'others ME + recentering', b); d2, p2 = pw('hybrid', 'own labelled', b); d3, p3 = pw('own labelled', 'others ME + recentering', b)
    print(f'{b:>6} {nt:6.0f} {nl:8.0f} {nt * 3 / 60:7.1f} {nt * PERIOD / 60:8.1f} | {o.mean():8.1f} ({o.notna().sum():2d}) {r.mean():15.1f} {h.mean():7.1f} | '
          f'{d1:+8.1f} {p1:6.3f} {d2:+8.1f} {p2:6.3f} {d3:+8.1f} {p3:6.3f}')
print('sig min = minutes of trial signal (trials x 3 s); sess min = trials x median trial period (a session with T1 trials only)')
print('p = paired Wilcoxon, two-sided, uncorrected (descriptive)')

rec3 = col('others ME + recentering', '60').mean()
own = {b: col('own labelled', b).mean() for b in [str(n) for n in NS] + ['all']}
first = next((b for b in own if own[b] >= rec3), None)
print(f'\nown labelled calibration reaches the 3-min unlabelled level ({rec3:.1f}%) at budget: {first or "not reached within MI runs 1-5"}')
first0 = next((b for b in own if own[b] >= col('others ME strict', '0').mean()), None)
print(f'own labelled calibration reaches the strict zero-calibration level at budget: {first0 or "not reached within MI runs 1-5"}')
log('total time')
print('===== A.8 FINISHED =====', flush=True)
