# R.1 follow-up, steps N5 (CPU) and N4 (GPU). Every choice below was fixed on 12 Nov 2026, before running
# (STEP_N1_N4_PREREG.md, amendment 2).
# N5: calibration curves with the strong own-calibration pipeline (TS+LDA, 8 sites, windows from 1.0 s = N1 candidate P1):
#     own labelled, others' ME strict, others' ME + recentering and hybrid at the A.8 budgets; the same with 0-3 s must
#     reproduce A.8. Outcome: the labelled budget at which a strong own calibration matches each zero-calibration mode.
# N4: ShallowConvNet eye and frontal controls (cell 4, strict, seeds 0-1, the A.4b-1/A.10b code unchanged).
import os, re, sys, json, time, copy, shutil, collections, warnings
import numpy as np, pandas as pd, mne
from scipy import stats, linalg
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
A1, A2B, A4B, OUT = f'{ROOT}/01_preprocessed', f'{ROOT}/03_A2b_electrodes', f'{ROOT}/07_A4b_deep', f'{ROOT}/15_R1_followup'
os.makedirs(OUT, exist_ok=True)
def log(msg):
    print(f'{msg} ({(time.time() - t0) / 60:.1f} min)', flush=True)
print(f'===== R.1 follow-up N5 + N4 started {time.strftime("%H:%M:%S")} =====', flush=True)

SUBJ, FS = list(range(1, 16)), 128
WARM, N0, DELTA = 60, 20, 5.0            # recentering warm-up (trials), hybrid prior weight, equivalence margin (points)
POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
CH = json.load(open(f'{A1}/ME_sub01_meta.json'))['channels']
EEG = CH[:61]
PIDX = [EEG.index(c) for c in POOL]
BANDS = ('low', 'mu_beta')
SOS = {'low': butter(4, (0.3, 3.0), btype='band', fs=FS, output='sos'),
       'mu_beta': butter(4, (8.0, 30.0), btype='band', fs=FS, output='sos')}
FBANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
FBSOS = [butter(4, b, btype='band', fs=FS, output='sos') for b in FBANDS]
WSTART = np.round(np.arange(21) * 0.1 * FS).astype(int); WLEN = FS
TEND = int(WSTART[-1] + WLEN)                                      # 384 samples = 3.0 s after the cue
TOFF = np.round(np.linspace(0, TEND - 1, 16)[1:]).astype(int)      # 15 samples, 0.2 to 3.0 s after the cue
FBWIN = (0, FS)                                                    # FBCSP windows: 0-3 s and 1-3 s
HAND = ['hand_close', 'hand_open']
SETS = {'1020_6': ['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'], '1020_9': ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']}
PIPES = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
PDESC = {'P0': 'TS+LDA, fold-specific 8 sites, 0-3 s (current reference)',
         'P1': 'TS+LDA, fold-specific 8 sites, windows from 1.0 s',
         'P2': 'TS+LDA, 31 sites, 0-3 s',
         'P3': 'TS+LDA, 31 sites, windows from 1.0 s',
         'P4': 'low-frequency amplitudes (Ofner-style), 61 channels',
         'P5': 'low-frequency amplitudes (Ofner-style), 31 sites',
         'P6': 'FBCSP (7 bands x 4 filters) + LDA, 31 sites, 0-3 s',
         'P7': 'FBCSP (7 bands x 4 filters) + LDA, 31 sites, 1-3 s'}
# ShallowConvNet per person (mean of 5 seeds), read from the A.10b result file
_B = pd.read_csv(f'{ROOT}/12_A10_deep_calibration/A10b_results.csv', float_precision='round_trip')
_B['subject'] = _B['subject'].astype(int)
DEEP = {k: _B[_B.method == m].groupby('subject').bal_acc.mean().reindex(SUBJ).values
        for k, m in (('deep_strict', 'strict'), ('deep_al10', 'aligned 10'), ('deep_al30', 'aligned 30'),
                     ('deep_al60', 'aligned 60'), ('deep_alall', 'aligned all'))}

# ---------- helpers (same as A.2b/A.3/A.8) ----------
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

def cfilt(X, sos):
    zi = sosfilt_zi(sos)[:, None, :] * X[:, :1][None]
    return sosfilt(sos, X, axis=1, zi=zi)[0]

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

def lda():
    return LDA(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])

def csp(C, yy):
    """CSP for one band: trace-normalised class means, fixed 10% shrinkage, 2 filters from each end."""
    M = []
    for c in np.unique(yy):
        Cc = C[yy == c]; Cc = Cc / np.trace(Cc, axis1=1, axis2=2)[:, None, None]
        Cm = Cc.mean(0); n = Cm.shape[0]
        M.append(0.9 * Cm + 0.1 * np.trace(Cm) / n * np.eye(n))
    _, V = linalg.eigh(M[1], M[0] + M[1])
    return V[:, [0, 1, -2, -1]]

def fbf(C, filt):
    out = []
    for k, Wk in enumerate(filt):
        v = np.einsum('ci,ncd,di->ni', Wk, C[:, k], Wk)
        out.append(np.log(v / v.sum(1, keepdims=True)))
    return np.hstack(out)

def wil(d):
    """Paired Wilcoxon, two-sided; exact without ties or zeros, otherwise normal approximation with tie correction."""
    d = np.round(np.asarray(d, float), 6)
    if not np.any(d != 0):
        return 1.0
    nz = d[d != 0]
    exact = len(nz) == len(d) and len(np.unique(np.abs(nz))) == len(nz)
    if exact:
        return float(stats.wilcoxon(d, method='exact').pvalue)
    try:
        return float(stats.wilcoxon(d, method='asymptotic', correction=False).pvalue)
    except ValueError:                                  # older scipy calls it 'approx'
        return float(stats.wilcoxon(d, method='approx', correction=False).pvalue)

def tost(d, m=DELTA):
    d = np.asarray(d, float); n = len(d); md = d.mean(); se = d.std(ddof=1) / np.sqrt(n); df = n - 1
    h = stats.t.ppf(0.95, df) * se
    p = max(stats.t.sf((md + m) / se, df), stats.t.cdf((md - m) / se, df))
    return md, md - h, md + h, p, max(abs(md - h), abs(md + h))

def ci95(x):
    x = np.asarray(x, float); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h

def holm(p):
    p = np.asarray(p, float); adj = np.empty(len(p)); run = 0.0
    for k, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(1.0, run)
    return adj

# ---------- 1) A.2b cache and bad channels ----------
DATA, BADS = {'ME': {}, 'MI': {}}, {}
for cond in ('ME', 'MI'):
    for s in SUBJ:
        fn = f'{A2B}/cache/{cond}_sub{s:02d}.npz'
        if not os.path.exists(fn):
            sys.exit(f'missing cache file: {fn}')
        DATA[cond][s] = dict(np.load(fn))
        BADS[(cond, s)] = [c for c in json.load(open(f'{A1}/{cond}_sub{s:02d}_meta.json'))['bad_channels'] if c in EEG]
log('A.2b cache loaded')

cls_all = sorted(set(np.concatenate([DATA['MI'][s]['cls'].astype(str) for s in SUBJ])))
REST = [c for c in cls_all if c not in HAND]
if len(REST) != 1 or not set(HAND) <= set(cls_all):
    sys.exit(f'STOP: unexpected T1 classes in the A.2b cache: {cls_all}')
REST = REST[0]
MOVS = set()
for cond in ('ME', 'MI'):
    for s in SUBJ:
        D = DATA[cond][s]; cl = D['cls'].astype(str)
        ym, yr = np.unique(D['y'][np.isin(cl, HAND)]), np.unique(D['y'][cl == REST])
        if len(ym) != 1 or len(yr) != 1 or ym[0] == yr[0]:
            sys.exit(f'STOP: label coding differs from the expected one ({cond} {s})')
        MOVS.add(ym[0])
if len(MOVS) != 1:
    sys.exit('STOP: movement label is not the same in every file')
MOV = MOVS.pop()
print(f'T1 classes: {HAND} vs {REST}; movement label = {MOV}', flush=True)

# ---------- 2) fold-specific 8-site montages (same parser and check as A.8) ----------
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
    sys.exit('STOP: could not identify the fold-specific montages (send me these lines)')
src, M = good[0]
MONT = {t: sorted(M[t][:8]) for t in SUBJ}
print(f'fold-specific montages from {src}', flush=True)

# ---------- 3) features ----------
def feats(cond, s, S, target, j0=0):
    """TS features as in A.2b/A.8; j0 = first window used (10 -> windows starting at or after 1.0 s)."""
    D, parts = DATA[cond][s], []
    for b in BANDS:
        if target and f'cov31raw_{b}' in D and any(POOL[i] in BADS[(cond, s)] for i in S):
            C = D[f'cov31raw_{b}'][:, j0:][..., S, :][..., S].astype(np.float64)
            W = interp_matrix([POOL[i] for i in S], BADS[(cond, s)])
            C = W @ C @ W.T
        else:
            C = D[f'cov31_{b}'][:, j0:][..., S, :][..., S].astype(np.float64)
        parts.append(car_logmean(C))
    return np.hstack(parts)

def library(cond, S, t):
    Xo, Xr, yo = [], [], []
    for s in SUBJ:
        if s == t:
            continue
        D = DATA[cond][s]; k = ~D['r4m'].astype(bool)
        X = feats(cond, s, S, False)[k]
        Xo.append(X); Xr.append(X - X.mean(0)); yo.append(D['y'][k])
    return np.vstack(Xo), np.vstack(Xr), np.concatenate(yo)


# =====================================================================================================================
# N5: calibration curves with the strong own-calibration pipeline (CPU)
# =====================================================================================================================
NS = [10, 20, 30, 45, 60, 'all']
def library_j(S, t, j0):
    Xo, Xr, yo = [], [], []
    for s in SUBJ:
        if s == t:
            continue
        D = DATA['ME'][s]; k = ~D['r4m'].astype(bool)
        X = feats('ME', s, S, False, j0)[k]
        Xo.append(X); Xr.append(X - X.mean(0)); yo.append(D['y'][k])
    return np.vstack(Xo), np.vstack(Xr), np.concatenate(yo)

R5 = []
for t in SUBJ:
    S = MONT[t]
    D = DATA['MI'][t]; y = D['y']; run = D['run']; r4m = D['r4m'].astype(bool)
    te = np.where(run >= 6)[0]; cal = np.where(run <= 5)[0]
    def ba(pred):
        return 100 * balanced_accuracy_score(y[te], pred)
    for j0, win in ((0, '0-3 s'), (10, 'from 1.0 s')):
        Xo, Xr, yo = library_j(S, t, j0)
        m_str, m_rec = lda().fit(Xo, yo), lda().fit(Xr, yo)
        F = feats('MI', t, S, True, j0)
        R5.append(dict(subject=t, window=win, budget='0', method='others ME strict', acc=ba(m_str.predict(F[te])), n_labelled=0))
        mu_o, Sig, cl = m_rec.means_, m_rec.covariance_, m_rec.classes_
        for N in NS:
            idx = cal if N == 'all' else cal[:N]
            mw = F[idx].mean(0); Xt = F[te] - mw
            R5.append(dict(subject=t, window=win, budget=str(N), method='others ME + recentering', acc=ba(m_rec.predict(Xt)),
                           n_labelled=0))
            lab = idx[~r4m[idx]]; yl = y[lab]
            ok = min(np.sum(yl == cl[0]), np.sum(yl == cl[1])) >= 2            # same rule as A.8
            R5.append(dict(subject=t, window=win, budget=str(N), method='own labelled',
                           acc=ba(lda().fit(F[lab], yl).predict(F[te])) if ok else np.nan, n_labelled=len(lab)))
            mu = mu_o.copy(); Xl = F[lab] - mw
            for ci, c in enumerate(cl):
                nc = int(np.sum(yl == c))
                if nc:
                    mu[ci] = (N0 * mu_o[ci] + nc * Xl[yl == c].mean(0)) / (N0 + nc)
            w = np.linalg.solve(Sig, mu[1] - mu[0]); b0 = -0.5 * (mu[1] + mu[0]) @ w
            R5.append(dict(subject=t, window=win, budget=str(N), method='hybrid',
                           acc=ba(np.where(Xt @ w + b0 > 0, cl[1], cl[0])), n_labelled=len(lab)))
    pd.DataFrame(R5).to_csv(f'{OUT}/N5_curves.csv', index=False)
log('N5 curves done')

R5 = pd.DataFrame(R5)
def cv(method, budget, win):
    return R5[(R5.method == method) & (R5.budget == budget) & (R5.window == win)].set_index('subject').acc.sort_index()
print('\n=== N5 anchors (0-3 s must reproduce A.8; from 1.0 s must reproduce N1 P1 and A.13a) ===')
A8DOC = {'own labelled': [56.5, 59.7, 61.3, 63.4, 64.4, 67.1], 'others ME + recentering': [63.9, 64.4, 64.9, 63.7, 65.4, 65.3],
         'hybrid': [65.5, 64.9, 65.2, 63.6, 66.3, 66.0]}
bad5 = 0
for m, docs in A8DOC.items():
    got = [cv(m, str(N), '0-3 s').mean() for N in NS]
    okk = all(abs(g - d) <= 0.3 for g, d in zip(got, docs)); bad5 += not okk
    print(f'  0-3 s {m:24s} ' + ' '.join(f'{g:5.1f}' for g in got) + f'   (A.8 {"/".join(map(str, docs))})  {"OK" if okk else "DIFFERENT"}')
for lab_, v, e in (('0-3 s strict', cv('others ME strict', '0', '0-3 s').mean(), 61.0),
                   ('from 1.0 s own labelled, all', cv('own labelled', 'all', 'from 1.0 s').mean(), 75.8),
                   ('from 1.0 s recentered 60', cv('others ME + recentering', '60', 'from 1.0 s').mean(), 68.6)):
    okk = abs(v - e) <= 0.3; bad5 += not okk
    print(f'  {lab_:32s} {v:5.1f}  (earlier {e})  {"OK" if okk else "DIFFERENT"}')
print('ALL N5 ANCHORS OK' if not bad5 else f'{bad5} N5 ANCHOR(S) DIFFERENT: do not interpret; send me the whole output')

print('\n=== N5: balanced accuracy vs labelled budget (windows from 1.0 s; mean of the people with both classes; n in brackets) ===')
print(f'{"budget":>6} | {"own labelled":>16} | {"others + recentering":>20} | {"hybrid":>8}')
own = {}
for N in NS:
    o = cv('own labelled', str(N), 'from 1.0 s'); own[str(N)] = o
    r = cv('others ME + recentering', str(N), 'from 1.0 s'); h = cv('hybrid', str(N), 'from 1.0 s')
    print(f'{str(N):>6} | {o.mean():9.1f} (n={o.notna().sum():2d}) | {r.mean():20.1f} | {h.mean():8.1f}')
s1 = cv('others ME strict', '0', 'from 1.0 s')
print(f'others ME strict, from 1.0 s: {s1.mean():.1f}')
MODES = [('TS+LDA zero calibration, 0-3 s (primary pipeline)', cv('others ME strict', '0', '0-3 s')),
         ('TS+LDA zero calibration, from 1.0 s', s1),
         ('TS+LDA recentered 60, from 1.0 s', cv('others ME + recentering', '60', 'from 1.0 s')),
         ('TS+LDA hybrid 60, from 1.0 s', cv('hybrid', '60', 'from 1.0 s')),
         ('ShallowConvNet zero calibration', pd.Series(DEEP['deep_strict'], index=SUBJ)),
         ('ShallowConvNet aligned 10', pd.Series(DEEP['deep_al10'], index=SUBJ)),
         ('ShallowConvNet aligned 60', pd.Series(DEEP['deep_al60'], index=SUBJ))]
print('\n=== N5 outcome: labelled budget at which the strong own calibration (from 1.0 s) first matches each mode (group means) ===')
eqrows = []
for name, v in MODES:
    first = next((N for N in NS if own[str(N)].mean() >= v.mean()), None)
    diffs = []
    for N in NS:
        o = own[str(N)]; d = (o - v.reindex(o.index)).dropna(); m, lo, hi = ci95(d.values)
        diffs.append(f'{str(N)}: {m:+.1f} [{lo:+.1f}, {hi:+.1f}]')
    txt = f'{first} labelled trials' if first is not None else 'not reached within MI runs 1-5'
    print(f'  {name:52s} {v.mean():5.1f}%  -> matched at {txt}')
    print(f'      own (from 1.0 s) - mode, 95% CI: ' + ' | '.join(diffs))
    eqrows.append(dict(mode=name, mean=v.mean(), matched_at=first))
pd.DataFrame(eqrows).to_csv(f'{OUT}/N5_matched_budget.csv', index=False)

# =====================================================================================================================
# N4: ShallowConvNet eye and frontal controls (GPU), A.4b-1 / A.10b code unchanged
# =====================================================================================================================
import torch, torch.nn as nn
from sklearn.metrics import balanced_accuracy_score as bal
DEV = 'cuda' if torch.cuda.is_available() else None
if DEV is None:
    sys.exit('N5 is saved. N4 needs a GPU: switch the runtime to L4 or T4 and run again (N5 will simply be recomputed)')
print('\nGPU:', torch.cuda.get_device_name(0), flush=True)
SEEDS4, MAX_EP, PATIENCE, BATCH, LR = (0, 1), 150, 20, 64, 1e-3
TS = {'ME': {}, 'MI': {}}
for cond in ('ME', 'MI'):
    for s in SUBJ:
        with np.load(f'{A4B}/cache_ts/{cond}_sub{s:02d}.npz') as z:
            TS[cond][s] = {k: z[k] for k in z.files}
print('cache_ts keys:', sorted(TS['MI'][1].keys()), '| X31 shape (MI, person 1):', TS['MI'][1]['X31'].shape, flush=True)

# --- functions copied unchanged from a4b1.py (via a10b.py) ---
QB2 = {}
def get_ts(cond, s, S, idx, within):
    D = TS[cond][s]
    raw = within and 'X31raw' in D and any(POOL[i] in BADS[(cond, s)] for i in S)
    X = D['X31raw' if raw else 'X31'][idx][:, S, :].astype(np.float64)
    if raw:
        X = interp_matrix([POOL[i] for i in S], BADS[(cond, s)]) @ X
    k = len(S)
    if k not in QB2:
        QB2[k] = np.linalg.qr(np.c_[np.ones(k), np.eye(k)[:, :k - 1]])[0][:, 1:]
    return QB2[k].T @ X

class Shallow(nn.Module):
    def __init__(self, C, T, n_out=2, F=40, k=13, pool=38, stride=8, p=0.5):
        super().__init__()
        self.ct = nn.Conv2d(1, F, (1, k)); self.cs = nn.Conv2d(F, F, (C, 1), bias=False)
        self.bn = nn.BatchNorm2d(F); self.pool = nn.AvgPool2d((1, pool), stride=(1, stride)); self.drop = nn.Dropout(p)
        with torch.no_grad():
            n = self.feat(torch.zeros(1, 1, C, T)).numel()
        self.fc = nn.Linear(n, n_out)
    def feat(self, x):
        x = self.bn(self.cs(self.ct(x)))
        return self.drop(torch.log(torch.clamp(self.pool(x * x), min=1e-6)))
    def forward(self, x):
        return self.fc(self.feat(x.unsqueeze(1)).flatten(1))

def predict(model, X):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(model(torch.tensor(X[i:i + 256], dtype=torch.float32, device=DEV)).argmax(1).cpu().numpy())
    return np.concatenate(out)

def train(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Shallow(Xtr.shape[1], Xtr.shape[2]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    w = torch.tensor(len(ytr) / (2.0 * np.bincount(ytr, minlength=2)), dtype=torch.float32, device=DEV)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV); yt = torch.tensor(ytr, device=DEV)
    best, state, wait, ep_best = -1.0, None, 0, 0
    for ep in range(MAX_EP):
        model.train(); perm = torch.randperm(len(yt), device=DEV)
        for i in range(0, len(yt), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 2:
                continue
            opt.zero_grad(); lossf(model(Xt[b]), yt[b]).backward(); opt.step()
        acc = bal(yva, predict(model, Xva))
        if acc > best:
            best, state, wait, ep_best = acc, copy.deepcopy(model.state_dict()), 0, ep + 1
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(state)
    return model, ep_best
# --- end of copied functions ---

# EOG epochs: from the raw recording, with the filter variant that reproduces the cached EEG epochs (chosen by reproduction only)
EOGI = [CH.index(c) for c in CH[61:64]]
print('EOG channels:', CH[61:64], flush=True)
def raw_epochs(cond, s, rows_idx, sos, use_zi):
    Z = np.load(f'{A1}/{cond}_sub{s:02d}.npz')
    E = pd.read_csv(f'{A1}/{cond}_sub{s:02d}_events.csv').sort_values(['run', 'trial']).reset_index(drop=True)
    out, keep = [], np.zeros(len(E), bool)
    for r in range(1, 11):
        X = Z[f'run{r}'].astype(np.float64)
        if X.shape[0] != len(CH):
            sys.exit(f'STOP: raw run has {X.shape[0]} channels, expected {len(CH)}')
        Xc = X[rows_idx]
        Y = cfilt(Xc, sos) if use_zi else sosfilt(sos, Xc, axis=1)
        rr = np.where(E.run.values == r)[0]; cues = E.cue_sample_128.values[rr].astype(int)
        ok = (cues >= 0) & (cues + TEND <= X.shape[1]); keep[rr[ok]] = True
        out += [Y[:, c:c + TEND] for c in cues[ok]]
    E = E[keep].reset_index(drop=True)
    t1 = np.isin(E['class'].values.astype(str), HAND + [REST])
    return E[t1].reset_index(drop=True), np.stack(out)[t1]

SOS40 = butter(4, (0.3, 40.0), btype='band', fs=FS, output='sos')
D1 = TS['MI'][1]; okc = [i for i, c in enumerate(POOL) if c not in BADS[('MI', 1)]]
ref = D1['X31raw'] if 'X31raw' in D1 else D1['X31']
choice = None
for use_zi in (True, False):
    E1, X1 = raw_epochs('MI', 1, PIDX, SOS40, use_zi)
    same_trials = (len(X1) == len(ref) and np.array_equal(E1.run.values, D1['run'])
                   and np.array_equal(np.isin(E1['class'].values.astype(str), HAND), D1['y'] == MOV))
    if not same_trials or X1.shape[1:] != ref.shape[1:]:
        print(f'  filter variant zi={use_zi}: trials or epoch shape differ from cache_ts ({X1.shape} vs {ref.shape})'); continue
    rel = np.max(np.abs(X1[:, okc] - ref[:, okc])) / np.max(np.abs(ref[:, okc]))
    print(f'  filter variant zi={use_zi}: largest relative difference to cache_ts {rel:.2e}', flush=True)
    if rel < 1e-4 and choice is None:
        choice = use_zi
EOG = None
if choice is None:
    print('EOG control NOT RUN: no filter variant reproduces the cached EEG epochs (pre-specified rule); frontal control only')
else:
    EOG = {'ME': {}, 'MI': {}}
    for cond in ('ME', 'MI'):
        for s in SUBJ:
            Es, Xe = raw_epochs(cond, s, EOGI, SOS40, choice)
            if (len(Xe) != len(TS[cond][s]['y']) or not np.array_equal(Es.run.values, TS[cond][s]['run'])
                    or not np.array_equal(np.isin(Es['class'].values.astype(str), HAND), TS[cond][s]['y'] == MOV)):
                sys.exit(f'STOP: EOG trials do not match cache_ts for {cond} {s}')
            EOG[cond][s] = Xe
    print(f'EOG epochs built for 30 sessions (filter variant zi={choice})', flush=True)
Q3 = np.linalg.qr(np.c_[np.ones(3), np.eye(3)[:, :2]])[0][:, 1:]
def get_eog(cond, s, S, idx, within):
    return Q3.T @ EOG[cond][s][idx]

FRONT5 = [POOL.index(c) for c in ('F3', 'F1', 'Fz', 'F2', 'F4')]
def run4(t, seed, getter, S):
    clean = lambda cond, s: np.where(~TS[cond][s]['r4m'])[0]
    Dt = TS['MI'][t]; ite = np.where(Dt['run'] >= 6)[0]
    Xte, yte = getter('MI', t, S, ite, True), Dt['y'][ite]
    others = [s for s in SUBJ if s != t]
    val = list(np.random.default_rng(100 * t + seed).choice(others, 2, replace=False))
    Xs, ys = [], []
    for s in others:
        if s in val:
            continue
        i = clean('ME', s); Xs.append(getter('ME', s, S, i, False)); ys.append(TS['ME'][s]['y'][i])
    Xv, yv = [], []
    for s in val:
        D = TS['MI'][s]; i = np.where(D['run'] >= 6)[0]; Xv.append(getter('MI', s, S, i, False)); yv.append(D['y'][i])
    Xtr, ytr, Xva, yva = np.concatenate(Xs), np.concatenate(ys), np.concatenate(Xv), np.concatenate(yv)
    sc = Xtr.std()
    model, ep = train(Xtr / sc, ytr, Xva / sc, yva, seed)
    return 100 * bal(yte, predict(model, Xte / sc)), ep

CTRL = [('frontal 5 only', get_ts, FRONT5)] + ([('EOG 3 only', get_eog, None)] if EOG is not None else [])
R4 = []
for t in SUBJ:
    for name, getter, S in CTRL:
        for seed in SEEDS4:
            a, ep = run4(t, seed, getter, S)
            R4.append(dict(subject=t, control=name, seed=seed, bal_acc=a, best_epoch=ep))
    pd.DataFrame(R4).to_csv(f'{OUT}/N4_deep_controls.csv', index=False)
    log(f'N4 person {t} done')
R4 = pd.DataFrame(R4)
print('\n=== N4: ShallowConvNet controls (cell 4, strict, seeds 0-1; mean of 15 people) ===')
for name, g in R4.groupby('control', sort=False):
    v = g.groupby('subject').bal_acc.mean()
    m, lo, hi = ci95(v.values)
    d = np.round(v.values - 50, 6); nz = d[d != 0]
    exact = len(nz) == len(d) and len(np.unique(np.abs(nz))) == len(nz)
    try:
        p = stats.wilcoxon(d, alternative='greater', method='exact' if exact else 'asymptotic').pvalue
    except ValueError:
        p = stats.wilcoxon(d, alternative='greater', method='exact' if exact else 'approx').pvalue
    passed = (m < 55) and (p >= 0.05)
    print(f'  {name:16s} {m:5.1f} [{lo:.1f}, {hi:.1f}]  above 50%: one-sided p = {p:.3f}  -> {"PASSES" if passed else "DOES NOT PASS"} '
          f'(rule: mean < 55% and p >= 0.05)')
log('total time')
print('===== R.1 follow-up N5 + N4 FINISHED =====', flush=True)
