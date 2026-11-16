# R.1 follow-up, steps N1-N3 (CPU). Every choice below was fixed on 12 Nov 2026, before running.
# N1: stronger own-calibration references (labelled MI runs 1-5 -> MI runs 6-10, T1): 8 fixed candidates,
#     a nested per-person choice (leave-one-run-out inside runs 1-5) and an optimistic upper bound.
# N2: specificity of the zero-calibration hand-vs-rest detector: how often it fires on the other four
#     imagined arm movements (elbow flexion/extension, supination/pronation) in MI runs 6-10.
# N3: Holm correction within the equivalence (TOST) families; TOST for the rule-defined 10-20 site sets.
import os, re, sys, json, time, collections, warnings
import numpy as np, pandas as pd, mne
from scipy import stats, linalg
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
A1, A2B, OUT = f'{ROOT}/01_preprocessed', f'{ROOT}/03_A2b_electrodes', f'{ROOT}/15_R1_followup'
os.makedirs(OUT, exist_ok=True)
def log(msg):
    print(f'{msg} ({(time.time() - t0) / 60:.1f} min)', flush=True)
print(f'===== R.1 follow-up N1-N3 started {time.strftime("%H:%M:%S")} =====', flush=True)

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

def raw_features(s, S):
    """From the A.1 recording of the new person's MI session (all 7 classes, same trial rule as the A.2b cache):
    (a) TS features on the montage S with the new person's interpolation rule (for N2, and to check the cache);
    (b) low-frequency amplitudes, CAR over the montage, 15 samples 0.2-3.0 s (61 channels and 31 sites);
    (c) filter-bank covariances on the 31 sites (7 bands, windows 0-3 s and 1-3 s)."""
    Z = np.load(f'{A1}/MI_sub{s:02d}.npz')
    E = pd.read_csv(f'{A1}/MI_sub{s:02d}_events.csv').sort_values(['run', 'trial']).reset_index(drop=True)
    bads = BADS[('MI', s)]
    W61, W31, WS = interp_matrix(EEG, bads), interp_matrix(POOL, bads), interp_matrix([POOL[i] for i in S], bads)
    mch = [PIDX[i] for i in S]
    keep = np.zeros(len(E), bool); MF, TD61, TD31, FB = [], [], [], []
    for r in range(1, 11):
        X = Z[f'run{r}'].astype(np.float64)
        rows = np.where(E.run.values == r)[0]
        cues = E.cue_sample_128.values[rows].astype(int)
        ok = (cues >= 0) & (cues + TEND <= X.shape[1])
        keep[rows[ok]] = True; cues = cues[ok]
        if not len(cues):
            continue
        ix = cues[:, None] + WSTART[None, :]
        parts = []
        for b in BANDS:
            Y = cfilt(X[mch], SOS[b])
            seg = np.stack([Y[:, i:i + WLEN] for i in ix.ravel()])
            C = (seg @ np.swapaxes(seg, 1, 2) / WLEN).reshape(len(cues), len(WSTART), len(S), len(S))
            parts.append(car_logmean(WS @ C @ WS.T))
        MF.append(np.hstack(parts))
        for Wm, chans, out in ((W61, list(range(61)), TD61), (W31, PIDX, TD31)):
            Xm = Wm @ X[chans]; Xm = Xm - Xm.mean(0, keepdims=True)
            Y = cfilt(Xm, SOS['low'])
            out.append(np.transpose(Y[:, cues[:, None] + TOFF[None, :]], (1, 0, 2)).reshape(len(cues), -1))
        X31 = W31 @ X[PIDX]; fb = np.zeros((len(cues), len(FBANDS), len(FBWIN), 31, 31))
        for k, sos in enumerate(FBSOS):
            Y = cfilt(X31, sos)
            for wi, a in enumerate(FBWIN):
                seg = np.stack([Y[:, c + a:c + TEND] for c in cues])
                fb[:, k, wi] = seg @ np.swapaxes(seg, 1, 2) / seg.shape[-1]
        FB.append(fb)
    E = E[keep].reset_index(drop=True)
    return E, np.vstack(MF), np.vstack(TD61), np.vstack(TD31), np.concatenate(FB)

# ---------- 4) one new person at a time ----------
rows, srows = [], []
for t in SUBJ:
    S = MONT[t]
    D = DATA['MI'][t]; y = D['y']; run = D['run']; r4m = D['r4m'].astype(bool)
    te = np.where(run >= 6)[0]; cal = np.where(run <= 5)[0]; lab = cal[~r4m[cal]]
    def ba(pred, idx=te):
        return 100 * balanced_accuracy_score(y[idx], pred)
    row = dict(subject=t)

    # zero calibration, recentering and hybrid (A.8/A.9 rules), cells 2 and 3 (A.4a)
    Xo, Xr, yo = library('ME', S, t)
    m_str, m_rec = lda().fit(Xo, yo), lda().fit(Xr, yo)
    F = feats('MI', t, S, True)
    row['strict'] = ba(m_str.predict(F[te]))
    for N in (10, 60, 'all'):
        idx = cal if N == 'all' else cal[:N]
        row[f'rec{N}'] = ba(m_rec.predict(F[te] - F[idx].mean(0)))
    idx = cal[:WARM]; mw60 = F[idx].mean(0); l60 = idx[~r4m[idx]]; yl = y[l60]
    mu_o, Sig, cl = m_rec.means_, m_rec.covariance_, m_rec.classes_
    mu = mu_o.copy(); Xl = F[l60] - mw60
    for ci, c in enumerate(cl):
        nc = int(np.sum(yl == c))
        if nc:
            mu[ci] = (N0 * mu_o[ci] + nc * Xl[yl == c].mean(0)) / (N0 + nc)
    w = np.linalg.solve(Sig, mu[1] - mu[0]); b0 = -0.5 * (mu[1] + mu[0]) @ w
    row['hyb60'] = ba(np.where((F[te] - mw60) @ w + b0 > 0, cl[1], cl[0]))
    Xo2, _, yo2 = library('MI', S, t)
    row['cell2'] = ba(lda().fit(Xo2, yo2).predict(F[te]))
    De = DATA['ME'][t]; ke = ~De['r4m'].astype(bool)
    row['cell3'] = ba(lda().fit(feats('ME', t, S, False)[ke], De['y'][ke]).predict(F[te]))

    # rule-defined 10-20 sets (A.9 V6), for N3
    for name, sites in SETS.items():
        Sx = sorted(POOL.index(c) for c in sites)
        Xa, Xb, ya = library('ME', Sx, t)
        Fx = feats('MI', t, Sx, True)
        row[f'{name}_strict'] = ba(lda().fit(Xa, ya).predict(Fx[te]))
        row[f'{name}_rec60'] = ba(lda().fit(Xb, ya).predict(Fx[te] - Fx[cal[:WARM]].mean(0)))
        row[f'{name}_own'] = ba(lda().fit(Fx[lab], y[lab]).predict(Fx[te]))

    # raw-data features, with a check that they reproduce the cache exactly (same trials, same order, same values)
    E, MF, TD61, TD31, FB = raw_features(t, S)
    ecl = E['class'].values.astype(str)
    t1 = np.isin(ecl, HAND + [REST])
    same = np.array_equal(E.run.values[t1], run) and np.array_equal(ecl[t1], D['cls'].astype(str))
    dev = float(np.max(np.abs(MF[t1] - F))) if same else np.inf
    row['cache_check_maxdiff'] = dev
    if not same or dev > 1e-2:
        print(f'subject {t}: raw trials do not match the A.2b cache (same order: {same}; max difference {dev:.2e})')
        print('  raw T1 runs/classes (first 8):', list(zip(E.run.values[t1][:8], ecl[t1][:8])))
        print('  cache runs/classes (first 8):', list(zip(run[:8], D['cls'].astype(str)[:8])))
        sys.exit('STOP: send me these lines')

    # N1: own-calibration candidates P0-P7, test result and leave-one-run-out score inside runs 1-5
    FT = {'P0': F, 'P1': feats('MI', t, S, True, 10),
          'P2': feats('MI', t, list(range(31)), True), 'P3': feats('MI', t, list(range(31)), True, 10),
          'P4': TD61[t1], 'P5': TD31[t1]}
    FBt = FB[t1]
    def fit_pred(p, tr, ts):
        if p in ('P6', 'P7'):
            C = FBt[:, :, 0 if p == 'P6' else 1]
            filt = [csp(C[tr, k], y[tr]) for k in range(len(FBANDS))]
            return lda().fit(fbf(C[tr], filt), y[tr]).predict(fbf(C[ts], filt))
        X = FT[p]
        return lda().fit(X[tr], y[tr]).predict(X[ts])
    for p in PIPES:
        row[f'own_{p}'] = ba(fit_pred(p, lab, te))
        pred = np.empty(len(cal), dtype=y.dtype)
        for r in range(1, 6):
            trr = cal[(run[cal] != r) & ~r4m[cal]]; pos = np.where(run[cal] == r)[0]
            pred[pos] = fit_pred(p, trr, cal[pos])
        row[f'loro_{p}'] = 100 * balanced_accuracy_score(y[cal], pred)
    best = max(PIPES, key=lambda p: (row[f'loro_{p}'], -PIPES.index(p)))
    row['nested_choice'] = best; row['own_nested'] = row[f'own_{best}']

    # N2: specificity (strict and recentered 60), test runs 6-10 only
    oth = ~t1 & (E.run.values >= 6); co = ecl[oth]
    for mode, mdl, shift in (('strict', m_str, 0.0), ('rec60', m_rec, mw60)):
        ph = mdl.predict(F[te] - shift); po = mdl.predict(MF[oth] - shift)
        r_ = dict(subject=t, mode=mode, hand_hit=100 * np.mean(ph[y[te] == MOV] == MOV),
                  rest_fa=100 * np.mean(ph[y[te] != MOV] == MOV), other_arm=100 * np.mean(po == MOV),
                  n_other=int(oth.sum()))
        for c in sorted(set(co)):
            r_[c] = 100 * np.mean(po[co == c] == MOV)
        srows.append(r_)
    rows.append(row)
    pd.DataFrame(rows).to_csv(f'{OUT}/N1N3_per_subject.csv', index=False)
    pd.DataFrame(srows).to_csv(f'{OUT}/N2_specificity.csv', index=False)
    log(f'subject {t} done (cache check max difference {dev:.1e}; nested choice {best})')

# ---------- 5) summaries ----------
A = pd.DataFrame(rows).set_index('subject').sort_index()
for k, v in DEEP.items():
    A[k] = v
print('\n=== anchors (must match earlier steps within 0.3 points) ===')
anchors = [('strict', 61.0), ('rec10', 63.9), ('rec60', 65.4), ('recall', 65.3), ('hyb60', 66.3), ('cell2', 62.0),
           ('cell3', 62.7), ('own_P0', 67.1), ('own_P2', 65.6), ('1020_6_strict', 61.2), ('1020_6_rec60', 63.7),
           ('1020_6_own', 66.3), ('1020_9_strict', 60.0), ('1020_9_rec60', 62.9), ('1020_9_own', 68.2),
           ('deep_strict', 66.3), ('deep_al10', 67.4), ('deep_al60', 68.1)]
bad = 0
for k, e in anchors:
    v = A[k].mean(); okk = abs(v - e) <= 0.3; bad += not okk
    print(f'  {k:16s} {v:5.1f}  (earlier {e})  {"OK" if okk else "DIFFERENT"}')
print('ALL ANCHORS OK' if not bad else f'{bad} ANCHOR(S) DIFFERENT: do not interpret; send me the whole output')

print('\n=== N1: own labelled calibration (MI runs 1-5 -> runs 6-10), balanced accuracy %, mean of 15 ===')
ref = A['own_P0']; out = []
for p in PIPES + ['nested']:
    x = A[f'own_{p}']; m, lo, hi = ci95(x)
    out.append(dict(candidate=p, description=PDESC.get(p, 'per-person choice by leave-one-run-out in runs 1-5'),
                    mean=round(m, 1), ci95=f'[{lo:.1f}, {hi:.1f}]', minus_P0=round((x - ref).mean(), 1),
                    p_vs_P0=round(wil(x - ref), 3) if p != 'P0' else np.nan,
                    loro_mean=round(A[f'loro_{p}'].mean(), 1) if p != 'nested' else np.nan))
N1 = pd.DataFrame(out); print(N1.to_string(index=False))
print('nested choice per person:', dict(collections.Counter(A.nested_choice)))
UB = max(PIPES, key=lambda p: A[f'own_{p}'].mean())
gap = A['own_nested'].mean() - ref.mean()
print(f'upper bound (best candidate chosen on the test runs, optimistic): {UB}, {A[f"own_{UB}"].mean():.1f}%')
print(f'pre-specified rule: nested reference - P0 = {gap:+.1f} points -> '
      f'{"the nested reference becomes the primary reference (gap >= 2)" if gap >= 2 else "P0 stays the reference (gap < 2)"}')

METHODS = [('strict', 'TS+LDA zero calibration'), ('rec10', 'TS+LDA recentered 10'), ('rec60', 'TS+LDA recentered 60'),
           ('recall', 'TS+LDA recentered all'), ('hyb60', 'TS+LDA hybrid 60'), ('deep_strict', 'ShallowConvNet zero calibration'),
           ('deep_al10', 'ShallowConvNet aligned 10'), ('deep_al30', 'ShallowConvNet aligned 30'),
           ('deep_al60', 'ShallowConvNet aligned 60'), ('deep_alall', 'ShallowConvNet aligned all')]
tost_rows = []
def family(name, pairs):
    res = [(lab_, *tost(np.asarray(a) - np.asarray(b))) for lab_, a, b in pairs]
    adj = holm([r[4] for r in res])
    print(f'\n--- {name} (TOST ±{DELTA:.0f} points, t-based; Holm within this family) ---')
    print(f'{"comparison":44s} {"diff":>6} {"90% CI":>16} {"p":>7} {"Holm p":>7} {"within":>7} verdict')
    for (lab_, md, lo, hi, p, wth), pa in zip(res, adj):
        v = 'equivalent' if pa < 0.05 else 'not shown'
        print(f'{lab_:44s} {md:+6.1f} {f"[{lo:+.1f}, {hi:+.1f}]":>16} {p:7.3f} {pa:7.3f} {wth:7.1f} {v}')
        tost_rows.append(dict(family=name, comparison=lab_, diff=md, ci90_lo=lo, ci90_hi=hi, p=p, p_holm=pa, within=wth))

print('\n=== N3: equivalence families ===')
family('A: TS+LDA (A.9)', [('recentered 60 - own (P0)', A.rec60, ref), ('recentered all - own (P0)', A.recall, ref),
                           ('recentered 10 - own (P0)', A.rec10, ref), ('hybrid 60 - own (P0)', A.hyb60, ref),
                           ('zero calibration - own (P0)', A.strict, ref), ('cell 4 - cell 2', A.strict, A.cell2),
                           ('cell 4 - cell 3', A.strict, A.cell3)])
family('B: ShallowConvNet (A.10)', [(f'{d} - own (P0)', A[k], ref) for k, d in METHODS if k.startswith('deep') and k != 'deep_al10'] +
       [('ShallowConvNet aligned 10 - own (P0)', A.deep_al10, ref)])
c_pairs = []
for name in SETS:
    for mode, base in (('strict', 'strict'), ('rec60', 'rec60'), ('own', 'own_P0')):
        c_pairs.append((f'{name} {mode} - selected 8', A[f'{name}_{mode}'], A[base]))
family('C: 10-20 sets vs selected 8 (Table 3)', c_pairs)
print('Table 3 Wilcoxon p (tie-robust): ' + ', '.join(f'{lab_.split(" - ")[0]} {wil(np.asarray(a) - np.asarray(b)):.3f}' for lab_, a, b in c_pairs))
family('D: vs the nested reference', [(f'{d} - nested ref', A[k], A.own_nested) for k, d in METHODS])
family(f'E: vs the upper bound ({UB}, optimistic)', [(f'{d} - {UB}', A[k], A[f'own_{UB}']) for k, d in METHODS])
pd.DataFrame(tost_rows).to_csv(f'{OUT}/N3_tost.csv', index=False)

print('\n=== N2: specificity of the zero-calibration detector (new person\'s MI runs 6-10; % of trials detected as movement) ===')
S2 = pd.DataFrame(srows); extra = [c for c in S2.columns if c not in ('subject', 'mode', 'hand_hit', 'rest_fa', 'other_arm', 'n_other')]
for mode, g in S2.groupby('mode', sort=False):
    g = g.sort_values('subject')
    print(f'\n{mode}: hand {g.hand_hit.mean():.1f} | rest {g.rest_fa.mean():.1f} | other arm movements {g.other_arm.mean():.1f} '
          f'(' + ', '.join(f'{c} {g[c].mean():.1f}' for c in extra) + f'); about {g.n_other.mean():.0f} trials per person')
    p_h, p_r = wil(g.other_arm - g.hand_hit), wil(g.other_arm - g.rest_fa)
    print(f'  other arm - hand: {(g.other_arm - g.hand_hit).mean():+.1f} (p = {p_h:.3f}); '
          f'other arm - rest: {(g.other_arm - g.rest_fa).mean():+.1f} (p = {p_r:.3f})')
    mid = (g.hand_hit.mean() + g.rest_fa.mean()) / 2
    if p_h >= 0.05:
        verdict = 'NOT hand-specific (fires on other arm movements as on hand movements)'
    elif g.other_arm.mean() < mid:
        verdict = 'hand-specific (other arm movements significantly below hand, and closer to rest)'
    else:
        verdict = 'partly specific (below hand, but closer to hand than to rest)'
    print(f'  pre-specified verdict: {verdict}')
print('\n=== check: A.11 ShallowConvNet at K = 12 vs the repeat run and A.10b (seeds/repeats 0-2) ===')
D13 = f'{ROOT}/14_A11_13'
LIB = pd.read_csv(f'{D13}/A1113_deep_library.csv', float_precision='round_trip')
PR = pd.read_csv(f'{D13}/A1113_deep_test_probs.csv', float_precision='round_trip')
print('  library columns:', list(LIB.columns))
rcol = next((c for c in LIB.columns if c in ('r', 'rep', 'repeat', 'seed')), None)
for lmode, pm in (('strict', 'strict'), ('aligned 60', 'aligned 60')):
    k12 = LIB[(LIB.K == 12) & (LIB['mode'] == lmode) & (LIB[rcol].isin([0, 1, 2]) if rcol else True)].groupby('subject').bal_acc.mean().mean()
    g = PR[(PR.method == pm) & (PR.seed.isin([0, 1, 2]))]
    rr = g.groupby(['subject', 'seed']).apply(lambda d: 100 * balanced_accuracy_score(d.y_true, (d.p_move > 0.5).astype(int))).groupby('subject').mean().mean()
    bb = _B[(_B.method == pm) & (_B.seed.isin([0, 1, 2]))].groupby('subject').bal_acc.mean().mean()
    print(f'  {lmode:11s}: A.11 K=12 {k12:.2f} | repeat run seeds 0-2 {rr:.2f} | A.10b seeds 0-2 {bb:.2f} | A.10b 5 seeds {_B[_B.method == pm].groupby("subject").bal_acc.mean().mean():.2f}')
log('total time')
print('===== R.1 follow-up N1-N3 FINISHED =====', flush=True)
