# Step A.3: self-calibration ceiling (cell 1) + Lee-protocol row (background, resumable)
import os, sys, json, shutil, time, warnings, numpy as np, pandas as pd, mne
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score as bal
from sklearn.model_selection import StratifiedKFold
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
el = lambda: f'{(time.time() - t0) / 60:.1f} min'
print(f'\n===== A.3 started {time.strftime("%H:%M:%S")} =====')

ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
A1 = f'{ROOT}/01_preprocessed'
if not os.path.exists(f'{A1}/MI_sub15.npz'):
    A1 = '/content/a1_preprocessed'
A2B, OUT = f'{ROOT}/03_A2b_electrodes', f'{ROOT}/04_A3_ceiling'
CACHE_T1, CACHE_X = f'{A2B}/cache', f'{OUT}/cache_lee'
os.makedirs(CACHE_X, exist_ok=True)
for f in (f'{A1}/MI_sub15.npz', f'{CACHE_T1}/MI_sub15.npz', f'{A2B}/A2b2_consensus.json'):
    if not os.path.exists(f):
        sys.exit(f'STOP: missing {f}')

FS, SUBJ = 128, list(range(1, 16))
CH = json.load(open(f'{A1}/ME_sub01_meta.json'))['channels']
EEG = CH[:61]
POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
PIDX = [EEG.index(c) for c in POOL]
CONS = json.load(open(f'{A2B}/A2b2_consensus.json'))['sites']
MONT = {'cons8': [POOL.index(c) for c in CONS], 'standard31': list(range(31))}
print('deployed montage:', CONS)
BANDS = ('low', 'mu_beta')
SOS = {'low': butter(4, (0.3, 3.0), btype='band', fs=FS, output='sos'),
       'mu_beta': butter(4, (8.0, 30.0), btype='band', fs=FS, output='sos')}
WSTART = np.round(np.arange(21) * 0.1 * FS).astype(int); WLEN = FS; WI = np.arange(21)
HAND, EXTRA = ['hand_close', 'hand_open'], ['elbow_extension', 'supination']
LCL = {'elbow_extension': 0, 'supination': 1, 'hand_open': 2, 'hand_close': 3}

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

def causal_band(X, b):
    zi = sosfilt_zi(SOS[b])[:, None, :] * X[:, :1][None]
    return sosfilt(SOS[b], X, axis=1, zi=zi)[0]

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

def slda_fit(X, y):
    """Shrinkage LDA, same as sklearn LDA(solver='lsqr', shrinkage='auto') with equal priors, any number of classes."""
    cl = np.unique(y); K = len(cl); d = X.shape[1]; lam = np.zeros(d); U, M = [], []
    for c in cl:
        Xc = X[y == c]; n = len(Xc); mu = Xc.mean(0); M.append(mu)
        sd = Xc.std(0); sd[sd == 0] = 1.0
        Z = (Xc - mu) / sd
        r2 = np.einsum('ij,ij->i', Z, Z)
        G = Z.T @ Z if d <= n else Z @ Z.T
        tr = r2.sum() / n; m = tr / d
        dl = np.sum(G ** 2) / n ** 2
        beta = (np.sum(r2 ** 2) / n - dl) / (d * n)
        delta = (dl - 2 * m * tr + d * m ** 2) / d
        a = 0.0 if beta == 0 else min(beta, delta) / delta        # Ledoit-Wolf shrinkage per class
        lam += a * m * sd ** 2 / K
        U.append(np.sqrt((1 - a) / (K * n)) * (Z * sd).T)
    U, M = np.hstack(U), np.column_stack(M)
    if d <= U.shape[1]:
        Wm = np.linalg.solve(np.diag(lam) + U @ U.T, M)
    else:
        LiM, LiU = M / lam[:, None], U / lam[:, None]
        Wm = LiM - LiU @ np.linalg.solve(np.eye(U.shape[1]) + U.T @ LiU, U.T @ LiM)
    return Wm, -0.5 * np.sum(M * Wm, axis=0), cl

def predict(X, mdl):
    Wm, b, cl = mdl
    return cl[np.argmax(X @ Wm + b, axis=1)]

def save_npz(fn, D):
    tmp = '/content/_a3_tmp.npz'
    np.savez(tmp, **D)
    try:
        shutil.copyfile(tmp, fn + '.part'); os.replace(fn + '.part', fn)
    except OSError:
        shutil.copyfile(tmp, fn)
    os.remove(tmp)

def load_npz(fn, need):
    if not os.path.exists(fn):
        return None
    try:
        with np.load(fn) as z:
            D = {k: z[k] for k in z.files}
        assert all(k in D for k in need)
        return D
    except Exception:
        os.remove(fn); return None

# 1) load the A.2b cache (T1 classes) and build a cache for the two extra Lee classes
DATA, XDATA, BADS = {'ME': {}, 'MI': {}}, {'ME': {}, 'MI': {}}, {}
for cond in ('ME', 'MI'):
    for s in SUBJ:
        BADS[(cond, s)] = [c for c in json.load(open(f'{A1}/{cond}_sub{s:02d}_meta.json'))['bad_channels'] if c in EEG]
        DATA[cond][s] = load_npz(f'{CACHE_T1}/{cond}_sub{s:02d}.npz', ('cov31_low', 'feat61_mu_beta', 'cls'))
        if DATA[cond][s] is None:
            sys.exit(f'STOP: A.2b cache file for {cond} {s} is missing or damaged')
        fn = f'{CACHE_X}/{cond}_sub{s:02d}.npz'
        D = load_npz(fn, ('cov31_low', 'cov31_mu_beta', 'feat61_low', 'feat61_mu_beta', 'cls'))
        if D is None:
            Z = np.load(f'{A1}/{cond}_sub{s:02d}.npz')
            E = pd.read_csv(f'{A1}/{cond}_sub{s:02d}_events.csv')
            E = E[E['class'].isin(EXTRA)].sort_values(['run', 'trial']).reset_index(drop=True)
            W61 = interp_matrix(EEG, BADS[(cond, s)])
            cov = {b: [] for b in BANDS}; keep = np.zeros(len(E), bool)
            for r in range(1, 11):
                X = Z[f'run{r}'].astype(np.float64)
                rows = np.where(E.run.values == r)[0]
                cues = E.cue_sample_128.values[rows]
                ok = (cues >= 0) & (cues + WSTART[-1] + WLEN <= X.shape[1])
                keep[rows[ok]] = True
                idx = cues[ok][:, None] + WSTART[None, :]
                for b in BANDS:
                    Y = causal_band(X, b)
                    S = np.stack([Y[:61, i:i + WLEN] for i in idx.ravel()])
                    cov[b].append((S @ np.swapaxes(S, 1, 2) / WLEN).reshape(len(idx), len(WSTART), 61, 61))
            E = E[keep].reset_index(drop=True)
            D = {'run': np.asarray(E.run, dtype=np.int64), 'r4m': np.asarray(E.r4m_rejected, dtype=bool),
                 'cls': np.array(E['class'].tolist(), dtype='U16')}
            for b in BANDS:
                C61 = W61 @ np.concatenate(cov[b]) @ W61.T
                D[f'cov31_{b}'] = C61[..., PIDX, :][..., PIDX].astype(np.float32)
                D[f'feat61_{b}'] = car_logmean(C61).astype(np.float32)
            save_npz(fn, D)
            print(f'extra classes {cond} subject {s:2d}: {len(E)} trials ({el()})')
        XDATA[cond][s] = D
print(f'data ready ({el()})')

def feats(D, cond, s, montage, idx, within=False):
    """Both bands. CAR over the montage. within=True: the person's bad channels are interpolated inside the montage."""
    parts = []
    for b in BANDS:
        if montage == 'all61':
            parts.append(D[f'feat61_{b}'][idx].astype(np.float64)); continue
        S = MONT[montage]
        raw = within and f'cov31raw_{b}' in D and any(POOL[i] in BADS[(cond, s)] for i in S)
        C = D[f'cov31raw_{b}' if raw else f'cov31_{b}'][np.ix_(idx, WI, S, S)].astype(np.float64)
        if raw:
            W = interp_matrix([POOL[i] for i in S], BADS[(cond, s)]); C = W @ C @ W.T
        parts.append(car_logmean(C))
    return np.hstack(parts)

def lee_data(cond, s, montage):
    D1, D2 = DATA[cond][s], XDATA[cond][s]
    i1 = np.where(np.isin(D1['cls'], HAND) & ~D1['r4m'])[0]
    i2 = np.where(~D2['r4m'])[0]
    X = np.vstack([feats(D1, cond, s, montage, i1), feats(D2, cond, s, montage, i2)])
    y = np.array([LCL[c] for c in np.r_[D1['cls'][i1], D2['cls'][i2]]])
    return X, y, np.r_[D1['run'][i1], D2['run'][i2]]

# self-check: fast multi-class LDA vs sklearn (subject 1, MI, 4 Lee classes, 31 sites)
Xc, yc, _ = lee_data('MI', 1, 'standard31')
tr, te = next(StratifiedKFold(5, shuffle=True, random_state=0).split(Xc, yc))
ref = LDA(solver='lsqr', shrinkage='auto', priors=np.ones(4) / 4).fit(Xc[tr], yc[tr]).predict(Xc[te])
print(f'fast LDA vs sklearn (4 classes): same decisions in {100 * np.mean(predict(Xc[te], slda_fit(Xc[tr], yc[tr])) == ref):.1f}% of trials')

# 2) ceiling: own MI runs 1-5 (clean, with labels) -> own MI runs 6-10 (all trials)
CEIL = f'{OUT}/A3_ceiling.csv'
if os.path.exists(CEIL):
    RC = pd.read_csv(CEIL)
else:
    rows = []
    for s in SUBJ:
        D = DATA['MI'][s]
        for task in ('T1', 'T2'):
            ok = np.ones(len(D['y']), bool) if task == 'T1' else np.isin(D['cls'], HAND)
            lab = D['y'] if task == 'T1' else (D['cls'] == 'hand_close').astype(int)
            tr = np.where(ok & (D['run'] <= 5) & ~D['r4m'])[0]
            te = np.where(ok & (D['run'] >= 6))[0]
            for montage in ('cons8', 'standard31', 'all61'):
                Xtr, Xte = feats(D, 'MI', s, montage, tr, True), feats(D, 'MI', s, montage, te, True)
                acc = 100 * bal(lab[te], predict(Xte, slda_fit(Xtr, lab[tr])))
                p = np.nan
                if montage == 'cons8' and task == 'T1':              # per-person permutation test (rule 3)
                    rng = np.random.default_rng(s)
                    null = [100 * bal(lab[te], predict(Xte, slda_fit(Xtr, rng.permutation(lab[tr]))))
                            for _ in range(1000)]
                    p = (1 + np.sum(np.array(null) >= acc)) / 1001
                rows.append({'subject': s, 'task': task, 'montage': montage, 'bal_acc': acc, 'p_perm': p,
                             'n_train': len(tr), 'n_test': len(te)})
        print(f'ceiling subject {s:2d} done ({el()})')
    RC = pd.DataFrame(rows); RC.to_csv(CEIL, index=False)

# 3) Lee-protocol row: 4 classes, own ME + own MI for training, own MI for testing; only the split differs
LEEF = f'{OUT}/A3_lee.csv'
RL = pd.read_csv(LEEF) if os.path.exists(LEEF) else pd.DataFrame(columns=['subject', 'montage', 'split', 'bal_acc'])
done = set(RL['subject'].astype(int)) if len(RL) else set()
for s in SUBJ:
    if s in done:
        continue
    rows = []
    for montage in ('all61', 'standard31', 'cons8'):
        Xe, ye, re_ = lee_data('ME', s, montage)
        Xi, yi, ri = lee_data('MI', s, montage)
        accs = []
        for seed in (0, 1):                                            # (a) random 80/20 trial split, as in Lee 2022
            pred = np.empty(len(yi), dtype=int)
            fe = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xe, ye))
            fi = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xi, yi))
            for k in range(5):
                mdl = slda_fit(np.vstack([Xe[fe[k][0]], Xi[fi[k][0]]]), np.r_[ye[fe[k][0]], yi[fi[k][0]]])
                pred[fi[k][1]] = predict(Xi[fi[k][1]], mdl)
            accs.append(100 * bal(yi, pred))
        rows.append({'subject': s, 'montage': montage, 'split': 'random (Lee)', 'bal_acc': np.mean(accs)})
        pred = np.empty(len(yi), dtype=int)                            # (b) run-wise: 2 runs out per fold
        for k in range(5):
            test_runs = (2 * k + 1, 2 * k + 2)
            ti, tm = np.isin(ri, test_runs), np.isin(re_, test_runs)
            mdl = slda_fit(np.vstack([Xe[~tm], Xi[~ti]]), np.r_[ye[~tm], yi[~ti]])
            pred[ti] = predict(Xi[ti], mdl)
        rows.append({'subject': s, 'montage': montage, 'split': 'run-wise (ours)', 'bal_acc': 100 * bal(yi, pred)})
    RL = pd.concat([RL, pd.DataFrame(rows)], ignore_index=True)
    RL.to_csv(LEEF, index=False)
    print(f'Lee row subject {s:2d} done ({el()})')
RL['subject'] = RL['subject'].astype(int); RL['bal_acc'] = RL['bal_acc'].astype(float)

# 4) summaries
def pw(x, ref=None):
    d = np.asarray(x) - (50 if ref is None else np.asarray(ref))
    if not np.any(d != 0):
        return 1.0
    return round(wilcoxon(d, alternative='greater' if ref is None else 'two-sided').pvalue, 4)

print('\n=== A.3-1: ceiling (own MI runs 1-5 with labels -> own MI runs 6-10), balanced accuracy %, 15 subjects ===')
out = []
for (task, montage), g in RC.groupby(['task', 'montage'], sort=False):
    a = g.bal_acc.values
    row = {'task': task, 'montage': montage, 'mean': round(a.mean(), 1), 'median': round(np.median(a), 1),
           'sd': round(a.std(ddof=1), 1), 'min': round(a.min(), 1), 'max': round(a.max(), 1),
           'n_above_50': int((a > 50).sum()), 'p_vs_50': pw(a)}
    if g.p_perm.notna().any():
        row['n_sig_perm_p<0.05'] = int((g.p_perm < 0.05).sum())
    out.append(row)
print(pd.DataFrame(out).to_string(index=False))
print('zero calibration for comparison (A.2b): honest estimate 60.9% (fold-specific sites), 31 sites 60.8%, 61 channels 58.3%')

print('\n=== A.3-2: Lee-protocol row (4 classes, own ME + MI training, own MI test), balanced accuracy %, chance 25 ===')
out = []
for montage, g in RL.groupby('montage', sort=False):
    r = g[g.split == 'random (Lee)'].sort_values('subject').bal_acc.values
    w = g[g.split == 'run-wise (ours)'].sort_values('subject').bal_acc.values
    out.append({'montage': montage, 'random_mean': round(r.mean(), 1), 'random_sd': round(r.std(ddof=1), 1),
                'runwise_mean': round(w.mean(), 1), 'runwise_sd': round(w.std(ddof=1), 1),
                'difference': round((r - w).mean(), 1), 'p_random_vs_runwise': pw(r, w)})
print(pd.DataFrame(out).to_string(index=False))
print('Lee 2022 reported 69% (their own model, 4-40 Hz, ICA, 0-4 s, random split)')
print(f'\ntotal time {el()}')
print('===== A.3 FINISHED =====')
