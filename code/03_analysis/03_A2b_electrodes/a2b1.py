# Step A.2b-1: electrode-study cache + reference montages + go/no-go (background, resumable)
import os, sys, json, shutil, time, warnings, numpy as np, pandas as pd, mne
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
print(f'\n===== A.2b-1 started {time.strftime("%H:%M:%S")} =====')

ROOT = '/content/drive/MyDrive/classification_MI-ME'
A1D, A1L = f'{ROOT}/30_v06_pipeline/01_preprocessed', '/content/a1_preprocessed'
OUT = f'{ROOT}/30_v06_pipeline/03_A2b_electrodes'
CACHE = f'{OUT}/cache'
os.makedirs(CACHE, exist_ok=True)
print(f"Drive free: {shutil.disk_usage('/content/drive/MyDrive').free / 1e9:.1f} GB (the cache needs about 1 GB)")

# 0) A.1 files must be on Drive (copy them from the Colab disk if needed)
need = [f'{c}_sub{s:02d}{e}' for c in ('ME', 'MI') for s in range(1, 16) for e in ('.npz', '_events.csv', '_meta.json')]
miss = [f for f in need if not os.path.exists(f'{A1D}/{f}')]
if miss and os.path.isdir(A1L):
    shutil.copytree(A1L, A1D, dirs_exist_ok=True)
    miss = [f for f in need if not os.path.exists(f'{A1D}/{f}')]
print(f'A.1 on Drive: {len(need) - len(miss)}/{len(need)} required files')
if not miss:
    A1 = A1D
elif os.path.isdir(A1L) and all(os.path.exists(f'{A1L}/{f}') for f in need):
    A1 = A1L
else:
    sys.exit('STOP: A.1 files are missing on Drive and on the Colab disk; A.1 has to be re-run.')

# 1) fixed settings (plan v0.7, section 4.2-4.3)
FS, SUBJ = 128, list(range(1, 16))
CH = json.load(open(f'{A1}/ME_sub01_meta.json'))['channels']
EEG = CH[:61]
POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
assert all(c in EEG for c in POOL), 'pool channel missing'
PIDX = [EEG.index(c) for c in POOL]
MONT = {'standard31': list(range(31)), 'frontal5': [POOL.index(c) for c in ['F3', 'F1', 'Fz', 'F2', 'F4']]}
BANDS = {'low': (0.3, 3.0), 'mu_beta': (8.0, 30.0)}
SOS = {b: butter(4, f, btype='band', fs=FS, output='sos') for b, f in BANDS.items()}
WSTART = np.round(np.arange(21) * 0.1 * FS).astype(int)      # 1-s windows every 0.1 s: cue+0.0 ... cue+3.0 s
WLEN = FS
T1 = {'hand_close': 1, 'hand_open': 1, 'rest': 0}

# 10-05 template positions for spherical-spline interpolation
mname = 'colin27_1005' if 'colin27_1005' in mne.channels.get_builtin_montages() else 'standard_1005'
info = mne.create_info(EEG, FS, 'eeg')
info.set_montage(mne.channels.make_standard_montage(mname), match_case=False)
POS = np.array([ch['loc'][:3] for ch in info['chs']])
_, ORIGIN, _ = mne.bem.fit_sphere_to_headshape(info, dig_kinds=('eeg',), units='m', verbose=False)

def interp_matrix(names, bads):
    """Linear map (n x n): each bad channel is replaced by its spherical-spline estimate from the good ones."""
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
    """Log-Euclidean tangent vector: upper triangle of logm(C), off-diagonals x sqrt(2)."""
    m = C.shape[-1]
    tr = np.trace(C, axis1=-2, axis2=-1)[..., None, None] / m
    w, V = np.linalg.eigh(C + 1e-5 * tr * np.eye(m))
    L = (V * np.log(np.maximum(w, 1e-12))[..., None, :]) @ np.swapaxes(V, -1, -2)
    iu = np.triu_indices(m)
    return L[..., iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2))

QB = {}
def car_logmean(C):
    """CAR over the given channels (exact projection), then the tangent vector, averaged over the 21 windows."""
    n = C.shape[-1]
    if n not in QB:
        QB[n] = np.linalg.qr(np.c_[np.ones(n), np.eye(n)[:, :n - 1]])[0][:, 1:]
    Q = QB[n]
    return logvec(Q.T @ C @ Q).mean(axis=1)

def save_npz(fn, D):                      # write locally first, then move to Drive (no half-written files)
    tmp = '/content/_a2b_tmp.npz'
    np.savez(tmp, **D)
    try:
        shutil.copyfile(tmp, fn + '.part'); os.replace(fn + '.part', fn)
    except OSError:
        shutil.copyfile(tmp, fn)
    os.remove(tmp)

def load_npz(fn):                         # returns None if the file is missing or damaged
    if not os.path.exists(fn):
        return None
    try:
        with np.load(fn) as z:
            D = {k: z[k] for k in z.files}
        assert 'cov31_low' in D and 'feat61_mu_beta' in D and 'eog_mu_beta' in D
        return D
    except Exception:
        os.remove(fn); return None

# 2) per-session cache: window covariances (31 pool sites, 3 EOG), 61-channel features
DATA, BADS = {'ME': {}, 'MI': {}}, {}
for cond in ('ME', 'MI'):
    for s in SUBJ:
        meta = json.load(open(f'{A1}/{cond}_sub{s:02d}_meta.json'))
        bads = [c for c in meta['bad_channels'] if c in EEG]
        BADS[(cond, s)] = bads
        fn = f'{CACHE}/{cond}_sub{s:02d}.npz'
        D = load_npz(fn)
        if D is not None:
            DATA[cond][s] = D; continue
        Z = np.load(f'{A1}/{cond}_sub{s:02d}.npz')
        E = pd.read_csv(f'{A1}/{cond}_sub{s:02d}_events.csv')
        E = E[E['class'].isin(T1)].sort_values(['run', 'trial']).reset_index(drop=True)
        W61 = interp_matrix(EEG, bads)
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
                S = np.stack([Y[:, i:i + WLEN] for i in idx.ravel()])
                cov[b].append((S @ np.swapaxes(S, 1, 2) / WLEN).reshape(len(idx), len(WSTART), 64, 64))
        E = E[keep].reset_index(drop=True)
        D = {'y': np.asarray(E['class'].map(T1), dtype=np.int64), 'run': np.asarray(E.run, dtype=np.int64),
             'r4m': np.asarray(E.r4m_rejected, dtype=bool), 'cls': np.array(E['class'].tolist(), dtype='U16')}
        for b in BANDS:
            C = np.concatenate(cov[b])
            C61 = W61 @ C[..., :61, :61] @ W61.T                    # bad channels interpolated from all 61
            D[f'cov31_{b}'] = C61[..., PIDX, :][..., PIDX].astype(np.float32)
            if cond == 'MI' and any(c in POOL for c in bads):        # raw copy for within-montage interpolation
                D[f'cov31raw_{b}'] = C[..., PIDX, :][..., PIDX].astype(np.float32)
            D[f'feat61_{b}'] = car_logmean(C61).astype(np.float32)
            D[f'eog_{b}'] = C[..., 61:, 61:].astype(np.float32)
        save_npz(fn, D)
        DATA[cond][s] = D
        print(f'{cond} subject {s:2d}: {len(E)} T1 trials ({(~keep).sum()} out of bounds), bad: {bads or "-"} '
              f'({(time.time() - t0) / 60:.1f} min)')
print(f'cache ready ({(time.time() - t0) / 60:.1f} min)')
pb = {k: [c for c in v if c in POOL] for k, v in BADS.items() if k[0] == 'MI' and any(c in POOL for c in v)}
print('MI sessions with bad channels inside the 31-site pool:', pb or 'none')

# 3) features per montage
def sub_feats(cond, s, S, b, target):
    D = DATA[cond][s]
    if target and f'cov31raw_{b}' in D and any(POOL[i] in BADS[(cond, s)] for i in S):
        C = D[f'cov31raw_{b}'][..., S, :][..., S].astype(np.float64)
        W = interp_matrix([POOL[i] for i in S], BADS[(cond, s)])  # target: interpolated within the montage
        C = W @ C @ W.T
    else:
        C = D[f'cov31_{b}'][..., S, :][..., S].astype(np.float64)
    return car_logmean(C)

def montage_feats(cond, s, montage, bands, target):
    D, parts = DATA[cond][s], []
    for b in bands:
        if montage == 'all61':
            parts.append(D[f'feat61_{b}'].astype(np.float64))
        elif montage == 'eog3':
            parts.append(logvec(D[f'eog_{b}'].astype(np.float64)).mean(axis=1))   # no CAR on EOG
        else:
            parts.append(sub_feats(cond, s, MONT[montage], b, target))
    return np.hstack(parts)

# 4) shrinkage LDA: same result as sklearn LDA(solver='lsqr', shrinkage='auto', priors=[.5, .5]), but faster
def slda_fit(X, y):
    d = X.shape[1]; lam = np.zeros(d); U, M = [], []
    for c in (0, 1):
        Xc = X[y == c]; n = len(Xc); mu = Xc.mean(0); M.append(mu)
        sd = Xc.std(0); sd[sd == 0] = 1.0
        Z = (Xc - mu) / sd
        G = Z @ Z.T
        tr = np.trace(G) / n; m = tr / d
        dl = np.sum(G ** 2) / n ** 2
        beta = (np.sum(np.diag(G) ** 2) / n - dl) / (d * n)
        delta = (dl - 2 * m * tr + d * m ** 2) / d
        a = 0.0 if beta == 0 else min(beta, delta) / delta        # Ledoit-Wolf shrinkage
        lam += 0.5 * a * m * sd ** 2
        U.append(np.sqrt(0.5 * (1 - a) / n) * (Z * sd).T)
    U, M = np.hstack(U), np.column_stack(M)
    if d <= U.shape[1]:
        Wm = np.linalg.solve(np.diag(lam) + U @ U.T, M)
    else:
        LiM, LiU = M / lam[:, None], U / lam[:, None]
        Wm = LiM - LiU @ np.linalg.solve(np.eye(U.shape[1]) + U.T @ LiU, U.T @ LiM)
    b = -0.5 * np.sum(M * Wm, axis=0)
    return Wm[:, 1] - Wm[:, 0], b[1] - b[0]

Xc = np.vstack([montage_feats('ME', s, 'standard31', ('low',), False) for s in SUBJ[:14]])
yc = np.concatenate([DATA['ME'][s]['y'] for s in SUBJ[:14]])
Xv = montage_feats('MI', 15, 'standard31', ('low',), True)
w, b = slda_fit(Xc, yc)
ref = LDA(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5]).fit(Xc, yc).decision_function(Xv)
print(f'fast LDA vs sklearn: same decisions in {100 * np.mean((Xv @ w + b > 0) == (ref > 0)):.1f}% of trials')

# 5) leave-one-subject-out: train on 14 others (clean trials), test on the new person's MI runs 6-10 (all trials)
def loso(train_cond, montage, bands, mode):
    Ftr = {s: montage_feats(train_cond, s, montage, bands, False) for s in SUBJ}
    Fte = {s: montage_feats('MI', s, montage, bands, True) for s in SUBJ}
    acc = {}
    for t in SUBJ:
        Xs, ys = [], []
        for s in SUBJ:
            if s == t:
                continue
            D, X = DATA[train_cond][s], Ftr[s]
            k = ~D['r4m']
            if mode == 'recentered':
                X = X - X[k].mean(0)
            Xs.append(X[k]); ys.append(D['y'][k])
        D, X = DATA['MI'][t], Fte[t]
        if mode == 'recentered':
            X = X - X[D['run'] <= 5].mean(0)          # unlabeled warm-up: the new person's MI runs 1-5
        te = D['run'] >= 6
        w, b = slda_fit(np.vstack(Xs), np.concatenate(ys))
        acc[t] = 100 * balanced_accuracy_score(D['y'][te], (X[te] @ w + b > 0).astype(int))
    return acc

CONFIGS = [('all61', ('low', 'mu_beta'), m) for m in ('strict', 'recentered')] + \
          [('standard31', bs, m) for bs in (('low', 'mu_beta'), ('low',), ('mu_beta',)) for m in ('strict', 'recentered')] + \
          [('frontal5', ('low', 'mu_beta'), 'strict'), ('eog3', ('low', 'mu_beta'), 'strict')]
RES = f'{OUT}/A2b_stage1_references.csv'
R = pd.read_csv(RES) if os.path.exists(RES) else pd.DataFrame(columns=['cell', 'montage', 'bands', 'mode', 'subject', 'bal_acc'])
done = set(zip(R['cell'], R['montage'], R['bands'], R['mode']))
for cell in ('ME', 'MI'):
    for montage, bands, mode in CONFIGS:
        key = (f'{cell} others -> MI', montage, '+'.join(bands), mode)
        if key in done:
            continue
        acc = loso(cell, montage, bands, mode)
        new = pd.DataFrame([{'cell': key[0], 'montage': montage, 'bands': key[2], 'mode': mode,
                             'subject': t, 'bal_acc': round(a, 1)} for t, a in acc.items()])
        R = pd.concat([R, new], ignore_index=True)
        R.to_csv(RES, index=False)                    # progress saved after every configuration
        print(f'{key} mean {new.bal_acc.mean():.1f}% ({(time.time() - t0) / 60:.1f} min)')

R = R.drop_duplicates(['cell', 'montage', 'bands', 'mode', 'subject'], keep='last')
R['bal_acc'] = R['bal_acc'].astype(float)
def summ(g):
    a = g.bal_acc.values
    return pd.Series({'mean': a.mean().round(1), 'median': np.median(a).round(1), 'sd': a.std(ddof=1).round(1),
                      'min': a.min(), 'max': a.max(), 'n_above_50': int((a > 50).sum()),
                      'p_wilcoxon': round(wilcoxon(a - 50, alternative='greater').pvalue, 4) if np.any(a != 50) else 1.0})
S = R.groupby(['cell', 'montage', 'bands', 'mode'], sort=False)[['bal_acc']].apply(summ).reset_index()
S['n_above_50'] = S['n_above_50'].astype(int)
S.to_csv(f'{OUT}/A2b_stage1_summary.csv', index=False)
print('\n=== A.2b-1: T1 (hand vs rest), new person\'s MI runs 6-10, balanced accuracy %, 15 subjects ===')
print(S.to_string(index=False))

g = S[(S.cell == 'ME others -> MI') & (S.montage == 'standard31') & (S.bands == 'low+mu_beta') & (S['mode'] == 'strict')].iloc[0]
go = (g['mean'] >= 55) and (g.p_wilcoxon < 0.05)
print(f"\nGo/no-go (strict, 31 sites, both bands): mean {g['mean']}%, p = {g.p_wilcoxon} -> "
      + ('GO: run the selection in strict mode' if go else 'STOP: signal too weak for a stable selection; decide together'))
print(f'total time {(time.time() - t0) / 60:.1f} min')
print('===== A.2b-1 FINISHED =====')
