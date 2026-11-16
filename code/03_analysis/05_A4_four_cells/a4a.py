# Step A.4a: four-cell zero-calibration table + controls (background, resumable)
import os, sys, json, shutil, time, warnings, numpy as np, pandas as pd, mne
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score as bal
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
el = lambda: f'{(time.time() - t0) / 60:.1f} min'
print(f'\n===== A.4a started {time.strftime("%H:%M:%S")} =====')

ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
A1 = f'{ROOT}/01_preprocessed'
if not os.path.exists(f'{A1}/MI_sub15.npz'):
    A1 = '/content/a1_preprocessed'
A2B, OUT = f'{ROOT}/03_A2b_electrodes', f'{ROOT}/05_A4_four_cells'
CACHE_T1, CACHE_PRE = f'{A2B}/cache', f'{OUT}/cache_precue'
os.makedirs(CACHE_PRE, exist_ok=True)
for f in (f'{A1}/MI_sub15.npz', f'{CACHE_T1}/MI_sub15.npz', f'{A2B}/A2b2_state.json', f'{A2B}/A2b2_consensus.json'):
    if not os.path.exists(f):
        sys.exit(f'STOP: missing {f}')

FS, SUBJ = 128, list(range(1, 16))
EEG = json.load(open(f'{A1}/ME_sub01_meta.json'))['channels'][:61]
POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
PIDX = [EEG.index(c) for c in POOL]
BANDS = ('low', 'mu_beta')
SOS = {'low': butter(4, (0.3, 3.0), btype='band', fs=FS, output='sos'),
       'mu_beta': butter(4, (8.0, 30.0), btype='band', fs=FS, output='sos')}
WLEN = FS
WI = np.arange(21)                                   # post-cue windows: starts at cue + 0.0 ... 2.0 s
W_POST05 = np.arange(5, 21)                          # control: starts at cue + 0.5 ... 2.0 s
PRE_START = np.round((np.arange(11) * 0.1 - 2.0) * FS).astype(int)   # control: starts at cue - 2.0 ... -1.0 s
T1MAP, HAND = {'hand_close': 1, 'hand_open': 1, 'rest': 0}, ['hand_close', 'hand_open']

PATHS = {int(k): v for k, v in json.load(open(f'{A2B}/A2b2_state.json'))['paths'].items()}
CONS = json.load(open(f'{A2B}/A2b2_consensus.json'))['sites']
print('consensus montage:', CONS)

mname = 'colin27_1005' if 'colin27_1005' in mne.channels.get_builtin_montages() else 'standard_1005'
info = mne.create_info(EEG, FS, 'eeg')
info.set_montage(mne.channels.make_standard_montage(mname), match_case=False)
POS = np.array([ch['loc'][:3] for ch in info['chs']])
_, ORIGIN, _ = mne.bem.fit_sphere_to_headshape(info, dig_kinds=('eeg',), units='m', verbose=False)
pc3 = POS[EEG.index('C3')]
NEAR8 = sorted(int(i) for i in np.argsort([np.linalg.norm(POS[EEG.index(c)] - pc3) for c in POOL], kind='stable')[:8])
FIXED = {'cons8': sorted(POOL.index(c) for c in CONS),
         'frontal5': [POOL.index(c) for c in ['F3', 'F1', 'Fz', 'F2', 'F4']],
         'central5': sorted(POOL.index(c) for c in ['Cz', 'CP3', 'C1', 'C2', 'C6']),
         'parietal3': sorted(POOL.index(c) for c in ['P3', 'Pz', 'P4']),
         'nearC3_8': NEAR8}

def montage(name, t):
    return 'eog3' if name == 'eog3' else (sorted(PATHS[t][:8]) if name == 'fold8' else FIXED[name])

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
        a = 0.0 if beta == 0 else min(beta, delta) / delta
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
    tmp = '/content/_a4_tmp.npz'
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

# 1) data: post-cue cache from A.2b, and a new pre-cue cache (control)
POST, PRE, BADS = {'ME': {}, 'MI': {}}, {'ME': {}, 'MI': {}}, {}
for cond in ('ME', 'MI'):
    for s in SUBJ:
        BADS[(cond, s)] = [c for c in json.load(open(f'{A1}/{cond}_sub{s:02d}_meta.json'))['bad_channels'] if c in EEG]
        POST[cond][s] = load_npz(f'{CACHE_T1}/{cond}_sub{s:02d}.npz', ('cov31_low', 'eog_mu_beta', 'cls', 'y'))
        if POST[cond][s] is None:
            sys.exit(f'STOP: A.2b cache for {cond} {s} is missing or damaged')
        fn = f'{CACHE_PRE}/{cond}_sub{s:02d}.npz'
        D = load_npz(fn, ('cov31_low', 'cov31_mu_beta', 'y', 'cls'))
        if D is None:
            Z = np.load(f'{A1}/{cond}_sub{s:02d}.npz')
            E = pd.read_csv(f'{A1}/{cond}_sub{s:02d}_events.csv')
            E = E[E['class'].isin(T1MAP)].sort_values(['run', 'trial']).reset_index(drop=True)
            bads = BADS[(cond, s)]; W61 = interp_matrix(EEG, bads)
            cov = {b: [] for b in BANDS}; keep = np.zeros(len(E), bool)
            for r in range(1, 11):
                X = Z[f'run{r}'].astype(np.float64)
                rows = np.where(E.run.values == r)[0]
                cues = E.cue_sample_128.values[rows]
                ok = (cues + PRE_START[0] >= 0) & (cues + WLEN <= X.shape[1])
                keep[rows[ok]] = True
                idx = cues[ok][:, None] + PRE_START[None, :]
                for b in BANDS:
                    Y = causal_band(X, b)
                    S = np.stack([Y[:61, i:i + WLEN] for i in idx.ravel()])
                    cov[b].append((S @ np.swapaxes(S, 1, 2) / WLEN).reshape(len(idx), len(PRE_START), 61, 61))
            E = E[keep].reset_index(drop=True)
            D = {'y': np.asarray(E['class'].map(T1MAP), dtype=np.int64), 'run': np.asarray(E.run, dtype=np.int64),
                 'r4m': np.asarray(E.r4m_rejected, dtype=bool), 'cls': np.array(E['class'].tolist(), dtype='U16')}
            for b in BANDS:
                C = np.concatenate(cov[b])
                C61 = W61 @ C @ W61.T
                D[f'cov31_{b}'] = C61[..., PIDX, :][..., PIDX].astype(np.float32)
                if cond == 'MI' and any(c in POOL for c in bads):
                    D[f'cov31raw_{b}'] = C[..., PIDX, :][..., PIDX].astype(np.float32)
            save_npz(fn, D)
            print(f'pre-cue {cond} subject {s:2d}: {len(E)} trials ({el()})')
        PRE[cond][s] = D
print(f'data ready ({el()})')

def F(src, cond, s, S, idx, within, wins):
    D, parts = src[cond][s], []
    for b in BANDS:
        if S == 'eog3':
            parts.append(logvec(D[f'eog_{b}'][np.ix_(idx, wins)].astype(np.float64)).mean(axis=1)); continue
        raw = within and f'cov31raw_{b}' in D and any(POOL[i] in BADS[(cond, s)] for i in S)
        C = D[f'cov31raw_{b}' if raw else f'cov31_{b}'][np.ix_(idx, wins, S, S)].astype(np.float64)
        if raw:
            W = interp_matrix([POOL[i] for i in S], BADS[(cond, s)]); C = W @ C @ W.T
        parts.append(car_logmean(C))
    return np.hstack(parts)

CELLS = {1: ('MI', 'self'), 2: ('MI', 'others'), 3: ('ME', 'self'), 4: ('ME', 'others')}
CNAME = {1: '1 MI self', 2: '2 MI others', 3: '3 ME self', 4: '4 ME others'}

def run_cell(cell, t, task='T1', mname='fold8', src=None, wins=WI, reject=False, nperm=0):
    src = POST if src is None else src
    cond, who = CELLS[cell]
    S = montage(mname, t)
    tmask = lambda D: np.ones(len(D['y']), bool) if task == 'T1' else np.isin(D['cls'], HAND)
    lab = lambda D, i: D['y'][i] if task == 'T1' else (D['cls'][i] == 'hand_close').astype(int)
    if who == 'self':
        D = src[cond][t]
        i = np.where(tmask(D) & ~D['r4m'] & ((D['run'] <= 5) if cond == 'MI' else True))[0]
        Xtr, ytr = F(src, cond, t, S, i, cond == 'MI', wins), lab(D, i)
    else:
        Xs, ys = [], []
        for s in SUBJ:
            if s != t:
                D = src[cond][s]; i = np.where(tmask(D) & ~D['r4m'])[0]
                Xs.append(F(src, cond, s, S, i, False, wins)); ys.append(lab(D, i))
        Xtr, ytr = np.vstack(Xs), np.concatenate(ys)
    D = src['MI'][t]
    i = np.where(tmask(D) & (D['run'] >= 6) & ((~D['r4m']) if reject else True))[0]
    Xte, yte = F(src, 'MI', t, S, i, True, wins), lab(D, i)
    acc = 100 * bal(yte, predict(Xte, slda_fit(Xtr, ytr)))
    if not nperm:
        return acc, np.nan, np.nan
    rng = np.random.default_rng(1000 * cell + t)
    null = np.array([100 * bal(yte, predict(Xte, slda_fit(Xtr, rng.permutation(ytr)))) for _ in range(nperm)])
    return acc, (1 + np.sum(null >= acc)) / (nperm + 1), null.mean()

# self-check: fast LDA vs sklearn (cell 4, person 15, fold-specific montage)
S = montage('fold8', 15)
Xs = np.vstack([F(POST, 'ME', s, S, np.where(~POST['ME'][s]['r4m'])[0], False, WI) for s in SUBJ[:14]])
ys = np.concatenate([POST['ME'][s]['y'][~POST['ME'][s]['r4m']] for s in SUBJ[:14]])
Xv = F(POST, 'MI', 15, S, np.where(POST['MI'][15]['run'] >= 6)[0], True, WI)
ref = LDA(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5]).fit(Xs, ys).predict(Xv)
print(f'fast LDA vs sklearn: same decisions in {100 * np.mean(predict(Xv, slda_fit(Xs, ys)) == ref):.1f}% of trials')

# 2) all analyses, saved after every person (resumable)
RES = f'{OUT}/A4a_results.csv'
R = pd.read_csv(RES) if os.path.exists(RES) else pd.DataFrame()
done = set(R['subject'].astype(int)) if len(R) else set()
for t in SUBJ:
    if t in done:
        continue
    rows = []
    def add(analysis, cell, acc, p, nullm):
        rows.append({'subject': t, 'analysis': analysis, 'cell': CNAME[cell], 'bal_acc': acc,
                     'p_perm': p, 'perm_null_mean': nullm})
    for c in CELLS:
        add('T1 fold8 (primary)', c, *run_cell(c, t, nperm=1000))
        add('T1 cons8 (optimistic)', c, *run_cell(c, t, mname='cons8'))
        add('T1 fold8, R4M-clean test', c, *run_cell(c, t, reject=True))
        add('T2 fold8', c, *run_cell(c, t, task='T2'))
    for c in (2, 4):
        for m in ('eog3', 'frontal5', 'central5', 'parietal3', 'nearC3_8'):
            add(f'control: {m}', c, *run_cell(c, t, mname=m))
        add('control: fold8 without first 0.5 s', c, *run_cell(c, t, wins=W_POST05))
        add('control: fold8 pre-cue (-2 to 0 s)', c, *run_cell(c, t, src=PRE, wins=np.arange(len(PRE_START))))
    R = pd.concat([R, pd.DataFrame(rows)], ignore_index=True)
    R.to_csv(RES, index=False)
    print(f'person {t:2d} done ({el()})')
R['subject'] = R['subject'].astype(int); R['bal_acc'] = R['bal_acc'].astype(float)
R = R.drop_duplicates(['subject', 'analysis', 'cell'], keep='last')

# 3) statistics
def pw(x, ref=None):
    d = np.asarray(x) - (50 if ref is None else np.asarray(ref))
    if not np.any(d != 0):
        return 1.0
    return wilcoxon(d, alternative='greater' if ref is None else 'two-sided').pvalue

def boot_ci(a, n=10000):
    m = np.median(np.random.default_rng(0).choice(a, (n, len(a)), replace=True), axis=1)
    return np.percentile(m, [2.5, 97.5])

def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj

def table(analysis):
    out = []
    for c in CELLS:
        g = R[(R.analysis == analysis) & (R.cell == CNAME[c])].sort_values('subject')
        a = g.bal_acc.values; lo, hi = boot_ci(a)
        row = {'cell': CNAME[c], 'mean': round(a.mean(), 1), 'median': round(np.median(a), 1),
               'median_95CI': f'{lo:.1f}-{hi:.1f}', 'sd': round(a.std(ddof=1), 1), 'min': round(a.min(), 1),
               'max': round(a.max(), 1), 'n_above_50': int((a > 50).sum()), 'p_vs_50': round(pw(a), 4)}
        if g.p_perm.notna().any():
            row['n_sig_perm'] = int((g.p_perm.astype(float) < 0.05).sum())
            row['perm_null_mean'] = round(g.perm_null_mean.astype(float).mean(), 1)
        out.append(row)
    return pd.DataFrame(out)

P = R[R.analysis == 'T1 fold8 (primary)'].pivot(index='subject', columns='cell', values='bal_acc').loc[SUBJ]
print('\n=== A.4a-1: four cells, T1 hand vs rest, fold-specific 8 sites (PRIMARY), new person MI runs 6-10, % ===')
print(table('T1 fold8 (primary)').to_string(index=False))
print('\nper person (* = permutation p < 0.05):')
PP = R[R.analysis == 'T1 fold8 (primary)'].pivot(index='subject', columns='cell', values='p_perm').loc[SUBJ]
print(pd.DataFrame({c: [f'{P.loc[s, c]:.1f}' + ('*' if float(PP.loc[s, c]) < 0.05 else '') for s in SUBJ]
                    for c in P.columns}, index=SUBJ).to_string())

pairs = [(1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (2, 3)]
raw = [pw(P[CNAME[a]].values, P[CNAME[b]].values) for a, b in pairs]
adj = holm(raw)
print('\n=== A.4a-2: cell comparisons (Wilcoxon, two-sided, Holm-corrected) ===')
print(pd.DataFrame([{'comparison': f'{CNAME[a]} vs {CNAME[b]}',
                     'mean_difference': round((P[CNAME[a]] - P[CNAME[b]]).mean(), 1),
                     'p_raw': round(r, 4), 'p_holm': round(h, 4)} for (a, b), r, h in zip(pairs, raw, adj)]).to_string(index=False))
c1, c2, c3, c4 = (P[CNAME[k]].mean() for k in (1, 2, 3, 4))
print('\n=== A.4a-3: where the accuracy is lost (mean points) ===')
print(f'changing the person:   with MI training {c1 - c2:+.1f} | with ME training {c3 - c4:+.1f}')
print(f'changing ME for MI:    own data {c1 - c3:+.1f} | other people {c2 - c4:+.1f}')
print(f'interaction (person x condition): {(c1 - c2) - (c3 - c4):+.1f}')

print('\n=== A.4a-4: same table, consensus 8 sites (optimistic: chosen with all 15) ===')
print(table('T1 cons8 (optimistic)')[['cell', 'mean', 'median', 'n_above_50', 'p_vs_50']].to_string(index=False))
print('\n=== A.4a-5: same table, R4M-clean test trials (secondary) ===')
print(table('T1 fold8, R4M-clean test')[['cell', 'mean', 'median', 'n_above_50', 'p_vs_50']].to_string(index=False))
print('\n=== A.4a-6: T2 open vs close, fold-specific 8 sites ===')
print(table('T2 fold8')[['cell', 'mean', 'median', 'median_95CI', 'n_above_50', 'p_vs_50']].to_string(index=False))

print('\n=== A.4a-7: controls (T1, strict), mean %, p vs 50 ===')
out = []
for an in [a for a in R.analysis.unique() if a.startswith('control')]:
    for c in (2, 4):
        a = R[(R.analysis == an) & (R.cell == CNAME[c])].bal_acc.values
        out.append({'control': an.replace('control: ', ''), 'cell': CNAME[c], 'mean': round(a.mean(), 1),
                    'n_above_50': int((a > 50).sum()), 'p_vs_50': round(pw(a), 4)})
print(pd.DataFrame(out).to_string(index=False))
print('near-C3 8 sites:', [POOL[i] for i in NEAR8])
print(f'\ntotal time {el()}')
print('===== A.4a FINISHED =====')
