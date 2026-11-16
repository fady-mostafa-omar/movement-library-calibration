# Step A.6: unsupervised adaptation (recentering) - how many minutes of unlabelled warm-up are needed
import os, sys, json, time, warnings, numpy as np, pandas as pd, mne
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score as bal
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
el = lambda: f'{(time.time() - t0) / 60:.1f} min'
print(f'\n===== A.6 started {time.strftime("%H:%M:%S")} =====')

ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
A1 = f'{ROOT}/01_preprocessed'
if not os.path.exists(f'{A1}/ME_sub01_meta.json'):
    A1 = '/content/a1_preprocessed'
A2B, A4, OUT = f'{ROOT}/03_A2b_electrodes', f'{ROOT}/05_A4_four_cells', f'{ROOT}/06_A6_adaptation'
for f in (f'{A1}/ME_sub01_meta.json', f'{A2B}/cache/MI_sub15.npz', f'{A2B}/A2b2_state.json', f'{A4}/A4a_results.csv'):
    if not os.path.exists(f):
        sys.exit(f'STOP: missing {f}')

FS, SUBJ = 128, list(range(1, 16))
EEG = json.load(open(f'{A1}/ME_sub01_meta.json'))['channels'][:61]
POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
BANDS, WI = ('low', 'mu_beta'), np.arange(21)
PATHS = {int(k): v for k, v in json.load(open(f'{A2B}/A2b2_state.json'))['paths'].items()}
NWARM = [0, 10, 20, 30, 60, 90]                      # warm-up trials (each = 3 s of signal); 0 = strict

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

DATA, BADS = {'ME': {}, 'MI': {}}, {}
for cond in ('ME', 'MI'):
    for s in SUBJ:
        with np.load(f'{A2B}/cache/{cond}_sub{s:02d}.npz') as z:
            DATA[cond][s] = {k: z[k] for k in z.files}
        BADS[(cond, s)] = [c for c in json.load(open(f'{A1}/{cond}_sub{s:02d}_meta.json'))['bad_channels'] if c in EEG]
print(f'cache loaded ({el()})')

def F(cond, s, S, idx, within):
    D, parts = DATA[cond][s], []
    for b in BANDS:
        raw = within and f'cov31raw_{b}' in D and any(POOL[i] in BADS[(cond, s)] for i in S)
        C = D[f'cov31raw_{b}' if raw else f'cov31_{b}'][np.ix_(idx, WI, S, S)].astype(np.float64)
        if raw:
            W = interp_matrix([POOL[i] for i in S], BADS[(cond, s)]); C = W @ C @ W.T
        parts.append(car_logmean(C))
    return np.hstack(parts)

CELLS = {2: ('MI', 'others'), 3: ('ME', 'self'), 4: ('ME', 'others')}
CNAME = {2: '2 MI others', 3: '3 ME self', 4: '4 ME others'}
RES = f'{OUT}/A6_results.csv'
R = pd.read_csv(RES) if os.path.exists(RES) else pd.DataFrame()
done = set(R['subject'].astype(int)) if len(R) else set()
for t in SUBJ:
    if t in done:
        continue
    S = sorted(PATHS[t][:8])                         # fold-specific montage (chosen without person t)
    D = DATA['MI'][t]
    te = np.where(D['run'] >= 6)[0]; Xte, yte = F('MI', t, S, te, True), D['y'][te]
    warm = np.where(D['run'] <= 5)[0]                # in time order (runs 1-5); labels never used
    Xw = F('MI', t, S, warm, True)
    rows = []
    for c, (cond, who) in CELLS.items():
        subs = [t] if who == 'self' else [s for s in SUBJ if s != t]
        Xs, ys = [], []
        for s in subs:
            i = np.where(~DATA[cond][s]['r4m'])[0]
            Xs.append(F(cond, s, S, i, False)); ys.append(DATA[cond][s]['y'][i])
        y = np.concatenate(ys)
        strict = slda_fit(np.vstack(Xs), y)
        recent = slda_fit(np.vstack([X - X.mean(0) for X in Xs]), y)     # each training session centred on itself
        for n in NWARM:
            if n == 0:
                acc = 100 * bal(yte, predict(Xte, strict))
            else:
                acc = 100 * bal(yte, predict(Xte - Xw[:n].mean(0), recent))
            rows.append({'subject': t, 'cell': CNAME[c], 'warmup_trials': n, 'minutes': n * 3 / 60, 'bal_acc': acc})
        rows.append({'subject': t, 'cell': CNAME[c], 'warmup_trials': -1, 'minutes': np.nan,
                     'bal_acc': 100 * bal(yte, predict(Xte - Xte.mean(0), recent))})   # adapts on the test runs, unlabelled
    R = pd.concat([R, pd.DataFrame(rows)], ignore_index=True)
    R.to_csv(RES, index=False)
    print(f'person {t:2d} done ({el()})')
R['subject'] = R['subject'].astype(int); R['bal_acc'] = R['bal_acc'].astype(float); R['warmup_trials'] = R['warmup_trials'].astype(int)
R = R.drop_duplicates(['subject', 'cell', 'warmup_trials'], keep='last')

A4R = pd.read_csv(f'{A4}/A4a_results.csv')
c1 = A4R[(A4R.analysis == 'T1 fold8 (primary)') & (A4R.cell == '1 MI self')].sort_values('subject').bal_acc.values

def pw(x, ref, alt):
    d = np.asarray(x) - np.asarray(ref)
    return 1.0 if not np.any(d != 0) else wilcoxon(d, alternative=alt).pvalue

print('\n=== A.6: recentering with unlabelled warm-up from the new person (T1, fold-specific 8 sites, MI runs 6-10, %) ===')
out = []
for c in CNAME.values():
    g = R[R.cell == c]
    base = g[g.warmup_trials == 0].sort_values('subject').bal_acc.values
    for n in NWARM + [-1]:
        a = g[g.warmup_trials == n].sort_values('subject').bal_acc.values
        out.append({'cell': c,
                    'warm-up': 'strict (none)' if n == 0 else ('during use (test runs)' if n == -1 else f'{n} trials'),
                    'minutes_of_signal': '-' if n <= 0 else f'{n * 3 / 60:.1f}',
                    'mean': round(a.mean(), 1), 'median': round(np.median(a), 1), 'n_above_50': int((a > 50).sum()),
                    'gain_vs_strict': '-' if n == 0 else round((a - base).mean(), 1),
                    'p_gain': '-' if n == 0 else round(pw(a, base, 'greater'), 4),
                    'p_vs_cell1': round(pw(a, c1, 'two-sided'), 4)})
print(pd.DataFrame(out).to_string(index=False))
print(f'cell 1 (own MI, full labelled calibration of about 27 min): mean {c1.mean():.1f}%')
print('note: "minutes" = minutes of trial signal (3 s per trial); in Ofner they are spread over a longer session')
print(f'\ntotal time {el()}')
print('===== A.6 FINISHED =====')
