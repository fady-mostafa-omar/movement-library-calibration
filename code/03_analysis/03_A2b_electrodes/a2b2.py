# Step A.2b-2: nested forward selection of electrodes (background, resumable)
import os, sys, json, time, warnings, numpy as np, pandas as pd, mne
from collections import Counter
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
el = lambda: f'{(time.time() - t0) / 60:.1f} min'
print(f'\n===== A.2b-2 started {time.strftime("%H:%M:%S")} =====')

ROOT = '/content/drive/MyDrive/classification_MI-ME'
OUT = f'{ROOT}/30_v06_pipeline/03_A2b_electrodes'
A1 = f'{ROOT}/30_v06_pipeline/01_preprocessed'
if not os.path.exists(f'{A1}/ME_sub01_meta.json'):
    A1 = '/content/a1_preprocessed'
CACHE = f'{OUT}/cache' if os.path.exists(f'{OUT}/cache/MI_sub15.npz') else '/content/a2b_cache'
if not os.path.exists(f'{CACHE}/MI_sub15.npz') or not os.path.exists(f'{A1}/ME_sub01_meta.json'):
    sys.exit('STOP: the A.2b-1 cache or the A.1 meta files are missing; run A.2b-1 again first.')
STATE, OUTER = f'{OUT}/A2b2_state.json', f'{OUT}/A2b2_outer.csv'
ANATF, CONS = f'{OUT}/A2b2_anatomical.csv', f'{OUT}/A2b2_consensus.json'
print('cache:', CACHE)

FS, SUBJ = 128, list(range(1, 16))
EEG = json.load(open(f'{A1}/ME_sub01_meta.json'))['channels'][:61]
POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
BANDS, WI, HAND = ('low', 'mu_beta'), np.arange(21), ['hand_close', 'hand_open']
COUNTS, KMAX = [4, 6, 8, 12, 16], 16

# template positions (10-05), spherical-spline interpolation, features, fast shrinkage LDA (same as A.2b-1)
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
    d = X.shape[1]; lam = np.zeros(d); U, M = [], []
    for c in (0, 1):
        Xc = X[y == c]; n = len(Xc); mu = Xc.mean(0); M.append(mu)
        sd = Xc.std(0); sd[sd == 0] = 1.0
        Z = (Xc - mu) / sd
        r2 = np.einsum('ij,ij->i', Z, Z)                      # diagonal of Z Z'
        K = Z.T @ Z if d <= n else Z @ Z.T                    # same Frobenius norm either way
        tr = r2.sum() / n; m = tr / d
        dl = np.sum(K ** 2) / n ** 2
        beta = (np.sum(r2 ** 2) / n - dl) / (d * n)
        delta = (dl - 2 * m * tr + d * m ** 2) / d
        a = 0.0 if beta == 0 else min(beta, delta) / delta    # Ledoit-Wolf shrinkage
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

# load the A.2b-1 cache
DATA, BADS = {'ME': {}, 'MI': {}}, {}
for cond in ('ME', 'MI'):
    for s in SUBJ:
        with np.load(f'{CACHE}/{cond}_sub{s:02d}.npz') as z:
            DATA[cond][s] = {k: z[k] for k in z.files}
        BADS[(cond, s)] = [c for c in json.load(open(f'{A1}/{cond}_sub{s:02d}_meta.json'))['bad_channels'] if c in EEG]
print(f'cache loaded ({el()})')

IDX = {}
for s in SUBJ:
    me, mi = DATA['ME'][s], DATA['MI'][s]
    IDX[('ME', s, 'trainT1')] = np.where(~me['r4m'])[0]
    IDX[('ME', s, 'trainT2')] = np.where(~me['r4m'] & np.isin(me['cls'], HAND))[0]
    IDX[('MI', s, 'trainT1')] = np.where(~mi['r4m'])[0]
    IDX[('MI', s, 'trainT2')] = np.where(~mi['r4m'] & np.isin(mi['cls'], HAND))[0]
    IDX[('MI', s, 'testT1')] = np.where(mi['run'] >= 6)[0]
    IDX[('MI', s, 'testT2')] = np.where((mi['run'] >= 6) & np.isin(mi['cls'], HAND))[0]
    IDX[('MI', s, 'warm')] = np.where(mi['run'] <= 5)[0]

def lab(cond, s, idx, task):
    D = DATA[cond][s]
    return D['y'][idx] if task == 'T1' else (D['cls'][idx] == 'hand_close').astype(int)

def F(cond, s, S, idx, target):
    """Both bands; CAR over the montage S; the new person's bad channels are interpolated within S."""
    D, parts = DATA[cond][s], []
    for b in BANDS:
        raw = target and f'cov31raw_{b}' in D and any(POOL[i] in BADS[(cond, s)] for i in S)
        C = D[f'cov31raw_{b}' if raw else f'cov31_{b}'][np.ix_(idx, WI, S, S)].astype(np.float64)
        if raw:
            W = interp_matrix([POOL[i] for i in S], BADS[(cond, s)]); C = W @ C @ W.T
        parts.append(car_logmean(C))
    return np.hstack(parts)

YME = {s: lab('ME', s, IDX[('ME', s, 'trainT1')], 'T1') for s in SUBJ}
YMI = {s: lab('MI', s, IDX[('MI', s, 'testT1')], 'T1') for s in SUBJ}

# self-check: fast LDA vs sklearn on the 31-site features
S31 = list(range(31))
Xc = np.vstack([F('ME', s, S31, IDX[('ME', s, 'trainT1')], False) for s in SUBJ[:14]])
yc = np.concatenate([YME[s] for s in SUBJ[:14]])
Xv = F('MI', 15, S31, IDX[('MI', 15, 'testT1')], True)
w, b = slda_fit(Xc, yc)
ref = LDA(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5]).fit(Xc, yc).decision_function(Xv)
print(f'fast LDA vs sklearn: same decisions in {100 * np.mean((Xv @ w + b > 0) == (ref > 0)):.1f}% of trials')

def inner_eval(S, folds):
    """Inner LOSO (ME of 13 others -> MI runs 6-10 of the validation person) for every outer fold in `folds`.
    One model per pair {outer, validation} serves both folds that need it."""
    Xme = {s: F('ME', s, S, IDX[('ME', s, 'trainT1')], False) for s in SUBJ}
    Xmi = {s: F('MI', s, S, IDX[('MI', s, 'testT1')], True) for s in SUBJ}
    out = {o: {} for o in folds}
    for p in {tuple(sorted((o, v))) for o in folds for v in SUBJ if v != o}:
        tr = [s for s in SUBJ if s not in p]
        w, b = slda_fit(np.vstack([Xme[s] for s in tr]), np.concatenate([YME[s] for s in tr]))
        for o, v in (p, p[::-1]):
            if o in out:
                out[o][v] = 100 * balanced_accuracy_score(YMI[v], (Xmi[v] @ w + b > 0).astype(int))
    return {o: np.array([out[o][v] for v in SUBJ if v != o]) for o in folds}

def outer_eval(S, o, task='T1', mode='strict', train_cond='ME'):
    """Train on the 14 others, test on person o's MI runs 6-10."""
    trk, tek = ('trainT2', 'testT2') if task == 'T2' else ('trainT1', 'testT1')
    Xs, ys = [], []
    for s in SUBJ:
        if s == o:
            continue
        i = IDX[(train_cond, s, trk)]
        X = F(train_cond, s, S, i, False)
        if mode == 'recentered':
            X = X - X.mean(0)
        Xs.append(X); ys.append(lab(train_cond, s, i, task))
    i = IDX[('MI', o, tek)]
    X = F('MI', o, S, i, True)
    if mode == 'recentered':
        X = X - F('MI', o, S, IDX[('MI', o, 'warm')], True).mean(0)   # unlabeled MI runs 1-5
    w, b = slda_fit(np.vstack(Xs), np.concatenate(ys))
    return 100 * balanced_accuracy_score(lab('MI', o, i, task), (X @ w + b > 0).astype(int))

def save_state(st):
    with open(STATE + '.part', 'w') as f:
        json.dump(st, f)
    os.replace(STATE + '.part', STATE)

# 1) nested forward selection: first the best pair, then one site at a time up to 16 (all folds in step)
st = json.load(open(STATE)) if os.path.exists(STATE) else {'paths': {}, 'inner': {}}
if not st['paths']:
    pairs = [(i, j) for i in range(31) for j in range(i + 1, 31)]
    best = {o: (-1.0, None, None) for o in SUBJ}
    for n_done, S in enumerate(pairs, 1):
        r = inner_eval(list(S), SUBJ)
        for o in SUBJ:
            if r[o].mean() > best[o][0]:
                best[o] = (r[o].mean(), S, r[o])
        if n_done % 93 == 0:
            print(f'pairs {n_done}/465 ({el()})')
    for o in SUBJ:
        st['paths'][str(o)] = [int(x) for x in best[o][1]]
        st['inner'][str(o)] = {'2': best[o][2].tolist()}
    save_state(st)
    print(f'size  2 done ({el()})')

for k in range(3, KMAX + 1):
    if all(len(st['paths'][str(o)]) >= k for o in SUBJ):
        continue
    need = {}
    for o in SUBJ:
        p = st['paths'][str(o)]
        for i in range(31):
            if i not in p:
                need.setdefault(tuple(sorted(p + [i])), []).append(o)
    best = {o: (-1.0, None, None) for o in SUBJ}
    for S, folds in need.items():
        r = inner_eval(list(S), folds)
        for o in folds:
            m = r[o].mean()
            add = (set(S) - set(st['paths'][str(o)])).pop()
            if m > best[o][0] or (m == best[o][0] and add < best[o][1]):
                best[o] = (m, add, r[o])
    for o in SUBJ:
        st['paths'][str(o)].append(int(best[o][1]))
        st['inner'][str(o)][str(k)] = best[o][2].tolist()
    save_state(st)
    print(f'size {k:2d} done: {len(need)} distinct candidate sets ({el()})')

paths = {o: st['paths'][str(o)] for o in SUBJ}
inner = {o: {int(k): np.array(v) for k, v in st['inner'][str(o)].items()} for o in SUBJ}

# 2) honest outer test with fold-specific selections (+ descriptive rows at the planned counts)
Ro = pd.read_csv(OUTER) if os.path.exists(OUTER) else pd.DataFrame(columns=['fold', 'size', 'analysis', 'bal_acc'])
done = set(Ro['fold'].astype(int)) if len(Ro) else set()
for o in SUBJ:
    if o in done:
        continue
    rows = []
    for k in range(2, KMAX + 1):
        S = sorted(paths[o][:k])
        rows.append({'fold': o, 'size': k, 'analysis': 'ME->MI T1 strict', 'bal_acc': outer_eval(S, o)})
        if k in COUNTS:
            rows += [{'fold': o, 'size': k, 'analysis': 'ME->MI T1 recentered', 'bal_acc': outer_eval(S, o, mode='recentered')},
                     {'fold': o, 'size': k, 'analysis': 'MI->MI T1 strict', 'bal_acc': outer_eval(S, o, train_cond='MI')},
                     {'fold': o, 'size': k, 'analysis': 'ME->MI T2 strict', 'bal_acc': outer_eval(S, o, task='T2')}]
    Ro = pd.concat([Ro, pd.DataFrame(rows)], ignore_index=True)
    Ro.to_csv(OUTER, index=False)
    print(f'outer fold {o:2d} evaluated ({el()})')
Ro = Ro.drop_duplicates(['fold', 'size', 'analysis'], keep='last')
Ro[['fold', 'size']] = Ro[['fold', 'size']].astype(int); Ro['bal_acc'] = Ro['bal_acc'].astype(float)

# 3) anatomical comparator: the N standard sites nearest to C3 (10-05 template)
pc3 = POS[EEG.index('C3')]
near = np.argsort([np.linalg.norm(POS[EEG.index(c)] - pc3) for c in POOL], kind='stable')
ANAT = {N: sorted(int(i) for i in near[:N]) for N in COUNTS}
if os.path.exists(ANATF):
    Ra = pd.read_csv(ANATF)
else:
    rows = []
    for N in COUNTS:
        S = ANAT[N]
        Xme = {s: F('ME', s, S, IDX[('ME', s, 'trainT1')], False) for s in SUBJ}
        Xmi = {s: F('MI', s, S, IDX[('MI', s, 'testT1')], True) for s in SUBJ}
        for t in SUBJ:
            tr = [s for s in SUBJ if s != t]
            w, b = slda_fit(np.vstack([Xme[s] for s in tr]), np.concatenate([YME[s] for s in tr]))
            rows.append({'size': N, 'subject': t,
                         'bal_acc': 100 * balanced_accuracy_score(YMI[t], (Xmi[t] @ w + b > 0).astype(int))})
    Ra = pd.DataFrame(rows); Ra.to_csv(ANATF, index=False)

# 4) one-SE rule per fold, honest estimate, consensus montage
def one_se(inn):
    means = {k: inn[k].mean() for k in COUNTS}
    kb = max(COUNTS, key=lambda k: (means[k], -k))
    se = inn[kb].std(ddof=1) / np.sqrt(len(inn[kb]))
    return min(k for k in COUNTS if means[k] >= means[kb] - se)

def pw(x, y=None):
    d = np.asarray(x) - (50 if y is None else np.asarray(y))
    if not np.any(d != 0):
        return 1.0
    return round(wilcoxon(d, alternative='greater' if y is None else 'two-sided').pvalue, 4)

kstar = {o: one_se(inner[o]) for o in SUBJ}
cnt = Counter(kstar.values())
Kc = max(sorted(cnt), key=lambda k: cnt[k])
freq = Counter(i for o in SUBJ for i in paths[o][:Kc])
pos_mean = {i: np.mean([paths[o].index(i) for o in SUBJ if i in paths[o][:Kc]]) for i in freq}
consensus = sorted(freq, key=lambda i: (-freq[i], pos_mean[i], i))[:Kc]
json.dump({'K': int(Kc), 'sites': [POOL[i] for i in consensus], 'freq': {POOL[i]: int(freq[i]) for i in freq},
           'kstar_per_fold': {str(o): int(kstar[o]) for o in SUBJ},
           'paths': {str(o): [POOL[i] for i in paths[o]] for o in SUBJ}}, open(CONS, 'w'), indent=1)

sel = Ro[Ro.analysis == 'ME->MI T1 strict'].pivot(index='fold', columns='size', values='bal_acc').loc[SUBJ]
an = Ra.pivot(index='subject', columns='size', values='bal_acc').loc[SUBJ]
curve = []
for k in range(2, KMAX + 1):
    row = {'sites': k, 'inner_mean': round(np.mean([inner[o][k].mean() for o in SUBJ]), 1),
           'outer_mean': round(sel[k].mean(), 1), 'outer_median': round(sel[k].median(), 1),
           'n_above_50': int((sel[k] > 50).sum()), 'p_vs_50': pw(sel[k].values)}
    if k in COUNTS:
        row.update({'near_C3_mean': round(an[k].mean(), 1), 'p_selected_vs_near_C3': pw(sel[k].values, an[k].values)})
    curve.append(row)
C = pd.DataFrame(curve); C.to_csv(f'{OUT}/A2b2_curve.csv', index=False)

ref = pd.read_csv(f'{OUT}/A2b_stage1_summary.csv')
ref = ref[(ref.cell == 'ME others -> MI') & (ref.bands == 'low+mu_beta') & (ref['mode'] == 'strict')]
print('\n=== A.2b-2: selection curve (ME others -> new person MI, T1, strict, balanced accuracy %) ===')
print(C.to_string(index=False))
print('references (A.2b-1): ' + ', '.join(f'{r.montage} {r["mean"]}%' for _, r in ref.iterrows()))

h = np.array([sel.loc[o, kstar[o]] for o in SUBJ])
print('\n=== one-SE rule ===')
print('chosen count per outer fold:', {o: kstar[o] for o in SUBJ})
print(f'honest estimate (each fold at its own count): mean {h.mean():.1f}%, median {np.median(h):.1f}%, '
      f'sd {h.std(ddof=1):.1f}, min {h.min():.1f}, max {h.max():.1f}, above 50: {(h > 50).sum()}/15, p = {pw(h)}')

print(f'\n=== consensus montage: {Kc} sites (most frequent count) ===')
print(', '.join(f'{POOL[i]} ({freq[i]}/15)' for i in consensus))
print('other sites at this count:', ', '.join(f'{POOL[i]} ({freq[i]})' for i in sorted(freq, key=lambda i: -freq[i])
                                              if i not in consensus) or '-')
print('near-C3 montages:', {N: [POOL[i] for i in ANAT[N]] for N in COUNTS})

D2 = Ro[Ro['size'].isin(COUNTS)].groupby(['analysis', 'size']).bal_acc.mean().round(1).unstack('size')
print('\n=== descriptive (fold-specific selections, mean %) ===')
print(D2.to_string())
print(f'\ntotal time {el()}')
print('===== A.2b-2 FINISHED =====')
