# Steps A.11-A.13, GPU part (pre-registered 12 Nov 2026, STEP_A11_13_PREREG.md). Exact A.10b training code (ShallowConvNet, cell 4).
# Part 1: strict and aligned (10, 60, all), 5 seeds; saves per-trial P(movement) for the new person and for the two
#         early-stopping people (A.12 thresholds, A.13b ensemble). Balanced accuracy is only checked against A.10b.
# Part 2: library-size curve (A.11): K = 3, 6, 9 training people, repeats r = 0-2, strict and aligned 60;
#         K = 12 is taken from part 1 seeds 0-2 (identical configuration to A.10b).
import os, sys, json, time, copy, shutil, warnings, numpy as np, pandas as pd, mne, torch, torch.nn as nn
from scipy import stats
from sklearn.metrics import balanced_accuracy_score as bal
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
el = lambda: f'{(time.time() - t0) / 60:.1f} min'
DEV = 'cuda' if torch.cuda.is_available() else None
print(f'===== A.11-13 GPU started {time.strftime("%H:%M:%S")} =====', flush=True)
if DEV is None:
    sys.exit('no GPU: switch the runtime to L4 or T4 and start again')
print('GPU:', torch.cuda.get_device_name(0), flush=True)

ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
A1, A2B, A4B = f'{ROOT}/01_preprocessed', f'{ROOT}/03_A2b_electrodes', f'{ROOT}/07_A4b_deep'
A10, OUT = f'{ROOT}/12_A10_deep_calibration', f'{ROOT}/14_A11_13'
os.makedirs(OUT, exist_ok=True)
PROB, VPROB, LIB = f'{OUT}/A1113_deep_test_probs.csv', f'{OUT}/A1113_deep_val_probs.csv', f'{OUT}/A1113_deep_library.csv'
FS, SUBJ = 128, list(range(1, 16))
EEG = json.load(open(f'{A1}/ME_sub01_meta.json'))['channels'][:61]
POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
PATHS = {int(k): v for k, v in json.load(open(f'{A2B}/A2b2_state.json'))['paths'].items()}
SEEDS, MAX_EP, PATIENCE, BATCH, LR = (0, 1, 2, 3, 4), 150, 20, 64, 1e-3     # A.4b-1 settings; seeds extended from 2 to 5
BUDGETS, KS, REPS = [10, 60, 'all'], [3, 6, 9], (0, 1, 2)

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

# A.4b-1 time-series cache (unchanged)
TS, BADS = {'ME': {}, 'MI': {}}, {}
for cond in ('ME', 'MI'):
    for s in SUBJ:
        BADS[(cond, s)] = [c for c in json.load(open(f'{A1}/{cond}_sub{s:02d}_meta.json'))['bad_channels'] if c in EEG]
        fn = f'{A4B}/cache_ts/{cond}_sub{s:02d}.npz'
        if not os.path.exists(fn):
            sys.exit(f'missing {fn}')
        with np.load(fn) as z:
            TS[cond][s] = {k: z[k] for k in z.files}
print(f'data ready ({el()})', flush=True)

# --- functions copied unchanged from a4b1.py ---
QB = {}
def get_ts(cond, s, S, idx, within):
    D = TS[cond][s]
    raw = within and 'X31raw' in D and any(POOL[i] in BADS[(cond, s)] for i in S)
    X = D['X31raw' if raw else 'X31'][idx][:, S, :].astype(np.float64)
    if raw:
        X = interp_matrix([POOL[i] for i in S], BADS[(cond, s)]) @ X
    k = len(S)
    if k not in QB:
        QB[k] = np.linalg.qr(np.c_[np.ones(k), np.eye(k)[:, :k - 1]])[0][:, 1:]
    return QB[k].T @ X

def ea(X):
    R = np.mean(X @ np.swapaxes(X, 1, 2), axis=0) / X.shape[2]
    w, V = np.linalg.eigh(R + 1e-6 * np.trace(R) / len(R) * np.eye(len(R)))
    return V @ np.diag(w ** -0.5) @ V.T

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

def proba(model, X):
    """P(movement) per trial (softmax of the same outputs that predict() takes the argmax of)."""
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(torch.softmax(model(torch.tensor(X[i:i + 256], dtype=torch.float32, device=DEV)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)

def fit(t, seed, mode, train_people=None):
    """As A.10b run(): early-stopping people from rng(100*t + seed); training on train_people (default: all the other 12)."""
    S = sorted(PATHS[t][:8])
    clean = lambda cond, s: np.where(~TS[cond][s]['r4m'])[0]
    others = [s for s in SUBJ if s != t]
    val = list(np.random.default_rng(100 * t + seed).choice(others, 2, replace=False))
    pool = [s for s in others if s not in val]
    people = pool if train_people is None else sorted(train_people)
    Xs, ys = [], []
    for s in people:
        i = clean('ME', s); X = get_ts('ME', s, S, i, False)
        if mode == 'aligned':
            X = ea(X) @ X
        Xs.append(X); ys.append(TS['ME'][s]['y'][i])
    Xv, yv, vs = [], [], []
    for s in val:
        D = TS['MI'][s]; i = np.where(D['run'] >= 6)[0]; X = get_ts('MI', s, S, i, False)
        if mode == 'aligned':
            X = ea(get_ts('MI', s, S, np.where(D['run'] <= 5)[0], False)) @ X
        Xv.append(X); yv.append(D['y'][i]); vs.append(np.full(len(i), s))
    Xtr, ytr, Xva, yva = np.concatenate(Xs), np.concatenate(ys), np.concatenate(Xv), np.concatenate(yv)
    sc = Xtr.std()
    model, ep = train(Xtr / sc, ytr, Xva / sc, yva, seed)
    return model, sc, ep, S, pool, (Xva / sc, yva, np.concatenate(vs))

def test_sets(t, S, mode, budgets):
    """The new person's MI runs 6-10: raw (strict) or aligned on the first N trials of MI runs 1-5."""
    Dt = TS['MI'][t]; ite = np.where(Dt['run'] >= 6)[0]; cal = np.where(Dt['run'] <= 5)[0]
    X0, y = get_ts('MI', t, S, ite, True), Dt['y'][ite]
    if mode == 'strict':
        return {'strict': X0}, y
    return {f'aligned {N}': ea(get_ts('MI', t, S, cal if N == 'all' else cal[:N], True)) @ X0 for N in budgets}, y

def save(df, fn):
    tmp = '/content/a1113_tmp.csv'; df.to_csv(tmp, index=False)
    shutil.copyfile(tmp, fn + '.part'); os.replace(fn + '.part', fn)

L = pd.read_csv(LIB) if os.path.exists(LIB) else pd.DataFrame()
done = set(L['subject'].astype(int)) if len(L) else set()
P = pd.read_csv(PROB) if os.path.exists(PROB) and done else pd.DataFrame()
V = pd.read_csv(VPROB) if os.path.exists(VPROB) and done else pd.DataFrame()
if len(P):
    P = P[P.subject.astype(int).isin(done)]; V = V[V.subject.astype(int).isin(done)]
for t in SUBJ:
    if t in done:
        print(f'person {t:2d} already done', flush=True); continue
    pr, vr, lr = [], [], []
    # part 1: full library (12 people), 5 seeds
    for seed in SEEDS:
        for mode in ('strict', 'aligned'):
            model, sc, ep, S, pool, (Xva, yva, vsub) = fit(t, seed, mode)
            sets, y = test_sets(t, S, mode, BUDGETS)
            for m, X in sets.items():
                p = proba(model, X / sc)
                pr.append(pd.DataFrame({'subject': t, 'seed': seed, 'method': m, 'trial': np.arange(len(y)),
                                        'y_true': y, 'p_move': p, 'best_epoch': ep}))
                if seed in REPS and m in ('strict', 'aligned 60'):
                    lr.append({'subject': t, 'K': 12, 'rep': seed, 'mode': m, 'bal_acc': 100 * bal(y, (p > 0.5).astype(int)),
                               'best_epoch': ep})
            vr.append(pd.DataFrame({'subject': t, 'seed': seed, 'mode': mode, 'val_subject': vsub, 'y_true': yva,
                                    'p_move': proba(model, Xva)}))
    # part 2: smaller libraries
    for K in KS:
        for r in REPS:
            others = [s for s in SUBJ if s != t]
            val = list(np.random.default_rng(100 * t + r).choice(others, 2, replace=False))
            pool = [s for s in others if s not in val]
            people = list(np.random.default_rng(1000 * t + 10 * K + r).choice(pool, K, replace=False))
            for mode in ('strict', 'aligned'):
                model, sc, ep, S, _, _ = fit(t, r, mode, train_people=people)
                sets, y = test_sets(t, S, mode, [60])
                for m, X in sets.items():
                    lr.append({'subject': t, 'K': K, 'rep': r, 'mode': m,
                               'bal_acc': 100 * bal(y, (proba(model, X / sc) > 0.5).astype(int)), 'best_epoch': ep})
    P = pd.concat([P] + pr, ignore_index=True); save(P, PROB)
    V = pd.concat([V] + vr, ignore_index=True); save(V, VPROB)
    L = pd.concat([L, pd.DataFrame(lr)], ignore_index=True); save(L, LIB)
    g = pd.DataFrame(lr).groupby(['mode', 'K']).bal_acc.mean()
    print(f'person {t:2d} done: strict K=3/6/9/12 ' + ' / '.join(f'{g[("strict", k)]:.1f}' for k in (3, 6, 9, 12))
          + f' ({el()})', flush=True)

# checks and summary
P['subject'] = P['subject'].astype(int); L['subject'] = L['subject'].astype(int)
P['pred'] = (P.p_move > 0.5).astype(int)
acc = P.groupby(['subject', 'seed', 'method']).apply(lambda d: 100 * bal(d.y_true, d.pred)).rename('bal_acc').reset_index()
B = pd.read_csv(f'{A10}/A10b_results.csv')
print('\n=== part 1 agreement with A.10b (balanced accuracy, 15 people x 5 seeds) ===')
for m in ['strict'] + [f'aligned {b}' for b in BUDGETS]:
    new = acc[acc.method == m].set_index(['subject', 'seed']).bal_acc
    old = B[B.method == m].set_index(['subject', 'seed']).bal_acc
    d = (new - old.reindex(new.index)).dropna()
    print(f'  {m:12s}: now {new.groupby(level=0).mean().mean():5.1f} vs A.10b {old.groupby(level=0).mean().mean():5.1f}; '
          f'identical in {int((d.abs() < 0.05).sum())}/{len(d)}, max abs difference {d.abs().max():.1f}')
print('\n=== A.11 ShallowConvNet library-size curve (mean of 15 people; each person = mean of 3 repeats) ===')
for m in ('strict', 'aligned 60'):
    row = []
    for K in KS + [12]:
        v = L[(L['mode'] == m) & (L.K == K)].groupby('subject').bal_acc.mean()
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        row.append(f'K={K:2d} {v.mean():5.1f} [{v.mean() - h:4.1f}, {v.mean() + h:4.1f}]')
    print(f'  {m:10s}: ' + ' | '.join(row))
print(f'\ntotal time {el()}')
print('===== A.11-13 GPU FINISHED =====', flush=True)
