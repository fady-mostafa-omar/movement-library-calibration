# Step A.10b: deep-model calibration curve with the exact A.4b-1 training code (ShallowConvNet, cell 4: others' ME -> new MI)
# strict and aligned (Euclidean alignment on the new person's first N MI trials, no labels); 5 seeds (0-4)
# seeds 0-1 must reproduce A.4b-1 (A4b1_results.csv); TOST (margin 5 points) against the TS+LDA full labelled calibration
import os, sys, json, time, copy, shutil, warnings, numpy as np, pandas as pd, mne, torch, torch.nn as nn
from scipy import stats
from sklearn.metrics import balanced_accuracy_score as bal
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
el = lambda: f'{(time.time() - t0) / 60:.1f} min'
DEV = 'cuda' if torch.cuda.is_available() else None
print(f'===== A.10b started {time.strftime("%H:%M:%S")} =====', flush=True)
if DEV is None:
    sys.exit('no GPU: switch the runtime to L4 or T4 and start again')
print('GPU:', torch.cuda.get_device_name(0), flush=True)

ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
A1, A2B, A4B = f'{ROOT}/01_preprocessed', f'{ROOT}/03_A2b_electrodes', f'{ROOT}/07_A4b_deep'
A8, A9, OUT = f'{ROOT}/10_A8_calibration', f'{ROOT}/11_A9_robustness', f'{ROOT}/12_A10_deep_calibration'
os.makedirs(OUT, exist_ok=True)
RES = f'{OUT}/A10b_results.csv'
FS, SUBJ = 128, list(range(1, 16))
EEG = json.load(open(f'{A1}/ME_sub01_meta.json'))['channels'][:61]
POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
PATHS = {int(k): v for k, v in json.load(open(f'{A2B}/A2b2_state.json'))['paths'].items()}
SEEDS, MAX_EP, PATIENCE, BATCH, LR = (0, 1, 2, 3, 4), 150, 20, 64, 1e-3     # A.4b-1 settings; seeds extended from 2 to 5
BUDGETS, DELTA = [10, 20, 30, 45, 60, 'all'], 5.0

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

def run(t, seed, mode):
    """Cell 4 exactly as in a4b1.prep(); the aligned test set is aligned on the first N trials of MI runs 1-5."""
    S = sorted(PATHS[t][:8])
    clean = lambda cond, s: np.where(~TS[cond][s]['r4m'])[0]
    Dt = TS['MI'][t]; ite = np.where(Dt['run'] >= 6)[0]; cal = np.where(Dt['run'] <= 5)[0]
    Xte0, yte = get_ts('MI', t, S, ite, True), Dt['y'][ite]
    others = [s for s in SUBJ if s != t]
    val = list(np.random.default_rng(100 * t + seed).choice(others, 2, replace=False))
    Xs, ys = [], []
    for s in others:
        if s in val:
            continue
        i = clean('ME', s); X = get_ts('ME', s, S, i, False)
        if mode == 'aligned':
            X = ea(X) @ X
        Xs.append(X); ys.append(TS['ME'][s]['y'][i])
    Xv, yv = [], []
    for s in val:
        D = TS['MI'][s]; i = np.where(D['run'] >= 6)[0]; X = get_ts('MI', s, S, i, False)
        if mode == 'aligned':
            X = ea(get_ts('MI', s, S, np.where(D['run'] <= 5)[0], False)) @ X
        Xv.append(X); yv.append(D['y'][i])
    Xtr, ytr, Xva, yva = np.concatenate(Xs), np.concatenate(ys), np.concatenate(Xv), np.concatenate(yv)
    sc = Xtr.std()
    model, ep = train(Xtr / sc, ytr, Xva / sc, yva, seed)
    if mode == 'strict':
        return {'strict': 100 * bal(yte, predict(model, Xte0 / sc))}, ep
    out = {}
    for N in BUDGETS:
        idx = cal if N == 'all' else cal[:N]
        Xte = ea(get_ts('MI', t, S, idx, True)) @ Xte0
        out[f'aligned {N}'] = 100 * bal(yte, predict(model, Xte / sc))
    return out, ep

R = pd.read_csv(RES) if os.path.exists(RES) else pd.DataFrame()
done = set(R['subject'].astype(int)) if len(R) else set()
for t in SUBJ:
    if t in done:
        print(f'person {t:2d} already done', flush=True); continue
    rows = []
    for seed in SEEDS:
        for mode in ('strict', 'aligned'):
            out, ep = run(t, seed, mode)
            rows += [{'subject': t, 'seed': seed, 'method': m, 'bal_acc': a, 'best_epoch': ep} for m, a in out.items()]
    R = pd.concat([R, pd.DataFrame(rows)], ignore_index=True)
    tmp = '/content/a10b_tmp.csv'; R.to_csv(tmp, index=False)
    shutil.copyfile(tmp, RES + '.part'); os.replace(RES + '.part', RES)
    g = pd.DataFrame(rows).groupby('method').bal_acc.mean()
    print(f'person {t:2d} done: strict {g["strict"]:.1f}, aligned all {g["aligned all"]:.1f} ({el()})', flush=True)
R['subject'] = R['subject'].astype(int); R['seed'] = R['seed'].astype(int)

# 1) reproduction check against A.4b-1 (seeds 0 and 1)
A4 = pd.read_csv(f'{A4B}/A4b1_results.csv')
A4 = A4[(A4['cell'] == '4 ME others') & (A4['model'] == 'ShallowConvNet')]
print('\n=== reproduction of A.4b-1 (ShallowConvNet, cell 4, seeds 0-1) ===')
for mode, m in (('strict', 'strict'), ('aligned', 'aligned all')):
    old = A4[A4['mode'] == mode].set_index(['subject', 'seed']).bal_acc
    new = R[(R.method == m) & (R.seed.isin([0, 1]))].set_index(['subject', 'seed']).bal_acc
    d = (new - old.reindex(new.index)).dropna()
    print(f'  {mode:8s}: mean of seeds 0-1 now {new.groupby(level=0).mean().mean():.1f} vs A.4b-1 {old.groupby(level=0).mean().mean():.1f}; '
          f'per subject-seed: identical in {int((d.abs() < 0.05).sum())}/{len(d)}, max abs difference {d.abs().max():.1f}')

# 2) 5-seed curve
def col(m):
    return R[R.method == m].groupby('subject').bal_acc.mean()
def pgt(a):
    a = np.asarray(a, float)
    return float(stats.wilcoxon(a - 50, alternative='greater').pvalue) if np.any(a != 50) else 1.0
A8R = pd.read_csv(f'{A8}/A8_results.csv')
A9R = pd.read_csv(f'{A9}/A9_results.csv')
def ts(budget):
    if budget == 'strict':
        g = A9R[(A9R.method == 'cell 4 others ME (strict)') & (A9R.montage == 'fold-specific 8') & (A9R.t_end == 3.0)]
    else:
        g = A8R[(A8R.method == 'others ME + recentering') & (A8R.budget.astype(str) == str(budget))]
    return g.set_index('subject').acc
own = A9R[(A9R.method == 'cell 1 own MI (labelled, ~27 min)') & (A9R.montage == 'fold-specific 8') & (A9R.t_end == 3.0)].set_index('subject').acc
print(f'\n=== A.10b: ShallowConvNet, others ME -> new person MI runs 6-10, fold-specific 8, mean of 15 people x {len(SEEDS)} seeds ===')
print(f'{"method":12s} {"mean":>6} {"seed SD":>8} {"median":>7} {">50":>4} {"p>50":>7} | {"TS+LDA":>6} {"deep-TS":>8} {"p":>6}')
for b in ['strict'] + BUDGETS:
    m = 'strict' if b == 'strict' else f'aligned {b}'
    a = col(m)
    sd = R[R.method == m].groupby('seed').bal_acc.mean().std()
    r = ts(b).reindex(a.index); d = (a - r).values
    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    print(f'{m:12s} {a.mean():6.1f} {sd:8.1f} {np.median(a):7.1f} {int((a > 50).sum()):4d} {pgt(a):7.4f} | {r.mean():6.1f} {d.mean():+8.1f} {p:6.3f}')
print('seed SD = SD across the 5 seeds of the group mean; TS+LDA = same budget (A.8 recentering; strict from A.9)')

def tost(d, delta=DELTA):
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    p1 = 1 - stats.t.cdf((m + delta) / se, n - 1); p2 = stats.t.cdf((m - delta) / se, n - 1)
    lo, hi = stats.t.interval(0.90, n - 1, loc=m, scale=se)
    return m, lo, hi, max(p1, p2)
print(f'\n=== equivalence with the TS+LDA full labelled calibration ({own.mean():.1f}%), margin +/-{DELTA:.0f} points (TOST, 5-seed means) ===')
for m in ['strict', 'aligned 10', 'aligned 30', 'aligned 60', 'aligned all']:
    a = col(m)
    mm, lo, hi, p = tost((a - own.reindex(a.index)).values)
    print(f'  ShallowConvNet {m:12s} - own calibration: {mm:+5.1f} [90% CI {lo:+5.1f}, {hi:+5.1f}], p TOST = {p:.3f} '
          f'-> {"EQUIVALENT" if p < 0.05 else "not shown"} (within {max(abs(lo), abs(hi)):.1f} points)')
print(f'\ntotal time {el()}')
print('===== A.10b FINISHED =====', flush=True)
