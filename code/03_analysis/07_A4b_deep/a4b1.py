# Step A.4b-1: EEGNet and ShallowConvNet on the four cells (strict and aligned), compared with TS+LDA
import os, sys, json, shutil, time, copy, warnings, numpy as np, pandas as pd, mne, torch, torch.nn as nn
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score as bal
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
el = lambda: f'{(time.time() - t0) / 60:.1f} min'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\n===== A.4b-1 started {time.strftime("%H:%M:%S")} on {DEV} =====')

ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
A1 = f'{ROOT}/01_preprocessed'
if not os.path.exists(f'{A1}/MI_sub15.npz'):
    A1 = '/content/a1_preprocessed'
A2B, A4, A6, OUT = (f'{ROOT}/03_A2b_electrodes', f'{ROOT}/05_A4_four_cells', f'{ROOT}/06_A6_adaptation',
                    f'{ROOT}/07_A4b_deep')
CACHE = f'{OUT}/cache_ts'
os.makedirs(CACHE, exist_ok=True)
for f in (f'{A1}/MI_sub15.npz', f'{A2B}/A2b2_state.json', f'{A4}/A4a_results.csv', f'{A6}/A6_results.csv'):
    if not os.path.exists(f):
        sys.exit(f'STOP: missing {f}')

FS, SUBJ, NT = 128, list(range(1, 16)), 3 * 128           # 3-s epochs after the cue (384 samples)
EEG = json.load(open(f'{A1}/ME_sub01_meta.json'))['channels'][:61]
POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
PIDX = [EEG.index(c) for c in POOL]
T1MAP = {'hand_close': 1, 'hand_open': 1, 'rest': 0}
SOS = butter(4, (0.3, 40.0), btype='band', fs=FS, output='sos')   # causal broadband for deep models
PATHS = {int(k): v for k, v in json.load(open(f'{A2B}/A2b2_state.json'))['paths'].items()}
SEEDS, MAX_EP, PATIENCE, BATCH, LR = (0, 1), 150, 20, 64, 1e-3

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

def save_npz(fn, D):
    tmp = '/content/_a4b_tmp.npz'
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

# 1) time-series cache: 31 pool sites, causal 0.3-40 Hz, cue to cue + 3 s, T1 trials
TS, BADS = {'ME': {}, 'MI': {}}, {}
for cond in ('ME', 'MI'):
    for s in SUBJ:
        BADS[(cond, s)] = [c for c in json.load(open(f'{A1}/{cond}_sub{s:02d}_meta.json'))['bad_channels'] if c in EEG]
        fn = f'{CACHE}/{cond}_sub{s:02d}.npz'
        D = load_npz(fn, ('X31', 'y', 'run', 'r4m'))
        if D is None:
            Z = np.load(f'{A1}/{cond}_sub{s:02d}.npz')
            E = pd.read_csv(f'{A1}/{cond}_sub{s:02d}_events.csv')
            E = E[E['class'].isin(T1MAP)].sort_values(['run', 'trial']).reset_index(drop=True)
            bads = BADS[(cond, s)]; W61 = interp_matrix(EEG, bads)
            keep_raw = cond == 'MI' and any(c in POOL for c in bads)
            Xi, Xr, keep = [], [], np.zeros(len(E), bool)
            for r in range(1, 11):
                X = Z[f'run{r}'][:61].astype(np.float64)
                Y = sosfilt(SOS, X, axis=1, zi=sosfilt_zi(SOS)[:, None, :] * X[:, :1][None])[0]
                Yi = W61 @ Y
                rows = np.where(E.run.values == r)[0]
                cues = E.cue_sample_128.values[rows]
                ok = (cues >= 0) & (cues + NT <= Y.shape[1])
                keep[rows[ok]] = True
                for c0 in cues[ok]:
                    Xi.append(Yi[PIDX, c0:c0 + NT])
                    if keep_raw:
                        Xr.append(Y[PIDX, c0:c0 + NT])
            E = E[keep].reset_index(drop=True)
            D = {'X31': np.array(Xi, dtype=np.float32), 'y': np.asarray(E['class'].map(T1MAP), dtype=np.int64),
                 'run': np.asarray(E.run, dtype=np.int64), 'r4m': np.asarray(E.r4m_rejected, dtype=bool)}
            if keep_raw:
                D['X31raw'] = np.array(Xr, dtype=np.float32)
            save_npz(fn, D)
            print(f'time series {cond} subject {s:2d}: {len(E)} trials ({el()})')
        TS[cond][s] = D
print(f'data ready ({el()})')

QB = {}
def get_ts(cond, s, S, idx, within):
    """Montage S, CAR by projection (k-1 virtual channels); the new person's bad channels interpolated within S."""
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
    """Euclidean alignment matrix (He & Wu 2020) from unlabelled trials X (n, c, T)."""
    R = np.mean(X @ np.swapaxes(X, 1, 2), axis=0) / X.shape[2]
    w, V = np.linalg.eigh(R + 1e-6 * np.trace(R) / len(R) * np.eye(len(R)))
    return V @ np.diag(w ** -0.5) @ V.T

class EEGNet(nn.Module):                                   # Lawhern et al. 2018 (EEGNet-8,2), dropout 0.25
    def __init__(self, C, T, n_out=2, F1=8, D=2, F2=16, k1=64, p=0.25):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(F1),
                                nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D),
                                nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p))
        self.b2 = nn.Sequential(nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                                nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
                                nn.AvgPool2d((1, 8)), nn.Dropout(p))
        with torch.no_grad():
            n = self.b2(self.b1(torch.zeros(1, 1, C, T))).numel()
        self.fc = nn.Linear(n, n_out)
    def forward(self, x):
        return self.fc(self.b2(self.b1(x.unsqueeze(1))).flatten(1))

class Shallow(nn.Module):                                  # Schirrmeister et al. 2017, kernels scaled to 128 Hz
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

ARCH = {'EEGNet': EEGNet, 'ShallowConvNet': Shallow}

def predict(model, X):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(model(torch.tensor(X[i:i + 256], dtype=torch.float32, device=DEV)).argmax(1).cpu().numpy())
    return np.concatenate(out)

def train(arch, Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ARCH[arch](Xtr.shape[1], Xtr.shape[2]).to(DEV)
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

def prep(cell, t, mode, seed):
    """Training / validation / test sets. The new person (t) never takes part in training or early stopping."""
    S = sorted(PATHS[t][:8])
    def clean(cond, s, extra=True):
        D = TS[cond][s]; return np.where(~D['r4m'] & extra)[0]
    Dt = TS['MI'][t]; ite = np.where(Dt['run'] >= 6)[0]
    Xte, yte = get_ts('MI', t, S, ite, True), Dt['y'][ite]
    if mode == 'aligned':
        Xte = ea(get_ts('MI', t, S, np.where(Dt['run'] <= 5)[0], True)) @ Xte      # unlabelled warm-up
    if cell == 1:                                    # own MI: runs 1-4 train, run 5 early stopping
        itr, iva = clean('MI', t, Dt['run'] <= 4), clean('MI', t, Dt['run'] == 5)
        Xtr, ytr = get_ts('MI', t, S, itr, True), Dt['y'][itr]
        Xva, yva = get_ts('MI', t, S, iva, True), Dt['y'][iva]
    elif cell == 3:                                  # own ME: runs 1-8 train, runs 9-10 early stopping
        Dm = TS['ME'][t]
        itr, iva = clean('ME', t, Dm['run'] <= 8), clean('ME', t, Dm['run'] >= 9)
        Xtr, ytr = get_ts('ME', t, S, itr, False), Dm['y'][itr]
        Xva, yva = get_ts('ME', t, S, iva, False), Dm['y'][iva]
        if mode == 'aligned':
            M = ea(get_ts('ME', t, S, clean('ME', t), False)); Xtr, Xva = M @ Xtr, M @ Xva
    else:                                            # others: 12 train, 2 for early stopping (their MI runs 6-10)
        cond = 'MI' if cell == 2 else 'ME'
        others = [s for s in SUBJ if s != t]
        val = list(np.random.default_rng(100 * t + seed).choice(others, 2, replace=False))
        Xs, ys = [], []
        for s in others:
            if s in val:
                continue
            i = clean(cond, s); X = get_ts(cond, s, S, i, False)
            if mode == 'aligned':
                X = ea(X) @ X
            Xs.append(X); ys.append(TS[cond][s]['y'][i])
        Xv, yv = [], []
        for s in val:
            D = TS['MI'][s]; i = np.where(D['run'] >= 6)[0]; X = get_ts('MI', s, S, i, False)
            if mode == 'aligned':
                X = ea(get_ts('MI', s, S, np.where(D['run'] <= 5)[0], False)) @ X
            Xv.append(X); yv.append(D['y'][i])
        Xtr, ytr, Xva, yva = np.concatenate(Xs), np.concatenate(ys), np.concatenate(Xv), np.concatenate(yv)
    sc = Xtr.std()                                   # one global scale from the training data only
    return Xtr / sc, ytr, Xva / sc, yva, Xte / sc, yte

RUNS = [(c, 'strict') for c in (1, 2, 3, 4)] + [(c, 'aligned') for c in (2, 3, 4)]
CNAME = {1: '1 MI self', 2: '2 MI others', 3: '3 ME self', 4: '4 ME others'}
RES = f'{OUT}/A4b1_results.csv'
R = pd.read_csv(RES) if os.path.exists(RES) else pd.DataFrame()
done = set(R['subject'].astype(int)) if len(R) else set()
for t in SUBJ:
    if t in done:
        continue
    rows = []
    for cell, mode in RUNS:
        for seed in SEEDS:
            Xtr, ytr, Xva, yva, Xte, yte = prep(cell, t, mode, seed)
            for arch in ARCH:
                model, ep = train(arch, Xtr, ytr, Xva, yva, seed)
                rows.append({'subject': t, 'cell': CNAME[cell], 'mode': mode, 'model': arch, 'seed': seed,
                             'bal_acc': 100 * bal(yte, predict(model, Xte)), 'best_epoch': ep})
    R = pd.concat([R, pd.DataFrame(rows)], ignore_index=True)
    R.to_csv(RES, index=False)
    print(f'person {t:2d} done ({el()})')
R['subject'] = R['subject'].astype(int); R['bal_acc'] = R['bal_acc'].astype(float)

# 2) summary and comparison with TS+LDA (A.4a strict; A.6 recentered with the same 90-trial warm-up)
A4R = pd.read_csv(f'{A4}/A4a_results.csv'); A4R = A4R[A4R.analysis == 'T1 fold8 (primary)']
A6R = pd.read_csv(f'{A6}/A6_results.csv'); A6R = A6R[A6R.warmup_trials == 90]
def ref_vals(cell, mode):
    if mode == 'strict':
        return A4R[A4R.cell == cell].sort_values('subject').bal_acc.values
    return A6R[A6R.cell == cell].sort_values('subject').bal_acc.values

def pw(d, alt):
    d = np.asarray(d)
    return 1.0 if not np.any(d != 0) else wilcoxon(d, alternative=alt).pvalue

print('\n=== A.4b-1: deep models vs TS+LDA (T1, fold-specific 8 sites, new person MI runs 6-10, %, mean of 2 seeds) ===')
out = []
for (cell, mode), g in R.groupby(['cell', 'mode'], sort=False):
    ref = ref_vals(cell, mode)
    out.append({'cell': cell, 'mode': mode, 'model': 'TS+LDA (ours)', 'mean': round(ref.mean(), 1),
                'median': round(np.median(ref), 1), 'n_above_50': int((ref > 50).sum()), 'p_vs_50': round(pw(ref - 50, 'greater'), 4),
                'diff_vs_TS+LDA': '-', 'p_vs_TS+LDA': '-', 'median_best_epoch': '-'})
    for arch, h in g.groupby('model', sort=False):
        a = h.groupby('subject').bal_acc.mean().loc[SUBJ].values
        out.append({'cell': cell, 'mode': mode, 'model': arch, 'mean': round(a.mean(), 1), 'median': round(np.median(a), 1),
                    'n_above_50': int((a > 50).sum()), 'p_vs_50': round(pw(a - 50, 'greater'), 4),
                    'diff_vs_TS+LDA': round((a - ref).mean(), 1), 'p_vs_TS+LDA': round(pw(a - ref, 'two-sided'), 4),
                    'median_best_epoch': int(h.best_epoch.median())})
print(pd.DataFrame(out).to_string(index=False))
print('aligned = Euclidean alignment with the unlabelled MI runs 1-5 (same warm-up as A.6, 90 trials);'
      ' TS+LDA aligned = A.6 recentering with 90 trials')
print(f'\ntotal time {el()}')
print('===== A.4b-1 FINISHED =====')
