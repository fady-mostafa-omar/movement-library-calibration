# Step A.4b-1c: controls for the deep models (cells 2 and 4, strict)
import os, sys, json, shutil, time, copy, warnings, numpy as np, pandas as pd, mne, torch, torch.nn as nn
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score as bal
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
el = lambda: f'{(time.time() - t0) / 60:.1f} min'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\n===== A.4b-1c started {time.strftime("%H:%M:%S")} on {DEV} =====')

ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
A1 = f'{ROOT}/01_preprocessed'
if not os.path.exists(f'{A1}/MI_sub15.npz'):
    A1 = '/content/a1_preprocessed'
A2B, A4, OUT = f'{ROOT}/03_A2b_electrodes', f'{ROOT}/05_A4_four_cells', f'{ROOT}/07_A4b_deep'
CACHE, CACHE_PRE = f'{OUT}/cache_ts', f'{OUT}/cache_ts_precue'
os.makedirs(CACHE_PRE, exist_ok=True)
for f in (f'{A1}/MI_sub15.npz', f'{CACHE}/MI_sub15.npz', f'{OUT}/A4b1_results.csv', f'{A4}/A4a_results.csv'):
    if not os.path.exists(f):
        sys.exit(f'STOP: missing {f}')

FS, SUBJ, NT, NPRE = 128, list(range(1, 16)), 3 * 128, 2 * 128
EEG = json.load(open(f'{A1}/ME_sub01_meta.json'))['channels'][:61]
POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
PIDX = [EEG.index(c) for c in POOL]
T1MAP = {'hand_close': 1, 'hand_open': 1, 'rest': 0}
SOS = butter(4, (0.3, 40.0), btype='band', fs=FS, output='sos')
PATHS = {int(k): v for k, v in json.load(open(f'{A2B}/A2b2_state.json'))['paths'].items()}
FIXED = {'central5': sorted(POOL.index(c) for c in ['Cz', 'CP3', 'C1', 'C2', 'C6']),
         'parietal3': sorted(POOL.index(c) for c in ['P3', 'Pz', 'P4'])}
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
    tmp = '/content/_a4b1c_tmp.npz'
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

# 1) post-cue cache (from A.4b-1) and a new pre-cue cache (cue - 2 s to cue)
TS, PRE, BADS = {'ME': {}, 'MI': {}}, {'ME': {}, 'MI': {}}, {}
for cond in ('ME', 'MI'):
    for s in SUBJ:
        BADS[(cond, s)] = [c for c in json.load(open(f'{A1}/{cond}_sub{s:02d}_meta.json'))['bad_channels'] if c in EEG]
        TS[cond][s] = load_npz(f'{CACHE}/{cond}_sub{s:02d}.npz', ('X31', 'y', 'run', 'r4m'))
        if TS[cond][s] is None:
            sys.exit(f'STOP: A.4b-1 cache for {cond} {s} is missing or damaged')
        fn = f'{CACHE_PRE}/{cond}_sub{s:02d}.npz'
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
                ok = (cues - NPRE >= 0) & (cues <= Y.shape[1])
                keep[rows[ok]] = True
                for c0 in cues[ok]:
                    Xi.append(Yi[PIDX, c0 - NPRE:c0])
                    if keep_raw:
                        Xr.append(Y[PIDX, c0 - NPRE:c0])
            E = E[keep].reset_index(drop=True)
            D = {'X31': np.array(Xi, dtype=np.float32), 'y': np.asarray(E['class'].map(T1MAP), dtype=np.int64),
                 'run': np.asarray(E.run, dtype=np.int64), 'r4m': np.asarray(E.r4m_rejected, dtype=bool)}
            if keep_raw:
                D['X31raw'] = np.array(Xr, dtype=np.float32)
            save_npz(fn, D)
            print(f'pre-cue {cond} subject {s:2d}: {len(E)} trials ({el()})')
        PRE[cond][s] = D
print(f'data ready ({el()})')

QB = {}
def get_ts(src, cond, s, S, idx, within):
    D = src[cond][s]
    raw = within and 'X31raw' in D and any(POOL[i] in BADS[(cond, s)] for i in S)
    X = D['X31raw' if raw else 'X31'][idx][:, S, :].astype(np.float64)
    if raw:
        X = interp_matrix([POOL[i] for i in S], BADS[(cond, s)]) @ X
    k = len(S)
    if k not in QB:
        QB[k] = np.linalg.qr(np.c_[np.ones(k), np.eye(k)[:, :k - 1]])[0][:, 1:]
    return QB[k].T @ X

class EEGNet(nn.Module):
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
    best, state, wait = -1.0, None, 0
    for ep in range(MAX_EP):
        model.train(); perm = torch.randperm(len(yt), device=DEV)
        for i in range(0, len(yt), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 2:
                continue
            opt.zero_grad(); lossf(model(Xt[b]), yt[b]).backward(); opt.step()
        acc = bal(yva, predict(model, Xva))
        if acc > best:
            best, state, wait = acc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(state)
    return model

def prep(cell, t, seed, S, src, tsl, shuffle=False):
    """Cells 2/4, strict. Same validation subjects as A.4b-1. The new person never trains or early-stops."""
    cond = 'MI' if cell == 2 else 'ME'
    others = [s for s in SUBJ if s != t]
    val = list(np.random.default_rng(100 * t + seed).choice(others, 2, replace=False))
    Xs, ys = [], []
    for s in others:
        if s in val:
            continue
        i = np.where(~src[cond][s]['r4m'])[0]
        Xs.append(get_ts(src, cond, s, S, i, False)[:, :, tsl]); ys.append(src[cond][s]['y'][i])
    Xv, yv = [], []
    for s in val:
        i = np.where(src['MI'][s]['run'] >= 6)[0]
        Xv.append(get_ts(src, 'MI', s, S, i, False)[:, :, tsl]); yv.append(src['MI'][s]['y'][i])
    i = np.where(src['MI'][t]['run'] >= 6)[0]
    Xte, yte = get_ts(src, 'MI', t, S, i, True)[:, :, tsl], src['MI'][t]['y'][i]
    Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
    if shuffle:
        ytr = np.random.default_rng(1000 * t + seed).permutation(ytr)
    sc = Xtr.std()
    return Xtr / sc, ytr, np.concatenate(Xv) / sc, np.concatenate(yv), Xte / sc, yte

CTRL = [('without first 0.5 s', 'fold8', 'post', slice(64, None), False),
        ('pre-cue (-2 to 0 s)', 'fold8', 'pre', slice(None), False),
        ('central5 only', 'central5', 'post', slice(None), False),
        ('parietal3 only', 'parietal3', 'post', slice(None), False),
        ('shuffled training labels', 'fold8', 'post', slice(None), True)]
CNAME = {2: '2 MI others', 4: '4 ME others'}
RES = f'{OUT}/A4b1c_results.csv'
R = pd.read_csv(RES) if os.path.exists(RES) else pd.DataFrame()
done = set(R['subject'].astype(int)) if len(R) else set()
for t in SUBJ:
    if t in done:
        continue
    rows = []
    for name, mont, srcname, tsl, shuf in CTRL:
        S = sorted(PATHS[t][:8]) if mont == 'fold8' else FIXED[mont]
        src = TS if srcname == 'post' else PRE
        for cell in (2, 4):
            for seed in SEEDS:
                Xtr, ytr, Xva, yva, Xte, yte = prep(cell, t, seed, S, src, tsl, shuf)
                for arch in ARCH:
                    acc = 100 * bal(yte, predict(train(arch, Xtr, ytr, Xva, yva, seed), Xte))
                    rows.append({'subject': t, 'control': name, 'cell': CNAME[cell], 'model': arch,
                                 'seed': seed, 'bal_acc': acc})
    R = pd.concat([R, pd.DataFrame(rows)], ignore_index=True)
    R.to_csv(RES, index=False)
    print(f'person {t:2d} done ({el()})')
R['subject'] = R['subject'].astype(int); R['bal_acc'] = R['bal_acc'].astype(float)

# 2) summary against the pre-specified rules
B = pd.read_csv(f'{OUT}/A4b1_results.csv'); B = B[B['mode'] == 'strict']
A4R = pd.read_csv(f'{A4}/A4a_results.csv')
LDA_CTRL = {'without first 0.5 s': 'control: fold8 without first 0.5 s', 'pre-cue (-2 to 0 s)': 'control: fold8 pre-cue (-2 to 0 s)',
            'central5 only': 'control: central5', 'parietal3 only': 'control: parietal3'}

def pw(d, alt):
    d = np.asarray(d)
    return 1.0 if not np.any(d != 0) else wilcoxon(d, alternative=alt).pvalue

print('\n=== A.4b-1c: controls for the deep models (strict, T1, new person MI runs 6-10, %, mean of 2 seeds) ===')
out = []
for cell in CNAME.values():
    for arch in ARCH:
        full = B[(B.cell == cell) & (B.model == arch)].groupby('subject').bal_acc.mean().loc[SUBJ].values
        out.append({'cell': cell, 'model': arch, 'control': 'full 0-3 s (A.4b-1)', 'mean': round(full.mean(), 1),
                    'n_above_50': int((full > 50).sum()), 'p_vs_50': round(pw(full - 50, 'greater'), 4),
                    'change_vs_full': '-', 'p_change': '-', 'TS+LDA_same_control': '-'})
        for name, *_ in CTRL:
            a = R[(R.cell == cell) & (R.model == arch) & (R.control == name)].groupby('subject').bal_acc.mean().loc[SUBJ].values
            lda = '-'
            if name in LDA_CTRL:
                lda = round(A4R[(A4R.analysis == LDA_CTRL[name]) & (A4R.cell == cell)].bal_acc.mean(), 1)
            out.append({'cell': cell, 'model': arch, 'control': name, 'mean': round(a.mean(), 1),
                        'n_above_50': int((a > 50).sum()), 'p_vs_50': round(pw(a - 50, 'greater'), 4),
                        'change_vs_full': round((a - full).mean(), 1), 'p_change': round(pw(a - full, 'two-sided'), 4),
                        'TS+LDA_same_control': lda})
T = pd.DataFrame(out)
print(T.to_string(index=False))

print('\n=== pre-specified rules (gain accepted as motor-related only if all three hold) ===')
for cell in CNAME.values():
    for arch in ARCH:
        g = T[(T.cell == cell) & (T.model == arch)].set_index('control')
        r1 = float(g.loc['without first 0.5 s', 'change_vs_full']) >= -2.0
        r2 = (g.loc['pre-cue (-2 to 0 s)', 'p_vs_50'] >= 0.05) and (g.loc['shuffled training labels', 'p_vs_50'] >= 0.05)
        r3 = g.loc['central5 only', 'p_vs_50'] < 0.05
        print(f'{cell:12s} {arch:15s} | first 0.5 s not needed: {r1} | pre-cue & shuffle at chance: {r2} | '
              f'central5 above chance: {r3} -> {"ACCEPTED" if (r1 and r2 and r3) else "NOT accepted"}')
print(f'\ntotal time {el()}')
print('===== A.4b-1c FINISHED =====')
