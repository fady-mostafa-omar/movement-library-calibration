# Step A.7-1: PhysioNet EEGMMIDB - controlled re-evaluation of zero-calibration ME->MI, left vs right fist
import os, sys, json, shutil, time, copy, warnings, numpy as np, pandas as pd, mne, torch, torch.nn as nn
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score as bal
from mne.datasets import eegbci
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
el = lambda: f'{(time.time() - t0) / 60:.1f} min'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\n===== A.7-1 started {time.strftime("%H:%M:%S")} on {DEV} =====')

ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
OUT = f'{ROOT}/08_A7_physionet'; CACHE = f'{OUT}/cache'; RAWDIR = '/content/mne_data'
os.makedirs(CACHE, exist_ok=True); os.makedirs(RAWDIR, exist_ok=True)

FS, NT = 160, 640                                        # PhysioNet: 160 Hz; 4-s epochs from the cue
RUNS = {3: 'ME', 7: 'ME', 11: 'ME', 4: 'MI', 8: 'MI', 12: 'MI'}   # left/right fist runs
KIND = {'T0': 'rest', 'T1': 'left', 'T2': 'right'}
BANDS = {'bb': (0.3, 40.0), 'low': (0.3, 3.0), 'mu_beta': (8.0, 30.0)}
SOS = {b: butter(4, f, btype='band', fs=FS, output='sos') for b, f in BANDS.items()}
WIN, STEP = FS, 16                                       # 1-s windows every 0.1 s
WSTARTS = np.arange(0, NT - WIN + 1, STEP)               # starts 0.0 ... 3.0 s (31 windows)
WSETS = {'full': np.arange(len(WSTARTS)), 'post': np.where(WSTARTS >= 80)[0]}   # post = starts >= 0.5 s

EYE25 = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
         'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8']
MOTOR21 = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
           'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
CONS8 = ['Cz', 'P4', 'CP3', 'C1', 'P3', 'C2', 'Pz', 'C6']
POOL31 = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
          'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
NEARC3_8 = ['FC5', 'FC3', 'FC1', 'C5', 'C3', 'C1', 'CP5', 'CP3']
FEATS = {'full': ['all64', 'noeye39', 'eye25', 'motor21', 'pool31', 'cons8', 'nearC3_8'],
         'post': ['all64', 'noeye39', 'rand25_0', 'rand25_1', 'rand25_2']}

def make_montages(ch):
    noeye = [c for c in ch if c not in EYE25]
    M = {'all64': list(ch), 'noeye39': noeye, 'eye25': EYE25, 'motor21': MOTOR21, 'pool31': POOL31,
         'cons8': CONS8, 'nearC3_8': NEARC3_8}
    for k in range(3):                                   # 25 random sites away from the eyes (fixed seeds)
        drop = set(np.random.default_rng(k).choice(noeye, 25, replace=False))
        M[f'rand25_{k}'] = [c for c in ch if c not in drop]
    for m, v in M.items():
        assert all(c in ch for c in v), f'montage {m}: missing channel'
    return M

QB = {}
def qproj(k):
    if k not in QB:
        QB[k] = np.linalg.qr(np.c_[np.ones(k), np.eye(k)[:, :k - 1]])[0][:, 1:]
    return QB[k]

def logvec(C):
    m = C.shape[-1]
    tr = np.trace(C, axis1=-2, axis2=-1)[..., None, None] / m
    w, V = np.linalg.eigh(C + 1e-5 * tr * np.eye(m))
    L = (V * np.log(np.maximum(w, 1e-12))[..., None, :]) @ np.swapaxes(V, -1, -2)
    iu = np.triu_indices(m)
    return L[..., iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2))

def save_npz(fn, D):
    tmp = '/content/_a7_tmp.npz'
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

# 1) download + cache (per subject): 0.3-40 Hz epochs for the deep model, tangent features for TS+LDA
CHF, EXCL = f'{OUT}/channels.json', f'{OUT}/excluded.json'
CH = json.load(open(CHF)) if os.path.exists(CHF) else None
excluded = json.load(open(EXCL)) if os.path.exists(EXCL) else {}
MONT = make_montages(CH) if CH else None
DATA = {}
for s in range(1, 110):
    if str(s) in excluded:
        continue
    fn = f'{CACHE}/S{s:03d}.npz'
    D = load_npz(fn, ('bb', 'kind', 'cond', 'f_all64_full'))
    if D is None:
        try:
            files = eegbci.load_data(s, list(RUNS), path=RAWDIR, update_path=False, verbose=False)
            ep = {b: [] for b in BANDS}; kind, cond, run = [], [], []
            for f in sorted(files):
                r = int(os.path.basename(f)[5:7])
                raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
                eegbci.standardize(raw)
                if raw.info['sfreq'] != FS:
                    raise ValueError(f"sampling rate {raw.info['sfreq']}")
                if CH is None:
                    CH = list(raw.ch_names); json.dump(CH, open(CHF, 'w')); MONT = make_montages(CH)
                if set(raw.ch_names) != set(CH):
                    raise ValueError('channel set differs')
                X = raw.get_data()[[raw.ch_names.index(c) for c in CH]] * 1e6
                Y = {b: sosfilt(SOS[b], X, axis=1, zi=sosfilt_zi(SOS[b])[:, None, :] * X[:, :1][None])[0] for b in BANDS}
                for a in raw.annotations:
                    if a['description'] not in KIND:
                        continue
                    st = int(round(a['onset'] * FS))
                    if st < 2 * FS or st + NT > X.shape[1]:          # skip filter start-up and truncated epochs
                        continue
                    for b in BANDS:
                        ep[b].append(Y[b][:, st:st + NT])
                    kind.append(KIND[a['description']]); cond.append(RUNS[r]); run.append(r)
            D = {'bb': np.array(ep['bb'], dtype=np.float16), 'kind': np.array(kind, dtype='U5'),
                 'cond': np.array(cond, dtype='U2'), 'run': np.array(run, dtype=np.int64)}
            for b in ('low', 'mu_beta'):
                E = np.array(ep[b], dtype=np.float32)
                C = np.stack([np.einsum('nct,ndt->ncd', E[:, :, w:w + WIN], E[:, :, w:w + WIN]) / WIN for w in WSTARTS], 1)
                needed = set(FEATS['full']) | set(FEATS['post'])
                for m in needed:
                    idx = [CH.index(c) for c in MONT[m]]; Q = qproj(len(idx))
                    Lw = logvec(Q.T @ C[..., idx, :][..., idx].astype(np.float64) @ Q)   # (n, windows, d)
                    for wset, ms in FEATS.items():
                        if m in ms:
                            D[f'f_{m}_{wset}_{b}'] = Lw[:, WSETS[wset]].mean(1).astype(np.float32)
            for wset, ms in FEATS.items():
                for m in ms:
                    D[f'f_{m}_{wset}'] = np.hstack([D.pop(f'f_{m}_{wset}_low'), D.pop(f'f_{m}_{wset}_mu_beta')])
            save_npz(fn, D)
            print(f'S{s:03d}: {len(kind)} epochs ({el()})')
        except Exception as e:
            excluded[str(s)] = str(e)[:120]; json.dump(excluded, open(EXCL, 'w'))
            print(f'S{s:03d} excluded: {excluded[str(s)]}')
            continue
    DATA[s] = D
SUBJ = sorted(DATA)
print(f'subjects ready: {len(SUBJ)}; excluded: {excluded} ({el()})')

# 2) zero-calibration ME->MI, left (0) vs right (1); subject-wise 10-fold CV
perm = np.random.default_rng(0).permutation(SUBJ)
FOLDS = [list(map(int, f)) for f in np.array_split(perm, 10)]

def lr_mask(D, c):
    return (D['cond'] == c) & np.isin(D['kind'], ['left', 'right'])

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

def lda_predict(X, mdl):
    Wm, b, cl = mdl
    return cl[np.argmax(X @ Wm + b, axis=1)]

CONDS = [('all64', 'full'), ('all64', 'post'), ('noeye39', 'full'), ('noeye39', 'post'),
         ('rand25_0', 'post'), ('rand25_1', 'post'), ('rand25_2', 'post'), ('eye25', 'full')]
TS_CONDS = CONDS + [('motor21', 'full')]
RES = f'{OUT}/A7_1_results.csv'
R = pd.read_csv(RES) if os.path.exists(RES) else pd.DataFrame(columns=['model', 'condition', 'fold', 'subject', 'bal_acc'])
done = set(zip(R['model'], R['condition'], R['fold'].astype(int))) if len(R) else set()

def add_rows(model, cname, k, accs):
    global R
    R = pd.concat([R, pd.DataFrame([{'model': model, 'condition': cname, 'fold': k, 'subject': s, 'bal_acc': a}
                                    for s, a in accs.items()])], ignore_index=True)
    R.to_csv(RES, index=False)

for m, w in TS_CONDS:                                            # TS+LDA (secondary model)
    cname = f'{m} {w}'
    for k, test in enumerate(FOLDS):
        if ('TS+LDA', cname, k) in done:
            continue
        train = [s for s in SUBJ if s not in test]
        X = np.vstack([DATA[s][f'f_{m}_{w}'][lr_mask(DATA[s], 'ME')] for s in train])
        y = np.concatenate([(DATA[s]['kind'][lr_mask(DATA[s], 'ME')] == 'right').astype(int) for s in train])
        mdl = slda_fit(X.astype(np.float64), y)
        accs = {}
        for s in test:
            mk = lr_mask(DATA[s], 'MI')
            accs[s] = 100 * bal((DATA[s]['kind'][mk] == 'right').astype(int),
                                lda_predict(DATA[s][f'f_{m}_{w}'][mk].astype(np.float64), mdl))
        add_rows('TS+LDA', cname, k, accs)
    print(f'TS+LDA {cname} done ({el()})')

class Shallow(nn.Module):                                        # ShallowConvNet, kernels scaled to 160 Hz
    def __init__(self, C, T, n_out=2, F=40, k=16, pool=48, stride=10, p=0.5):
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

def deep_predict(model, X):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(model(torch.tensor(X[i:i + 256], dtype=torch.float32, device=DEV)).argmax(1).cpu().numpy())
    return np.concatenate(out)

def deep_train(Xtr, ytr, Xva, yva, seed=0, max_ep=100, patience=15):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Shallow(Xtr.shape[1], Xtr.shape[2]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    w = torch.tensor(len(ytr) / (2.0 * np.bincount(ytr, minlength=2)), dtype=torch.float32, device=DEV)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV); yt = torch.tensor(ytr, device=DEV)
    best, state, wait = -1.0, None, 0
    for ep in range(max_ep):
        model.train(); p = torch.randperm(len(yt), device=DEV)
        for i in range(0, len(yt), 64):
            b = p[i:i + 64]
            if len(b) < 2:
                continue
            opt.zero_grad(); lossf(model(Xt[b]), yt[b]).backward(); opt.step()
        acc = bal(yva, deep_predict(model, Xva))
        if acc > best:
            best, state, wait = acc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    del Xt
    return model

def deep_xy(subjects, c, m, w):
    idx = [CH.index(ch) for ch in MONT[m]]; Q = qproj(len(idx)); tsl = slice(80, None) if w == 'post' else slice(None)
    Xs, ys = [], []
    for s in subjects:
        mk = lr_mask(DATA[s], c)
        X = DATA[s]['bb'][mk][:, idx, :].astype(np.float32)
        Xs.append((Q.T.astype(np.float32) @ X)[:, :, tsl]); ys.append((DATA[s]['kind'][mk] == 'right').astype(int))
    return Xs, ys

for m, w in CONDS:                                               # ShallowConvNet (primary model)
    cname = f'{m} {w}'
    for k, test in enumerate(FOLDS):
        if ('ShallowConvNet', cname, k) in done:
            continue
        train = [s for s in SUBJ if s not in test]
        val = list(map(int, np.random.default_rng(k).choice(train, 8, replace=False)))
        Xs, ys = deep_xy([s for s in train if s not in val], 'ME', m, w)
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
        Xv, yv = deep_xy(val, 'MI', m, w)
        sc = Xtr.std()
        model = deep_train(Xtr / sc, ytr, np.concatenate(Xv) / sc, np.concatenate(yv))
        Xt, yt = deep_xy(test, 'MI', m, w)
        add_rows('ShallowConvNet', cname, k, {s: 100 * bal(y, deep_predict(model, X / sc)) for s, X, y in zip(test, Xt, yt)})
        del Xtr, Xs
        torch.cuda.empty_cache()
    print(f'ShallowConvNet {cname} done ({el()})')

# 3) summary and the pre-specified verdict
R['bal_acc'] = R['bal_acc'].astype(float); R['subject'] = R['subject'].astype(int)
R = R.drop_duplicates(['model', 'condition', 'subject'], keep='last')
def pw(d, alt):
    d = np.asarray(d)
    return 1.0 if not np.any(d != 0) else wilcoxon(d, alternative=alt).pvalue
print(f'\n=== A.7-1: PhysioNet, zero-calibration ME->MI, left vs right fist, {len(SUBJ)} subjects, balanced accuracy % ===')
out = []
for model in ('ShallowConvNet', 'TS+LDA'):
    for cname in [f'{m} {w}' for m, w in TS_CONDS]:
        a = R[(R.model == model) & (R.condition == cname)].bal_acc.values
        if len(a):
            out.append({'model': model, 'condition': cname, 'n': len(a), 'mean': round(a.mean(), 1),
                        'median': round(np.median(a), 1), 'n_above_50': int((a > 50).sum()), 'p_vs_50': round(pw(a - 50, 'greater'), 4)})
print(pd.DataFrame(out).to_string(index=False))
print('Perez-Velasco et al. 2025 reported 86.2% for this scenario (their model)')

print('\n=== pre-specified verdict (large effect: drop_eye >= 10 AND drop_eye - drop_random >= 5) ===')
for model in ('ShallowConvNet', 'TS+LDA'):
    P = R[R.model == model].pivot(index='subject', columns='condition', values='bal_acc').loc[SUBJ]
    full, eye = P['all64 full'], P['noeye39 post']
    rnd = P[['rand25_0 post', 'rand25_1 post', 'rand25_2 post']].mean(1)
    de, dr = (full - eye).mean(), (full - rnd).mean()
    big = de >= 10 and (de - dr) >= 5
    print(f'{model:15s} full {full.mean():.1f} | no eye-sites + no first 0.5 s {eye.mean():.1f} (drop {de:.1f}) | '
          f'random 25 removed + no first 0.5 s {rnd.mean():.1f} (drop {dr:.1f}) | difference {de - dr:.1f}, '
          f'p = {pw(rnd - eye, "two-sided"):.4f} -> {"LARGE eye/artifact effect" if big else "no large effect"}'
          + ('  (PRIMARY)' if model == 'ShallowConvNet' else ''))
print(f'\ntotal time {el()}')
print('===== A.7-1 FINISHED =====')
