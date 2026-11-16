# Step A.10: deep-model calibration curve on Ofner T1 (ShallowConvNet, fold-specific 8 sites), approved 10 Nov 2026
# strict zero calibration and Euclidean alignment with warm-up budgets of the new person's first N MI trials (no labels);
# TOST (margin 5 points) against the TS+LDA full labelled calibration (cell 1 from A.9)
import os, re, sys, json, glob, time, shutil, collections, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, mne
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from mne.channels.interpolation import _make_interpolation_matrix
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
t0 = time.time()
ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
A1, A2B, A4B = f'{ROOT}/01_preprocessed', f'{ROOT}/03_A2b_electrodes', f'{ROOT}/07_A4b_deep'
A9, OUT = f'{ROOT}/11_A9_robustness', f'{ROOT}/12_A10_deep_calibration'
os.makedirs(OUT, exist_ok=True)
RES = f'{OUT}/A10_results.csv'
def log(msg):
    print(f'{msg} ({(time.time() - t0) / 60:.1f} min)', flush=True)
print(f'===== A.10 started {time.strftime("%H:%M:%S")} =====', flush=True)
if not torch.cuda.is_available():
    sys.exit('no GPU: switch the runtime to L4 or T4 and start again')
dev = 'cuda'; torch.backends.cudnn.benchmark = True
print('GPU:', torch.cuda.get_device_name(0), flush=True)

SUBJ, FS, DELTA = list(range(1, 16)), 128, 5.0
BUDGETS = [10, 20, 30, 45, 60, 'all']
SEEDS = [0, 1]
POOL = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
CH = json.load(open(f'{A1}/ME_sub01_meta.json'))['channels']
EEG = CH[:61]
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

# 1) labels and trial order from the A.2b cache; signals from the A.4b-1 cache (checked to match)
LAB, BADS = {}, {}
for cond in ('ME', 'MI'):
    for s in SUBJ:
        Z = np.load(f'{A2B}/cache/{cond}_sub{s:02d}.npz')
        LAB[(cond, s)] = {'y': Z['y'].astype(np.int64), 'run': Z['run'].astype(int), 'r4m': Z['r4m'].astype(bool)}
        BADS[(cond, s)] = [c for c in json.load(open(f'{A1}/{cond}_sub{s:02d}_meta.json'))['bad_channels'] if c in EEG]
files = sorted(glob.glob(f'{A4B}/cache_ts/*.npz'))
if not files:
    sys.exit(f'no files in {A4B}/cache_ts: send me this line')
print('A.4b cache example files:', [os.path.basename(f) for f in files[:4]], flush=True)
Z0 = np.load(files[0])
print('keys:', {k: (Z0[k].shape, str(Z0[k].dtype)) for k in Z0.files}, flush=True)
def find(cond, s):
    names = {os.path.basename(f): f for f in files}
    for n in (f'{cond}_sub{s:02d}.npz', f'sub{s:02d}_{cond}.npz', f'{cond}_S{s:02d}.npz', f'{cond}_{s:02d}.npz'):
        if n in names:
            return names[n]
    hits = [f for b, f in names.items() if re.search(rf'(^|[^A-Za-z]){cond}([^A-Za-z]|$)', b)
            and re.search(rf'(?<!\d)0?{s}(?!\d)', re.sub(cond, '', b))]
    return hits[0] if len(hits) == 1 else None
SIG = {}
for cond in ('ME', 'MI'):
    for s in SUBJ:
        f = find(cond, s)
        if f is None:
            sys.exit(f'no A.4b cache file for {cond} subject {s}: send me the file list above')
        Z = np.load(f)
        if 'X31' not in Z.files:
            sys.exit(f'{os.path.basename(f)} has no X31 array: send me the keys above')
        X = Z['X31']
        if X.ndim != 3 or X.shape[1] != 31 or X.shape[0] != len(LAB[(cond, s)]['y']):
            sys.exit(f'{os.path.basename(f)}: X31 shape {X.shape} does not match {len(LAB[(cond, s)]["y"])} A.2b trials')
        if 'y' in Z.files and not np.array_equal(Z['y'].astype(np.int64), LAB[(cond, s)]['y']):
            sys.exit(f'{os.path.basename(f)}: labels differ from the A.2b cache')
        SIG[(cond, s)] = {'X31': X, 'X31raw': Z['X31raw'] if 'X31raw' in Z.files else None}
T = SIG[('ME', 1)]['X31'].shape[2]
print(f'deep cache matches the A.2b trial order; {T} samples per trial', flush=True)

# 2) fold-specific montages (A.8 parser and check)
def flat(x):
    out = []
    for v in x:
        out += flat(v) if isinstance(v, (list, tuple)) else [v]
    return out
def to_idx(v):
    if isinstance(v, dict):
        v = next((v[k] for k in ('path', 'order', 'sites', 'selected', 'selection') if isinstance(v.get(k), (list, tuple))), None)
    if not isinstance(v, (list, tuple)):
        return None
    res = []
    for e in flat(v):
        if isinstance(e, str) and e in POOL:
            res.append(POOL.index(e))
        elif isinstance(e, (int, float)) and float(e).is_integer() and 0 <= int(e) < 31:
            res.append(int(e))
        else:
            return None
    return res
EXPECT = {'Cz': 13, 'P4': 13, 'CP3': 12, 'C1': 11, 'P3': 11, 'C2': 7}
def valid(m):
    if sorted(m) != SUBJ or any(v is None or len(v) < 8 for v in m.values()):
        return False
    cnt = collections.Counter(POOL[i] for v in m.values() for i in v[:8])
    return all(cnt.get(k, 0) == n for k, n in EXPECT.items())
cands = []
def walk(o, where):
    if isinstance(o, dict):
        try:
            keys = sorted(int(k) for k in o)
        except (ValueError, TypeError):
            keys = None
        if keys == SUBJ:
            cands.append((where, {int(k): to_idx(v) for k, v in o.items()}))
        for k, v in o.items():
            if isinstance(v, (dict, list, tuple)):
                walk(v, f'{where}/{k}')
    elif isinstance(o, (list, tuple)):
        if len(o) == 15 and all(isinstance(v, (list, tuple, dict)) for v in o):
            cands.append((where, {i + 1: to_idx(v) for i, v in enumerate(o)}))
        for i, v in enumerate(o):
            if isinstance(v, (dict, list, tuple)):
                walk(v, f'{where}[{i}]')
st = json.load(open(f'{A2B}/A2b2_state.json')) if os.path.exists(f'{A2B}/A2b2_state.json') else {}
walk(st, 'A2b2_state.json')
if os.path.exists(f'{A2B}/A2b2_outer.csv'):
    O = pd.read_csv(f'{A2B}/A2b2_outer.csv')
    scol = next((c for c in O.columns if c.lower() in ('subject', 'test_subject', 'outer', 'outer_subject')), None)
    for c in O.columns:
        if scol and O[c].dtype == object:
            m = {}
            for _, r in O.iterrows():
                toks = [x for x in re.split(r'[^A-Za-z0-9]+', str(r[c])) if x in POOL]
                if len(toks) == 8:
                    m[int(r[scol])] = [POOL.index(x) for x in toks]
            if len(m) == 15:
                cands.append((f'A2b2_outer.csv:{c}', m))
good = [(w, m) for w, m in cands if valid(m)]
if not good:
    sys.exit('could not identify the fold-specific montages (A.8 parser): send me this line')
MONT = {t: sorted(good[0][1][t][:8]) for t in SUBJ}

# 3) signals: montage, CAR by projection, optional Euclidean alignment
QB = {}
def Qm(n):
    if n not in QB:
        QB[n] = np.linalg.qr(np.c_[np.ones(n), np.eye(n)[:, :n - 1]])[0][:, 1:].astype(np.float32)
    return QB[n]
def sig(cond, s, S, target):
    d = SIG[(cond, s)]
    if target and d['X31raw'] is not None and any(POOL[i] in BADS[(cond, s)] for i in S):
        X = d['X31raw'][:, S, :].astype(np.float32)
        W = interp_matrix([POOL[i] for i in S], BADS[(cond, s)]).astype(np.float32)
        X = np.matmul(W, X)
    else:
        X = d['X31'][:, S, :].astype(np.float32)
    return np.matmul(Qm(len(S)).T, X)
def ea_ref(X):
    Xr = X.transpose(1, 0, 2).reshape(X.shape[1], -1).astype(np.float64)
    w, V = np.linalg.eigh(Xr @ Xr.T / Xr.shape[1])
    w = np.maximum(w, 1e-10 * w.max())
    return ((V / np.sqrt(w)) @ V.T).astype(np.float32)

class Shallow(nn.Module):
    def __init__(self, C, T, F=40, k=13, pool=38, stride=8):        # kernels scaled to 128 Hz (as in A.4b-1)
        super().__init__()
        self.temp = nn.Conv2d(1, F, (1, k))
        self.spat = nn.Conv2d(F, F, (C, 1), bias=False)
        self.bn = nn.BatchNorm2d(F)
        self.pool = nn.AvgPool2d((1, pool), (1, stride))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(F * ((T - k + 1 - pool) // stride + 1), 2)
    def forward(self, x):
        x = self.bn(self.spat(self.temp(x.unsqueeze(1))))
        x = torch.log(torch.clamp(self.pool(x * x), min=1e-6))
        return self.fc(self.drop(x.flatten(1)))
@torch.no_grad()
def logits(net, X):
    net.eval()
    return torch.cat([net(X[i:i + 512]) for i in range(0, len(X), 512)])

def train(Xtr, ytr, Xes, yes, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    sc = float(np.sqrt(np.mean(np.concatenate([x.reshape(-1) for x in Xtr]).astype(np.float64) ** 2)))
    Xt = torch.from_numpy((np.concatenate(Xtr) / sc).astype(np.float32)).to(dev); yt = torch.from_numpy(np.concatenate(ytr)).to(dev)
    Xv = torch.from_numpy((np.concatenate(Xes) / sc).astype(np.float32)).to(dev); yv = torch.from_numpy(np.concatenate(yes)).to(dev)
    ct, cv = torch.bincount(yt, minlength=2).float(), torch.bincount(yv, minlength=2).float()
    lossf = nn.CrossEntropyLoss(weight=ct.sum() / (2 * ct)); vw = cv.sum() / (2 * cv)
    net = Shallow(Xt.shape[1], Xt.shape[2]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, wait = np.inf, 0
    state = {n: v.detach().clone() for n, v in net.state_dict().items()}
    for ep in range(150):
        net.train()
        perm = torch.randperm(len(yt), device=dev)
        for i in range(0, len(perm) - 1, 64):
            b = perm[i:i + 64]
            opt.zero_grad(set_to_none=True)
            lossf(net(Xt[b]), yt[b]).backward()
            opt.step()
        vl = nn.functional.cross_entropy(logits(net, Xv), yv, weight=vw).item()
        if vl < best - 1e-4:
            best, wait = vl, 0
            state = {n: v.detach().clone() for n, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= 20:
                break
    net.load_state_dict(state)
    return net, sc

# 4) one new person at a time (resumable)
rows = pd.read_csv(RES).to_dict('records') if os.path.exists(RES) else []
done = {int(r['subject']) for r in rows}
for t in SUBJ:
    if t in done:
        log(f'subject {t} already done'); continue
    S = MONT[t]
    others = [s for s in SUBJ if s != t]
    es = list(np.random.default_rng(t).choice(others, 2, replace=False))   # early stopping: 2 training subjects' MI runs 6-10
    tr = {}
    for s in others:
        k = ~LAB[('ME', s)]['r4m']
        tr[s] = (sig('ME', s, S, False)[k], LAB[('ME', s)]['y'][k])
    esd = {}
    for s in es:
        X = sig('MI', s, S, False); L = LAB[('MI', s)]
        esd[s] = (X, L['y'], L['run'])
    Xt_all = sig('MI', t, S, True); Lt = LAB[('MI', t)]
    te = Lt['run'] >= 6; yte = Lt['y'][te]
    cal = np.where(Lt['run'] <= 5)[0]
    acc = collections.defaultdict(list)
    for seed in SEEDS:
        # strict
        net, sc = train([v[0] for v in tr.values()], [v[1] for v in tr.values()],
                        [esd[s][0][esd[s][2] >= 6] for s in es], [esd[s][1][esd[s][2] >= 6] for s in es], seed)
        p = logits(net, torch.from_numpy((Xt_all[te] / sc).astype(np.float32)).to(dev)).argmax(1).cpu().numpy()
        acc['strict'].append(100 * balanced_accuracy_score(yte, p))
        # aligned: each training session on itself; early-stopping subjects on their MI runs 1-5; the new person on the first N trials
        Xa = [np.matmul(ea_ref(v[0]), v[0]) for v in tr.values()]
        Xea = [np.matmul(ea_ref(esd[s][0][esd[s][2] <= 5]), esd[s][0][esd[s][2] >= 6]) for s in es]
        net, sc = train(Xa, [v[1] for v in tr.values()], Xea, [esd[s][1][esd[s][2] >= 6] for s in es], seed)
        for N in BUDGETS:
            idx = cal if N == 'all' else cal[:N]
            Xte = np.matmul(ea_ref(Xt_all[idx]), Xt_all[te])
            p = logits(net, torch.from_numpy((Xte / sc).astype(np.float32)).to(dev)).argmax(1).cpu().numpy()
            acc[f'aligned {N}'].append(100 * balanced_accuracy_score(yte, p))
        torch.cuda.empty_cache()
    for m, v in acc.items():
        rows.append(dict(subject=t, method=m, acc=float(np.mean(v)), n_seeds=len(v)))
    tmp = '/content/a10_tmp.csv'; pd.DataFrame(rows).to_csv(tmp, index=False)
    shutil.copyfile(tmp, RES + '.part'); os.replace(RES + '.part', RES)
    log(f'subject {t} done: strict {np.mean(acc["strict"]):.1f}, aligned all {np.mean(acc["aligned all"]):.1f}')

# 5) summary
R = pd.DataFrame(rows)
def col(m):
    return R[R.method == m].set_index('subject').acc
def pgt(a):
    a = np.asarray(a, float)
    return float(stats.wilcoxon(a - 50, alternative='greater').pvalue) if np.any(a != 50) else 1.0
print('\n=== anchors (A.4b-1, ShallowConvNet, cell 4; deep models vary slightly between runs, tolerance 1 point) ===')
for m, e in [('strict', 66.5), ('aligned all', 67.9)]:
    v = col(m).mean()
    print(f'  {m:12s} {v:5.1f}  (A.4b-1: {e})  {"OK" if abs(v - e) <= 1.0 else "DIFFERENT"}')

A9R = pd.read_csv(f'{A9}/A9_results.csv')
def a9(method):
    g = A9R[(A9R.method == method) & (A9R.montage == 'fold-specific 8') & (A9R.t_end == 3.0)]
    return g.set_index('subject').acc
own, ts_strict, ts_rec60 = a9('cell 1 own MI (labelled, ~27 min)'), a9('cell 4 others ME (strict)'), a9('recentered 60 trials')
print(f'\n=== A.10: ShallowConvNet calibration curve (others ME -> new person MI runs 6-10, fold-specific 8, mean of 15, {len(SEEDS)} seeds) ===')
print(f'{"method":14s} {"mean":>6} {"median":>7} {">50":>4} {"p>50":>7} | vs TS+LDA same budget')
for m in ['strict'] + [f'aligned {N}' for N in BUDGETS]:
    a = col(m)
    ref = ts_strict if m == 'strict' else None
    extra = ''
    if ref is not None:
        d = (a - ref.reindex(a.index)).values
        extra = f'{d.mean():+.1f} (p = {stats.wilcoxon(d).pvalue:.3f})'
    elif m == 'aligned 60':
        d = (a - ts_rec60.reindex(a.index)).values
        extra = f'{d.mean():+.1f} vs TS+LDA recentered 60 (p = {stats.wilcoxon(d).pvalue:.3f})'
    print(f'{m:14s} {a.mean():6.1f} {np.median(a):7.1f} {int((a > 50).sum()):4d} {pgt(a):7.4f} | {extra}')

def tost(d, delta=DELTA):
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    p1 = 1 - stats.t.cdf((m + delta) / se, n - 1); p2 = stats.t.cdf((m - delta) / se, n - 1)
    lo, hi = stats.t.interval(0.90, n - 1, loc=m, scale=se)
    return m, lo, hi, max(p1, p2)
print(f'\n=== equivalence with the TS+LDA full labelled calibration (cell 1, 67.1%), margin +/-{DELTA:.0f} points (TOST) ===')
for m in ['strict', 'aligned 10', 'aligned 60', 'aligned all']:
    a = col(m)
    mm, lo, hi, p = tost((a - own.reindex(a.index)).values)
    print(f'  ShallowConvNet {m:12s} - own calibration: {mm:+5.1f} [90% CI {lo:+5.1f}, {hi:+5.1f}], p TOST = {p:.3f} '
          f'-> {"EQUIVALENT" if p < 0.05 else "not shown"} (within {max(abs(lo), abs(hi)):.1f} points)')
log('total time')
print('===== A.10 FINISHED =====', flush=True)
