# Step A.7-2a: PhysioNet right fist vs rest (T1-like), zero-calibration ME -> MI (approved 10 Nov 2026)
import os, sys, json, glob, time, shutil
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

t0 = time.time()
ROOT = '/content/drive/MyDrive/classification_MI-ME/30_v06_pipeline'
D7, OUT = f'{ROOT}/08_A7_physionet', f'{ROOT}/09_A7_2a_physionet_rest'
os.makedirs(OUT, exist_ok=True)
RES = f'{OUT}/A72a_results.csv'
def log(msg):
    print(f'{msg} ({(time.time() - t0) / 60:.1f} min)', flush=True)
print(f'===== A.7-2a started {time.strftime("%H:%M:%S")} =====', flush=True)
if not torch.cuda.is_available():
    sys.exit('no GPU: switch the runtime to L4 or T4 and start again')
dev = 'cuda'
torch.backends.cudnn.benchmark = True
print('GPU:', torch.cuda.get_device_name(0), flush=True)

# 1) channels and montages
ch = json.load(open(f'{D7}/channels.json'))
if isinstance(ch, dict):
    ch = next((v for v in ch.values() if isinstance(v, list) and len(v) == 64), None)
if not isinstance(ch, list) or len(ch) != 64:
    sys.exit('unexpected content in channels.json: send me this line')
CH = [str(c) for c in ch]
UP = [c.upper() for c in CH]
def chidx(names):
    miss = [n for n in names if n.upper() not in UP]
    if miss:
        sys.exit(f'channels not found: {miss}; available: {CH}')
    return [UP.index(n.upper()) for n in names]
POOL31 = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz',
          'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4']
MONT = {'cons8': ['Cz', 'P4', 'CP3', 'C1', 'P3', 'C2', 'Pz', 'C6'],        # our montage (A.2b consensus)
        'nearC3_8': ['FC5', 'FC3', 'FC1', 'C5', 'C3', 'C1', 'CP5', 'CP3'],   # anatomical comparator
        'pool31': POOL31, 'all64': CH,
        'central5': ['Cz', 'CP3', 'C1', 'C2', 'C6'], 'parietal3': ['P3', 'Pz', 'P4'],
        'occ5': ['O1', 'Oz', 'O2', 'PO7', 'PO8']}                            # visual control
MIDX = {k: chidx(v) for k, v in MONT.items()}

# 2) A.7-1 cache: right fist (T2) and rest (T0) epochs only
files = sorted(glob.glob(f'{D7}/cache/S*.npz'))
if not files:
    sys.exit('no cache files found in 08_A7_physionet/cache')
Z0 = np.load(files[0])
print('cache keys:', {k: (Z0[k].shape, str(Z0[k].dtype)) for k in Z0.files}, flush=True)
if 'bb' not in Z0.files or 'kind' not in Z0.files or 'cond' not in Z0.files or 'run' not in Z0.files:
    sys.exit('cache keys differ from what this script expects: send me the line above')
kinds = set(np.unique(Z0['kind'].astype(str)).tolist())
conds = set(np.unique(Z0['cond'].astype(str)).tolist())
print('kind values:', sorted(kinds), '| cond values:', sorted(conds), flush=True)
REST = next((k for k in ('rest', 'T0') if k in kinds), None)
RIGHT = next((k for k in ('right', 'T2') if k in kinds), None)
if REST is None or RIGHT is None or not {'ME', 'MI'} <= conds:
    sys.exit('cannot identify rest/right in "kind" or ME/MI in "cond": send me these lines')
TSM = ['cons8', 'pool31', 'all64', 'nearC3_8']
FK = {}
for m in TSM:
    ks = sorted(k for k in Z0.files if k.startswith(f'f_{m}_') and 'full' in k)
    if not ks:
        sys.exit(f'no cached TS+LDA features for {m}; keys: {Z0.files}')
    FK[m] = ks
print('TS+LDA feature keys:', FK, flush=True)

DATA, skipped = {}, {}
for f in files:
    s = int(os.path.basename(f)[1:4])
    Z = np.load(f)
    kind, cond, run = Z['kind'].astype(str), Z['cond'].astype(str), Z['run'].astype(int)
    k = (kind == REST) | (kind == RIGHT)
    te = k & (cond == 'MI') & np.isin(run, (8, 12))
    wu = k & (cond == 'MI') & (run == 4)
    if len(np.unique(kind[te])) < 2 or wu.sum() < 5 or not np.any(k & (cond == 'ME')):
        skipped[s] = 'missing test classes or warm-up run'
        continue
    d = {'y': (kind[k] == RIGHT).astype(np.int64), 'cond': cond[k], 'run': run[k], 'bb': Z['bb'][k]}
    for m in TSM:
        d[f'f_{m}'] = np.hstack([Z[key][k].reshape(int(k.sum()), -1).astype(np.float64) for key in FK[m]])
    DATA[s] = d
SUBJ = sorted(DATA)
def cnt(s, mask):
    return int(mask.sum()), int(DATA[s]['y'][mask].sum())
me = np.array([cnt(s, DATA[s]['cond'] == 'ME') for s in SUBJ])
tt = np.array([cnt(s, (DATA[s]['cond'] == 'MI') & np.isin(DATA[s]['run'], (8, 12))) for s in SUBJ])
log(f'subjects ready: {len(SUBJ)} (skipped: {skipped or "none"}); per subject: ME {me[:, 0].mean():.0f} epochs '
    f'({me[:, 1].mean():.0f} right), MI test (runs 8+12) {tt[:, 0].mean():.0f} ({tt[:, 1].mean():.0f} right)')
FOLDS = np.array_split(np.random.default_rng(0).permutation(SUBJ), 10)

# 3) helpers
QB = {}
def Q(n):
    """CAR as an exact projection onto the space orthogonal to the all-ones vector (n x n-1)."""
    if n not in QB:
        QB[n] = np.linalg.qr(np.c_[np.ones(n), np.eye(n)[:, :n - 1]])[0][:, 1:].astype(np.float32)
    return QB[n]

def slda_fit(X, y):
    """Shrinkage LDA, identical to sklearn LDA(solver='lsqr', shrinkage='auto', equal priors)."""
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

def ea_ref(X):
    """Euclidean alignment matrix R^(-1/2) from the mean covariance of the epochs X (n, ch, t)."""
    Xr = X.transpose(1, 0, 2).reshape(X.shape[1], -1).astype(np.float64)
    w, V = np.linalg.eigh(Xr @ Xr.T / Xr.shape[1])
    w = np.maximum(w, 1e-10 * w.max())
    return ((V / np.sqrt(w)) @ V.T).astype(np.float32)

def shuf(s, y):
    return np.random.default_rng(1000 + s).permutation(y)

# self-check: fast LDA vs sklearn on real features
tst0 = set(FOLDS[0].tolist()); tr0 = [s for s in SUBJ if s not in tst0]
Xc = np.vstack([DATA[s]['f_cons8'][DATA[s]['cond'] == 'ME'] for s in tr0])
yc = np.concatenate([DATA[s]['y'][DATA[s]['cond'] == 'ME'] for s in tr0])
Xv = np.vstack([DATA[s]['f_cons8'][DATA[s]['cond'] == 'MI'] for s in sorted(tst0)])
same = np.mean(predict(Xv, slda_fit(Xc, yc)) == LDA(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5]).fit(Xc, yc).predict(Xv))
print(f'fast LDA vs sklearn: same decisions in {100 * same:.1f}% of trials', flush=True)

# 4) TS+LDA (secondary): cached log-Euclidean features, 0-4 s
def ts_run(m, mode, shuffle):
    acc = {}
    for test in FOLDS:
        test = set(test.tolist()); Xs, ys = [], []
        for s in SUBJ:
            if s in test:
                continue
            mk = DATA[s]['cond'] == 'ME'
            X, y = DATA[s][f'f_{m}'][mk], DATA[s]['y'][mk]
            if mode == 'recentered':
                X = X - X.mean(0)
            Xs.append(X); ys.append(shuf(s, y) if shuffle else y)
        mdl = slda_fit(np.vstack(Xs), np.concatenate(ys))
        for s in test:
            d = DATA[s]; mk = (d['cond'] == 'MI') & np.isin(d['run'], (8, 12))
            X = d[f'f_{m}'][mk]
            if mode == 'recentered':
                X = X - d[f'f_{m}'][(d['cond'] == 'MI') & (d['run'] == 4)].mean(0)   # unlabelled warm-up: MI run 4
            acc[s] = 100 * balanced_accuracy_score(d['y'][mk], predict(X, mdl))
    return acc

# 5) ShallowConvNet (primary), kernels scaled to 160 Hz
class Shallow(nn.Module):
    def __init__(self, C, T, F=40, k=16, pool=48, stride=10):
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
    return torch.cat([net(X[i:i + 512]) for i in range(0, len(X), 512)])   # small batches: no memory spikes

def prep(m, mode, start):
    """CAR within the montage; 'aligned': ME aligned on its own ME epochs, MI aligned on MI run 4 (unlabelled)."""
    q = Q(len(MIDX[m])).T
    P = {}
    for s in SUBJ:
        d = DATA[s]
        X = np.matmul(q, d['bb'][:, MIDX[m], start:].astype(np.float32))
        if mode == 'aligned':
            me_, mi_ = d['cond'] == 'ME', d['cond'] == 'MI'
            X[me_] = np.matmul(ea_ref(X[me_]), X[me_])
            X[mi_] = np.matmul(ea_ref(X[mi_ & (d['run'] == 4)]), X[mi_])
        P[s] = X
    return P

def run_shallow(m, mode, start, shuffle):
    P = prep(m, mode, start)
    acc = {}
    for k, test in enumerate(FOLDS):
        test = set(test.tolist()); tr = [s for s in SUBJ if s not in test]
        es = np.random.default_rng(k).choice(tr, 8, replace=False).tolist()     # early stopping on 8 training subjects' MI
        Xtr = [P[s][DATA[s]['cond'] == 'ME'] for s in tr]
        ytr = [DATA[s]['y'][DATA[s]['cond'] == 'ME'] for s in tr]
        Xes = [P[s][DATA[s]['cond'] == 'MI'] for s in es]
        yes = [DATA[s]['y'][DATA[s]['cond'] == 'MI'] for s in es]
        if shuffle:
            ytr = [shuf(s, y) for s, y in zip(tr, ytr)]
            yes = [shuf(s, y) for s, y in zip(es, yes)]
        sc = float(np.sqrt(sum(float(np.sum(x.astype(np.float64) ** 2)) for x in Xtr) / sum(x.size for x in Xtr)))
        Xt = torch.from_numpy((np.concatenate(Xtr) / sc).astype(np.float32)).to(dev)
        yt = torch.from_numpy(np.concatenate(ytr)).to(dev)
        Xv = torch.from_numpy((np.concatenate(Xes) / sc).astype(np.float32)).to(dev)
        yv = torch.from_numpy(np.concatenate(yes)).to(dev)
        torch.manual_seed(k)
        ct = torch.bincount(yt, minlength=2).float(); cv = torch.bincount(yv, minlength=2).float()
        lossf = nn.CrossEntropyLoss(weight=ct.sum() / (2 * ct))
        vw = cv.sum() / (2 * cv)
        net = Shallow(Xt.shape[1], Xt.shape[2]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        best, wait = np.inf, 0
        state = {n: v.detach().clone() for n, v in net.state_dict().items()}
        for ep in range(100):
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
                if wait >= 15:
                    break
        net.load_state_dict(state)
        for s in test:
            mk = (DATA[s]['cond'] == 'MI') & np.isin(DATA[s]['run'], (8, 12))
            X = torch.from_numpy((P[s][mk] / sc).astype(np.float32)).to(dev)
            acc[s] = 100 * balanced_accuracy_score(DATA[s]['y'][mk], logits(net, X).argmax(1).cpu().numpy())
        del Xt, Xv, yt, yv, net, opt
        torch.cuda.empty_cache()
    return acc

# 6) run all conditions (resumable: finished conditions are read from the results file)
def cname(m, mode, start=0, shuffle=False):
    return f'{m} {mode}' + (' post' if start else '') + (' shuffled' if shuffle else '')
TS_CONDS = [(m, md, False) for md in ('strict', 'recentered') for m in TSM] + [('cons8', 'strict', True)]
SH_CONDS = [('cons8', 'strict', 0, False), ('cons8', 'strict', 80, False), ('cons8', 'strict', 0, True),
            ('central5', 'strict', 0, False), ('parietal3', 'strict', 0, False), ('occ5', 'strict', 0, False),
            ('nearC3_8', 'strict', 0, False), ('pool31', 'strict', 0, False), ('all64', 'strict', 0, False),
            ('cons8', 'aligned', 0, False), ('nearC3_8', 'aligned', 0, False), ('pool31', 'aligned', 0, False),
            ('all64', 'aligned', 0, False)]
rows = pd.read_csv(RES).to_dict('records') if os.path.exists(RES) else []
done = {(r['model'], r['condition']) for r in rows}
def save(model, cond, acc):
    rows.extend({'model': model, 'condition': cond, 'subject': int(s), 'bal_acc': round(float(a), 2)} for s, a in acc.items())
    tmp = '/content/a72a_results_tmp.csv'
    pd.DataFrame(rows).to_csv(tmp, index=False)
    shutil.copyfile(tmp, RES + '.part'); os.replace(RES + '.part', RES)

for m, md, sh in TS_CONDS:
    c = cname(m, md, 0, sh)
    if ('TS+LDA', c) not in done:
        save('TS+LDA', c, ts_run(m, md, sh))
    log(f'TS+LDA {c} done')
for m, md, st, sh in SH_CONDS:
    c = cname(m, md, st, sh)
    if ('ShallowConvNet', c) not in done:
        save('ShallowConvNet', c, run_shallow(m, md, st, sh))
    log(f'ShallowConvNet {c} done')

# 7) summary and the pre-specified verdict
R = pd.DataFrame(rows)
def pgt(a):
    a = np.asarray(a, float)
    return float(wilcoxon(a - 50, alternative='greater').pvalue) if np.any(a != 50) else 1.0
def summ(g):
    a = g.bal_acc.values
    return pd.Series({'n': len(a), 'mean': round(a.mean(), 1), 'median': round(float(np.median(a)), 1),
                      'n_above_50': int((a > 50).sum()), 'p_vs_50': round(pgt(a), 4)})
S = R.groupby(['model', 'condition'], sort=False)[['bal_acc']].apply(summ).reset_index()
S['n'] = S['n'].astype(int); S['n_above_50'] = S['n_above_50'].astype(int)
S.to_csv(f'{OUT}/A72a_summary.csv', index=False)
pd.set_option('display.width', 200)
print('\n=== A.7-2a: PhysioNet, right fist vs rest, zero-calibration ME->MI, test = MI runs 8+12, balanced accuracy % ===')
print(S.to_string(index=False))

def col(model, cond):
    return R[(R.model == model) & (R.condition == cond)].set_index('subject').bal_acc
def pdiff(model, c1, c2, alt='two-sided'):
    x, z = col(model, c1), col(model, c2)
    dl = (x - z.reindex(x.index)).dropna().values
    return dl.mean(), (float(wilcoxon(dl, alternative=alt).pvalue) if np.any(dl != 0) else 1.0)

M = 'ShallowConvNet'
a = col(M, 'cons8 strict'); mean = a.mean(); p_main = pgt(a)
m_sh, p_sh = col(M, 'cons8 strict shuffled').mean(), pgt(col(M, 'cons8 strict shuffled'))
m_post = col(M, 'cons8 strict post').mean(); drop = mean - m_post
c5, o5 = col(M, 'central5 strict'), col(M, 'occ5 strict'); p_c5 = pgt(c5)
d_co, p_co = pdiff(M, 'central5 strict', 'occ5 strict', 'greater')
c_gt_o = d_co > 0 and p_co < 0.05
d_mont = mean - col(M, 'nearC3_8 strict').mean()
fail = p_main >= 0.05 or p_sh < 0.05 or not c_gt_o
full = (not fail) and mean >= 60 and drop <= 2 and p_c5 < 0.05
verdict = 'FAILS' if fail else ('GENERALIZES' if full else 'PARTIAL')
yn = lambda b: 'yes' if b else 'NO'
print('\n=== pre-specified verdict (approved 10 Nov 2026): ShallowConvNet, our 8 sites, strict ===')
print(f'mean {mean:.1f}% (>= 60: {yn(mean >= 60)}); above chance p = {p_main:.4f} ({yn(p_main < 0.05)})')
print(f'shuffled labels {m_sh:.1f}%, p = {p_sh:.4f} (at chance: {yn(p_sh >= 0.05)})')
print(f'without first 0.5 s {m_post:.1f}% (drop {drop:.1f}; <= 2: {yn(drop <= 2)})')
print(f'central 5 alone {c5.mean():.1f}% (p = {p_c5:.4f}); occipital 5 alone {o5.mean():.1f}% (p = {pgt(o5):.4f}); '
      f'central minus occipital {d_co:+.1f}, p = {p_co:.4f} (central higher: {yn(c_gt_o)})')
print(f'-> {verdict}')
if not fail and drop > 2:
    print(f'   {drop:.1f} points disappear without the first 0.5 s: attributed to the response to the target onset')
print(f'montage: our 8 minus near-C3 8 = {d_mont:+.1f} points -> '
      + ('not Ofner-specific' if d_mont >= -2 else 'worse than near-C3 by more than 2 points'))
for model, c1, c2 in [('ShallowConvNet', 'cons8 aligned', 'cons8 strict'), ('TS+LDA', 'cons8 recentered', 'cons8 strict'),
                      ('ShallowConvNet', 'all64 strict', 'cons8 strict')]:
    dm, pp = pdiff(model, c1, c2)
    print(f'{model}: {c1} minus {c2} = {dm:+.1f} points (p = {pp:.4f})')
print(f"TS+LDA shuffled labels: {col('TS+LDA', 'cons8 strict shuffled').mean():.1f}%, p = {pgt(col('TS+LDA', 'cons8 strict shuffled')):.4f}")
print('reference: A.7-1 left vs right, all 64, ShallowConvNet 81.0%; Ofner T1 cell 4, ShallowConvNet 66.5%')
log('total time')
print('===== A.7-2a FINISHED =====', flush=True)
