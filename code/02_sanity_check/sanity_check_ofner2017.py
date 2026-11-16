# Sanity check: Ofner et al. (2017) within-person decoding on our preprocessed data, before any new analysis.
# Ofner's method: 0.3-3 Hz zero-phase Butterworth (4th order, second-order sections), common average reference over the
# good EEG channels, 7 time points 0.1 s apart (0.6-s span) around a fixed centre, shrinkage LDA (lsqr, automatic
# shrinkage, equal priors). Centre: executed movement (ME) = glove onset (rest: cue + the person's mean ME latency);
# imagined movement (MI) = cue + the person's mean ME latency, one value for all classes (no class-dependent timing).
#   6 movements: ME +0.13 s, MI -0.06 s; movement vs rest: ME -0.13 s, MI 0.0 s. Mean latency over artifact-free trials.
# Evaluation: 10 x 10-fold cross-validation (Ofner's protocol; movement vs rest uses an equal random subsample of
# movement trials) and leave-one-run-out (balanced accuracy, all trials). Controls: EOG only, frontal 5 only.
# Checked on 15 Nov 2026 against the saved results of the first run: leave-one-run-out identical in 60/60 values and
# six-movement 10x10 identical; movement-vs-rest 10x10 differs slightly per person (group means within 0.4 points)
# because the random subsample of movement trials of the first run was not recorded.
#
# Usage: python sanity_check_ofner2017.py --data <preprocessed folder> --out <output folder> [--subjects ...]
#        [--check <folder with A2_per_subject.csv and A2_MI_alignment_check.csv>]
import os, json, time, argparse
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
ap = argparse.ArgumentParser()
ap.add_argument('--data', default=os.path.join(REPO, 'data', 'preprocessed'))
ap.add_argument('--out', default=os.path.join(REPO, 'outputs', 'sanity_check'))
ap.add_argument('--subjects', type=int, nargs='*', default=list(range(1, 16)))
ap.add_argument('--check', default=None, help='optional folder with the saved results of the first run (02_A2_sanity)')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
t0 = time.time()

FS, OFF = 128, np.arange(-3, 4) * 0.1
MOV = ['elbow_flexion', 'elbow_extension', 'supination', 'pronation', 'hand_close', 'hand_open']
FRONTAL = ['F3', 'F1', 'Fz', 'F2', 'F4']
TASKS = ('mov_vs_mov', 'mov_vs_rest')
CENTRE = {('ME', 'mov_vs_mov'): 0.13, ('ME', 'mov_vs_rest'): -0.13, ('MI', 'mov_vs_mov'): -0.06, ('MI', 'mov_vs_rest'): 0.0}
OFNER = {('ME', 'mov_vs_mov'): 55, ('ME', 'mov_vs_rest'): 86, ('MI', 'mov_vs_mov'): 27, ('MI', 'mov_vs_rest'): 73}
SOS = butter(4, (0.3, 3.0), btype='band', fs=FS, output='sos')

def load(c, s):
    meta = json.load(open(f'{args.data}/{c}_sub{s:02d}_meta.json'))
    Z = np.load(f'{args.data}/{c}_sub{s:02d}.npz')
    return meta, {r: Z[f'run{r}'].astype(np.float64) for r in range(1, 11)}, pd.read_csv(f'{args.data}/{c}_sub{s:02d}_events.csv')

def mean_latency(s):
    E = pd.read_csv(f'{args.data}/ME_sub{s:02d}_events.csv')
    m = E['class'].isin(MOV) & E.onset_latency_s.notna() & ~E.r4m_rejected.astype(bool)
    return float(E.onset_latency_s[m].mean())

def prep(meta, runs):
    ch, bads = meta['channels'], {b.lower() for b in meta['bad_channels']}
    eeg = [i for i, n in enumerate(ch) if not n.lower().startswith('eog')]
    eog = [i for i, n in enumerate(ch) if n.lower().startswith('eog')]
    good = [i for i in eeg if ch[i].lower() not in bads]               # bad channels are left out
    names = [ch[i].lower() for i in good]
    filt = lambda X: sosfiltfilt(SOS, X, axis=1)
    P = {r: (filt(X[good] - X[good].mean(0, keepdims=True)), filt(X[eog])) for r, X in runs.items()}
    return P, [names.index(c.lower()) for c in FRONTAL if c.lower() in names]

def features(c, s, E, P, fr, task):
    ml = mean_latency(s)
    E = E[~E.r4m_rejected.astype(bool)]
    if task == 'mov_vs_mov':
        E = E[E['class'].isin(MOV)]
    lat = (np.where(E['class'].isin(MOV), E.onset_latency_s, ml) if c == 'ME' else np.full(len(E), ml)).astype(float)
    ok = ~np.isnan(lat); E, lat = E[ok], lat[ok]                       # ME movement trials need a glove onset
    t = lat + CENTRE[(c, task)]
    idx = np.round((E.cue_time_s.values[:, None] + t[:, None] + OFF[None]) * FS).astype(int)
    F = {'eeg': [], 'eog': [], 'frontal': []}; keep = np.zeros(len(E), bool)
    for i, (r, ix) in enumerate(zip(E.run.values, idx)):
        Xe, Xo = P[r]
        if ix.min() < 0 or ix.max() >= Xe.shape[1]:
            continue
        keep[i] = True
        F['eeg'].append(Xe[:, ix].ravel()); F['eog'].append(Xo[:, ix].ravel()); F['frontal'].append(Xe[fr][:, ix].ravel())
    E = E[keep]
    y = (np.array([MOV.index(x) for x in E['class']]) if task == 'mov_vs_mov'
         else E['class'].isin(MOV).astype(int).values)
    return {k: np.array(v) for k, v in F.items()}, y, E.run.values

def lda(k):
    return LDA(solver='lsqr', shrinkage='auto', priors=np.ones(k) / k)

def loro(X, y, run):
    pred = np.empty_like(y)
    for r in range(1, 11):
        te = run == r
        if te.any():
            pred[te] = lda(len(np.unique(y))).fit(X[~te], y[~te]).predict(X[te])
    return 100 * balanced_accuracy_score(y, pred)

def cv10x10(X, y, task):
    acc = []
    for rep in range(10):
        idx = np.arange(len(y))
        if task == 'mov_vs_rest':        # rest vs an equal random subsample of movement trials
            rng = np.random.RandomState(rep); mv, rs = idx[y == 1], idx[y == 0]
            idx = np.sort(np.r_[rng.choice(mv, len(rs), replace=False), rs])
        Xs, ys = X[idx], y[idx]
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=rep).split(Xs, ys):
            acc.append(accuracy_score(ys[te], lda(len(np.unique(ys))).fit(Xs[tr], ys[tr]).predict(Xs[te])))
    return 100 * np.mean(acc)

res = []
for s in args.subjects:
    for c in ('ME', 'MI'):
        meta, runs, E = load(c, s); P, fr = prep(meta, runs)
        for task in TASKS:
            F, y, run = features(c, s, E, P, fr, task)
            for ch in ('eeg', 'eog', 'frontal'):
                res.append(dict(subject=s, cond=c, task=task, chans=ch, n_trials=len(y), n_features=F[ch].shape[1],
                                acc_10x10=round(cv10x10(F[ch], y, task), 1),
                                acc_loro=round(loro(F[ch], y, run), 1) if ch == 'eeg' else np.nan))
    print(f'subject {s} done ({(time.time() - t0) / 60:.1f} min)', flush=True)
R = pd.DataFrame(res); R.to_csv(f'{args.out}/sanity_per_subject.csv', index=False)
G = R.groupby(['cond', 'task', 'chans'], sort=False).agg(acc_10x10=('acc_10x10', 'mean'), acc_loro=('acc_loro', 'mean')).round(1).reset_index()
G['ofner_2017'] = [OFNER.get((a, b)) if h == 'eeg' else None for a, b, h in zip(G.cond, G.task, G.chans)]
G.to_csv(f'{args.out}/sanity_group.csv', index=False)
print(G.to_string(index=False))

if args.check:      # compare with the saved results of the first run (ME: A2_per_subject; MI: final rows of the alignment check)
    A = pd.read_csv(f'{args.check}/A2_per_subject.csv'); A = A[A.cond == 'ME']
    M = pd.read_csv(f'{args.check}/A2_MI_alignment_check.csv'); M = M[M['mode'] == 'subject_mean'].assign(cond='MI')
    K = pd.concat([A, M], ignore_index=True)
    J = R.merge(K, on=['subject', 'cond', 'task', 'chans'], suffixes=('', '_saved'))
    same = lambda a, b: (a.isna() & b.isna()) | (np.abs(a - b) < 0.05)
    lo = same(J.acc_loro, J.acc_loro_saved); cv = same(J.acc_10x10, J.acc_10x10_saved)
    mm = J.task == 'mov_vs_mov'
    print(f'\ncompared with the saved results ({len(J)} rows):')
    print(f'  leave-one-run-out identical: {int(lo.sum())}/{len(J)}')
    print(f'  10x10, 6 movements identical: {int(cv[mm].sum())}/{int(mm.sum())}')
    d = (J.acc_10x10 - J.acc_10x10_saved)[~mm]
    print(f'  10x10, movement vs rest: mean |difference| {d.abs().mean():.2f} points (random subsample of the first run '
          f'not recorded; group means are compared in the README)')
    if 'n_trials_saved' in J:
        e = J[J.cond == 'ME']
        print(f'  trial and feature counts identical (ME): {int(((e.n_trials == e.n_trials_saved) & (e.n_features == e.n_features_saved)).sum())}/{len(e)}')
