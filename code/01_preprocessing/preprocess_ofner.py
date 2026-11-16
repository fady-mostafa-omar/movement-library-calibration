# Preprocessing of the Ofner et al. (2017) upper-limb data set (Section 2.1 of the paper).
# Raw GDF (512 Hz) -> per session (condition x subject): 10 runs at 128 Hz + one events table + one meta file.
#   per run: crop 3 s at both ends, remove each channel's median, causal 4th-order Butterworth low-pass at 45 Hz
#            (zero initial state), keep every 4th sample (512 -> 128 Hz); 61 EEG + 3 EOG channels in a fixed order.
#   events:  the 42 cues of each run (6 per class); cue time relative to the cropped run; cue sample at 128 Hz
#            (numpy rounding, half to even).
#   registry: artifact flags of trials (R4M), glove onset latencies (executed movement), bad channels and the channel
#            order come from the frozen registry in results/registry/ (built before this study; see the README).
# This script reproduces the preprocessed files used for every analysis exactly (all 30 sessions; signal difference 0,
# identical events), checked on 15 Nov 2026 against the files used for the paper.
#
# Usage:
#   python preprocess_ofner.py --raw <folder with the 300 GDF files> --out <output folder> [--subjects 1 2 ...]
#                              [--check <folder with reference files to compare against>]
import os, sys, json, time, argparse
import numpy as np, pandas as pd
import mne
from scipy.signal import butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
p = argparse.ArgumentParser()
p.add_argument('--raw', default=os.path.expanduser('~/mne_data/MNE-upperlimb-data/zenodo/834976'),
               help='folder with motorexecution_subjectS_runR.gdf and motorimagination_subjectS_runR.gdf (Zenodo 834976)')
p.add_argument('--out', default=os.path.join(REPO, 'data', 'preprocessed'))
p.add_argument('--registry', default=os.path.join(REPO, 'results', 'registry'))
p.add_argument('--subjects', type=int, nargs='*', default=list(range(1, 16)))
p.add_argument('--check', default=None, help='optional folder with reference preprocessed files (same names) to compare')
args = p.parse_args()

mne.set_log_level('ERROR'); t0 = time.time()
RAW, NEW = args.raw, args.out
os.makedirs(NEW, exist_ok=True)
FS_RAW, DEC, CROP_S = 512, 4, 3.0
COND = {'ME': 'motorexecution', 'MI': 'motorimagination'}
CLASS = {'1536': 'elbow_flexion', '1537': 'elbow_extension', '1538': 'supination', '1539': 'pronation',
         '1540': 'hand_close', '1541': 'hand_open', '1542': 'rest'}
CH = json.load(open(f'{args.registry}/channels.json'))   # 61 EEG + 3 EOG, canonical order (as in the paper's files)
SOS = butter(4, 45, btype='low', fs=FS_RAW, output='sos')

# 1) frozen registry: R4M trial flags, glove onset latencies, bad channels
REG = pd.read_csv(f'{args.registry}/registry_trials.csv')
RBD = pd.read_csv(f'{args.registry}/registry_bad_channels.csv', keep_default_na=False)
BADS = {(r.condition, int(r.subject)): [x for x in str(r.bad_channels).split(';') if x] for r in RBD.itertuples()}
print(f'registry: {len(REG)} trials, {int(REG.r4m_rejected.sum())} R4M-rejected, {len(BADS)} sessions')

# 2) channel picking (subject 1 ME has generic channel names: map by position as in subject 2, EOG order reversed)
REF = {c: [n.lower() for n in mne.io.read_raw_gdf(f'{RAW}/{COND[c]}_subject2_run1.gdf').ch_names] for c in COND}
EOG = [n for n in CH if n.startswith('eog')]
REV = dict(zip(EOG, EOG[::-1]))
def pick(raw, c):
    low = {n.lower(): i for i, n in enumerate(raw.ch_names)}
    if all(n.lower() in low for n in CH):
        return [low[n.lower()] for n in CH], 'names'
    return [REF[c].index(REV.get(n, n).lower()) for n in CH], 'position (EOG reversed)'

def cue_events(raw):
    ev, ed = mne.events_from_annotations(raw)
    code = {v: k for k, v in ed.items()}
    return [(int(e[0]), CLASS[code[e[2]]]) for e in ev if code[e[2]] in CLASS]

# 3) one session: crop 3 s each side, median removal, causal low-pass 45 Hz (zero initial state), every 4th sample
def build(c, s):
    runs, rows, how, nan = {}, [], set(), 0
    k = int(CROP_S * FS_RAW)
    for r in range(1, 11):
        raw = mne.io.read_raw_gdf(f'{RAW}/{COND[c]}_subject{s}_run{r}.gdf', preload=True)
        assert raw.info['sfreq'] == FS_RAW
        idx, h = pick(raw, c); how.add(h)
        X = raw.get_data(picks=idx)[:, k:-k]
        nan += int(np.isnan(X).sum())
        X = sosfilt(SOS, X - np.median(X, axis=1, keepdims=True), axis=1)[:, ::DEC]
        runs[f'run{r}'] = X.astype(np.float32)
        for t, (smp, cl) in enumerate(cue_events(raw), 1):
            rows.append(dict(condition=c, subject=s, run=r, trial=t, **{'class': cl},
                             cue_time_s=round((smp - k) / FS_RAW, 6),
                             cue_sample_128=int(np.round((smp - k) / DEC))))   # numpy: half to even
    E = pd.DataFrame(rows)
    R = REG[(REG.condition == c) & (REG.subject == s)][['run', 'trial', 'class', 'r4m_rejected', 'onset_latency_s']]
    E = E.merge(R, on=['run', 'trial'], how='left', suffixes=('', '_reg'))
    regm = int((E['class'] != E['class_reg']).sum())
    np.savez(f'{NEW}/{c}_sub{s:02d}.npz', **runs)
    E[['condition', 'subject', 'run', 'trial', 'class', 'cue_time_s', 'cue_sample_128', 'r4m_rejected',
       'onset_latency_s']].to_csv(f'{NEW}/{c}_sub{s:02d}_events.csv', index=False)
    json.dump(dict(version='A1_rebuild_v1', condition=c, subject=s, channels=CH, sfreq=FS_RAW // DEC,
                   units='uV (GDF physical)', crop_s_each_side=CROP_S, dc_removal='per-run channel median',
                   anti_alias='causal Butterworth low-pass 45 Hz, order 4, zero initial state, then every 4th sample 512->128',
                   reference='original recording reference (CAR applied later over the chosen montage)',
                   bad_channels=BADS[(c, s)]), open(f'{NEW}/{c}_sub{s:02d}_meta.json', 'w'), indent=1)
    return runs, E, how, nan, regm

# 4) build, and optionally compare with reference files
n_ok = n_all = 0
for s in args.subjects:
    for c in COND:
        runs, E, how, nan, regm = build(c, s)
        line = f'{c} subject {s:2d}: channels by {sorted(how)} | events {len(E)} | registry class mismatches {regm} | NaN {nan}'
        if args.check:
            Z = np.load(f'{args.check}/{c}_sub{s:02d}.npz')
            sig = max(float(np.max(np.abs(runs[q].astype(np.float64) - Z[q]))) if runs[q].shape == Z[q].shape else np.inf
                      for q in runs)
            O = pd.read_csv(f'{args.check}/{c}_sub{s:02d}_events.csv')
            same = len(O) == len(E)
            cls = int((O['class'].values != E['class'].values).sum()) if same else -1
            dt = float(np.max(np.abs(O.cue_time_s.values - E.cue_time_s.values))) if same else np.inf
            smp = int((O.cue_sample_128.values != E.cue_sample_128.values).sum()) if same else -1
            ok = sig < 1e-3 and same and cls == 0 and dt < 1e-6 and smp == 0 and regm == 0 and nan == 0
            n_ok += ok; n_all += 1
            line += (f' | {"PASS" if ok else "CHECK"}: signal max diff {sig:.2e} uV, class {cls}, cue time {dt:.1e} s, '
                     f'cue sample {smp}')
        print(f'{line} | {(time.time() - t0) / 60:.1f} min', flush=True)
if args.check:
    print(f'\n{n_ok}/{n_all} sessions identical to the reference files')
