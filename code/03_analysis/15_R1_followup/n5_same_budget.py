# Same-budget comparison of own labelled calibration and the hybrid (Section 3.3 and Table S4b of the paper).
# For each window setting and budget N: own(N) - hybrid(N) per person (people with an own calibration at that budget),
# paired mean, t-based 95% CI and a two-sided tie-robust Wilcoxon p. Reads N5_curves.csv (from n4_n5.py) and writes
# N5_same_budget.csv next to it. Repository rewrite of the correction step of 13 Nov 2026 (n5_fix.py, not kept);
# it reproduces the saved N5_same_budget.csv (checked with --check).
#
# Usage: python n5_same_budget.py --dir <folder with N5_curves.csv> [--check <saved N5_same_budget.csv>]
import os, argparse
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--dir', default=os.path.abspath(os.path.join(HERE, '..', '..', '..', 'results', '15_R1_followup')))
ap.add_argument('--out', default=None, help='output CSV (default: <dir>/N5_same_budget.csv)')
ap.add_argument('--check', default=None, help='optional saved N5_same_budget.csv to compare with')
args = ap.parse_args()

def wil_robust(d):
    """Two-sided Wilcoxon signed-rank test; differences rounded to 1e-6; exact without tied or zero differences,
    otherwise the normal approximation with tie correction (same rule as the rest of the paper)."""
    d = np.round(np.asarray(d, float), 6); d = d[~np.isnan(d)]; nz = d[d != 0]
    if len(nz) == 0:
        return 1.0
    if len(np.unique(np.abs(nz))) == len(nz) and len(nz) == len(d):
        return float(stats.wilcoxon(d, method='exact').pvalue)
    try:
        return float(stats.wilcoxon(d, method='asymptotic').pvalue)
    except ValueError:                                   # older scipy: 'approx'
        return float(stats.wilcoxon(d, method='approx').pvalue)

N5 = pd.read_csv(f'{args.dir}/N5_curves.csv', float_precision='round_trip')
N5['budget'] = N5['budget'].astype(str)
def curve(method, budget, win):
    return N5[(N5.method == method) & (N5.budget == budget) & (N5.window == win)].groupby('subject').acc.mean().sort_index()

rows = []
for win in ('from 1.0 s', '0-3 s'):
    for b in ('10', '20', '30', '45', '60', 'all'):
        own, hyb = curve('own labelled', b, win), curve('hybrid', b, win)
        pair = own.dropna().index.intersection(hyb.dropna().index)
        d = (own[pair] - hyb[pair]).values; n = len(d); m = d.mean()
        h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
        rows.append(dict(window=win, budget=b, n=n, own_mean=own[pair].mean(), hybrid_mean=hyb[pair].mean(),
                         own_minus_hybrid=m, ci95_lo=m - h, ci95_hi=m + h, wilcoxon_p=wil_robust(d),
                         hybrid_mean_all15=hyb.mean(), own_mean_all_available=own.dropna().mean()))
R = pd.DataFrame(rows)
out = args.out or f'{args.dir}/N5_same_budget.csv'
R.to_csv(out, index=False)
print(R.round(3).to_string(index=False))
if args.check:
    S = pd.read_csv(args.check, float_precision='round_trip'); S['budget'] = S['budget'].astype(str)
    M = R.merge(S, on=['window', 'budget'], suffixes=('', '_saved'))
    cols = ['n', 'own_mean', 'hybrid_mean', 'own_minus_hybrid', 'ci95_lo', 'ci95_hi', 'wilcoxon_p', 'hybrid_mean_all15',
            'own_mean_all_available']
    dev = max(float(np.max(np.abs(M[c].astype(float) - M[f'{c}_saved'].astype(float)))) for c in cols)
    print(f'\ncompared with {args.check}: {len(M)}/{len(S)} rows matched, largest difference {dev:.2e} '
          + ('-> identical' if len(M) == len(S) and dev < 1e-9 else '-> DIFFERENT'))
