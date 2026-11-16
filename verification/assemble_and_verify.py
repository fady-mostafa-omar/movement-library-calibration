# assemble_and_verify.py: builds the public repository of Paper 2 from the project folder on Google Drive and checks it.
# Run once on Google Colab (Drive mounted) from the unpacked repository files:
#     python /content/paper2_repo/verification/assemble_and_verify.py
# It only READS the project folder; everything it writes goes to the local repository folder, plus one zip file in
# <project>/60_repo_check/. Steps:
#   1. copy the analysis scripts (as run) and the small per-person result files (CSV/JSON/MD/TXT/LOG, <= 25 MB) of every
#      step folder; no EEG data, caches, model weights or figures;
#   2. export the frozen trial/bad-channel registry and the channel order;
#   3. check the repository versions of the scripts: preprocessing and sanity check on subject 1 against the saved files,
#      the same-budget comparison against the saved file, and the full figure and table script from results/;
#   4. write requirements.txt (imported packages, versions of this runtime), MANIFEST.csv and the zip.
import os, sys, re, json, glob, time, shutil, hashlib, zipfile, subprocess, importlib.metadata as md
import pandas as pd

t0 = time.time()
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
PROJ = '/content/drive/MyDrive/classification_MI-ME'
PIPE = next((p for p in (f'{PROJ}/30_v06_pipeline', f'{PROJ}/30_v06_pipline') if os.path.isdir(p)), None)
RAW = f'{PROJ}/01_data_inventory/raw_ofner_data/MNE-upperlimb-data/zenodo/834976'
CHECK = f'{PROJ}/60_repo_check'
WORK = '/content/repo_verification_tmp'
LOGF = f'{REPO}/verification/VERIFY_LOG.txt'
assert PIPE, f'project folder not found under {PROJ}'
os.makedirs(f'{REPO}/verification', exist_ok=True); os.makedirs(WORK, exist_ok=True)
LOG = open(LOGF, 'w', encoding='utf-8')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
say(f'Repository verification, {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}')

# ---------------- 1) scripts and result files ----------------
SKIP_DIRS = {'01_preprocessed', '00_inherited', '13_P2_figures', '15_P2_tables'}
SKIP_PY = {'make_tables.py'}
DATA_EXT = {'.csv', '.json', '.md', '.txt', '.log', '.tsv'}
MAXB = 25 * 1024 ** 2
n_py = n_res = 0; skipped = {}
for d in sorted(os.listdir(PIPE)):
    src = f'{PIPE}/{d}'
    if not os.path.isdir(src) or d in SKIP_DIRS or d.startswith(('.', '_')):
        continue
    for root, _, fs in os.walk(src):
        for f in sorted(fs):
            p = os.path.join(root, f); rel = os.path.relpath(p, src); ext = os.path.splitext(f)[1].lower()
            b = os.path.getsize(p)
            if ext == '.py' and f not in SKIP_PY:
                dst = f'{REPO}/code/03_analysis/{d}/{rel}'
                if os.path.exists(dst):
                    continue                                   # a repository file of the same name is kept
                n_py += 1
            elif ext in DATA_EXT and b <= MAXB:
                dst = f'{REPO}/results/{d}/{rel}'; n_res += 1
            else:
                key = f'{ext or "(none)"}' + (' > 25 MB' if ext in DATA_EXT else '')
                skipped[key] = skipped.get(key, 0) + 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(p, dst)
say(f'\n[1] copied {n_py} analysis scripts and {n_res} result files from {PIPE}')
say('    not copied (by type):', ', '.join(f'{k}: {v}' for k, v in sorted(skipped.items())) or 'none')
say('    analysis scripts:', ', '.join(sorted(os.path.relpath(p, f'{REPO}/code/03_analysis')
                                          for p in glob.glob(f'{REPO}/code/03_analysis/*/**/*.py', recursive=True))))
if not glob.glob(f'{REPO}/results/00_data_audit/*checksum*'):
    say('    NOTE: no checksum file in results/00_data_audit (the README mentions one): send me this line')

# evidence of the full rebuild of 15 Nov 2026 (all 30 sessions / 15 people), if it was saved
for sub in ('02_sanity',):
    for p in glob.glob(f'{CHECK}/{sub}/*.csv'):
        os.makedirs(f'{REPO}/verification/sanity_rebuild_15nov2026', exist_ok=True)
        shutil.copy2(p, f'{REPO}/verification/sanity_rebuild_15nov2026/')

# ---------------- 2) registry and channel order ----------------
OLD = f'{PIPE}/01_preprocessed'
RG = f'{REPO}/results/registry'; os.makedirs(RG, exist_ok=True)
RT, RB = f'{CHECK}/registry_trials.csv', f'{CHECK}/registry_bad_channels.csv'
if os.path.exists(RT) and os.path.exists(RB):
    shutil.copy2(RT, RG); shutil.copy2(RB, RG); how = 'copied from 60_repo_check'
else:
    rt, rb = [], []
    for c in ('ME', 'MI'):
        for s in range(1, 16):
            E = pd.read_csv(f'{OLD}/{c}_sub{s:02d}_events.csv')
            rt.append(E[['condition', 'subject', 'run', 'trial', 'class', 'r4m_rejected', 'onset_latency_s']])
            m = json.load(open(f'{OLD}/{c}_sub{s:02d}_meta.json'))
            rb.append({'condition': c, 'subject': s, 'bad_channels': ';'.join(m['bad_channels'])})
    pd.concat(rt, ignore_index=True).to_csv(f'{RG}/registry_trials.csv', index=False)
    pd.DataFrame(rb).to_csv(f'{RG}/registry_bad_channels.csv', index=False); how = 'built from the saved events and meta files'
json.dump(json.load(open(f'{OLD}/ME_sub02_meta.json'))['channels'], open(f'{RG}/channels.json', 'w'), indent=0)
REG = pd.read_csv(f'{RG}/registry_trials.csv')
say(f'\n[2] registry ({how}): {len(REG)} trials, {int(REG.r4m_rejected.sum())} rejected; '
    f'{len(json.load(open(f"{RG}/channels.json")))} channels')
open(f'{RG}/README.md', 'w').write(
    '# Frozen registry\n`registry_trials.csv`: one row per cued trial (condition, subject, run 1-10, trial 1-42, class), the '
    'artifact flag used for training data (`r4m_rejected`) and the glove onset latency of executed movements '
    '(`onset_latency_s`, s after the cue). `registry_bad_channels.csv`: bad channels per session. `channels.json`: channel '
    'order of the preprocessed files (61 EEG + 3 EOG). Built before this study (Section 2.1 of the paper) and used unchanged.\n')

# ---------------- privacy check of the copied text files ----------------
hits = []
for p in glob.glob(f'{REPO}/**/*', recursive=True):
    if (os.path.isfile(p) and os.path.splitext(p)[1].lower() in DATA_EXT | {'.py'} and os.path.getsize(p) < 5e6
            and os.path.abspath(p) != os.path.abspath(__file__)):
        t = open(p, encoding='utf-8', errors='ignore').read()
        for pat in (r'[\w.+-]+@[\w-]+\.[\w.]+', r'(?i)paper ?3|50_paper3', r'(?i)api[_-]?key|token\s*=|password'):
            for m_ in set(re.findall(pat, t)):
                hits.append(f'{os.path.relpath(p, REPO)}: {m_}')
say(f'\n[privacy] {len(hits)} e-mail address(es), Paper 3 mention(s) or key(s) found in the repository files'
    + ('' if not hits else ' (send me these lines):\n    ' + '\n    '.join(sorted(hits)[:40])))

# ---------------- 3) checks ----------------
def run(title, args, env=None):
    say(f'\n[3] {title}\n    $ python {" ".join(os.path.relpath(a, REPO) if a.startswith(REPO) else a for a in args)}')
    e = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1', **(env or {}))
    pr = subprocess.Popen([sys.executable] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e, cwd=REPO)
    out = []
    for line in pr.stdout:
        out.append(line.rstrip()); print('    ' + line.rstrip(), flush=True)
    pr.wait()
    LOG.write('\n'.join('    ' + l for l in out) + '\n'); LOG.flush()
    say(f'    exit code {pr.returncode} | {(time.time() - t0) / 60:.1f} min')
    return out

try:
    import mne  # noqa
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mne'])
C = f'{REPO}/code'
run('preprocessing, subject 1 (both sessions), against the files used for the paper',
    [f'{C}/01_preprocessing/preprocess_ofner.py', '--raw', RAW, '--out', f'{WORK}/preprocessed', '--subjects', '1',
     '--check', OLD])
run('sanity check, subject 1, against the saved results of the first run',
    [f'{C}/02_sanity_check/sanity_check_ofner2017.py', '--data', f'{WORK}/preprocessed', '--out', f'{WORK}/sanity',
     '--subjects', '1', '--check', f'{REPO}/results/02_A2_sanity'])
run('same-budget comparison, recomputed from N5_curves.csv and compared with the saved N5_same_budget.csv',
    [f'{C}/03_analysis/15_R1_followup/n5_same_budget.py', '--dir', f'{REPO}/results/15_R1_followup',
     '--out', f'{REPO}/verification/N5_same_budget_recomputed.csv',
     '--check', f'{REPO}/results/15_R1_followup/N5_same_budget.csv'])
out = run('figures and tables from results/ (full run)', [f'{C}/figures_tables/make_figures_tables.py'],
          env=dict(P2_RESULTS=f'{REPO}/results', P2_FIG_OUT=f'{REPO}/outputs/figures', P2_TAB_OUT=f'{REPO}/outputs/tables'))

# figure previews for the README
os.makedirs(f'{REPO}/docs/figures', exist_ok=True)
for p in glob.glob(f'{REPO}/outputs/figures/*_preview.png'):
    shutil.copy2(p, f'{REPO}/docs/figures/')

# ---------------- 4) requirements, manifest, zip ----------------
DIST = {'sklearn': 'scikit-learn', 'PIL': 'pillow', 'mne': 'mne', 'torch': 'torch', 'numpy': 'numpy', 'scipy': 'scipy',
        'pandas': 'pandas', 'matplotlib': 'matplotlib'}
import ast
LOCAL = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f'{REPO}/**/*.py', recursive=True)}
STD = set(sys.stdlib_module_names) | {'google', '__future__'} | LOCAL
mods = set()
for p in glob.glob(f'{C}/**/*.py', recursive=True):
    try:
        tree = ast.parse(open(p, encoding='utf-8', errors='ignore').read())
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module and n.level == 0]
    except SyntaxError:                                   # notebook-style lines (!pip ...): fall back to a regex
        names = re.findall(r'^\s*(?:import|from)\s+([A-Za-z_][\w.]*)', open(p, encoding='utf-8', errors='ignore').read(), re.M)
    mods |= {x.split('.')[0] for x in names if x.split('.')[0] not in STD}
pkgs = md.packages_distributions(); req = []
for m_ in sorted(mods):
    for dname in ([DIST[m_]] if m_ in DIST else pkgs.get(m_, [])):
        try:
            req.append(f'{dname}=={md.version(dname)}'); break
        except md.PackageNotFoundError:
            pass
    else:
        req.append(f'# {m_}: not installed in this runtime')
open(f'{REPO}/requirements.txt', 'w').write(
    f'# Packages imported by the code, with the versions of the Google Colab runtime used for the verification run\n'
    f'# (Python {sys.version.split()[0]}, {time.strftime("%Y-%m-%d")}). The final analysis run used Python 3.13, NumPy 2.4,\n'
    f'# SciPy 1.16, scikit-learn 1.6 and PyTorch 2.14 (paper, Section 2.12).\n' + '\n'.join(sorted(set(req))) + '\n')
say('\n[4] requirements.txt:', ', '.join(sorted(set(req))))

LOG.close()
rows = []
for root, dirs, fs in os.walk(REPO):
    dirs[:] = sorted(x for x in dirs if x != '__pycache__' and not (root == REPO and x in ('outputs', 'data')))
    for f in sorted(fs):
        p = os.path.join(root, f); r = os.path.relpath(p, REPO)
        if r != 'MANIFEST.csv' and not f.endswith('.pyc'):
            rows.append(dict(path=r, bytes=os.path.getsize(p), sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest()))
pd.DataFrame(rows).to_csv(f'{REPO}/MANIFEST.csv', index=False)
ZIP = '/content/paper2_repo.zip'
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for r in rows + [dict(path='MANIFEST.csv')]:
        z.write(f'{REPO}/{r["path"]}', f'paper2_repo/{r["path"]}')
os.makedirs(CHECK, exist_ok=True); shutil.copy2(ZIP, f'{CHECK}/paper2_repo.zip')

# ---------------- summary ----------------
log = open(LOGF, encoding='utf-8').read()
pick = lambda pat: [l.strip() for l in log.split('\n') if re.search(pat, l)]
print('\n================ SUMMARY (send me these lines) ================')
for l in (pick(r'sessions identical') + pick(r'leave-one-run-out identical|6 movements identical|movement vs rest: mean')
          + pick(r'rows matched') + pick(r'checks OK') + pick(r'conclusion\(s\) change') + pick(r'FAILED')
          + pick(r'\[privacy\]') + [l.strip() for l in log.split('\n') if l.strip() in set(hits)] + pick(r'NOTE:')):
    print('  ' + l)
print(f'  repository: {len(rows) + 1} files, {sum(r["bytes"] for r in rows) / 1e6:.1f} MB -> {ZIP} '
      f'(copy in 60_repo_check) | {(time.time() - t0) / 60:.1f} min')
