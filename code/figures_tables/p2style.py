# Shared style for the Paper 2 figures (BSPC / Elsevier artwork rules). Imported by make_figures_tables.py. Version 2 (12 Nov 2026).
# Rules: text >= 7 pt (6 pt minimum for minor labels), widths 90 / 140 / 190 mm, line widths 0.1-1.5 pt, Arial (or the
# metrically identical Liberation Sans) embedded, RGB, no transparency. Export: PDF + EPS (vector), TIFF 1000 dpi (LZW), PNG preview.
# Repository version: the input and output folders are set by environment variables (defaults: the repository folders).
#   P2_RESULTS  folder with the per-person result files (default: <repo>/results)
#   P2_FIG_OUT  folder for the figures (default: <repo>/outputs/figures)
#   P2_TAB_OUT  folder for the tables (default: <repo>/outputs/tables)
import os, io, glob, subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import to_rgb
from PIL import Image

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
ROOT = os.environ.get('P2_RESULTS', os.path.join(_REPO, 'results'))
FIG = os.environ.get('P2_FIG_OUT', os.path.join(_REPO, 'outputs', 'figures'))
TAB = os.environ.get('P2_TAB_OUT', os.path.join(_REPO, 'outputs', 'tables'))
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# font: Arial if it can be installed (Debian/Ubuntu, e.g. Google Colab), otherwise Liberation Sans (same metrics);
# DejaVu Sans only as a glyph fallback
if not glob.glob('/usr/share/fonts/truetype/msttcorefonts/Arial.ttf') and os.path.exists('/usr/bin/apt-get'):
    subprocess.run('echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections; '
                   'apt-get -qq update; apt-get -qq install -y ttf-mscorefonts-installer fonts-liberation', shell=True,
                   capture_output=True)
for f in glob.glob('/usr/share/fonts/truetype/msttcorefonts/Arial*.ttf') + glob.glob('/usr/share/fonts/truetype/liberation/LiberationSans-*.ttf'):
    fm.fontManager.addfont(f)
FONT = 'Arial' if any(f.name == 'Arial' for f in fm.fontManager.ttflist) else 'Liberation Sans'
print('font used:', FONT)

MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM          # Elsevier single, 1.5 and double column widths (inches)

C = dict(ts='#2a78d6', deep='#eb6834', hybrid='#1baf7a',
         ink='#0b0b0b', ink2='#52514e', muted='#6f6e69',
         grid='#e1e0d9', axis='#9f9e96', fill='#f2f1ed', chance='#898781')

def light(color, f=0.82):
    """Colour mixed with white (opaque replacement for transparency)."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)

FS, FS_SMALL, FS_TITLE, FS_LETTER = 7, 6.5, 8, 9
mpl.rcParams.update({
    'font.family': [FONT, 'DejaVu Sans'], 'font.size': FS,
    'axes.titlesize': FS_TITLE, 'axes.labelsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS, 'legend.fontsize': FS,
    'text.color': C['ink'], 'axes.labelcolor': C['ink'],
    'axes.edgecolor': C['axis'], 'axes.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'xtick.color': C['ink2'], 'ytick.color': C['ink2'],
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.2, 'lines.markersize': 4, 'legend.frameon': False, 'patch.linewidth': 0.6,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.facecolor': 'white',
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def letter(fig, x, y, l, title=None):
    """Panel letter (bold, 9 pt) and an optional panel title (8 pt), in figure coordinates."""
    fig.text(x, y, l, fontsize=FS_LETTER, weight='bold', va='top', ha='left')
    if title:
        fig.text(x + 0.022, y - 0.003, title, fontsize=FS_TITLE, va='top', ha='left')

def save(fig, name):
    """PDF and EPS (vector), TIFF at 1000 dpi (RGB, LZW) for the journal, and a 300-dpi PNG preview for review."""
    fig.savefig(f'{FIG}/{name}.pdf')
    fig.savefig(f'{FIG}/{name}.eps')
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=1000); buf.seek(0)
    im = Image.open(buf).convert('RGB'); im.save(f'{FIG}/{name}.tif', compression='tiff_lzw', dpi=(1000, 1000))
    fig.savefig(f'{FIG}/{name}_preview.png', dpi=300)
    w, h = im.size
    print(f'saved {name}: .pdf .eps .tif ({w} x {h} px = {w / 1000 * 25.4:.0f} x {h / 1000 * 25.4:.0f} mm at 1000 dpi) and _preview.png')
