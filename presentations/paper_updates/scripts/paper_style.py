"""Shared matplotlib style for the camera-ready figures (NeurIPS: 5.5in text width).

Method colours were checked with the dataviz palette validator (Machado 2009
CVD simulation, OKLab dE x100), all pairs within each figure's method set:
  Fig. 5/10  {Simple FT, EWC, LoRA, FLAME-CL}: worst CVD dE 9.1, normal dE 22.9 (pass)
  new-baseline fig {FLAME-CL, TRGE, SAME, LwF}: worst CVD dE 11.1, normal dE 19.2 (pass)
  LoRA-variants fig {FLAME-CL, LoRA, LoRA enc+MoE}: worst CVD dE 8.8, normal dE 24.9 (pass)
The previous paper colours failed (EWC orange vs LoRA green: CVD dE 3.6;
EWC orange vs FLAME-CL red: normal dE 13.7). Colours keep the hue names the
captions already use (FLAME-CL red, LoRA green, Simple FT blue, EWC orange).
Every method also has its own dash pattern, so identity never rests on colour.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TEXT_W = 5.5  # inches

INK = '#1f1f1d'        # primary text
INK_2 = '#52514e'      # secondary text (axis labels, direct labels)
GRID = '#e4e3df'

METHOD_COLOR = {
    'simple_ft': '#2a78d6',
    'ewc': '#eda100',
    'lora': '#1baf7a',
    'ours': '#c8293a',
    'trge': '#8e44ad',
    'same': '#b8860b',
    'lwf': '#e87ba4',
    'lifelong_pt': '#6250d6',   # only ever shown next to FLAME-CL (Fig. 8): CVD dE 25.0
    'lora_encmoe': '#0f5f73',   # dark teal (LoRA family); only in the LoRA-variants figure
}
METHOD_LS = {
    'ours': '-',
    'simple_ft': (0, (1.2, 1.4)),
    'ewc': (0, (4, 1.6)),
    'lora': (0, (5, 1.5, 1.2, 1.5)),
    'trge': (0, (4, 1.6)),
    'same': (0, (1.2, 1.4)),
    'lwf': (0, (5, 1.5, 1.2, 1.5)),
    'lifelong_pt': (0, (4, 1.6)),
    'lora_encmoe': (0, (2.4, 1.2)),
}
TASK_MARKER = {
    'pheno': 'o', 'los': 's', 'ihm': '^', 'mortality': 'D',
    'readmission': 'v', 'density': 'P', 'birads': 'X', 'risk': '*',
}
TASK_ORDER = ['pheno', 'los', 'ihm', 'mortality', 'readmission', 'density', 'birads', 'risk']


def apply():
    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Nimbus Roman', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'legend.fontsize': 6.5,
        'legend.title_fontsize': 7,
        'axes.edgecolor': '#8a8984',
        'axes.linewidth': 0.6,
        'axes.labelcolor': INK_2,
        'xtick.color': '#8a8984',
        'ytick.color': '#8a8984',
        'xtick.labelcolor': INK_2,
        'ytick.labelcolor': INK_2,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.color': GRID,
        'grid.linewidth': 0.5,
        'text.color': INK,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
        'figure.dpi': 150,
    })
