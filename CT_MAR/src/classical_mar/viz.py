# Shared matplotlib rcParams for figures. `params`: small, LaTeX-rendered, high-DPI
# plots sized for a paper column. `surface_params`: larger style for 3-D/surface plots.
params = {'figure.figsize': (3.487, 2.155129),
         'axes.labelsize': 8,
         'axes.titlesize':8,
         'xtick.labelsize':8,
         'ytick.labelsize':8,
         'text.latex.preamble':r"\usepackage{amsmath}",
         'legend.loc': 'best',
         'legend.shadow': False,
         'legend.fontsize': 8,
         'text.usetex' : True,
         'figure.dpi': 100,
         'savefig.dpi': 1200,
         'savefig.bbox': 'tight'}

surface_params = {'figure.figsize': (12.0, 8.0),         
         'axes.labelsize': 16,
         'axes.titlesize':16,
         'xtick.labelsize':12,
         'ytick.labelsize':12,
         #'text.latex.preamble':r"\usepackage{amsmath}",
         'legend.loc': 'best',
         'legend.shadow': False,
         'legend.fontsize': 16,
         #'text.usetex' : True,
         'figure.dpi': 100,
         'savefig.dpi': 1200,
         'savefig.bbox': 'tight'}



