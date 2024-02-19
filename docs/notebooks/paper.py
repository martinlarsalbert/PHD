import ipynbname
import os.path
from matplotlib.figure import Figure
import matplotlib as plt
import sympy as sp
import numpy as np
import pandas as pd
from phd.paper.equation import to_latex
from IPython.display import display, Latex, Math, HTML

from phd.paper.equation import to_latex

import logging

log = logging.getLogger(__name__)


# plt.style.use("paper")
import arviz as az

# az.style.use("arviz-grayscale")
az.style.use("arviz-white")
plt.rcParams["figure.dpi"] = 150
textsize = 9
plt.rcParams["axes.labelsize"] = textsize
plt.rcParams["axes.titlesize"] = textsize
plt.rcParams["legend.fontsize"] = textsize
plt.rcParams["xtick.labelsize"] = textsize
plt.rcParams["ytick.labelsize"] = textsize

latex_textheight = 7.59108
latex_textwidth = 5.39749
plt.rcParams["figure.figsize"] = (latex_textwidth,0.4*latex_textheight)
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['lines.markersize'] = 4
plt.rcParams['grid.linewidth'] = 0.1

#plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

paper_path = r"/home/maa/dev/PHD/docs/System-identification-for-a-physically-correct-ship-manoeuvring-model-in-wind-conditions"


def file_name_with_nb_ref(file_name: str) -> str:
    """It is good to keep a reference to the notebook that created a figure.
    This method creates one like: file_name_with_nb_ref = {notebook_name}.{file_name}

    Parameters
    ----------
    file_name : str
        _description_

    Returns
    -------
    str
        file_name_with_nb_ref
    """
    nb_fname = ipynbname.name()
    return f"{nb_fname}.{file_name}"


def file_path_with_nb_ref(file_name: str, directory="figures") -> str:
    directory_path = os.path.join(paper_path, directory)

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    return os.path.join(directory_path, file_name_with_nb_ref(file_name=file_name))


def save_fig(fig: Figure, file_name: str):
    file_path = file_path_with_nb_ref(file_name=file_name, directory="figures")
    fig.savefig(file_path)

def scale_figure(fig,scale=1):
    size = np.array(fig.get_size_inches())
    new_size=scale*size
    fig.set_size_inches(new_size)

def save_eq(eq: sp.Eq, file_name: str = None, subs={}, replace_latex={}):
    display(HTML(file_name_with_nb_ref(file_name=file_name)))
    eq_latex = to_latex(eq=eq, subs=subs)
    
    for old,replace in replace_latex.items():
        eq_latex = eq_latex.replace(old,replace)
    
    display(Math(eq_latex))
    display(HTML("<br>"))

    if file_name is None:
        return eq_latex

    file_name_ext = f"{file_name}.tex"
    path = file_path_with_nb_ref(file_name=file_name_ext, directory="equations")

    dir_path = os.path.split(path)[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        # print(f"Makedir:{dir_path}")

    # print(f"Writing:{path}")
    with open(path, mode="w") as file:
        file.write(eq_latex)

def table_columns(df,columns=2):
    df = df.copy()
    df.reset_index(inplace=True, drop=True)
    N = len(df)
    n = int(np.ceil(N/columns))

    i=0
    table = df[i*n:(i+1)*n].reset_index(drop=True)
    for i in range(1,columns):
        new = df[i*n:(i+1)*n].reset_index(drop=True)
        table = pd.merge(left=table, right=new, how='left', left_index=True,right_index=True, suffixes=('',str(i)))
    
    return table