import ipynbname
import os.path
from matplotlib.figure import Figure
import matplotlib as plt
import sympy as sp
from phd.paper.equation import to_latex
from IPython.display import display, Latex, Math, HTML

from phd.paper.equation import to_latex

import logging
log = logging.getLogger(__name__)


# plt.style.use("paper")
import arviz as az

az.style.use("arviz-grayscale")
plt.rcParams["figure.dpi"] = 300

paper_path = r'/home/maa/dev/PHD/docs/System-identification-for-a-physically-correct-ship-manoeuvring-model-in-wind-conditions'


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


def save_eq(eq: sp.Eq, file_name:str=None, subs={}):
    display(HTML(file_name_with_nb_ref(file_name=file_name)))
    eq_latex = to_latex(eq=eq, subs=subs)
    display(Math(eq_latex))
    display(HTML('<br>'))
    
    
    if file_name is None:
        return eq_latex
    
    file_name_ext = f"{file_name}.tex"
    path = file_path_with_nb_ref(file_name=file_name_ext, directory='equations')
    
    dir_path = os.path.split(path)[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        #print(f"Makedir:{dir_path}")
    
    #print(f"Writing:{path}")
    with open(path, mode='w') as file:
        file.write(eq_latex)    
    
        
