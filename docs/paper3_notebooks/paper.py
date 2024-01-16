import ipynbname
import os.path
from matplotlib.figure import Figure
import matplotlib as plt
import sympy as sp

# plt.style.use("paper")
import arviz as az

az.style.use("arviz-grayscale")
plt.rcParams["figure.dpi"] = 300

paper_path = os.path.normpath("../../docs/paper3/")


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


def save_eq(eq: sp.Eq, file_name: str, label: str = None):
    file_path = file_path_with_nb_ref(file_name=file_name, directory="equations")
    eq_latex = sp.latex(eq)

    if label is None:
        label = file_name

    latex = (
        r"""\begin{equation}
    \begin{aligned}
"""
        + eq_latex
        + r"""
    \end{aligned}
    \label{eq:"""
        + label
        + r"""}
\end{equation}"""
    )

    with open(file_path, mode="w") as file:
        file.write(latex)
