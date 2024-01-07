from sympy import latex
import sympy as sp

def to_latex_equation(eq:sp.Eq, label:str)-> str:
    eq_latex = to_latex(eq)
    eq_label=f"eq:{label}"
    s_latex = "\\begin{equation}\n" + "\\label{" + eq_label + "}\n" + eq_latex + "\n \\end{equation}\n"
    
    return s_latex

def to_latex(eq:sp.Eq)-> str:
    eq_latex = latex(eq)    
    return eq_latex