from sympy import latex
import sympy as sp

def to_latex_equation(eq:sp.Eq, label:str)-> str:
    eq_latex = to_latex(eq)
    eq_label=f"eq:{label}"
    s_latex = "\\begin{equation}\n" + "\\label{" + eq_label + "}\n" + eq_latex + "\n \\end{equation}\n"
    
    return s_latex


standard_substitutions={
    'delta':r'\delta',
}

def to_latex(eq:sp.Eq, do_standard_substitutions=True)-> str:
    eq_latex = latex(eq)    
    
    eq_latex_subs = str(eq_latex)
    if do_standard_substitutions:
        for old,new in standard_substitutions.items():
            eq_latex_subs=eq_latex_subs.replace(old,new)
    
    return eq_latex_subs