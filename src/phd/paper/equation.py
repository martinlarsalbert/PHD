from sympy import latex
import sympy as sp
import sympy.physics.mechanics as me
from sympy import Eq
from vessel_manoeuvring_models.substitute_dynamic_symbols import eq_dottify

def to_latex_equation(eq:sp.Eq, label:str)-> str:
    eq_latex = to_latex(eq)
    eq_label=f"eq:{label}"
    s_latex = "\\begin{equation}\n" + "\\label{" + eq_label + "}\n" + eq_latex + "\n \\end{equation}\n"
    
    return s_latex


standard_substitutions={
    #'alpha':r'\alpha',
    #'alpha_fport':r'\alpha_f_port',
    #'delta':r'\delta',
    'infty':r'\infty',
    'thrust':'T',
    
}

def to_latex(eq:sp.Eq, do_standard_substitutions=True, subs={})-> str:
    eq_latex = latex(eq)    
    
    eq_latex_subs = str(eq_latex)
    if do_standard_substitutions:
        substitutions = standard_substitutions.copy()
        substitutions.update(subs)
        
        for old,new in substitutions.items():
            eq_latex_subs=eq_latex_subs.replace(old,new)
    
    return eq_latex_subs

