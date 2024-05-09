from sympy import latex
import sympy as sp
import sympy.physics.mechanics as me
from sympy import Eq

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

def dynamic_symbol_dot(symbol, to_variable_name=False)->sp.Symbol:
    """Convert dynamic symbol or derivate to a symbol with dots...

    Args:
        symbol (_type_): _description_

    Returns:
        sp.Symbol: sympy symbol
    """
    
    t = sp.symbols('t')
    
    if isinstance(symbol,sp.Derivative):
        name = symbol.args[0].name
        
        if symbol.args[1][0]!=t:
            return symbol

        order = symbol.args[1][1]

        return dotted_symbol(name, order, to_variable_name=to_variable_name)

    if isinstance(symbol,sp.Symbol):
        return sp.symbols(symbol.name)

    return symbol

def dotted_symbol(name:str, order:int, to_variable_name=False):

    assert order >= 0, "order cannot be negative"
    
    if order==0:
        return name
    
    if to_variable_name:
        return sp.symbols(f'{name}{order}d')
    else:
        ds = "d"*(order-1)
        return sp.symbols(fr'\{ds}dot{{{name}}}')

def search(expression, subs={}, to_variable_name=False):
    
    ## phi
    if hasattr(expression,'name'):
        if expression == me.dynamicsymbols(expression.name):
            subs[expression] = sp.symbols(expression.name)

    # phi1d,...
    if isinstance(expression, sp.Derivative):
        subs[expression]=dynamic_symbol_dot(expression, to_variable_name=to_variable_name)
    else:
        for arg in expression.args:
            search(arg, to_variable_name=to_variable_name)

    return subs

def eq_dottify(expression, to_variable_name=False)->sp.Equality:
    """Convert dynamic symbols or derivatives to dotted symbols

    Args:
        expression (_type_): _description_

    Returns:
        sp.Equality: _description_
    """
    subs=search(expression, to_variable_name=to_variable_name)
    if isinstance(expression, sp.Equality):
        return Eq(expression.lhs.subs(subs), expression.rhs.subs(subs), evaluate=False)
    else:
        return expression.subs(subs)