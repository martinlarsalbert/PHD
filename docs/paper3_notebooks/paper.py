import os.path
import ipynbname
import os.path
import sympy as sp
from IPython.display import display
import logging
log = logging.getLogger(__name__)

from phd.paper.equation import to_latex

#path = os.path.dirname(__file__)
path = r'/home/maa/dev/PHD/docs/paper'

def file_name_with_nb_ref(file_name:str)->str:
    nb_fname = ipynbname.name()
    return f"{nb_fname}-{file_name}"

def file_path_with_nb_ref(file_name:str, subfolder='')->str:
    return os.path.join(path,subfolder,file_name_with_nb_ref(file_name=file_name))

def equation(eq:sp.Eq, file_name:str=None):
    display(eq)
    eq_latex = to_latex(eq=eq)
    
    if file_name is None:
        return eq_latex
    
    path = file_path_with_nb_ref(file_name=file_name, subfolder='equations')
    
    dir_path = os.path.split(path)[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        #print(f"Makedir:{dir_path}")
    
    #print(f"Writing:{path}")
    with open(path, mode='w') as file:
        file.write(eq_latex)    
    
        
    