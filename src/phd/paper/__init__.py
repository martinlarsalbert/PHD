import ipynbname
import os.path
#path = ""  # To be overwritten

def file_name_with_nb_ref(file_name:str)->str:
    nb_fname = ipynbname.name()
    return f"{nb_fname}-{file_name}"

def file_path_with_nb_ref(file_name:str)->str:
    global path
    return os.path.join(path,file_name_with_nb_ref(file_name=file_name))
    