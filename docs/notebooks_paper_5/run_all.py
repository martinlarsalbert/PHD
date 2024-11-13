import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os.path

pwd = os.path.dirname(__file__)

notebook_filename = r"results_optiwise_ID.ipynb"

file_path = os.path.join(pwd,notebook_filename)
file_path_out = os.path.join(pwd,"output.ipynb")


#with open(file_path) as f:
#    nb = nbformat.read(f, as_version=4)
#    
##ep = ExecutePreprocessor(timeout=600, kernel_name='Kedro (phd)')
#ep = ExecutePreprocessor(timeout=600)
#ep.preprocess(nb, {'metadata': {'path': ''}})

import papermill as pm

pm.execute_notebook(
   file_path,
   file_path_out,
   kernel_name='Python3'
)