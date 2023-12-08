import pytest
from phd.pipelines.models.models_wPCC import ModelSemiempiricalCovered, MainModel
#import src
import os.path
import yaml
import dill
import sympy as sp

import logging
log = logging.getLogger()

@pytest.fixture
def ship_data():
    #path_tests = os.path.join(src.__path__,'tests')
    log.info("Load shipdata")
    path_tests=os.path.dirname(__file__)   
    print(path_tests)
    path_ship_data = os.path.join(path_tests,"ship_data.yml")
    with open(path_ship_data,mode='r') as file:    
        ship_data = yaml.safe_load(file)
    yield ship_data
    
@pytest.fixture
def model(ship_data):
    log.info("Creating a model")
    model = ModelSemiempiricalCovered(ship_data=ship_data, create_jacobians=True)
    yield model

def test_pickle_model(ship_data,tmp_path):
    
    #model = ModelSemiempiricalCovered(ship_data=ship_data, create_jacobians=False)
    model = MainModel(create_jacobians=False)
    file_path = str(tmp_path / 'model.pkl')
    model.save(path="model.pkl")
    
    #model2 = model.load(path = file_path)
    
    
class A():
    
    def __getstate__(self):
        
        return self.__dict__
    
    def __setstate__(self,d):
        
        self.__dict__ = d
    
def test_pickle(tmp_path):
    
    a = A()
    
    file_path = tmp_path / 'test.yml'
    with open(file_path, mode="wb") as file:
        dill.dump(a, file=file)

    with open(file_path, mode="rb") as file:
        a2 = dill.load(file=file)
        
        
def test_pickle_function():
    
    x = sp.symbols("x")
    
    dill.dumps(x)
    
    f = sp.Function("f")(x)
    f.__class__.__module__ = None
        
    dill.dumps(f)
    
    