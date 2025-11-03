import numpy as np
import pandas as pd
import statsmodels.api as sm
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary
from sklearn.preprocessing import StandardScaler


def do_fit(X_selected, X_scaled_selected, y_selected, threshold=0.00005, good_columns=[]):
    
    if len(good_columns) == 0:
        optimizer = ps.STLSQ(threshold=threshold)
        optimizer.fit(X_scaled_selected,y_selected)

        mask = np.abs(optimizer.coef_)[0] > 0
        assert np.sum(mask) > 0, f"Did not find any coefficients among:{X_selected.columns}"
        good_columns = X_selected.columns[mask]
    
    X_good = X_selected[good_columns].copy()

    fit = sm.OLS(y_selected,X_good).fit()

    return fit

def do_fit_pipeline(fit_pipeline:dict, X:pd.DataFrame, X_scaled:pd.DataFrame, y=pd.Series, include_bias=False):
    X = X.copy()
    X_scaled=X_scaled.copy()
    y=y.copy()
    
    derivatives = {}
    fits = {}

    for name, meta_data in fit_pipeline.items():

        mask = meta_data['mask']

        X_selected = X.loc[mask]
        X_scaled_selected = X_scaled.loc[mask]
        y_selected = y[mask]

        threshold = meta_data.get('threshold',0.00005)
        good_columns = meta_data.get('good_columns',[])
        try:
            fit = do_fit(X_selected=X_selected, X_scaled_selected=X_scaled_selected, y_selected=y_selected, threshold=threshold, good_columns=good_columns)
        except AssertionError:
            raise AssertionError(f"Failed on:{name}")
        
        fits[name] = fit

        parameters = fit.params.copy()
        derivatives.update(parameters)
        columns = list(parameters.keys())

        X_ = X[columns].copy()
        
        if not include_bias:
            if '1' in X_:
                X_['1'] = 0

        y_pred = fit.predict(X_)
        y-=y_pred

        # Remove the regressed parameters from the parameter library
        columns_remove = columns.copy()
        if '1' in columns_remove:
            columns_remove.remove('1')  # Don't remove '1'
        
        X.drop(columns=columns_remove, inplace=True)
        X_scaled.drop(columns=columns_remove, inplace=True)
        
    return derivatives, fits