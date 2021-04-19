###################
### OLS fitting ###
###################

'''
OLS fitting functions 
'''

import numpy as np
import statsmodels.api as sm

def ols_fit(x,y):
    '''
    Calculate OLS fit of a line aking use of statsmodels.api
    
    Parameters:
    ----------
        x: exog of line
        y: endog of line
    
    Returns:
    -------
        m: slope of OLS line
        b: intercept of OLS line
    '''
    
    # fit model
    model = np.array([x]).T
    model = sm.add_constant(model)
    model_fit = sm.OLS(y,model).fit()
    mb = np.asarray([model_fit.params[1], model_fit.params[0]])
    err = np.asarray(model_fit.bse[::-1])    
    return mb, err