###################
### MLE Fitting ###
###################

'''
Functions to do MLE 
'''

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binned_statistic

def mad(dat):
    '''
    Calculate the median absolute deviation (MAD)
    
    Parameters:
    ----------
        dat: array-like object containing data
    
    Returns:
    -------
        dev: MAD(dat)
    '''
    dev = np.median(np.absolute(dat - np.median(dat)))
    return dev

def linmodl(m,b,x):
    '''
    Slope-intercept form of a line
    
    Parameters:
    ----------
        m: slope of line
        x: exog of line
        b: intercept of line
        
    Returns:
    -------
        y: endog of line
    '''
    y = m*x + b
    return y

def lnL(theta,x,y,yerr):
    '''
    Log likelihood for linmodl
    
    Parameters:
    ----------
        theta: parameters to plug into linmodl (m,b)
        x: exog of line
        y: endog of line
        yerr: endog error
    
    Returns:
    -------
        lnl: log likelihood 
    '''
    
    m, b = theta
    modl = linmodl(m,b,x)
    inv_sig2 = np.reciprocal(np.square(yerr))
    lnl = -0.5 * np.sum(np.multiply(np.square((y - modl)),inv_sig2) - np.log(inv_sig2/(2*np.pi)))
    return lnl

def mle_fit(exog,endog,endog_err):
    '''
    Do a MLE fit of median abundances as function (y = mx + b) of radius
    using unbinned data
    
    Parameters:
    ----------
        exog: radii for stars
        endog: abundances for stars
        endog_err: errors in abund
    Returns:
    -------
        med_m: slope of line fit
        med_b: intercept of line fit
    '''
    
    #Calculate bins
    bins = np.arange(np.floor(np.min(exog)),np.ceil(np.max(exog)),1.0) 
    
    #Calcuradius[cln]late median value for each bin
    bin_stats, _, _ = binned_statistic(exog,endog,statistic='median',bins=bins)
    
    #Calculate spread (MAD) in values in each bin
    bin_stats_err, _, _ = binned_statistic(exog,endog,statistic=lambda y: np.median(np.absolute(y-np.median(y))),
                                           bins=bins)
    
    #Initialize MLE calculation
    med_exog = np.arange(len(bin_stats))+0.5
    med_endog = bin_stats
    med_endog_err = bin_stats_err
    
    med_m_guess = (bin_stats[1]-bin_stats[0])/(med_exog[1]-med_exog[0])
    med_b_guess = bin_stats[0]
    
    # minimize MLE and find slopes and intercepts
    nll = lambda *args: -lnL(*args)
    med_guess = np.array([med_m_guess, med_b_guess])
    med_soln = minimize(nll, med_guess, args=(med_exog, med_endog, med_endog_err))
    med_m_ml, med_b_ml = med_soln.x
    
    return med_m_ml, med_b_ml