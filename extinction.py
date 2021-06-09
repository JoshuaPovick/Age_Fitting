import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import binned_statistic

# Effective Wavelengths of different passbands
leff = {'BP':0.5387,'G':0.6419,'RP':0.7667,'J':1.2345,'H':1.6393,'K':2.1757} # mircons

############################
### Cardelli et al. 1989 ###
############################

def cardelli_a(x):
    '''
    a(x) function from Cardelli et al. 1989
    
    Input:
    -----
        x: effective wavelength in units of 1/micron
        
    Output:
    ------
        a: a function value  
    '''
    if 0.3 <= x < 1.1:
        a = 0.574*(x**1.61)
        return a
    
    elif 1.1 <= x < 3.3:
        y = x - 1.82
        a = (1.+0.17699*y-0.50477*(y**2)-0.02427*(y**3)+0.72085*(y**4)+
             0.01979*(y**5)-0.77530*(y**6)+0.32999*(y**7))
        return a
    
    elif 3.3 <= x < 8.0:
        if x < 5.9:
            a = 1.752-0.136*x-0.104/((x-4.67)**2+0.341)
            return a
        
        else:
            fa = -0.04473*((x-5.9)**2)+0.1207*((x-5.9)**3)
            a = 1.752-0.136*x-0.104/((x-4.67)**2+0.341)+fa
            return a       
    
def cardelli_b(x):
    '''
    b(x) function from Cardelli et al. 1989
    
    Input:
    -----
        x: effective wavelength in units of 1/micron
        
    Output:
    ------
        b: b function value 
    '''
    if 0.3 <= x < 1.1:
        b = -0.527*(x**1.61)
        return b
    
    elif 1.1 <= x <= 3.3:
        y = x - 1.82
        b = (1.41338*y+2.28305*(y**2)+1.07233*(y**3)-5.38434*(y**4)-
                0.62251*(y**5)+5.30260*(y**6)-2.09002*(y**7))
        return b
    
    elif 3.3 <= x < 8.0:
        if x < 5.9:
            b = -3.090+1.825*x+1.206/((x-4.62)**2+0.263)
            return b
        
        else:
            fb = 0.2130*((x-5.9)**2)+0.1207*((x-5.9)**3)
            b = -3.090+1.825*x+1.206/((x-4.62)**2+0.263)+fb
            return b
    
def cardelli_alav(wave,rv):
    '''
    Calculate A\lambda/Av
    
    Inputs:
    ------
        wave: effective wavelength in units of micron
        rv: Rv value (=Av/E(B_V))
        
    Output:
    ------
        alav: A\lambda/Av
    '''
    x=1/wave
    alav = cardelli_a(x)+cardelli_b(x)/rv
    return alav

def cardelli_alebv(wave,rv):
    '''
    Calculate relative extinction to E(B-V)
    
    Inputs:
    ------
        wave: effective wavelength in units of micron
        rv: Rv value (=Av/E(B_V))
        
    Output:
    ------
        alebv: A\lambda/E(B-V)
    '''
    x=1/wave
    alebv = cardelli_a(x)*rv+cardelli_b(x)/rv
    return alebv

def cardelli_e12ebv(wave_blue,wave_red,rv):
    '''
    Calculate E(1-2)/E(B-V)
    
    Inputs:
    ------
        wave_blue: effective wavelength in units of micron
        wave_red: effective wavelength in units of micron
        rv: Rv value (=Av/E(B_V))
    
    Output:
    ------
        e12ebv: E(1-2)/E(B-V)
    '''
    xblue=1/wave_blue
    xred=1/wave_red
    a2a1rv = (cardelli_a(xblue) - cardelli_a(xred))*rv
    b2b1 = cardelli_b(xblue) - cardelli_b(xred)
    e12ebv = a2a1rv + b2b1
    return e12ebv
    
def cardelli_e12av(wave_blue,wave_red,rv):
    '''
    Calculate E(1-2)/Av
    
    Inputs:
    ------
        wave_blue: effective wavelength of bluer band in units of micron
        wave_red: effective wavelength of redder band in units of micron
        rv: Rv value (=Av/E(B_V))
    
    Output:
    ------
        e12av: E(1-2)/Av
    '''
    xblue=1/wave_blue
    xred=1/wave_red
    a2a1 = cardelli_a(xblue) - cardelli_a(xred)
    b2b1rv = (cardelli_b(xblue) - cardelli_b(xred))/rv
    e12av = a2a1 + b2b1rv
    return e12av

##############################
### Indebetouw et al. 2005 ###
##############################

def ind_alak(wave):
    '''
    Calculate the relative extinction to Ak using Indebetouw et al. 2005.
    This uses GLIMPSE data and has only been verified in IR.
    
    Input:
    -----
        wave: effective wavelength in microns of the passband
        
    Output:
    ------
        alak: A\lambda/Ak
    '''
    # 0.61 +/- 0.04; -2.22 +/- 0.17; 1.21 +/- 0.23
    alak = 10**(0.61-2.22*np.log10(wave)+1.21*(np.log10(wave)**2))
    return alak


########################
### Wang et al. 2019 ###
########################

def wang_alav(wave):
    '''
    Calculate the relative extinction to Av accoding to Wang et al. 2019
    
    Input:
    -----
        wave: effective wavelength in microns of the passband
        
    Output:
    ------
        alav: A\lambda/Av
    '''
    
    if 0.3 <= wave < 1.0:
        y = 1/wave - 1.82
        alav = (1.0+0.7499*y-0.1086*(y**2)-0.08909*(y**3)+0.02905*(y**4)+
                0.01069*(y**5)+0.001707*(y**6)-0.001002*(y**7))
        return alav
    
    elif 1.0 <= wave < 1.0:
        #0.3722 +/- 0.0026; -2.070 +/- 0.030
        alav = 0.3722*(wave**-2.070)
        return alav