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
### Other Fucntions  ###
########################

def closest(val,dat):
    '''
    find value closest to the given one
    
    Inputs:
    ------
        val: given value to find/get closest to
        dat: vals to search through
    
    Output:
    ------
        value in dat closest to val
    '''
    val = np.asarray(val)
    dat = np.asarray(dat)
    
    abs_diff = np.abs(dat - val)
    return dat[abs_diff.argmin()]

def parsec_teff_2_jk(teff,feh,age,isochrones):
    '''
    Calculate the intrinsic J - K from a star's given temperture and metallicity using PARSEC isochrones
    
    Inputs:
    ------
        teff: temperature of star
        feh: metallicity of star
        age: age in Gyr to use 
        isochrones: set of parsec isochrones
    Output:
    ------
        jk: expected J - K for the given temperature
    '''
    
    single = isochrones[np.where((isochrones['logAge']==closest(np.log10(age*10**9),isochrones['logAge']))&
                                 (isochrones['MH']==closest(feh,isochrones['MH'])))]
    
    sidx = np.argsort(single['logTe'])
    slogTe = single['logTe'][sidx]
    sjk = (single['Jmag']-single['Ksmag'])[sidx]
    
    spl = InterpolatedUnivariateSpline(slogTe[::2],sjk[::2])
    
    jk = spl(np.log10(teff))
    return jk

def parsec_teff2color(teff,feh,blue,red,age,isochrones):
    '''
    Calculate the intrinsic color from a star's given temperture and metallicity using PARSEC isochrones
    
    Inputs:
    ------
        teff: temperature of star
        feh: metallicity of star
        blue: [str] bluer magnitude
        red: [str] redder magnitude
        age: age in Gyr to use 
        isochrones: set of parsec isochrones
    Output:
    ------
        blue_red: expected intrinsic color for the given temperature
    '''
    
    single = isochrones[np.where((isochrones['logAge']==closest(np.log10(age*10**9),isochrones['logAge']))&
                                 (isochrones['MH']==closest(feh,isochrones['MH'])))]
    
    sidx = np.argsort(single['logTe'])
    slogTe = single['logTe'][sidx]
    scolor = (single[blue]-single[red])[sidx]
    
    spl = InterpolatedUnivariateSpline(slogTe[::2],scolor[::2])
    
    blue_red = spl(np.log10(teff))
    return blue_red

def ccline(x,cer,x0,y0):
    '''
    Calculate a line to trace back to the stellar locus in color-color space. The intersection is where the 
    unreddened star should lie on the stellar locus.
    
    Inputs:
    ------
        x: values to plug in [1 - 2]
        cer: color excess ratio [E(3 - 4)/E(1 - 2)]
        x0: color of some point [1 - 2]_0
        y0: color of some point [3 - 4]_0
    '''
    return np.add(np.multiply(cer,x),np.subtract(y0,np.multiply(cer,x0)))

def parsec_locus_jhjk(x0,y0,cer,feh,age,isochrones):
    '''
    Find the where a star would be located on the stellar locus in (J - K)-(J - H) color space.
    (J - K) is the abscissa and (J - H) is the ordinate.
    
    Inputs:
    ------
        x0: J - K color of star
        y0: J - H color of star
        cer: color excess ratio [E(J - H)/E(J - K)]
        feh: metallicity of star
        age: age in Gyr
        isochrones: set of parsec isochrones
    
    Outputs:
    -------
        jk_cross: J - K of intersection point
        jh_cross: J - H of intersection point
    '''
    single = isochrones[np.where((isochrones['logAge']==closest(np.log10(age*10**9),isochrones['logAge']))&
                             (isochrones['MH']==closest(feh,isochrones['MH'])))]
    xs = (single['Jmag'] - single['Ksmag'])
    ys = (single['Jmag'] - single['Hmag'])
    
    bins = np.arange(np.min(xs),np.max(xs),0.005)
    binned_color = binned_statistic(xs, ys, statistic='median', bins=bins).statistic
    
    fin = np.where((np.isfinite(bins[:-1])==True)&(np.isfinite(binned_color)==True))
    
    bins = bins[:-1][fin]
    binned_color = binned_color[fin]
    
    # Interpolate Isochrone
    spline = InterpolatedUnivariateSpline(bins,binned_color,ext=0)
    y_spl = spline(bins)
    
    # Find the crossing point
    if y0 - spline(x0)>0:
        # what to do if star is above isochrone in color-color space
        return np.squeeze([-9999.0,-9999.0])

    else:
        # Find roots in difference between the ccline and isochrone
        func = y_spl - ccline(bins,cer,x0,y0)
        func_spl = InterpolatedUnivariateSpline(bins,func,ext=0)
        jk_cross = func_spl.roots()
        jh_cross = spline(jk_cross)
        
        if len(jk_cross)==0:
            # If cross12 is empty 
            return np.squeeze([-9999.0,-9999.0])
        
        else:
            # Return Cartesian coordinates of crossing point in color-color space
            return np.squeeze(list(zip(jk_cross, jh_cross)))
        
def parsec_locus(x0,y0,mag1,mag2,mag3,mag4,cer,feh,age,parsec):
    '''
    Find the where a star would be located on the stellar locus in (1 - 2)-(3 - 4) color space.
    (1 - 2) is the abscissa and (3 - 4) is the ordinate.
    
    Inputs:
    ------
        x0: (1 - 2) color of star off locus
        y0: (3 - 4) color of star off locus
        mag1: [str] first magnitude in (1 - 2) color
        mag2: [str] second magnitude in (1 - 2) color
        mag3: [str] first magnitude in (3 - 4) color
        mag4: [str] second magnitude in (3 - 4) color
        cer: color excess ratio [E(3 - 4)/E(1 - 2)]
        feh: metallicity of star
        age: age in Gyr
        parsec: set of parsec isochrones
    
    Outputs:
    -------
        cross12: (1 - 2) of intersection point
        cross34: (3 - 4) of intersection point
    '''
    
    single = parsec[np.where((parsec['logAge']==closest(np.log10(age*10**9),parsec['logAge']))&
                             (parsec['MH']==closest(feh,parsec['MH'])))]
    xs = (single[mag1] - single[mag2])
    ys = (single[mag3] - single[mag4])
    
    bins = np.arange(np.min(xs),np.max(xs),0.005)
    binned_color = binned_statistic(xs, ys, statistic='median', bins=bins).statistic
    
    fin = np.where((np.isfinite(bins[:-1])==True)&(np.isfinite(binned_color)==True))
    
    bins = bins[:-1][fin]
    binned_color = binned_color[fin]
    
    # Interpolate Isochrone
    spline = InterpolatedUnivariateSpline(bins,binned_color,ext=0)
    y_spl = spline(bins)
    
    # Find the crossing point
    if y0 - spline(x0)>0:
        # what to do if star is above isochrone in color-color space
        return np.squeeze([-9999.0,-9999.0])

    else:
        # Find roots in difference between the ccline and isochrone
        func = y_spl - ccline(bins,cer,x0,y0)
        func_spl = InterpolatedUnivariateSpline(bins,func,ext=0)
        cross12 = func_spl.roots()
        cross34 = spline(cross12)
        
        if len(cross12)==0:
            # If cross12 is empty 
            return np.squeeze([-9999.0,-9999.0])
        
        else:
            # Return Cartesian coordinates of crossing point in color-color space
            return np.squeeze(list(zip(cross12, cross34)))