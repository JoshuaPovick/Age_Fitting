#!/usr/bin/env python

import numpy as np
import dlnpyutils.utils as dln

def salarisfeh(feh,al):
    '''
    Calculate the Salaris corrected metallicity from Salaris et al. 1993
   
    Inputs:
    ------
    feh: [Fe/H]
    al: [alpha/Fe]
   
    Output:
    ------
    salfeh: Salaris Corrected [Fe/H]

    '''

    salfeh = feh+np.log10(0.638*(10**(al))+0.362)
    return salfeh

def orig_salariscna(abund,mc=False):
    '''
    Calculate the Salaris corrected [Fe/H] according to Salaris et al. 1993 with Piersanti et al. 2007 and 
    Asplund et al. 2009. Also C and N have been added to the alpha elements and Ne has been excluded.
        
    Inputs:
    ------
    abund: [9x2 array] first column is [Fe/H],[C/Fe],[N/Fe],[O/Fe],[Mg/Fe],[Si/Fe],[S/Fe],[Ca/Fe],[Ti/Fe] 
       and second column is the errors
        
    Output:
    ------
    calc_salfeh: Salaris corrected metallicity
    calc_salfeh_err: error in corrected metallicity
    '''
        
    ### Salaris coefficients
    # (atomic_wgts/hydrogen_wgt) = (C,N,O,Mg,Si,S,Ca,Ti)/H
    asplund = np.array([8.43,7.83,8.69,7.60,7.51,7.12,6.34,4.95])
    mass_ratio = np.array([12.011,14.007,15.999,24.305,28.085,32.06,40.078,47.867])/1.008 #IUPAC
    # with Ne
    # (atomic_wgts/hydrogen_wgt) = (C,N,O,Ne,Mg,Si,S,Ca,Ti)/H    
    #asplund = np.array([8.43,7.83,8.69,7.84,7.60,7.51,7.12,6.34,4.95])
    #mass_ratio = np.array([12.011,14.007,15.999,20.1797,24.305,28.085,32.06,40.078,47.867])/1.008 #IUPAC    
    ZX_sol = 0.0181 # (Z/X) Asplund et al. 2009
    XZ_k = np.multiply(10**(asplund-12.0),mass_ratio/ZX_sol)
    sal_a = np.sum(XZ_k)
    sal_b = 1 - sal_a
        
    ### Alpha+C+N
    wgts = asplund/np.sum(asplund)

    
    ### Replace bad values with solar
    for i in range(len(abund[:,0])):
        if abund[i,0] < -10. or abund[i,0] > 10. or np.isfinite(abund[i,0])==False:
            abund[i,0] = 0.0
        if abund[i,1] < -10. or abund[i,1] > 10. or np.isfinite(abund[i,1])==False:
            abund[i,1] = 0.0 
    
    feh = abund[0,0]
    feh_err = abund[0,1]
    cnalpha = abund[1:,0]
    
    cnafe = np.log10(np.sum(np.multiply(10**cnalpha,wgts)))
    salfeh = feh + np.log10(sal_a*10**(cnafe)+sal_b)    
    
    ### MC for Salaris Correction
    if mc:
        cnalpha_err = abund[1:,1]    
        nsamples = 1000
        salfehdist = 999999.0*np.ones(nsamples)
        
        noisyfeh = np.random.normal(feh,feh_err,nsamples)
        for i in range(nsamples):
            noisycnalpha = 999999.0*np.ones(len(cnalpha))
            for j in range(len(cnalpha)):
                noisycnalpha[j] = np.random.normal(cnalpha[j],cnalpha_err[j])
            
            cnafe = np.log10(np.sum(np.multiply(10**noisycnalpha,wgts)))
            salfehdist[i] = noisyfeh[i] + np.log10(sal_a*10**(cnafe)+sal_b)

        calc_salfeh_err = dln.mad(salfehdist)
        
        return salfeh, calc_salfeh_err

    return salfeh


def salariscna(abund,abunderr=None):
    '''
    Calculate the Salaris corrected [Fe/H] according to Salaris et al. 1993 with Piersanti et al. 2007 and 
    Asplund et al. 2009. Also C and N have been added to the alpha elements and Ne has been excluded.
        
    Inputs:
    ------
    abund: [4x2 array] first column is [Fe/H],[C/Fe],[N/Fe],[alpha/Fe]
       and second column is the errors
        
    Output:
    ------
    calc_salfeh: Salaris corrected metallicity
    calc_salfeh_err: error in corrected metallicity
    '''
        
    ### Salaris coefficients
    # (atomic_wgts/hydrogen_wgt) = (C,N,O,Mg,Si,S,Ca,Ti)/H
    #asplund = np.array([8.43,7.83,8.69,7.60,7.51,7.12,6.34,4.95])
    #mass_ratio = np.array([12.011,14.007,15.999,24.305,28.085,32.06,40.078,47.867])/1.008 #IUPAC
    # with Ne
    # (atomic_wgts/hydrogen_wgt) = (C,N,O,Ne,Mg,Si,S,Ca,Ti)/H    
    asplund = np.array([8.43,7.83,8.69,7.93,7.60,7.51,7.12,6.34,4.95])
    mass_ratio = np.array([12.011,14.007,15.999,20.1797,24.305,28.085,32.06,40.078,47.867])/1.008 #IUPAC    
    ZX_sol = 0.0181 # (Z/X) Asplund et al. 2009
    XZ_k = np.multiply(10**(asplund-12.0),mass_ratio/ZX_sol)
    sal_a = np.sum(XZ_k)
    sal_b = 1 - sal_a
        
    ### Alpha+C+N
    #wgts = asplund/np.sum(asplund)
    # add together the alpha weights
    wgts = np.array([asplund[0],asplund[1],np.sum(asplund[2:])])/np.sum(asplund)
    
    feh = abund[0]
    cnalpha = abund[1:]
    
    #cnafe = np.log10(np.sum((10**cnalpha)*wgts)))
    #salfeh = feh + np.log10(sal_a*10**(cnafe)+sal_b)

    cnafe = np.sum((10**cnalpha)*wgts)
    salfeh = feh + np.log10(sal_a*cnafe+sal_b)        
    
    # Uncertainties
    if abunderr is not None:
        feherr = abunderr[0]
        cnafe_err = np.sqrt(np.sum((10**cnalpha*np.log(10)*abunderr[1:]*wgts)**2))
        corr_err = cnafe_err/((sal_a*cnafe*sal_b)*np.log(10))
        salfeherr = np.sqrt(feherr**2+corr_err**2)

        return salfeh, salfeherr
    
    else:
        return salfeh
