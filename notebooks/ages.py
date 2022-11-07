#!/usr/bin/env python

import os
import numpy as np
from astropy.table import Table,vstack


def age(tab,iso=None):
    """
    Get age for a star using isochrones.
    """

    data = tab.copy()
    for c in data.colnames: data[c].name = c.lower()
    
    if iso is None:
        iso = Table.read('/Users/nidever/isochrone/parsec_gaiaedr3_2mass/parsec_gaiaedr3_2mass.fits')
        # MH = -2.0 to +0.25 in 0.25 dex steps
        # age = 500 Myr to 10 Gyr in 500 Myr steps
    for c in iso.colnames: iso[c].name = c.lower()
    if 'teff' not in iso.colnames:
        iso['teff'] = 10**iso['logte']
    if 'feh' not in iso.colnames:
        iso['feh'] = iso['mh']
    if 'agemyr' not in iso.colnames:
        iso['agemyr'] = 10**iso['logage']/1e6

    # Star the output table
    dt = [('agemyr',float),('feh',float),('label',float),('teff',float),('logg',float),
          ('obs',(float,9)),('resid',(float,9)),('chisq',float)]
    out = np.zeros(len(data),dtype=np.dtype(dt))
    out = Table(out)
    for j in range(8): out[j][:] = np.nan
    
    # Loop over stars
    for i in range(len(data)):
        dist = data['distance'][i]  # distance in kpc
        distmod = 5*np.log10(dist)+10
        ext = data['ext'][i]
        
        # Array of observables
        #  Gaia EDR3 and 2MASS photometry, Teff, logg and [Fe/H]
        colnames = ['g_bp','g','g_rp','j','h','k','teff','logg','salaris_fe_h']
        obs = np.zeros(len(colnames),float)
        for j in range(len(colnames)): obs[j]=data[colnames[j]][i]
        errcolnames = ['g_bp_err','g_err','g_rp_err','j_err','h_err','k_err','teff_err','logg_err','salaris_fe_h_err']
        obserr = np.zeros(len(errcolnames),float)
        for j in range(len(errcolnames)): obserr[j]=data[errcolnames[j]][i]
        # Make sure the uncertainties are realistic
        obserr[0:6] = np.maximum(obserr[0:6],5e-3)  # cap phot errors at 5e-3
        obserr[6] = np.maximum(obserr[6],25)        # cap teff errors at 25 K
        obserr[7] = np.maximum(obserr[7],0.05)      # cap logg errors at 0.05
        obserr[8] = np.maximum(obserr[8],0.1)       # cap feh errosr at 0.1

        # Select region around the star in Teff, logg and [Fe/H]
        ind, = np.where((np.abs(10**iso['logte']-data['teff'][i]) < 100) &
                        (np.abs(iso['logg']-data['logg'][i]) < 0.2) &
                        (np.abs(iso['feh']-data['salaris_fe_h'][i]) < 0.3))
        if len(ind)==0:
            ind, = np.where((np.abs(10**iso['logte']-data['teff'][i]) < 300) &

                            (np.abs(iso['logg']-data['logg'][i]) < 0.4) &
                            (np.abs(iso['feh']-data['salaris_fe_h'][i]) < 0.5))
        if len(ind)==0:
            ind, = np.where((np.abs(10**iso['logte']-data['teff'][i]) < 500) &
                            (np.abs(iso['logg']-data['logg'][i]) < 0.6) &
                            (np.abs(iso['feh']-data['salaris_fe_h'][i]) < 0.8))            
        if len(ind)==0:
            print('No close isochrone points.  Skipping')
            continue
        iso1 = iso[ind]
        # Get array of observables for the isochrone data
        colnames = ['g_bpmag','gmag','g_rpmag','jmag','hmag','ksmag','teff','logg','feh']
        iso_obs = np.zeros((len(ind),len(colnames)),float)
        for j in range(len(colnames)): iso_obs[:,j] = iso1[colnames[j]]

        # Extinction and distance modulus
        for j in range(6):
            # Extinction
            iso_obs[:,j] += ext[j]
            # Distance modulus
            iso_obs[:,j] += distmod
                    
        # Chi-squared
        chisq = np.sum((iso_obs - obs.reshape(1,-1))**2 / obserr.reshape(1,-1), axis=1)
        # Find best isochrone point
        bestind = np.argmin(chisq)
        resid = obs-iso_obs[bestind,:]
        
        # Output table
        out['agemyr'][i] = iso1['agemyr'][bestind]
        out['feh'][i] = iso1['feh'][bestind]
        out['label'][i] = iso1['label'][bestind]
        out['teff'][i] = iso1['teff'][bestind]
        out['logg'][i] = iso1['logg'][bestind]
        out['obs'][i] = iso_obs[bestind,:]
        out['resid'][i] = resid
        out['chisq'][i] = np.min(chisq)

        print('%6d %8.2f %8.2f %4s %8.2f' % (i+1,out['agemyr'][i],out['feh'][i],out['label'][i],out['chisq'][i]))

    return out
