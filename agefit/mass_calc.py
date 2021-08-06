#!/usr/bin/env python
# coding: utf-8

# In[2]:


###Calculate APOGEE field mass with PARSEC isochrones
import astropy
from astropy.io import fits, ascii
from astropy.table import Table
import numpy as np
from scipy.interpolate import interp1d

def findclosestparsec(age,metallicity,parsec_ascii_table):
    
    #find closest parsec isochrone z fraction
    age = np.log10(age*(10**9))
    parsec_mets = np.unique(np.asarray((np.log10(parsec_ascii_table['Zini']/0.02))))
    met_index = np.absolute(parsec_mets-metallicity*np.ones(len(parsec_mets))).argmin()
    z_iso = round(0.02*(10**parsec_mets[met_index]),10)
    
    #find closest parsec isochrone age
    parsec_ages = np.unique(np.asarray(parsec_ascii_table['logAge']))
    age_index = np.absolute(parsec_ages-age*np.ones(len(parsec_ages))).argmin()
    age_iso = parsec_ages[age_index]
    
    #pick out isochrone
    #single_iso = np.where((parsec_ascii_table['Zini']==z_iso)&
    #                      (parsec_ascii_table['logAge']==age_iso)&
    #                      (parsec_ascii_table['Hmag']<=maxabsH))
    
    return age_iso, z_iso

def fieldmass(age_iso,z_iso,maxabsH,minabsH,selectfunc,number,parsec_ascii_table):
    
    single_iso = np.where((parsec_ascii_table['Zini']==z_iso)&
                          (parsec_ascii_table['logAge']==age_iso))#&
                          #(parsec_ascii_table['Hmag']<=maxabsH))
    
    #upto = np.where(parsec_ascii_table[single_iso]['Hmag']==np.min(parsec_ascii_table[single_iso]['Hmag']))
    #new_parsec = parsec_ascii_table[single_iso][0:int(np.squeeze(upto))]
    
    x1 = parsec_ascii_table[single_iso]['Hmag']
    y1 = parsec_ascii_table[single_iso]['int_IMF']
    
    points = zip(x1, y1)
    
    points = sorted(points, key=lambda point: point[0])
    
    x2, y2 = zip(*points)
    
    inter_iso = interp1d(x2,y2,kind='cubic',bounds_error=False,fill_value='extrapolate')#,assume_sorted=False)
    
    #inter_iso = interp1d(parsec_ascii_table[single_iso]['Hmag'][::-1],
                         #parsec_ascii_table[single_iso]['int_IMF'][::-1],kind='cubic',
                         #bounds_error=False,fill_value='extrapolate',assume_sorted=False) 
                          
                          #fill_value=(np.nan,np.nan)
    
#     inter_iso = interp1d(new_parsec['Hmag'][::-1],new_parsec['int_IMF'][::-1],kind='cubic',bounds_error=False,
#                          fill_value='extrapolate',assume_sorted=False) #fill_value=(np.nan,np.nan)
    
    diff_IMF = np.absolute(inter_iso(maxabsH)-inter_iso(minabsH))
    
    return (selectfunc*number)/diff_IMF


# In[ ]:


# def apogee_field_mass(age,metallicity,maxabsH,minabsH,selectfunc,number,parsec_path):
    
#     """
#     This is used to calculate the mass of stars in a field
#     - age: age of stars to use to pick out isochrone
#     - metallicity: metallicity of stars to use to pick out isochrone
#     - maxabsH: max value of H for the field 
#     - minabsH: min vale of H for the field
#     - selectfunc: selection function for the field
#     - number: number of stars observed
#     - parsec_path: dat file of PARSEC isochrones
#     """
    
#     ###Load in isochrones from parsec
#     parsecall = ascii.read(parsec_path, format='basic', delimiter='\s')
#     rgb = np.where(parsecall['label']==3)
#     parsec = parsecall[rgb]
    
#     #find closest parsec isochrone z fraction
#     age = np.log10(age*(10**9))
#     parsec_mets = np.unique(np.asarray((np.log10(parsec['Zini']/0.02))))
#     met_index = np.absolute(parsec_mets-metallicity*np.ones(len(parsec_mets))).argmin()
#     z_iso = 0.02*(10**parsec_mets[met_index])
    
#     #find closest parsec isochrone age
#     parsec_ages = np.unique(np.asarray(parsec['logAge']))
#     age_index = np.absolute(parsec_ages-age*np.ones(len(parsec_ages))).argmin()
#     age_iso = parsec_ages[age_index]
    
#     #pick out isochrone, interpolate, and find difference
#     single_iso = np.where((parsec['Zini']==z_iso)&(parsec['logAge']==age_iso)&(parsec['Hmag']<=maxabsH))

#     upto = np.where(parsec[single_iso]['Hmag']==min(parsec[single_iso]['Hmag']))
#     new_parsec = parsec[single_iso][0:int(np.squeeze(upto))]
    
#     inter_iso = interp1d(new_parsec['Hmag'][::-1],new_parsec['int_IMF'][::-1],kind='cubic',bounds_error=False,
#                          fill_value='extrapolate',assume_sorted=False) #fill_value=(np.nan,np.nan)
    
#     diff_IMF = np.absolute(inter_iso(maxabsH)-inter_iso(minabsH))
    
#     return (selectfunc*number)/diff_IMF

