#!/usr/bin/env python
# coding: utf-8

# In[1]:


#########################
### Data Manipulation ###
#########################

import numpy as np

def brtfntlmc_bins(apogee_data):
    '''
    Bin data by APOGEE Fields and Brt/Fnt RGB
    
    apogee_data: fits file of apogee data
    '''
    
    field_labels = ['30Dor','LMC1','LMC2','LMC3','LMC4','LMC5','LMC6','LMC7','LMC8','LMC9','LMC10','LMC11',
                    'LMC12','LMC13','LMC14','LMC15','LMC16','LMC17']
    
    binned_data = dict()
    
    #find faint and bright stars and bin apogee
    fnts = np.where(apogee_data['TARGET_NAME']=='FntRGB')
    brts = np.where(apogee_data['TARGET_NAME']=='BrtRGB')
    
    fnts_apogee = apogee_data[fnts]
    brts_apogee = apogee_data[brts]
    
    #find different fields in the fnt and brt bins and add to dictionary
    for i in range(len(field_labels)):
        fnt_key = '{}'.format(field_labels[i]) + '_fnt'
        fnt_val = np.where(fnts_apogee['FIELD'] == field_labels[i])
        binned_data[fnt_key] = fnt_val
        
        brt_key = '{}'.format(field_labels[i]) + '_brt'
        brt_val = np.where(brts_apogee['FIELD'] == field_labels[i])
        binned_data[brt_key] = brt_val
    
    return binned_data

def mad(dat):
    '''
    Calculate the median absolute deviation of 1d array-like object dat
    '''
    return np.median(np.absolute(dat - np.median(dat)))


# In[ ]:


#############################
### Field Mass Calculation ###
#############################

###Calculate APOGEE field mass with PARSEC isochrones
import astropy
from astropy.io import fits, ascii
from astropy.table import Table
import numpy as np
from scipy.interpolate import interp1d

def findclosestparsec(parsec_ascii_table,age,feh):
    '''
    Find closest PARSEC isochrone and return the age and z fraction
    
    parsec_ascii_table: ascii table of isochrone to use (label = 3, RGB stars)
    age: age of field to use
    feh: metallicity 
    '''
    
    #find closest parsec isochrone z fraction
#     parsec_mets = np.unique(np.asarray((np.log10(parsec_ascii_table['Zini']/0.02))))
#     met_index = np.absolute(parsec_mets-feh*np.ones(len(parsec_mets))).argmin()
#     feh_iso = round(0.02*(10**parsec_mets[met_index]),10)
    parsec_mets = np.unique(np.asarray(parsec_ascii_table['MH']))
    mets_index = np.absolute(parsec_mets-feh*np.ones(len(parsec_mets))).argmin()
    feh_iso = parsec_mets[mets_index]
    
    #find closest parsec isochrone age
    age = np.log10(age*(10**9))
    parsec_ages = np.unique(np.asarray(parsec_ascii_table['logAge']))
    age_index = np.absolute(parsec_ages-age*np.ones(len(parsec_ages))).argmin()
    age_iso = parsec_ages[age_index]
    
    return age_iso, feh_iso

def fieldmass(age_iso,feh_iso,absH,sf_brt,sf_fnt,parsec_ascii_table):
    
    '''
    Calculate Mass of APOGEE Field with PARSEC isochrones:
    
      SF_brt x N_brt + SF_fnt x N_fnt
    ----------------------------------- = field mass in mSol
    intIMF(minH_brt) - intIMF(maxH_fnt)
    
    age_iso: Age of PARSEC isochrone to use
    z_iso: Z fraction of PARSEC isochrone to use
    
    absH: arraylike object that contains all the Hmags for the whole field
    
    sf_brt: arraylike object that contains selection function for brt stars
    sf_fnt: arraylike object that contains selection function for fnt stars
    
    parsec_ascii_table: ascii table of isochrone to use (label = 3, RGB stars)
    
    '''
    
    single_iso = np.where((parsec_ascii_table['MH']==feh_iso)&
                          (parsec_ascii_table['logAge']==age_iso))#&
                          #(parsec_ascii_table['Hmag']<=np.max(absH)))
    
    x1 = parsec_ascii_table[single_iso]['Hmag']
    y1 = parsec_ascii_table[single_iso]['int_IMF']
    
    points = zip(x1, y1)
    
    points = sorted(points, key=lambda point: point[0])
    
    x2, y2 = zip(*points)
    x2 = np.asarray(x2)
    y2 = np.asarray(y2)
    
    Hcut = np.where(x2<-2)
    x2 = x2[Hcut]
    y2 = y2[Hcut]
    
    inter_iso = interp1d(x2,y2,kind='cubic',bounds_error=False,fill_value='extrapolate',assume_sorted=False)
    
    diff_IMF = np.absolute(inter_iso(np.max(absH))-inter_iso(np.min(absH)))
    
    if len(sf_fnt) == 0:
        numerator = sf_brt[0]*len(sf_brt) 
        
    else:
        numerator = sf_brt[0]*len(sf_brt) + sf_fnt[0]*len(sf_fnt)
    
    return numerator/diff_IMF


# In[ ]:


#################
#### Geometry ###
#################
#
#def LMCdisk_cart(ra, dec):
#    
#    '''
#    Calculate the position of stars in the LMC disk plane with 
#    center at the LMC center in cartesian coordinates (x, y).
#    This also calculates the distance to the individual stars.
#    
#    This follows van der Marel and Cioni 2001 
#    
#    Input
#    - ra: right ascension of stars
#    - dec: declination of stars
#    
#    Output
#    - x_m: x coordinate
#    - y_m: y coordinate
#    - dis: distance to LMC star
#    '''
#    alph0 = np.radians(82.25) #right ascension of center of LMC
#    delt0 = np.radians(-69.50) #declination of center of LMC
#    pa = np.radians(149.23+90.00) #146.37 #position angle of line of nodes
#    io = np.radians(25.86) #27.81 #inclination of LMC disk
#    d0 = 49.90 #distance to center of LMC
#    
#    #convert to radians
#    ra = np.radians(ra)
#    dec = np.radians(dec)
#    sd = np.sin(delt0)
#    cd = np.cos(delt0)
#    
#    cr = cd*np.cos(dec)*np.cos(ra-alph0)+sd*np.sin(dec)
#    srcp = -np.cos(dec)*np.sin(ra-alph0)
#    srsp = cd*np.sin(dec) - sd*np.cos(dec)*np.cos(ra-alph0)
#    dis = d0*np.cos(io)/(np.cos(io)*cr - np.sin(io)*np.cos(pa)*srsp + np.sin(io)*np.sin(pa)*srcp)
#    
#    x_m = dis*srcp
#    y_m = dis*(np.cos(io)*srsp + np.sin(io)*cr) - d0*np.sin(io)
#    
#    return x_m, y_m, dis
#
#import astropy
#from astropy.io import fits, ascii
#from astropy.table import Table
#import numpy as np
#from scipy.spatial import ConvexHull
#
## def apogee_field_area(field,data):
#    
##     """
##     This calculates the areal extent of an APOGEE field (kpc^2)
##     - field: string of name of field to get area
##     - data: table of data for field
##     """
#    
##     fld = np.where(data['FIELD']==field)
#    
##     ###calculate the centroid
##     x_m0, y_m0, dist = LMCdisk_cart(data[fld]['RA'],data[fld]['DEC'])
#    
##     points = []
##     for j in range(len(np.squeeze(fld))):
##         points.append([x_m0[j],y_m0[j]])
##     points = np.asarray(points)
#    
##     #find exterior points and area using Convex Hull algorithm 
##     hull = ConvexHull(points)
#    
##     return hull.volume
#
#
##################
#### STAR AGES ###
##################
#
## ### Age model
## def find_age(lt,k,feh,lg):
##     #['lt' 'k' 'feh' 'lg' 'ltlg' 'klg' 'fehlg' 'k2' 'lg2']
##     p = [2.34865658e+01,7.73561422e-01,3.31229224e+00,-3.92253575e-02,-4.71932940e+00,
##          -8.67142123e-01,-6.83193259e-01,1.93831669e-02,1.39372987e-01,8.31208860e-01]
##     age = p[0] + p[1]*lt + p[2]*k + p[3]*feh + p[4]*lg + p[5]*np.multiply(lt,lg) + p[6]*np.multiply(k,lg) + p[7]*np.multiply(feh,lg) + p[8]*np.square(k) + p[9]*np.square(lg)
##     return age
#
#####################################
#### Get Uncertainties: Add Noise ###
#####################################
#
#def add_noise(quant,quant_err):
#    '''
#    Add noise to data and return new values
#    
#    Parameters:
#        quant: 1d array-like data to add noise to
#        quant_err: 1d array-like object of errors for quant
#    
#    return: 1d array-like object of data with added noise
#    
#    '''
#    
#    noise = np.random.normal(0,quant_err)
#    new = quant + noise
#    return new
#
#def add_pos_noise(quant,quant_err):
#    '''
#    Add noise to data and return new values that are only positive
#    
#    Parameters:
#        quant: 1d array-like data to add noise to
#        quant_err: 1d array-like object of errors for quant
#    
#    return: 1d array-like object of data with added noise
#    
#    '''
#    
#    noise = np.absolute(np.random.normal(0,quant_err))
#    new = quant + noise
#    return new
#
#def sal_noise(cfe,cfeERR,nfe,nfeERR,feh,fehERR,mh,mhERR):
#    '''
#    Calculate noisy values for Salaris calculation with C and N.
#    This does not take into account actually plugging in [M/H],
#    for that use add_noise.
#    
#    Parameters:
#        cfe: 1d array-like object of carbon abundances
#        cfeERR: 1d array-like object of carbon abundance errors
#        nfe: 1d array-like object of nitrogen abundances
#        nfeERR: 1d array-like object of nitrogen abundance errors
#        feh: 1d array-like object of iron abundances
#        fehERR: 1d array-like object of iron abundance errors
#        
#    Return:
#        noisy Salaris correction ffac
#    '''
#    
#    sol_C = 0.28115244582676185 #solar carbon abundance
#    sol_N = 0.06901474154376043 #solar nitrogen abundance
#    
#    # Calculate [C/M] and [N/M] with respective errors
#    cm = cfe + feh - mh
#    nm = nfe + feh - mh
#    CMERR = np.sqrt((cfeERR)**2+(fehERR)**2+(mhERR)**2)
#    NMERR = np.sqrt((nfeERR)**2+(fehERR)**2+(mhERR)**2)
#    
#    # Calculate X fractions for C and N with respective errors
#    x_C = sol_C*10**(cm)
#    x_N = sol_N*10**(nm)
#    
#    x_CERR = 10**(cm)*np.log(10)*CMERR
#    x_NERR = 10**(nm)*np.log(10)*NMERR
#    
#    # Calcuate f factor in Salaris correction with respective errors
#    ffac = (x_C+x_N)/(sol_C+sol_N) #factor from Salaris correction
#    ffacERR = np.sqrt((x_CERR)**2+(x_NERR)**2)/(sol_C+sol_N)
#    
#    # Add noise to calculated f factor
#    ffacnoise = np.random.normal(0, np.absolute(0.434*(ffacERR/ffac))) 
#    
#    return ffac + ffacnoise
#
##############
#### Other ###
##############
#
#def sal(MH,aM):
#    '''
#    Calculate the Salaris correction to the overall metallicity
#    '''
#    return MH + np.log(0.638*(10**(aM))+0.362)
#
##calculate absolute mag
#def absmag(magnitude,distance):
#    '''
#    - magnitude: apparent magnitude of star
#    - distance: distance to star in kpc
#    Calculate the absolute magnitude of star
#    '''
#    absm = []
#    absm.append(magnitude-5.0*np.log10(distance*1000)+5.0)
#    absm = np.squeeze(np.array(absm))
#    return absm
